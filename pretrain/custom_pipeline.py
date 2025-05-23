import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor 
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps, FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor

from custom_model import CustomTransformer2DModel
from custom_encoder import FusionModule
from custom_scheduler import StochasticRFOvershotDiscreteScheduler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomPipeline(FluxPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        fusion_module: FusionModule,
        transformer: CustomTransformer2DModel,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, StochasticRFOvershotDiscreteScheduler],
    ):
        super(FluxPipeline, self).__init__()

        self.register_modules(
            vae=vae,
            fusion_module=fusion_module,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64
    
    def check_inputs(
        self,
        height,
        width,
        c_refs_pil,
        s_refs_pil,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        if len(c_refs_pil) == 0 or len(s_refs_pil) == 0:
            raise ValueError("At least one content and style image is required.")
        
    @staticmethod
    def _prepare_latent_image_ids(n_glyphs, height, width, device, dtype):
        latent_image_ids_list = []
        for seq_idx in range(n_glyphs):
            latent_image_ids = torch.zeros(height // 2, width // 2, 3)
            latent_image_ids[..., 0] = seq_idx
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

            latent_image_ids = latent_image_ids.reshape(-1, 3) # (l, 3)
            latent_image_ids_list.append(latent_image_ids)

        latent_image_ids = torch.cat(latent_image_ids_list, dim=0).to(device=device, dtype=dtype) # (n*l, 3)

        return latent_image_ids
    
    @staticmethod
    def _pack_latents(latents, num_channels_latents, height, width):
        bsz, n_glyphs = latents.shape[:2]
        latents = latents.view(bsz, n_glyphs, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
        latents = latents.reshape(bsz, n_glyphs, (height // 2) * (width // 2), num_channels_latents * 4)
        latents = latents.reshape(bsz, -1, latents.shape[-1])

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        height = height // vae_scale_factor
        width = width // vae_scale_factor

        bsz = latents.shape[0]
        channels = latents.shape[-1]
        latents = latents.reshape(bsz, -1, height*width, channels)

        n_glyphs = latents.shape[1]
        latents = latents.view(bsz, n_glyphs, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
        latents = latents.reshape(bsz, n_glyphs, channels // (2 * 2), height * 2, width * 2) # (b, n, c, h, w)

        return latents
        
    def prepare_latents(
        self,
        bsz,
        n_glyphs,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (bsz, n_glyphs, num_channels_latents, height, width)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, num_channels_latents, height, width) # (b, n*l, d)

        latent_image_ids = self._prepare_latent_image_ids(n_glyphs, height, width, device, dtype)

        return latents, latent_image_ids
    
    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        c_refs_pil: Union[List[List[Image.Image]]],
        s_refs_pil: Optional[List[List[Image.Image]]],
        lamda_gate: float = 1.0,
        cfg_scale: float = 3.5,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            c_refs_pil,
            s_refs_pil,
        )

        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        bsz = len(c_refs_pil)
        n_glyphs = max(len(item) for item in c_refs_pil)

        # 2. Get embeddings
        device = self._execution_device
        dtype = self.transformer.dtype
        (
            content_embeds,
            style_embeds,
            pooled_style_embeds, 
            cond_ids,
        ) = self.fusion_module(
            content_images=c_refs_pil,
            style_images=s_refs_pil,
        )

        neg_c_refs_pil = [[Image.new("RGB", img.size, (255, 255, 255)) for img in item] for item in c_refs_pil]
        neg_s_refs_pil = [[Image.new("RGB", img.size, (255, 255, 255)) for img in item] for item in s_refs_pil]
        (
            neg_content_embeds,
            neg_style_embeds,
            neg_pooled_style_embeds, 
            neg_cond_ids,
        ) = self.fusion_module(
            content_images=neg_c_refs_pil,
            style_images=neg_s_refs_pil,
        )

        # 3. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            bsz,
            n_glyphs,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (height//16) * (width//16)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    pooled_projections=pooled_style_embeds,
                    content_hidden_states=content_embeds,
                    style_hidden_states=style_embeds,
                    cond_ids=cond_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    lamda_gate=lamda_gate,
                )[0]

                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    pooled_projections=neg_pooled_style_embeds,
                    content_hidden_states=neg_content_embeds,
                    style_hidden_states=neg_style_embeds,
                    cond_ids=neg_cond_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    lamda_gate=lamda_gate,
                )[0]

                noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor) # (b, n, c, h, w)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.reshape(-1, *latents.shape[2:]) # (b*n, c, h, w)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            images = []
            for i in range(bsz):
                image_batch = image[i*n_glyphs:(i+1)*n_glyphs][:len(c_refs_pil[i])]
                images.append(image_batch)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return FluxPipelineOutput(images=images)
