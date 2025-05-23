import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import AutoImageProcessor, CLIPVisionModel, Dinov2Model
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


### image embeddings
def get_image_embeds(
    image_encoder,
    image_inputs: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or image_encoder.device
    dtype = dtype or image_encoder.dtype

    # Get image embeddings
    image_inputs = image_inputs.to(dtype=dtype, device=device)
    image_embeds = image_encoder(image_inputs, output_hidden_states=True)

    # Use pooled output of CLIPImageModel
    pooled_image_embeds = image_embeds.pooler_output
    pooled_image_embeds = pooled_image_embeds.to(dtype=dtype, device=device)

    # Use last hidden state of CLIPImageModel
    image_embeds = image_embeds.last_hidden_state[:,1:,:]
    image_embeds = image_embeds.to(dtype=dtype, device=device)

    return pooled_image_embeds, image_embeds

def pad_and_stack(tensors, dim=0):

    max_size = max(t.shape[dim] for t in tensors)
    padded = []
    for t in tensors:
        pad_size = max_size - t.shape[dim]
        if pad_size > 0:
            pad = []
            for i in reversed(range(t.dim())):
                if i == dim:
                    pad.extend([0, pad_size])
                else:
                    pad.extend([0, 0])
            t = F.pad(t, pad)
        padded.append(t)
    
    stacked = torch.stack(padded, dim=0)

    return stacked, max_size


class FusionModule(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    @register_to_config
    def __init__(
        self,
        attn_heads: int = 16,
        dim_head: int = 256,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        dino_model_name: str = "facebook/dinov2-large",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        self.pretrained_module_names = ["clip_image_processor", "dino_image_processor", "clip_image_encoder", "dino_image_encoder"]
        self.clip_image_processor = AutoImageProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.dino_image_processor = AutoImageProcessor.from_pretrained(dino_model_name, use_fast=True)

        self.clip_image_encoder = CLIPVisionModel.from_pretrained(clip_model_name, cache_dir=cache_dir, device_map="cpu")
        self.dino_image_encoder = Dinov2Model.from_pretrained(dino_model_name, cache_dir=cache_dir, device_map="cpu")
    
        dino_hidden_dim = self.dino_image_encoder.config.hidden_size
        clip_hidden_dim = self.clip_image_encoder.config.hidden_size
        assert dino_hidden_dim == clip_hidden_dim

        self.content_mapper = nn.Sequential(
            BasicTransformerBlock(
                dim=dino_hidden_dim,
                num_attention_heads=attn_heads,
                attention_head_dim=dim_head,
            ),
            nn.LayerNorm(dino_hidden_dim, elementwise_affine=False, eps=1e-6),
        )
        self.style_mapper = nn.Sequential(
            BasicTransformerBlock(
                dim=clip_hidden_dim,
                num_attention_heads=attn_heads,
                attention_head_dim=dim_head,
            ),
            nn.LayerNorm(clip_hidden_dim, elementwise_affine=False, eps=1e-6),
        )
        self.pooled_style_mapper = nn.Sequential(
            FeedForward(dim=clip_hidden_dim),
            nn.LayerNorm(clip_hidden_dim, elementwise_affine=False, eps=1e-6),
        )

        # Initialize parameters
        self.content_mapper.apply(self._init_weights)
        self.style_mapper.apply(self._init_weights)
        self.pooled_style_mapper.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def state_dict(self, *args, **kwargs):

        sd = super().state_dict(*args, **kwargs)
        for key in list(sd.keys()):
            if any([key.startswith(module_name) for module_name in self.pretrained_module_names]):
                sd.pop(key)
                
        return sd

    def get_cond_ids(self, n_glyphs, l, type="content"):
        
        if type == "content":
            scale = int(l**0.5)
            cond_ids_list = []
            for seq_idx in range(n_glyphs):
                cond_ids = torch.zeros(scale, scale, 3) # [s, s, 3]
                cond_ids[..., 0] = seq_idx # use seq_idx to divide glyphs
                cond_ids[..., 1] = cond_ids[..., 1] + torch.arange(scale)[:, None]
                cond_ids[..., 2] = cond_ids[..., 2] + torch.arange(scale)[None, :]

                cond_ids = cond_ids.view(-1, 3) # (l, 3)
                cond_ids_list.append(cond_ids)
            cond_ids = torch.cat(cond_ids_list, dim=0) # (n*l, 3)
        elif type == "style":
            cond_ids = torch.zeros(n_glyphs*l, 3) # (m*l, 3)
        else:
            raise ValueError(f"Invalid type: {type}")

        return cond_ids

    def forward(
        self,
        content_images: List[List[Image.Image]],
        style_images: List[List[Image.Image]],
        *args, **kwargs,
    ):
        content_embeds_list = []
        style_embeds_list = []
        pooled_style_embeds_list = []
        for content_image, style_image in zip(content_images, style_images):
            content_image_inputs = self.dino_image_processor(images=content_image, return_tensors="pt").pixel_values
            style_image_inputs = self.clip_image_processor(images=style_image, return_tensors="pt").pixel_values

            # Get content embeddings
            _, content_embeds = get_image_embeds(
                image_encoder=self.dino_image_encoder,
                image_inputs=content_image_inputs,
            )
            content_embeds = self.content_mapper(content_embeds) # (n, l, d)

            # Get style embeddings
            pooled_style_embeds, style_embeds = get_image_embeds(
                image_encoder=self.clip_image_encoder,
                image_inputs=style_image_inputs,
            )
            pooled_style_embeds = self.pooled_style_mapper(pooled_style_embeds) # (m, d)
            pooled_style_embeds = pooled_style_embeds.mean(dim=0, keepdim=False) # (d)
            style_embeds = self.style_mapper(style_embeds) # (m, l, d)

            content_embeds_list.append(content_embeds)
            style_embeds_list.append(style_embeds)
            pooled_style_embeds_list.append(pooled_style_embeds)

        b, l, d = len(content_embeds_list), content_embeds_list[0].shape[1], content_embeds_list[0].shape[2]

        content_embeds, c_max_size = pad_and_stack(content_embeds_list, dim=0) # (b, N, l, d)
        style_embeds, s_max_size = pad_and_stack(style_embeds_list, dim=0) # (b, M, l, d)
        content_embeds = content_embeds.view(b, -1, d) # (b, N*l, d)
        style_embeds = style_embeds.view(b, -1, d) # (b, M*l, d)
        pooled_style_embeds = torch.stack(pooled_style_embeds_list, dim=0) # (b, d)

        # Get RoPE embeddings
        c_cond_ids = self.get_cond_ids(c_max_size, l, type="content").to(content_embeds)
        s_cond_ids = self.get_cond_ids(s_max_size, l, type="style").to(style_embeds)
        cond_ids = (c_cond_ids, s_cond_ids)

        return content_embeds, style_embeds, pooled_style_embeds, cond_ids