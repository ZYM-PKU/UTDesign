#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import time
import gc
import logging
import math
import os
import sys
sys.path.append(os.getcwd())
import shutil
import random
import argparse
import itertools
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from mmengine.config import Config

import torch
import torch.utils.checkpoint
import torch.utils.data as data
import numpy as np

import lpips
import transformers
import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    convert_unet_state_dict_to_peft,
)
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from custom_encoder import FusionModule
from custom_model import CustomTransformer2DModel
from custom_pipeline import CustomPipeline
from custom_dataset.merged_dataset import MergedGlyphDataset
from custom_dataset.utils.helper import composite_image, general_collate_fn
from trans_vae.custom_model import TransAutoencoderKL
from torchvision.utils import save_image

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"


def parse_config(path=None):
    
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str)
        args = parser.parse_args()
        path = args.config_dir
    config = Config.fromfile(path)
    
    config.config_dir = path

    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        config.local_rank = -1

    return config

def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_model_hook_partial(models, weights, output_dir):
    if accelerator.is_main_process and len(weights) > 0:
        transformer_lora_layers_to_save = None
        for model in models:
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
            elif isinstance(unwrap_model(model), type(unwrap_model(fusion_module))):
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "fusion_module"))
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        CustomPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
        )

def load_model_hook_partial(models, input_dir):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
            lora_state_dict = CustomPipeline.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        elif isinstance(unwrap_model(model), type(unwrap_model(fusion_module))):
            load_model = FusionModule.from_pretrained(input_dir, subfolder="fusion_module")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict(), strict=False)
            del load_model
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

def initialize_all_models(config, accelerator):

    logger.info(f"[MODEL] load vae")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.get("cache_dir", None),
    )
    vae.enable_slicing()

    logger.info(f"[MODEL] load mmdit")
    transformer: CustomTransformer2DModel = CustomTransformer2DModel.from_pretrained(
        config.stage1_model_path,
        subfolder="transformer",
    )
    if config.get("fuse_lora_paths", None):
        for idx, fuse_lora_path in enumerate(config.fuse_lora_paths):
            logger.info(f"loading lora from {fuse_lora_path}")
            lora_state_dict = CustomPipeline.lora_state_dict(fuse_lora_path)
            CustomPipeline.load_lora_into_transformer(lora_state_dict, None, transformer, f"fuse_lora_{idx}")
        transformer.fuse_lora()
        transformer.unload_lora()

    ref_transformer = copy.deepcopy(transformer)

    logger.info(f"[MODEL] load fusion module")
    fusion_module: FusionModule = FusionModule.from_pretrained(
        config.fusion_model_path,
        subfolder="fusion_module",
        low_cpu_mem_usage=False, # disable this to avoid BUGs
    )

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    ref_transformer.requires_grad_(False)
    fusion_module.requires_grad_(False)
    if not config.get("freeze_fusion", False):
        fusion_module.perceiver_resampler.requires_grad_(True)

    logger.info(f"[MODEL] move models to cuda")
    vae.to(accelerator.device, dtype=config.weight_dtype)
    ref_transformer.to(accelerator.device, dtype=config.weight_dtype)

    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        ref_transformer.enable_gradient_checkpointing()
        fusion_module.enable_gradient_checkpointing()
    
    # now we will add new LoRA weights to the attention layers
    logger.info(f"[MODEL] add lora in mmdit")
    target_modules = []
    # transformer_blocks
    module_names = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"]
    for name, _ in transformer.transformer_blocks.named_modules():
        if any([name.endswith(n) for n in module_names]):
            target_modules.append("transformer_blocks." + name)
    # single_transformer_blocks
    module_names = ["to_k", "to_q", "to_v", "proj_out"]
    for name, _ in transformer.single_transformer_blocks.named_modules():
        if any([name.endswith(n) for n in module_names]):
            target_modules.append("single_transformer_blocks." + name)

    transformer_lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # Make sure the trainable params are in float32.  
    logger.info(f"[MODEL] cast training params to fp32")
    if config.cast_training_params:
        models = [transformer, fusion_module]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    
    return vae, transformer, ref_transformer, fusion_module

def get_trainable_params(transformer, fusion_module, config, accelerator):

    # Optimization parameters
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_parameters, "lr": config.lr}
    params_to_optimize = [transformer_parameters_with_lr]

    if not config.get("freeze_fusion", False):
        fusion_module_parameters = list(filter(lambda p: p.requires_grad, fusion_module.parameters()))
        fusion_module_parameters_with_lr = {"params": fusion_module_parameters, "lr": config.lr_fusion}
        params_to_optimize.append(fusion_module_parameters_with_lr)
    
    if accelerator.is_main_process:
        for i, param_set in enumerate(params_to_optimize):
            num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
            accelerator.print(f"Trainable Params Set {i}: {num_params:.2f}M")
    
    return params_to_optimize

def get_sigmas(timesteps, accelerator, noise_scheduler_copy, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def log_validation(val_dataset, pipeline, config, accelerator):
    logger.info(f"Running validation...")
    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    # run inference
    image_logs = []
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
    for case_id in config.validation_case_indices:
        val_case = val_dataset[case_id]
        crop_gt_pil = val_case["crop_img"]
        crop_bg_pil = val_case["crop_bg"]
        prompt = val_case["prompt"]
        label = val_case["label"]
        c_refs_w_pil = val_case["c_refs_pil"]
        seg_bboxs = val_case["seg_bboxs"]
        r_crop_bbox = val_case["r_crop_bbox"]
 
        with torch.autocast(accelerator.device.type, dtype=torch.bfloat16):
            output = pipeline(
                height=config.resolution,
                width=config.resolution,
                c_refs_pil=[c_refs_w_pil],
                bg_pil=[crop_bg_pil],
                reg_bboxs=[r_crop_bbox],
                prompt=[prompt],
                label=[label],
                generator=generator,
            )
            images = output.images[0]   # list of list of PIL

        c_refs_concat = np.concatenate([np.array(img) for img in c_refs_w_pil], axis=1)
        c_refs_concat_pil = Image.fromarray(c_refs_concat)
        crop_output_pil = composite_image(crop_bg_pil, images, seg_bboxs)

        image_logs.append(
            { 
                "bg": crop_bg_pil,
                "c_refs": c_refs_concat_pil,
                "gt": crop_gt_pil,
                "output": crop_output_pil,
            }
        )

    for tracker in accelerator.trackers:
        assert tracker.name == "wandb"
        
        dict_to_log = {}
        for sample_idx, log in enumerate(image_logs):
            formatted_images = []

            for key, value in log.items():
                if isinstance(value, Image.Image):
                    value = wandb.Image(value, caption=key)
                    formatted_images.append(value)

            dict_to_log[f"sample_{sample_idx}"] = formatted_images

        tracker.log(dict_to_log)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train(global_step, first_epoch, config, accelerator, \
           train_dataloader, optimizer, lr_scheduler, transformer, ref_transformer, fusion_module, vae, \
            noise_scheduler_copy, noise_scheduler, progress_bar):

    for epoch in range(first_epoch, config.num_train_epochs):
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        transformer.train()
        fusion_module.train()
        for step_idx, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, fusion_module]
            with accelerator.accumulate(models_to_accumulate):

                gt_w = batch["win_gt"].to(dtype=vae.dtype) # (b, n, 3, 256, 256)
                gt_l = batch["lose_gt"].to(dtype=vae.dtype) # (b, n, 3, 256, 256)
                gt = torch.cat([gt_w, gt_l], dim=0)
                c_refs_pil = batch["c_refs_pil"]
                s_refs_pil = batch["s_refs_pil"]
                bg_pil = batch["bg"]
                r_crop_bboxs = batch["r_crop_bbox"]
                prompt = batch["prompt"]
                label = batch["label"]

                bsz, n_glyphs = gt_w.shape[:2]
                valid_lens = [len(labels) for labels in batch["labels"]]
                valid_mask = torch.zeros(bsz, n_glyphs)
                for i, valid_len in enumerate(valid_lens):
                    valid_mask[i, :valid_len] = 1
                valid_mask = torch.cat([valid_mask, valid_mask], dim=0)
                
                # Get conditional embeddings
                content_embeds, style_embeds, pooled_style_embeds, cond_ids = fusion_module(
                    content_images=c_refs_pil,
                    bg_images=bg_pil,
                    reg_bboxs=r_crop_bboxs,
                    prompts=prompt,
                    labels=label,
                )

                # Convert images to latent space
                gt = gt.reshape(-1, *gt.shape[2:]) # (b*n, 3, 256, 256)
                model_input = vae.encode(gt).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=content_embeds.dtype)
                model_input = model_input.reshape(bsz*2, n_glyphs, *model_input.shape[1:]) # (b, n, c, h, w)
        
                latent_image_ids = CustomPipeline._prepare_latent_image_ids(
                    n_glyphs,
                    model_input.shape[3],
                    model_input.shape[4],
                    accelerator.device,
                    cond_ids[0].dtype,
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=config.logit_mean,
                    logit_std=config.logit_std,
                    mode_scale=config.mode_scale,
                )

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                timesteps = torch.cat([timesteps, timesteps], dim=0)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, accelerator, noise_scheduler_copy, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = CustomPipeline._pack_latents(
                    noisy_model_input,
                    num_channels_latents=model_input.shape[2],
                    height=model_input.shape[3],
                    width=model_input.shape[4],
                )

                # double the conds to match the input
                content_embeds = torch.cat([content_embeds, content_embeds], dim=0)
                style_embeds = torch.cat([style_embeds, style_embeds], dim=0)
                pooled_style_embeds = torch.cat([pooled_style_embeds, pooled_style_embeds], dim=0)

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    content_hidden_states=content_embeds,
                    style_hidden_states=style_embeds,
                    pooled_projections=pooled_style_embeds,
                    timestep=timesteps / 1000,
                    img_ids=latent_image_ids,
                    cond_ids=cond_ids,
                    return_dict=False,
                )[0]

                # reference model
                with torch.no_grad():
                    # Get editing conditional embeddings
                    content_embeds, style_embeds, pooled_style_embeds, cond_ids = fusion_module(
                        content_images=c_refs_pil,
                        style_images=s_refs_pil,
                    )
                    # double the conds to match the input
                    content_embeds = torch.cat([content_embeds, content_embeds], dim=0).to(dtype=config.weight_dtype).detach()
                    style_embeds = torch.cat([style_embeds, style_embeds], dim=0).to(dtype=config.weight_dtype).detach()
                    pooled_style_embeds = torch.cat([pooled_style_embeds, pooled_style_embeds], dim=0).to(dtype=config.weight_dtype).detach()
                    packed_noisy_model_input = packed_noisy_model_input.to(dtype=config.weight_dtype)
                    timesteps = timesteps.to(dtype=config.weight_dtype)

                    model_pred_ref = ref_transformer(
                        hidden_states=packed_noisy_model_input,
                        content_hidden_states=content_embeds,
                        style_hidden_states=style_embeds,
                        pooled_projections=pooled_style_embeds,
                        timestep=timesteps / 1000,
                        img_ids=latent_image_ids,
                        cond_ids=cond_ids,
                        return_dict=False,
                    )[0].detach()

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input 

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
                model_pred = CustomPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[3] * vae_scale_factor / 2),
                    width=int(model_input.shape[4] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )
                model_pred_ref = CustomPipeline._unpack_latents(
                    model_pred_ref,
                    height=int(model_input.shape[3] * vae_scale_factor / 2),
                    width=int(model_input.shape[4] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )
                # model loss
                model_loss = (model_pred.float() - target.float()) ** 2
                model_loss = torch.mean(
                    (weighting.float() * model_loss).reshape(bsz*2, n_glyphs, -1),
                    -1,
                )
                model_loss = model_loss * valid_mask.to(model_loss)
                model_loss = model_loss.mean(dim=-1)
                model_loss_w, model_loss_l = model_loss.chunk(2, dim=0)
                raw_model_loss = 0.5 * (model_loss_w.mean() + model_loss_l.mean())
                model_diff = model_loss_w - model_loss_l

                # reference model loss
                model_loss_ref = (model_pred_ref.float() - target.float()) ** 2
                model_loss_ref = torch.mean(
                    (weighting.float() * model_loss_ref).reshape(bsz*2, n_glyphs, -1),
                    -1,
                )
                model_loss_ref = model_loss_ref * valid_mask.to(model_loss_ref)
                model_loss_ref = model_loss_ref.mean(dim=-1)
                model_loss_ref_w, model_loss_ref_l = model_loss_ref.chunk(2, dim=0)
                raw_model_loss_ref = model_loss_ref.mean()
                model_diff_ref = model_loss_ref_w - model_loss_ref_l

                scale_term = -0.5 * config.beta_dpo
                inside_term = scale_term * (model_diff - model_diff_ref)
                inside_term = torch.nan_to_num(inside_term, nan=0.0, posinf=0.0, neginf=0.0)
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                loss = (-1 * F.logsigmoid(inside_term)).clip(0.0, 5.0).mean()

                avg_loss = accelerator.reduce(loss, reduction='mean').item()
                avg_model_mse = accelerator.reduce(raw_model_loss, reduction='mean').item()
                avg_ref_mse = accelerator.reduce(raw_model_loss_ref, reduction='mean').item()
                avg_acc = accelerator.reduce(implicit_acc, reduction='mean').item()

                train_loss += avg_loss
                implicit_acc_accumulated += avg_acc

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(transformer.parameters(), fusion_module.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process or accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % config.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(config.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= config.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                            extra_kwargs = {'exclude_frozen_parameters': True}
                        else:
                            extra_kwargs = {}
                        accelerator.save_state(save_path, **extra_kwargs)
                        logger.info(f"[Rank{accelerator.process_index}] saved state to {save_path}", main_process_only=False)

                        if accelerator.is_main_process and accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                            transformer_lora_layers_to_save = get_peft_model_state_dict(unwrap_model(transformer))
                            CustomPipeline.save_lora_weights(
                                save_path,
                                transformer_lora_layers=transformer_lora_layers_to_save,
                            )
                            unwrap_model(fusion_module).save_pretrained(os.path.join(save_path, "fusion_module"))
                            
                logs = {
                    "train/loss": train_loss,
                    "train/model_mse": avg_model_mse,
                    "train/ref_mse": avg_ref_mse,
                    "train/implicit_acc": implicit_acc_accumulated,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step >= config.max_train_steps:
                    break
                
                if accelerator.is_main_process:
                    if len(config.validation_case_indices) > 0 and (global_step % config.validation_steps == 0 or global_step == 1): # or global_step == 1
                        
                        trans_vae = TransAutoencoderKL.from_pretrained(
                            config.trans_vae_dir,
                            subfolder="trans_vae",
                        )
                        # create pipeline
                        pipeline = CustomPipeline(
                            vae=trans_vae,
                            transformer=unwrap_model(transformer),
                            fusion_module=unwrap_model(fusion_module),
                            scheduler=noise_scheduler,
                        )
                        log_validation(
                            val_dataset=val_dataset,
                            pipeline=pipeline, 
                            config=config, 
                            accelerator=accelerator, 
                        )


if __name__ == "__main__":

    config = parse_config()

    logging_dir = Path(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS. 
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    config.weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        config.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        config.weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and config.weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=config.cache_dir if hasattr(config, "cache_dir") else None
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Initialize all models
    vae, transformer, ref_transformer, fusion_module = initialize_all_models(config, accelerator)

    accelerator.register_save_state_pre_hook(save_model_hook_partial)
    accelerator.register_load_state_pre_hook(load_model_hook_partial)

    # Get trainable parameters
    params_to_optimize = get_trainable_params(transformer, fusion_module, config, accelerator)

    # Optimizer
    if config.get("optimizer", None) == "prodigy":
        import prodigyopt # type: ignore
        optimizer_class = prodigyopt.Prodigy
        optimizer = optimizer_class(
            params_to_optimize,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            beta3=config.prodigy_beta3,
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
            decouple=config.prodigy_decouple,
            use_bias_correction=config.prodigy_use_bias_correction,
            safeguard_warmup=config.prodigy_safeguard_warmup,
        )
    else:
        if config.get("use_8bit_adam", None):
            import bitsandbytes as bnb # type: ignore
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    train_dataset = MergedGlyphDataset(
        datype="train",
        resolution=config.resolution,
        dataset_names=config.dataset_names,
        dataset_cfgs=config.dataset_cfgs,
    )
    train_dataloader = data.DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=config.dataloader_shuffle, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=config.dataloader_pin_memory,
        collate_fn=general_collate_fn,
    )
    val_dataset = train_dataset
    if accelerator.is_main_process:
        for dataset in train_dataset.datasets:
            print(f"[DATA] samples in {dataset.data_root}: {len(dataset.index_map)}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.train_batch_size # mute annoying deepspeed errors
    transformer, fusion_module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, fusion_module, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers, also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(copy.deepcopy(config))
        accelerator.init_trackers(
            project_name=config.tracker_project_name, 
            config=tracker_config, 
            init_kwargs={"wandb": {"entity": config.wandb_entity, "name": config.wandb_job_name}},
        )

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                extra_kwargs = {'load_module_strict': False}
            else:
                extra_kwargs = {}

            accelerator.load_state(os.path.join(config.output_dir, path), **extra_kwargs)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    train(global_step, first_epoch, config, accelerator, \
           train_dataloader, optimizer, lr_scheduler, transformer, ref_transformer, fusion_module, vae, \
              noise_scheduler_copy, noise_scheduler, progress_bar)

    # finally, save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, f"checkpoint-final")
        accelerator.save_state(save_path)
        logger.info(f"Final checkpoints is saved to {save_path}")

    accelerator.end_training()