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

import transformers
import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file

from custom_encoder import FusionModule, EmbeddingDiscriminator
from custom_dataset.merged_dataset import MergedGlyphDataset
from custom_dataset.utils.helper import general_collate_fn

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
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(fusion_module))):
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "fusion_module"))
            elif isinstance(unwrap_model(model), type(unwrap_model(emb_discriminator))):
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "emb_discriminator"))
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

def load_model_hook_partial(models, input_dir):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model), type(unwrap_model(fusion_module))):
            load_model = FusionModule.from_pretrained(input_dir, subfolder="fusion_module")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict(), strict=False)
        elif isinstance(unwrap_model(model), type(unwrap_model(emb_discriminator))):
            load_model = EmbeddingDiscriminator.from_pretrained(input_dir, subfolder="emb_discriminator")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

        del load_model

def initialize_all_models(config, accelerator):

    logger.info(f"[MODEL] load fusion module")
    fusion_module: FusionModule = FusionModule.from_pretrained(
        config.fusion_model_path,
        subfolder="fusion_module",
        vlm_model_name=config.vlm_model_name,
        pr_depth=config.pr_depth,
        pr_seq_len=config.pr_seq_len,
        clip_cache_dir=config.cache_dir_clip,
        vlm_cache_dir=config.cache_dir_vlm,
        low_cpu_mem_usage=False, # disable this to avoid BUGs
    )

    fusion_module.requires_grad_(False)
    fusion_module.perceiver_resampler.requires_grad_(True)

    logger.info(f"[MODEL] load discriminator")
    emb_discriminator = EmbeddingDiscriminator.from_config(config.disc_cfg)

    # Make sure the trainable params are in float32.  
    logger.info(f"[MODEL] cast_training_params to fp32")
    if config.cast_training_params:
        models = [fusion_module, emb_discriminator]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    
    return fusion_module, emb_discriminator

def get_trainable_params(fusion_module, config, accelerator):

    fusion_module_parameters = list(filter(lambda p: p.requires_grad, fusion_module.parameters()))

    # Optimization parameters
    fusion_module_parameters_with_lr = {"params": fusion_module_parameters, "lr": config.lr_fusion}
    params_to_optimize = [fusion_module_parameters_with_lr]
    
    if accelerator.is_main_process:
        for i, param_set in enumerate(params_to_optimize):
            num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
            accelerator.print(f"Trainable Params Set {i}: {num_params:.2f}M")
    
    return params_to_optimize

def log_validation(val_dataset, fusion_module, config, accelerator, global_step):
    logger.info(f"Running validation...")

    l2_loss = 0.0
    pooled_cosine_sim = 0.0
    for case_id in tqdm(config.validation_case_indices):
        val_case = val_dataset[case_id]
        s_refs_pil = val_case["s_refs_pil"]
        bg_pil = val_case["crop_bg"]
        r_crop_bbox = val_case["r_crop_bbox"]
        prompt = val_case["prompt"]
        label = val_case["label"]

        n_glyphs = config.pr_seq_len//256
        valid_mask = torch.zeros(1, n_glyphs)
        valid_mask[0, :len(label)] = 1

        with torch.no_grad():
            (
                style_embeds_1, 
                style_embeds_2, 
                pooled_style_embeds_1, 
                pooled_style_embeds_2
            ) = fusion_module(
                style_images=[s_refs_pil],
                bg_images=[bg_pil],
                reg_bboxs=[r_crop_bbox],
                prompts=[prompt],
                labels=[label],
            )

        loss = F.mse_loss(style_embeds_1, style_embeds_2, reduction="none")
        loss = loss.reshape(1, n_glyphs, -1).mean(dim=-1)
        loss = loss * valid_mask.to(loss)
        loss = loss.mean()
        l2_loss += loss
        pooled_cosine_sim += F.cosine_similarity(pooled_style_embeds_1, pooled_style_embeds_2, dim=-1).mean()
    
    l2_loss /= len(config.validation_case_indices)
    pooled_cosine_sim /= len(config.validation_case_indices)

    log_dict = {
        "eval/l2_loss": l2_loss.detach().item(),
        "eval/pooled_cosine_sim": pooled_cosine_sim.detach().item(),
    }
    accelerator.log(log_dict, step=global_step)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train(global_step, first_epoch, config, accelerator, \
           train_dataloader, optimizer, d_optimizer, lr_scheduler, fusion_module, emb_discriminator, progress_bar):

    for epoch in range(first_epoch, config.num_train_epochs):
        fusion_module.train()
        emb_discriminator.train()

        for step_idx, batch in enumerate(train_dataloader):
            models_to_accumulate = [fusion_module, emb_discriminator]
            with accelerator.accumulate(models_to_accumulate):

                s_refs_pil = batch["s_refs_pil"]
                bg_pil = batch["bg"]
                r_crop_bboxs = batch["r_crop_bbox"]
                prompt = batch["prompt"]
                label = batch["label"]

                bsz, n_glyphs = len(s_refs_pil), config.pr_seq_len//256
                valid_lens = [len(labels) for labels in batch["labels"]]
                valid_mask = torch.zeros(bsz, n_glyphs)
                for i, valid_len in enumerate(valid_lens):
                    valid_mask[i, :valid_len] = 1

                # Get conditional embeddings
                (
                    style_embeds_1, 
                    style_embeds_2, 
                    pooled_style_embeds_1, 
                    pooled_style_embeds_2
                ) = fusion_module(
                    style_images=s_refs_pil,
                    bg_images=bg_pil,
                    reg_bboxs=r_crop_bboxs,
                    prompts=prompt,
                    labels=label,
                )

                ### discriminator
                d_optimizer.zero_grad()
                real_labels = torch.ones(pooled_style_embeds_1.size(0), 1, device=pooled_style_embeds_1.device)
                fake_labels = torch.zeros(pooled_style_embeds_2.size(0), 1, device=pooled_style_embeds_2.device)
                d_real = emb_discriminator(pooled_style_embeds_1.detach())
                d_fake = emb_discriminator(pooled_style_embeds_2.detach())
                d_loss = F.binary_cross_entropy(d_real, real_labels) + F.binary_cross_entropy(d_fake, fake_labels)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(emb_discriminator.parameters(), config.max_grad_norm_disc)
                d_optimizer.step()

                # L2 loss
                l2_loss = F.mse_loss(style_embeds_1, style_embeds_2, reduction="none")
                l2_loss = l2_loss.reshape(bsz, n_glyphs, -1).mean(dim=-1)
                l2_loss = l2_loss * valid_mask.to(l2_loss)
                l2_loss = l2_loss.mean()

                # pooled l2 loss
                pooled_l2_loss = F.mse_loss(pooled_style_embeds_1, pooled_style_embeds_2, reduction="mean")

                # CLIP loss
                pooled_style_embeds_1_norm = F.normalize(pooled_style_embeds_1, dim=-1)
                pooled_style_embeds_2_norm = F.normalize(pooled_style_embeds_2, dim=-1)
                logits_per_sample = pooled_style_embeds_1_norm @ pooled_style_embeds_2_norm.T
                labels = torch.arange(logits_per_sample.size(0), device=logits_per_sample.device)
                loss_i = F.cross_entropy(logits_per_sample, labels)
                loss_j = F.cross_entropy(logits_per_sample.T, labels)
                contrastive_loss = ((loss_i + loss_j) / 2).mean()

                # gan loss
                gan_loss = F.binary_cross_entropy(emb_discriminator(pooled_style_embeds_2), real_labels)

                loss = l2_loss + pooled_l2_loss * config.lambda_pooled_l2 \
                      + gan_loss * config.lambda_gan + contrastive_loss * config.lambda_clip

                avg_loss = accelerator.reduce(loss, reduction="mean").item()
                avg_l2_loss = accelerator.reduce(l2_loss, reduction="mean").item()
                avg_pooled_l2_loss = accelerator.reduce(pooled_l2_loss, reduction="mean").item()
                avg_clip_loss = accelerator.reduce(contrastive_loss, reduction="mean").item()
                avg_gan_loss = accelerator.reduce(gan_loss, reduction="mean").item()
                avg_d_loss = accelerator.reduce(d_loss, reduction="mean").item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(fusion_module.parameters(), emb_discriminator.parameters())
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
                            unwrap_model(fusion_module).save_pretrained(os.path.join(save_path, "fusion_module"))
                
                logs = {
                    "train/loss": avg_loss, 
                    "train/l2_loss": avg_l2_loss,
                    "train/pooled_l2_loss": avg_pooled_l2_loss,
                    "train/clip_loss": avg_clip_loss,
                    "train/gan_loss": avg_gan_loss,
                    "train/d_loss": avg_d_loss,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break
                
                if accelerator.is_main_process:
                    if len(config.validation_case_indices) > 0 and (global_step % config.validation_steps == 0 or global_step == 1): # or global_step == 1
                        
                        log_validation(
                            val_dataset=val_dataset,
                            fusion_module=unwrap_model(fusion_module), 
                            config=config, 
                            accelerator=accelerator, 
                            global_step=global_step,
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

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Initialize all models
    fusion_module, emb_discriminator = initialize_all_models(config, accelerator)

    accelerator.register_save_state_pre_hook(save_model_hook_partial)
    accelerator.register_load_state_pre_hook(load_model_hook_partial)

    # Get trainable parameters
    params_to_optimize = get_trainable_params(fusion_module, config, accelerator)

    # Optimizer
    if config.get("optimizer", None) == "prodigy":
        import prodigyopt # type: ignore
        optimizer_class = prodigyopt.Prodigy
        optimizer = optimizer_class(
            params_to_optimize,
            lr=config.lr_fusion,
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

    d_optimizer = torch.optim.AdamW(emb_discriminator.parameters(), lr=config.lr_disc)

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
        persistent_workers=False,
        collate_fn=general_collate_fn,
    )
    val_dataset = MergedGlyphDataset(
        datype="val",
        resolution=config.resolution,
        dataset_names=config.dataset_names,
        dataset_cfgs=config.dataset_cfgs,
    )
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
    fusion_module, emb_discriminator, optimizer, d_optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        fusion_module, emb_discriminator, optimizer, d_optimizer, train_dataloader, lr_scheduler
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
           train_dataloader, optimizer, d_optimizer, lr_scheduler, fusion_module, emb_discriminator, progress_bar)

    # finally, save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, f"checkpoint-final")
        accelerator.save_state(save_path)
        logger.info(f"Final checkpoints is saved to {save_path}")

    accelerator.end_training()