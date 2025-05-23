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
import lpips
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from mmengine.config import Config

import torch
import torch.utils.checkpoint
import torch.utils.data as data
import numpy as np

from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import diffusers
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import VaeImageProcessor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from custom_model import TransAutoencoderKL
from custom_dataset.synth_dataset import SynthGlyphSingleDataset
from custom_dataset.utils.helper import render_chs, general_collate_fn
from custom_dataset.utils.augment import aug_chs

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
            if isinstance(unwrap_model(model), type(unwrap_model(trans_vae))):
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "trans_vae"))
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

def load_model_hook_partial(models, input_dir):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model), type(unwrap_model(trans_vae))):
            load_model = TransAutoencoderKL.from_pretrained(input_dir, subfolder="trans_vae")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

        del load_model

def initialize_all_models(config, accelerator):

    logger.info(f"[INFO] start load trans vae")
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.get("cache_dir", None),
    )
    trans_vae: TransAutoencoderKL = TransAutoencoderKL.from_config(vae.config)
    miss_keys, unexpected_keys = trans_vae.load_state_dict(vae.state_dict(), strict=False)
    print(f"Loaded TransAutoencoderKL with miss_keys: {miss_keys} and unexpected_keys: {unexpected_keys}")
    trans_vae.enable_slicing()
    del vae

    trans_vae.requires_grad_(False)
    trans_vae.decoder.requires_grad_(True)

    if config.gradient_checkpointing:
        trans_vae.enable_gradient_checkpointing()

    trans_vae.to(accelerator.device, dtype=config.weight_dtype)

    # Make sure the trainable params are in float32.  
    logger.info(f"[INFO] cast_training_params to fp32")
    if config.cast_training_params:
        models = [trans_vae]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    
    return trans_vae

def get_trainable_params(trans_vae, config, accelerator):

    trans_vae_parameters = list(filter(lambda p: p.requires_grad, trans_vae.parameters()))

    # Optimization parameters
    trans_vae_parameters_with_lr = {"params": trans_vae_parameters, "lr": config.lr}
    params_to_optimize = [trans_vae_parameters_with_lr]
    
    if accelerator.is_main_process:
        for i, param_set in enumerate(params_to_optimize):
            num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
            print(f"Trainable Params Set {i}: {num_params:.2f}M")
    
    return params_to_optimize

def log_validation(trans_vae, config, accelerator, global_step):
    logger.info(f"Running validation...")
    vae_processor = VaeImageProcessor()

    # run inference
    image_logs = []
    all_psnrs = []
    all_ssims = []
    for validation_case in config.validation_cases:
        text = validation_case["text"]
        font_base = validation_case["font_base"]
        aug_seed = validation_case.get("aug_seed", -1)

        font_path = os.path.join(config.font_root, font_base)
        ch_pil, ch_w_pil, ch_a_mask = render_chs(list(text), font_path, resolution=config.resolution)
        if aug_seed >= 0:
            ch_pil, ch_w_pil, ch_a_mask = aug_chs(ch_pil, aug_seed)

        ch = vae_processor.preprocess(image=ch_w_pil)
        ch *= ch_a_mask # make it gray!

        ch_a_mask = (ch_a_mask * 2) - 1
        ch_rgba = torch.cat([ch, ch_a_mask], dim=1)

        with torch.no_grad():
            with torch.autocast(accelerator.device.type, dtype=torch.bfloat16):
                output = trans_vae(ch.to(accelerator.device)).sample

        ch_rgba_pil = vae_processor.postprocess(ch_rgba)
        output_pil = vae_processor.postprocess(output)
        ch_concat = np.concatenate([np.array(img) for img in ch_rgba_pil], axis=1)
        output_concat = np.concatenate([np.array(img) for img in output_pil], axis=1)
        ch_rgba_concat_pil = Image.fromarray(ch_concat)
        output_concat_pil = Image.fromarray(output_concat)
        
        ch_a_concat_pil = Image.fromarray(ch_concat[:,:,3])
        output_a_concat_pil = Image.fromarray(output_concat[:,:,3])

        psnrs = [peak_signal_noise_ratio(np.array(c), np.array(o)) for c, o in zip(ch_rgba_pil, output_pil)]
        ssims = [structural_similarity(np.array(c.convert("L")), np.array(o.convert("L"))) for c, o in zip(ch_rgba_pil, output_pil)]
        all_psnrs.append(sum(psnrs) / len(psnrs))
        all_ssims.append(sum(ssims) / len(ssims))

        image_logs.append(
            { 
                "text": text,
                "ch_rgba": ch_rgba_concat_pil,
                "output": output_concat_pil,
                "ch_a": ch_a_concat_pil,
                "output_a": output_a_concat_pil,
            }
        )

    avg_psnr = sum(all_psnrs) / len(all_psnrs)
    avg_ssim = sum(all_ssims) / len(all_ssims)
    accelerator.log({"avg_psnr": avg_psnr, "avg_ssim": avg_ssim}, step=global_step)

    for tracker in accelerator.trackers:
        assert tracker.name == "wandb"
        
        dict_to_log = {}
        for sample_idx, log in enumerate(image_logs):
            formatted_images = []
            text = log["text"]
            
            for key, value in log.items():
                if isinstance(value, Image.Image):
                    value = wandb.Image(value, caption=key)
                    formatted_images.append(value)

            dict_to_log[f"sample_{sample_idx}"] = formatted_images

        tracker.log(dict_to_log)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train(global_step, first_epoch, config, accelerator, \
           train_dataloader, optimizer, lr_scheduler, trans_vae, progress_bar):
    
    lpips_loss_fn = lpips.LPIPS(net="squeeze").to(accelerator.device)

    for epoch in range(first_epoch, config.num_train_epochs):
        trans_vae.train()

        for step_idx, batch in enumerate(train_dataloader):
            models_to_accumulate = [trans_vae]
            with accelerator.accumulate(models_to_accumulate):

                gt = batch["gt"].to(dtype=torch.float32)
                gt_rgba = batch["gt_rgba"].to(dtype=torch.float32)

                # model feedforward
                model_pred = trans_vae(gt, return_dict=False)[0]

                # calculate loss
                target = gt_rgba
                mse_loss = F.mse_loss(model_pred[:,:3], target[:,:3], reduction="mean")
                lpips_loss = lpips_loss_fn(model_pred[:,:3], target[:,:3]).mean()
                a_loss = F.mse_loss(model_pred[:,3:], target[:,3:], reduction="mean")

                loss = (
                    mse_loss + config.lpips_scale * lpips_loss + config.a_scale * a_loss
                )
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = trans_vae.parameters()
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
                            unwrap_model(trans_vae).save_pretrained(os.path.join(save_path, "trans_vae"))

                loss_value = loss.detach().item()
                loss_value = 0.0 if loss_value > 10.0 else loss_value
                logs = {"loss": loss_value, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break
                
                if accelerator.is_main_process:
                    if config.validation_cases is not None and (global_step % config.validation_steps == 0 or global_step == 1): # or global_step == 1
                        
                        log_validation(
                            trans_vae = unwrap_model(trans_vae), 
                            config=config, 
                            accelerator=accelerator, 
                            global_step=global_step,
                        )


if __name__ == "__main__":

    config = parse_config()

    logging_dir = Path(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
        diffusers.utils.logging.set_verbosity_info()
    else:
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
    trans_vae = initialize_all_models(config, accelerator)

    accelerator.register_save_state_pre_hook(save_model_hook_partial)
    accelerator.register_load_state_pre_hook(load_model_hook_partial)

    # Get trainable parameters
    params_to_optimize = get_trainable_params(trans_vae, config, accelerator)

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
    train_dataset = SynthGlyphSingleDataset(
        datype="all",
        **config.dataset_cfg
    )
    train_dataloader = data.DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=config.dataloader_shuffle, 
        num_workers=config.dataloader_num_workers, 
        pin_memory=config.dataloader_pin_memory,
        collate_fn=general_collate_fn,
    )

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
    trans_vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        trans_vae, optimizer, train_dataloader, lr_scheduler
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
           train_dataloader, optimizer, lr_scheduler, trans_vae, progress_bar)

    # finally, save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, f"checkpoint-final")
        accelerator.save_state(save_path)
        logger.info(f"Final checkpoints is saved to {save_path}")

    accelerator.end_training()