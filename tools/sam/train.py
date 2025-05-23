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
from transformers import SamProcessor, SamModel

from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from transformers.utils import check_min_version
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import cast_training_params

from custom_dataset.synth_dataset import SynthGlyphRegionDataset
from custom_dataset.utils.helper import general_collate_fn

if is_wandb_available():
    import wandb

# Will error if the minimal version of transformers is not installed. Remove at your own risks.
check_min_version("4.46.0")

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
            if isinstance(unwrap_model(model), type(unwrap_model(sam_model))):
                unwrap_model(model).save_pretrained(output_dir)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

def load_model_hook_partial(models, input_dir):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        if isinstance(unwrap_model(model), type(unwrap_model(sam_model))):
            load_model = SamModel.from_pretrained(input_dir)
            model.load_state_dict(load_model.state_dict())
        else:
            raise ValueError(f"Unsupported model found: {type(model)=}")

        del load_model

def initialize_all_models(config, accelerator):

    logger.info(f"[INFO] start load sam model and processor")
    sam_model: SamModel = SamModel.from_pretrained(
        config.pretrained_model_name_or_path, 
        cache_dir=config.cache_dir,
    )
    sam_processor: SamProcessor = SamProcessor.from_pretrained(
        config.pretrained_model_name_or_path, 
        cache_dir=config.cache_dir,
    )

    sam_model.requires_grad_(True)
    sam_model.vision_encoder.requires_grad_(False)
    sam_model.prompt_encoder.requires_grad_(False)
    if config.train_encoder:
        sam_model.vision_encoder.neck.requires_grad_(True)
        for layer in sam_model.vision_encoder.layers[-config.num_train_layers:]:
            layer.requires_grad_(True)

    # if config.gradient_checkpointing:
    #     sam_model.gradient_checkpointing_enable()

    # Make sure the trainable params are in float32.  
    if config.cast_training_params:
        logger.info(f"[INFO] cast_training_params to fp32")
        models = [sam_model]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    
    return sam_model, sam_processor

def get_trainable_params(sam_model, config, accelerator):

    sam_model_parameters = list(filter(lambda p: p.requires_grad, sam_model.parameters()))

    # Optimization parameters
    sam_model_parameters_with_lr = {"params": sam_model_parameters, "lr": config.lr}
    params_to_optimize = [sam_model_parameters_with_lr]
    
    if accelerator.is_main_process:
        for i, param_set in enumerate(params_to_optimize):
            num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
            print(f"Trainable Params Set {i}: {num_params:.2f}M")
    
    return params_to_optimize

def log_validation(val_dataset, sam_model, sam_processor, config, accelerator):
    logger.info(f"Running validation...")

    # run inference
    image_logs = []
    for case_id in range(config.num_validation_cases):
        val_case = val_dataset[case_id]
        image = val_case["composite_img"]
        seg_bboxs = val_case["seg_bboxs"]
        gt_a_mask = val_case["composite_a_mask"].unsqueeze(1)

        model_inputs = sam_processor(image, input_boxes=[seg_bboxs], return_tensors="pt").to(accelerator.device)
        model_pred = sam_model(**model_inputs, multimask_output=False)
        pred_masks = sam_processor.post_process_masks(
            model_pred.pred_masks.cpu(), 
            model_inputs["original_sizes"], 
            model_inputs["reshaped_input_sizes"],
            binarize=False,
        )[0]
        pred_masks = pred_masks.clamp(min=0.0, max=1.0)

        image_logs.append(
            {   
                "image": image,
                "gt_a_mask": gt_a_mask,
                "pred_masks": pred_masks,
            }
        )

    for tracker in accelerator.trackers:
        assert tracker.name == "wandb"
        
        dict_to_log = {}
        for sample_idx, log in enumerate(image_logs):
            formatted_images = []

            for key, value in log.items():
                value = wandb.Image(value, caption=key)
                formatted_images.append(value)

            dict_to_log[f"sample_{sample_idx}"] = formatted_images

        tracker.log(dict_to_log)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train(global_step, first_epoch, config, accelerator, val_dataset, \
           train_dataloader, optimizer, lr_scheduler, sam_model, sam_processor, progress_bar):

    for epoch in range(first_epoch, config.num_train_epochs):
        sam_model.train()

        for step_idx, batch in enumerate(train_dataloader):
            models_to_accumulate = [sam_model]
            with accelerator.accumulate(models_to_accumulate):

                image = batch["composite_img"]
                gt_a_mask = batch["composite_a_mask"][0].unsqueeze(1)
                seg_bboxs = batch["seg_bboxs"]

                model_inputs = sam_processor(image, input_boxes=seg_bboxs, return_tensors="pt").to(accelerator.device)
                model_pred = sam_model(**model_inputs, multimask_output=False)
                pred_masks = sam_processor.post_process_masks(
                    model_pred.pred_masks,
                    model_inputs["original_sizes"], 
                    model_inputs["reshaped_input_sizes"],
                    binarize=False,
                )[0]
                pred_masks = pred_masks.clamp(min=0.0, max=1.0) # value clip

                # MSE Loss
                mse_loss = F.mse_loss(pred_masks, gt_a_mask, reduction='mean')
                # Focal Loss
                bce_loss = F.binary_cross_entropy(pred_masks, gt_a_mask, reduction='none')
                pt = torch.where(gt_a_mask == 1, pred_masks, 1 - pred_masks)
                focal_loss = config.focal_alpha * (1 - pt) ** config.focal_gamma * bce_loss
                focal_loss = focal_loss.mean()
                # Dice Loss
                intersection = (pred_masks * gt_a_mask).sum()
                union = pred_masks.sum() + gt_a_mask.sum()
                dice_loss = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)

                loss = mse_loss + config.lamda_focal * focal_loss + config.lamda_dice * dice_loss 

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = sam_model.parameters()
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
                            unwrap_model(sam_model).save_pretrained(save_path)
                            
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break
                
                if accelerator.is_main_process:
                    if config.num_validation_cases > 0 and (global_step % config.validation_steps == 0 or global_step == 1): # or global_step == 1
                        
                        log_validation(
                            val_dataset=val_dataset,
                            sam_model=unwrap_model(sam_model), 
                            sam_processor=sam_processor,
                            config=config, 
                            accelerator=accelerator, 
                        )


if __name__ == "__main__":

    config = parse_config()

    logging_dir = Path(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    else:
        transformers.utils.logging.set_verbosity_error()

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
    sam_model, sam_processor = initialize_all_models(config, accelerator)

    accelerator.register_save_state_pre_hook(save_model_hook_partial)
    accelerator.register_load_state_pre_hook(load_model_hook_partial)

    # Get trainable parameters
    params_to_optimize = get_trainable_params(sam_model, config, accelerator)

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
    train_dataset = SynthGlyphRegionDataset(
        datype="train", 
        augments_per_region=10,
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
    val_dataset = SynthGlyphRegionDataset(
        datype="val", 
        augments_per_region=1,
        **config.dataset_cfg,
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
    sam_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        sam_model, optimizer, train_dataloader, lr_scheduler
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

    train(global_step, first_epoch, config, accelerator, val_dataset, \
           train_dataloader, optimizer, lr_scheduler, sam_model, sam_processor, progress_bar)

    # finally, save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, f"checkpoint-final")
        accelerator.save_state(save_path)
        logger.info(f"Final checkpoints is saved to {save_path}")

    accelerator.end_training()