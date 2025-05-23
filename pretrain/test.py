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

import os
import sys
sys.path.append(os.getcwd())
import gc
import argparse
from tqdm import tqdm
from mmengine.config import Config

import random
import numpy as np

import torch
from PIL import Image

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import check_min_version

from custom_encoder import FusionModule
from custom_model import CustomTransformer2DModel
from custom_pipeline import CustomPipeline
from custom_scheduler import StochasticRFOvershotDiscreteScheduler
from custom_dataset.synth_dataset import SynthGlyphDataset
from custom_dataset.utils.helper import general_collate_fn, convert_gray_bg_to_white
from trans_vae.custom_model import TransAutoencoderKL

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

def initialize_pipeline(config, args):

    if config.get("trans_vae_dir", None) is not None:
        vae = TransAutoencoderKL.from_pretrained(
            config.trans_vae_dir,
            subfolder="trans_vae"
        )
    else:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="vae",
            revision=config.revision,
            variant=config.variant,
            cache_dir=config.get("cache_dir", None),
        )
    vae.enable_slicing()

    if args.use_overshooting:
        noise_scheduler = StochasticRFOvershotDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, 
            subfolder="scheduler",
            cache_dir=config.get("cache_dir", None),
        )
        noise_scheduler.set_c(2.0)
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, 
            subfolder="scheduler",
            cache_dir=config.get("cache_dir", None),
        )

    transformer = CustomTransformer2DModel.from_pretrained(args.ckpt_dir, subfolder="transformer")
    fusion_module = FusionModule.from_pretrained(args.ckpt_dir, subfolder="fusion_module")

    pipeline = CustomPipeline(
        vae=vae,
        transformer=transformer,
        fusion_module=fusion_module,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(torch.device("cuda", index=args.gpu_id))
    pipeline.enable_model_cpu_offload(gpu_id=args.gpu_id) # save vram

    return pipeline

def test(args):

    if args.seed is not None:
        seed_everything(args.seed)

    args.base_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    cfg_path = f"{args.base_name}/config/{args.name}.py"
    config = parse_config(cfg_path)

    if args.ckpt_idx is not None:
        args.ckpt_dir = f"work_dirs/{args.base_name}_{args.name}/checkpoint-{args.ckpt_idx}"
    else:
        args.ckpt_dir = f"work_dirs/{args.base_name}_{args.name}/checkpoint-final"

    args.save_dir = f"outputs/{args.datype}_{args.base_name}_{args.name}{args.variant}"
    os.makedirs(os.path.join(args.save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "pred"), exist_ok=True)

    # Dataset and DataLoaders creation:
    config.dataset_cfg.pop("samples_per_style")
    config.dataset_cfg.pop("interval_c_refs")
    config.dataset_cfg.pop("interval_s_refs")
    dataset_cfg = dict(
        datype=args.datype,
        samples_per_style=args.samples_per_style,
        interval_c_refs=args.interval_c_refs,
        interval_s_refs=args.interval_s_refs,
        **config.dataset_cfg,
    )
    test_dataset = SynthGlyphDataset(**dataset_cfg)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=general_collate_fn,
    )

    # Initialize pipeline
    pipeline = initialize_pipeline(config, args)

    # Start generation
    generator = torch.Generator(device=torch.device("cuda", index=args.gpu_id)).manual_seed(args.seed) if args.seed else None
    for idx, batch in tqdm(enumerate(test_dataloader)):
        style_idx = idx // args.samples_per_style
        sample_idx = idx % args.samples_per_style
        if style_idx >= args.cases: break

        this_index = f"style_{style_idx}_sample_{sample_idx}"
        os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)

        if config.get("trans_vae_dir", None) is not None:
            if os.path.exists(os.path.join(args.save_dir, this_index, "output_cb.png")):
                continue
        else:
            if os.path.exists(os.path.join(args.save_dir, this_index, "output.png")):
                continue

        gt_pil = batch["gt_pil"][0]
        c_refs_pil = batch["c_refs_pil"][0]
        s_refs_pil = batch["s_refs_pil"][0]
        
        print(f"Style {style_idx} Sample {sample_idx}:")
        output = pipeline(
            height=config.resolution,
            width=config.resolution,
            cfg_scale=args.cfg,
            lamda_gate=args.lamda_gate,
            c_refs_pil=[c_refs_pil],
            s_refs_pil=[s_refs_pil],
            num_inference_steps=args.steps,
            generator=generator,
        )
        images = output.images[0]

        white_bg = Image.new("RGBA", (config.resolution, config.resolution), (255, 255, 255, 255))
        cb_bg = Image.open("./assets/checkerboard.png").convert("RGBA").resize((config.resolution, config.resolution))
        cb_pil = [Image.alpha_composite(cb_bg, img) for img in gt_pil]
        gt_pil = [Image.alpha_composite(white_bg, img) for img in gt_pil]

        if config.get("trans_vae_dir", None) is not None:
            cb_images = [Image.alpha_composite(cb_bg, img) for img in images]
            images = [Image.alpha_composite(white_bg, img) for img in images]
        else:
            images = convert_gray_bg_to_white(images)

        # Save images for evaluation
        for idx, pil in enumerate(gt_pil):
            pil.save(os.path.join(args.save_dir, "gt", f"{this_index}_{idx}.png"))
        for idx, pil in enumerate(images):
            pil.save(os.path.join(args.save_dir, "pred", f"{this_index}_{idx}.png"))

        # Concatenate images horizontally
        gt_concat = np.concatenate([np.array(img) for img in gt_pil], axis=1)
        gt_cb_concat = np.concatenate([np.array(img) for img in cb_pil], axis=1)
        c_refs_concat = np.concatenate([np.array(img) for img in c_refs_pil], axis=1)
        s_refs_concat = np.concatenate([np.array(img) for img in s_refs_pil], axis=1)
        output_concat = np.concatenate([np.array(img) for img in images], axis=1)

        # Convert back to PIL images
        gt_concat_pil = Image.fromarray(gt_concat)
        gt_cb_concat_pil = Image.fromarray(gt_cb_concat)
        c_refs_concat_pil = Image.fromarray(c_refs_concat)
        s_refs_concat_pil = Image.fromarray(s_refs_concat)
        output_concat_pil = Image.fromarray(output_concat)
 
        # Save concatenated images
        gt_concat_pil.save(os.path.join(args.save_dir, this_index, "gt.png"))
        gt_cb_concat_pil.save(os.path.join(args.save_dir, this_index, "gt_cb.png"))
        c_refs_concat_pil.save(os.path.join(args.save_dir, this_index, "c_refs.png"))
        s_refs_concat_pil.save(os.path.join(args.save_dir, this_index, "s_refs.png")) 
        output_concat_pil.save(os.path.join(args.save_dir, this_index, "output.png"))

        # save outputs with checkerboard background
        if config.get("trans_vae_dir", None) is not None:
            output_cb_concat = np.concatenate([np.array(img) for img in cb_images], axis=1)
            output_cb_concat_pil = Image.fromarray(output_cb_concat)
            output_cb_concat_pil.save(os.path.join(args.save_dir, this_index, "output_cb.png"))


    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datype", type=str, default="val", choices=["val", "test"]) 
    parser.add_argument("--name", type=str, default="{path to base model}") 
    parser.add_argument("--ckpt_idx", type=int, default=None)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--samples_per_style", type=int, default=4)
    parser.add_argument("--interval_c_refs", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--interval_s_refs", type=int, nargs=2, default=[5, 10])
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--lamda_gate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_overshooting", type=int, default=1)
    args = parser.parse_args()

    test(args)