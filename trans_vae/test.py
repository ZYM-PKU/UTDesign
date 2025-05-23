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
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image

from diffusers.utils import check_min_version

from custom_model import TransAutoencoderKL
from custom_dataset.synth_dataset import SynthGlyphSingleDataset
from custom_dataset.utils.helper import general_collate_fn

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

    args.save_dir = f"outputs/{args.type}_{args.base_name}_{args.name}{args.variant}"
    os.makedirs(os.path.join(args.save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "pred"), exist_ok=True)

    # Model
    trans_vae: TransAutoencoderKL = TransAutoencoderKL.from_pretrained(args.ckpt_dir, subfolder="vae")
    trans_vae.enable_slicing()

    # Dataset and DataLoaders creation:
    test_dataset = SynthGlyphSingleDataset(
        datype="all",
        **config.dataset_cfg,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=general_collate_fn,
    )

    # Start generation
    for idx, batch in tqdm(enumerate(test_dataloader)):
        if idx >= args.cases: break

        this_index = f"sample_{idx}"
        os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)

        if os.path.exists(os.path.join(args.save_dir, this_index, "output_a.png")):
            continue

        gt = batch["gt"]
        gt_rgba = batch["gt_rgba"]
        
        output = trans_vae(gt, return_dict=False)[0]
        gt_rgba_pil = test_dataset.vae_processor.postprocess(gt_rgba)

        gt_rgba_pil = test_dataset.vae_processor.postprocess(gt_rgba)
        output_pil = test_dataset.vae_processor.postprocess(output)
        gt_concat = np.concatenate([np.array(img) for img in gt_rgba_pil], axis=1)
        output_concat = np.concatenate([np.array(img) for img in output_pil], axis=1)
        gt_rgba_concat_pil = Image.fromarray(gt_concat)
        output_concat_pil = Image.fromarray(output_concat)
        gt_a_concat_pil = Image.fromarray(gt_concat[:,:,3])
        output_a_concat_pil = Image.fromarray(output_concat[:,:,3])

        # Save images for evaluation
        for idx, pil in enumerate(gt_rgba_pil):
            pil.save(os.path.join(args.save_dir, "gt", f"{this_index}_{idx}.png"))
        for idx, pil in enumerate(output_pil):
            pil.save(os.path.join(args.save_dir, "pred", f"{this_index}_{idx}.png"))

        # Save concatenated images
        gt_rgba_concat_pil.save(os.path.join(args.save_dir, this_index, "gt.png"))
        output_concat_pil.save(os.path.join(args.save_dir, this_index, "output.png"))
        gt_a_concat_pil.save(os.path.join(args.save_dir, this_index, "gt_a.png"))
        output_a_concat_pil.save(os.path.join(args.save_dir, this_index, "output_a.png"))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="test") 
    parser.add_argument("--name", type=str, default="{path to vae model}") 
    parser.add_argument("--ckpt_idx", type=int, default=None)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    test(args)