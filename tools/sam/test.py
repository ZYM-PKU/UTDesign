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
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image

from transformers.utils import check_min_version
from transformers import SamProcessor, SamModel

from custom_dataset.synth_dataset import SynthGlyphRegionDataset
from custom_dataset.utils.helper import general_collate_fn

# Will error if the minimal version of transformers is not installed. Remove at your own risks.
check_min_version("4.46.0")

# python trans_vae/test.py --name sft_skip_connect_bs16_lr1e-5_res512 --ckpt_idx 150000 --gpu_id 0

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

    cfg_path = f"tools/{args.base_name}/config/{args.name}.py"
    config = parse_config(cfg_path)

    if args.ckpt_idx is not None:
        args.ckpt_dir = f"work_dirs/{args.base_name}_{args.name}/checkpoint-{args.ckpt_idx}"
    else:
        args.ckpt_dir = f"work_dirs/{args.base_name}_{args.name}/checkpoint-final"

    args.save_dir = f"outputs/{args.type}_{args.base_name}_{args.name}{args.variant}"
    os.makedirs(os.path.join(args.save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "pred"), exist_ok=True)

    device = torch.device("cuda", index=args.gpu_id)

    # Model
    sam_model: SamModel = SamModel.from_pretrained(args.ckpt_dir).to(device)
    sam_processor: SamProcessor = SamProcessor.from_pretrained(
        config.pretrained_model_name_or_path, 
        cache_dir=config.cache_dir,
    )

    # Dataset and DataLoaders creation:
    dataset_cfg = config.dataset_cfg
    test_dataset = SynthGlyphRegionDataset(
        datype="val", 
        augments_per_region=1,
        **dataset_cfg,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
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

        if os.path.exists(os.path.join(args.save_dir, this_index, "output.png")):
            continue

        image = batch["composite_img"]
        gt_a_mask = batch["composite_a_mask"][0].unsqueeze(1)
        seg_bboxs = batch["seg_bboxs"]

        model_inputs = sam_processor(image, input_boxes=seg_bboxs, return_tensors="pt")

        model_pred = sam_model(
            pixel_values=model_inputs["pixel_values"].to(device),
            input_boxes=model_inputs["input_boxes"].to(device),
            multimask_output=False,
        )
        pred_masks = model_pred.pred_masks
        pred_masks = sam_processor.post_process_masks(
            pred_masks, 
            model_inputs["original_sizes"], 
            model_inputs["reshaped_input_sizes"],
            binarize=False,
        )[0]
        # Min-max normalization
        pred_masks = pred_masks.relu()
        pred_masks = (pred_masks - pred_masks.min()) / (pred_masks.max() - pred_masks.min() + 1e-6)

        # Save images for evaluation
        image[0].save(os.path.join(args.save_dir, this_index, "input.png"))
        save_image(gt_a_mask, os.path.join(args.save_dir, this_index, "gt_a_mask.png"))
        save_image(pred_masks, os.path.join(args.save_dir, this_index, "pred_a_mask.png"))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="val") 
    parser.add_argument("--name", type=str, default="sft_adamw_bs1_lr1e-5_res1024") 
    parser.add_argument("--ckpt_idx", type=int, default=70000)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    test(args)