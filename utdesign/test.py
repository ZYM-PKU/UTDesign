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

import json
import math
import random
import numpy as np

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import check_min_version
from safetensors.torch import load_file

from custom_encoder import FusionModule
from custom_model import CustomTransformer2DModel
from custom_pipeline import CustomPipeline
from custom_scheduler import StochasticRFOvershotDiscreteScheduler
from trans_vae.custom_model import TransAutoencoderKL

from custom_dataset.merged_dataset import MergedGlyphDataset
from custom_dataset.utils.helper import get_char_dict, expand_bbox, get_bbox_from_points, get_rescaled_bbox, render_chs, crop_chs, composite_image, concat_images
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, signal

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

def initialize_dataset(args):

    print("Loading dataset...")

    dataset_names = ["CollectedGlyphDataset"] * len(args.test_data_roots)
    dataset_cfgs = [
        dict(
            data_root=data_root,
            asset_root=args.asset_root,
            std_font_path=args.std_font_path,
            expand_ratio=args.expand_ratio,
            min_box_size=args.min_box_size,
            max_box_items=args.max_box_items,
            min_image_size=args.min_image_size,
            max_image_size=args.max_image_size,
            min_alpha_ratio=args.min_alpha_ratio,
            min_distance=distance,
            force_valid=args.force_valid,
        )
        for data_root, distance in zip(args.test_data_roots, args.min_distances)
    ]
    test_dataset = MergedGlyphDataset(
        datype="val",
        resolution=args.resolution,
        dataset_names=dataset_names,
        dataset_cfgs=dataset_cfgs,
    )
    
    test_dataset.annos = []
    for i, dataset in enumerate(test_dataset.datasets):
        num_valid_cases = len(dataset.annos)
        random.shuffle(dataset.annos)
        dataset.annos = dataset.annos[:args.max_cases_per_source[i]]
        print(f"{dataset.data_root}: filter {len(dataset.annos)} cases from {num_valid_cases} cases.")
        test_dataset.annos += dataset.annos

    return test_dataset

def random_switch(labels, char_dict, ratio=1.0):

    num_chars = char_dict["num"]
    en_chars = char_dict["en"]
    common_cn_chars = char_dict["common"]
    cn_chars = char_dict["cn"]
    
    new_labels = []
    num_switch = math.ceil(len(labels) * ratio)
    switch_indices = random.sample(range(len(labels)), num_switch)
    for idx, label in enumerate(labels):
        if idx in switch_indices:
            if label in num_chars:
                new_label = random.choice(num_chars)
            elif label in en_chars:
                new_label = random.choice(en_chars)
            elif label in cn_chars:
                new_label = random.choice(common_cn_chars)
            else:
                print(f"Unknown char: {label}")
                new_label = " "
        else:
            new_label = label
        new_labels.append(new_label)
    
    return new_labels

def initialize_labels(test_dataset, args):

    print("Loading ground truth labels...")
    char_dict = get_char_dict(args.char_root)
    if os.path.exists(args.labels_gt_path):
        with open(args.labels_gt_path, "r") as f:
            labels_gt = json.load(f)
    else:
        labels_gt = {}
        annos = test_dataset.annos
        for case_idx, anno in enumerate(tqdm(annos)):
            gt_dict = {}
            gt_dict["caption"] = anno["caption"]
            for crop_idx, bbox in enumerate(anno["bboxs"]):
                labels = list(bbox["label"])
                new_labels = random_switch(labels, char_dict, ratio=args.switch_ratio)
                gt_item = dict(
                    labels=labels,
                    new_labels=new_labels,
                )
                gt_dict[f"crop_{crop_idx}"] = gt_item
            labels_gt[f"case_{case_idx}"] = gt_dict

        with open(args.labels_gt_path, "w") as f:
            json.dump(labels_gt, f, indent=4, ensure_ascii=False)
            f.flush()
    
    return labels_gt

def initialize_pipeline(config, args, gpu_id=0):

    print(f"Loading pipeline on gpu {gpu_id}...")

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

    transformer = CustomTransformer2DModel.from_pretrained(config.stage1_model_path, subfolder="transformer")
    if config.get("fuse_lora_paths", None):
        for idx, fuse_lora_path in enumerate(config.fuse_lora_paths):
            print(f"loading lora from {fuse_lora_path}")
            lora_state_dict = CustomPipeline.lora_state_dict(fuse_lora_path)
            CustomPipeline.load_lora_into_transformer(lora_state_dict, None, transformer, f"fuse_lora_{idx}")
        transformer.fuse_lora()
        transformer.unload_lora()
        
    fusion_model_path = config.fusion_model_path if args.use_base or config.get("freeze_fusion", False) else args.ckpt_dir
    fusion_module: FusionModule = FusionModule.from_pretrained(
        fusion_model_path,
        subfolder="fusion_module",
        low_cpu_mem_usage=False, # disable this to avoid BUGs
        clip_cache_dir=config.cache_dir_clip,
        vlm_cache_dir=config.cache_dir_vlm,
    )

    pipeline = CustomPipeline(
        vae=vae,
        transformer=transformer,
        fusion_module=fusion_module,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(dtype=torch.bfloat16, device=torch.device("cuda", index=gpu_id))
    pipeline.enable_model_cpu_offload(gpu_id=gpu_id) # save vram
    if not args.use_base and config.rank is not None:
        pipeline.load_lora_weights(args.ckpt_dir)

    return pipeline

def generate_case(case_idx, anno, img, bg, gt, labels_gt, curr_dataset, pipeline, generator, args):

    w, h = img.size
    case_gt_pil = []
    case_crop_gt_pil = []
    case_c_refs_w_pil = []
    case_s_refs_w_pil = []
    case_crop_bg_pil = []
    case_prompt = []
    case_label = []
    case_seg_bboxs = []
    case_global_seg_bboxs = []
    case_r_crop_bboxs = []
    for crop_idx, bbox in enumerate(anno["bboxs"]):
        labels = labels_gt[f"case_{case_idx}"][f"crop_{crop_idx}"]["labels"] if args.task == "generation" \
              else labels_gt[f"case_{case_idx}"][f"crop_{crop_idx}"]["new_labels"]
        prompt = f"{anno['caption']} Text to render: {anno['trans_labels'][crop_idx]}"
        label = bbox["label"]

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), img.height, img.width, ratio=curr_dataset.expand_ratio, no_random=True)
        r_crop_bbox = get_rescaled_bbox([x1, y1, x2, y2], w, h)
        crop_img = img.crop((x1, y1, x2, y2))
        crop_bg = bg.crop((x1, y1, x2, y2))
        crop_gt = gt.crop((x1, y1, x2, y2))

        seg_bboxs = []
        global_seg_bboxs = []
        for seg in bbox["items"]:
            seg_bbox = get_bbox_from_points(seg["points"], x1, y1)
            global_seg_bbox = get_bbox_from_points(seg["points"])
            seg_bboxs.append(seg_bbox)
            global_seg_bboxs.append(global_seg_bbox)

        _, c_refs_w_pil, _ = render_chs(labels, font_path=curr_dataset.std_font_path, resolution=curr_dataset.resolution)
        _, s_refs_w_pil, _ = crop_chs(crop_img, seg_bboxs, resolution=curr_dataset.resolution)
        gt_pil, _, _ = crop_chs(crop_gt, seg_bboxs, resolution=curr_dataset.resolution)

        case_gt_pil.append(gt_pil)
        case_crop_gt_pil.append(crop_img)
        case_crop_bg_pil.append(crop_bg)
        case_prompt.append(prompt)
        case_label.append(label)
        case_c_refs_w_pil.append(c_refs_w_pil)
        case_s_refs_w_pil.append(s_refs_w_pil)
        case_seg_bboxs.append(seg_bboxs)
        case_global_seg_bboxs.append(global_seg_bboxs)
        case_r_crop_bboxs.append(r_crop_bbox)

    case_bg_pil = [bg] * len(anno["bboxs"])
    if args.task == "generation":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipeline(
                height=args.resolution,
                width=args.resolution,
                cfg_scale=args.cfg,
                lamda_gate=args.lamda_gate,
                c_refs_pil=case_c_refs_w_pil,
                bg_pil=case_bg_pil,
                reg_bboxs=case_r_crop_bboxs,
                prompt=case_prompt,
                label=case_label,
                num_inference_steps=args.steps,
                generator=generator,
            )
    elif args.task == "editing":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipeline(
                height=args.resolution,
                width=args.resolution,
                cfg_scale=args.cfg,
                lamda_gate=args.lamda_gate,
                c_refs_pil=case_c_refs_w_pil,
                s_refs_pil=case_s_refs_w_pil,
                num_inference_steps=args.steps,
                generator=generator,
            )
    case_images = output.images

    return case_images, case_gt_pil, case_crop_bg_pil, case_crop_gt_pil, case_seg_bboxs, case_global_seg_bboxs

def test_worker(test_dataset, labels_gt, pipeline, args, start_idx, end_idx, gpu_id=0):

    print(f"Testing cases for {args.task} from {start_idx} to {end_idx}")
    generator = torch.Generator(device=torch.device("cuda", index=gpu_id)).manual_seed(args.seed) if args.seed else None
    for case_idx in tqdm(range(start_idx, end_idx)):
        if stop_event is not None and stop_event.is_set():
            print(f"Worker on gpu {gpu_id} received stop signal, exiting...")
            return False
        
        if os.path.exists(os.path.join(args.save_dir, "pred", f"{case_idx}.png")):
            continue

        anno = test_dataset.annos[case_idx]
        curr_dataset = test_dataset.get_curr_dataset(case_idx)
        img_path = os.path.join(curr_dataset.image_root, anno["image_name"])
        bg_path = os.path.join(curr_dataset.bg_root, anno["image_name"])
        gt_path = os.path.join(curr_dataset.gt_root, anno["image_name"])
        with Image.open(img_path) as im:
            img = im.convert("RGBA").copy()
        with Image.open(bg_path) as im:
            bg = im.convert("RGBA").copy()
        with Image.open(gt_path) as im:
            gt = im.convert("RGBA").copy()
        
        case_images, case_gt_pil, case_crop_bg_pil, case_crop_gt_pil, case_seg_bboxs, case_global_seg_bboxs = \
            generate_case(case_idx, anno, img, bg, gt, labels_gt, curr_dataset, pipeline, generator, args)

        final_output_pil = bg.copy()
        # Save crop images
        for crop_idx, (images, gt_pil, crop_bg_pil, crop_gt_pil, seg_bboxs, global_seg_bboxs) in \
            enumerate(zip(case_images, case_gt_pil, case_crop_bg_pil, case_crop_gt_pil, case_seg_bboxs, case_global_seg_bboxs)):

            crop_index = f"case_{case_idx}_crop_{crop_idx}"
            os.makedirs(os.path.join(args.save_dir, crop_index), exist_ok=True)

            gt_concat_pil = concat_images(gt_pil)
            output_concat_pil = concat_images(images)
            crop_output_pil = composite_image(crop_bg_pil, images, seg_bboxs)
            temp_output_pil = composite_image(bg, images, global_seg_bboxs)
            final_output_pil = composite_image(final_output_pil, images, global_seg_bboxs)

            gt_concat_pil.save(os.path.join(args.save_dir, crop_index, "gt.png"))
            output_concat_pil.save(os.path.join(args.save_dir, crop_index, "output.png"))
            crop_gt_pil.save(os.path.join(args.save_dir, crop_index, "crop_gt.png"))
            crop_output_pil.save(os.path.join(args.save_dir, crop_index, "crop_output.png"))
            temp_output_pil.save(os.path.join(args.save_dir, crop_index, "temp_output.png"))
        
        # Save final images
        img.convert("RGB").save(os.path.join(args.save_dir, "gt", f"{case_idx}.png"))
        final_output_pil.save(os.path.join(args.save_dir, "pred", f"{case_idx}.png"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del pipeline

    return True

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

    args.save_dir = f"outputs/{args.datype}_{args.task}_{args.base_name}_{args.name}{args.variant}"
    os.makedirs(os.path.join(args.save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "pred"), exist_ok=True)

    # Initialize dataset
    test_dataset = initialize_dataset(args)
    # Initialize labels
    labels_gt = initialize_labels(test_dataset, args)

    # Start generation (multiprocessing)
    num_samples = len(test_dataset.annos)
    split_len = num_samples // len(args.gpu_ids)
    with ThreadPoolExecutor(max_workers=10) as executor:
        pipelines = []
        for gpu_id in args.gpu_ids:
            pipeline = initialize_pipeline(config, args, gpu_id)
            pipelines.append(pipeline)
        futures = []
        for wid, (gpu_id, pipeline) in enumerate(zip(args.gpu_ids, pipelines)):
            start_idx = wid * split_len
            end_idx = (wid + 1) * split_len if wid < len(args.gpu_ids) - 1 else num_samples
            futures.append(executor.submit(test_worker, test_dataset, labels_gt, pipeline, args, start_idx, end_idx, gpu_id))

        for future in tqdm(as_completed(futures), total=len(futures)):
            success = future.result()
        executor.shutdown(cancel_futures=True)

    print("\033[32mALL DONE\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--task", type=str, default="editing", choices=["generation", "editing"])
    parser.add_argument("--datype", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--name", type=str, default="{path to final model}") 
    parser.add_argument("--ckpt_idx", type=int, default=None)
    parser.add_argument("--use_base", action="store_true")
    parser.add_argument("--variant", type=str, default="")
    # dataset
    parser.add_argument("--labels_gt_path", type=str, default="./assets/labels_gt.json")
    parser.add_argument("--asset_root", type=str, default="./assets")
    parser.add_argument("--std_font_path", type=str, default="./assets/NotoSansSC-Regular.ttf")
    parser.add_argument("--char_root", type=str, default="./assets/chars")
    parser.add_argument("--test_data_roots", type=str, nargs="+", default=["{path to test data1}", "{path to test data2}", "..."])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--expand_ratio", type=float, default=0.0)
    parser.add_argument("--min_box_size", type=int, default=32)
    parser.add_argument("--max_box_items", type=int, default=50)
    parser.add_argument("--min_image_size", type=int, default=480)
    parser.add_argument("--max_image_size", type=int, default=4096)
    parser.add_argument("--min_alpha_ratio", type=float, default=0.0)
    parser.add_argument("--min_distances", type=float, nargs='+', default=[0.3, 0.4, 0.4, 0.0])
    parser.add_argument("--force_valid", type=bool, default=True)
    parser.add_argument("--max_cases_per_source", type=int, nargs='+', default=[300, 200, 300, 200])
    parser.add_argument("--switch_ratio", type=float, default=0.5)
    # inference
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--lamda_gate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--use_overshooting", type=int, default=1)
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[4])
    args = parser.parse_args()

    stop_event = threading.Event()
    def signal_handler(signum, frame):
        print("\033[1;31mKeyboard interrupt detected. Stopping all tasks...\033[0m")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)

    test(args)