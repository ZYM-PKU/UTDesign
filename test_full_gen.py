import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "utdesign"))
import gc
import argparse
from tqdm import tqdm

import itertools
import json
import random
import numpy as np

import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.utils import check_min_version
from openai import OpenAI

from tools.llamafact.utils import get_layout_plan_stage1, get_layout_plan_stage2, draw_bboxs

from utdesign.custom_encoder import FusionModule
from utdesign.custom_model import CustomTransformer2DModel
from utdesign.custom_pipeline import CustomPipeline
from utdesign.custom_scheduler import StochasticRFOvershotDiscreteScheduler
from trans_vae.custom_model import TransAutoencoderKL

from custom_dataset.utils.helper import render_chs, composite_image, get_rescaled_bbox, composite_prompt

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

def initialize_pipeline(args, gpu_id=0):

    print(f"Loading pipeline on gpu {gpu_id}...")

    vae = TransAutoencoderKL.from_pretrained(
        args.ckpt_dir,
        subfolder="trans_vae",
    )
    vae.enable_slicing()

    if args.use_overshooting:
        noise_scheduler = StochasticRFOvershotDiscreteScheduler.from_pretrained(
            args.base_model,
            subfolder="scheduler",
            cache_dir=args.cache_dir,
        )
        noise_scheduler.set_c(2.0)
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.base_model,
            subfolder="scheduler",
            cache_dir=args.cache_dir,
        )

    transformer = CustomTransformer2DModel.from_pretrained(
        args.ckpt_dir, 
        subfolder="transformer",
    )
        
    fusion_module: FusionModule = FusionModule.from_pretrained(
        args.ckpt_dir,
        subfolder="fusion_module",
        low_cpu_mem_usage=False, # disable this to avoid BUGs
    )

    pipeline = CustomPipeline(
        vae=vae,
        transformer=transformer,
        fusion_module=fusion_module,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(dtype=torch.bfloat16, device=torch.device("cuda", index=gpu_id))
    # pipeline.enable_model_cpu_offload(gpu_id=gpu_id) # save vram
    pipeline.load_lora_weights(args.ckpt_dir)

    return pipeline

def test_worker(annos, t2i_pipe, pl_client, pipeline, args, gpu_id=0):

    generator = torch.Generator(device=torch.device("cuda", index=gpu_id)).manual_seed(args.seed) if args.seed else None
    for case_idx in tqdm(range(len(annos))):
        for sample_idx in range(args.num_sample_per_case):
            anno = annos[case_idx]
            this_index = f"case_{case_idx}_sample_{sample_idx}"
            os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)

            width, height = anno["width"], anno["height"]
            ori_width, ori_height = width, height
            width, height = (width // 16)*16, (height // 16)*16
            labels = anno["labels"]
            caption = anno["caption"]
            n_lines = len(labels)

            case_ch_refs = []
            case_c_refs_w_pil = []
            for label in labels:
                ch_refs, c_refs_w_pil, _ = render_chs(list(label), font_path=args.std_font_path, resolution=args.resolution)
                case_ch_refs.append(ch_refs)
                case_c_refs_w_pil.append(c_refs_w_pil)

            # step 1: generate background using t2i
            instruct = "Create a raw design image (background and main foreground objects) with no text or typography included."
            prompt = instruct + " " + caption
            with torch.no_grad():
                bg = t2i_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=28,
                    max_sequence_length=512,
                    generator=generator,
                ).images[0]
            bg = bg.resize((width, height))
            bg.save(os.path.join(args.save_dir, this_index, "bg.png"))

            # step 2: generate layout
            case_crop_bg_pil = []
            case_prompt = []
            case_label = []
            case_crop_bboxs = []
            case_global_seg_bboxs = []
            case_r_crop_bboxs = []
            crop_bboxs = get_layout_plan_stage1(pl_client, args.planner_model, labels, bg, anno["caption"])

            for crop_bbox, label in zip(crop_bboxs, labels):
                x1, y1, x2, y2 = crop_bbox
                r_crop_bbox = get_rescaled_bbox([x1, y1, x2, y2], width, height)
                crop_bg = bg.crop(crop_bbox)

                instruction = "Design the style for the text rendered in the design image."
                prompt = composite_prompt(caption, [label], instruction)
                
                x_offset, y_offset = crop_bbox[:2]
                seg_bboxs = get_layout_plan_stage2(pl_client, args.planner_model, list(label), crop_bg)
                global_seg_bboxs = [[bbox[0]+x_offset, bbox[1]+y_offset, bbox[2]+x_offset, bbox[3]+y_offset] for bbox in seg_bboxs]

                case_crop_bg_pil.append(crop_bg)
                case_prompt.append(prompt)
                case_label.append(label)
                case_crop_bboxs.append(crop_bbox)
                case_global_seg_bboxs.append(global_seg_bboxs)
                case_r_crop_bboxs.append(r_crop_bbox)

            layout_ref = draw_bboxs(bg.copy(), args.std_font_path, case_label, case_crop_bboxs, list(itertools.chain(*case_global_seg_bboxs)))
            layout_ref.save(os.path.join(args.save_dir, this_index, "layout.png"))

            # step 3: render glyphs
            case_bg_pil = [bg] * n_lines
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
            glyph_images = output.images
            final_output_pil = bg.copy()

            for idx, (images, global_seg_bboxs) in enumerate(zip(glyph_images, case_global_seg_bboxs)):
                images_concat_pil = Image.new("RGBA", (args.resolution*len(images), args.resolution), (0, 0, 0, 0))
                for i, image in enumerate(images):
                    x, y = i * args.resolution, 0
                    images_concat_pil.paste(image, (x, y))
                images_concat_pil.save(os.path.join(args.save_dir, this_index, f"output_{idx}.png"))
                final_output_pil = composite_image(final_output_pil, images, global_seg_bboxs)

            final_output_pil = final_output_pil.resize((ori_width, ori_height), Image.LANCZOS)
            final_output_pil.save(os.path.join(args.save_dir, this_index, "pred.png"))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    del t2i_pipe, pipeline

def test_full_gen(args):

    if args.seed is not None:
        seed_everything(args.seed)

    args.save_dir = f"outputs/test_full_gen{args.variant}"
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize dataset
    annos = []
    with open(args.anno_path, "r") as f:
        annos = json.load(f)

    # Initialize planner
    pl_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")

    # Initialize pipeline
    t2i_pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
    t2i_pipe = t2i_pipe.to(torch.device("cuda", index=args.gpu_id))
    t2i_pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)
    pipeline = initialize_pipeline(args, args.gpu_id)

    test_worker(annos, t2i_pipe, pl_client, pipeline, args, args.gpu_id)

    print("\033[32mALL DONE\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--planner_model", type=str, default="vllm_layout_planner")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/utdesign_l16+8_lora64_res256")
    parser.add_argument("--cache_dir", type=str, default="/blob/pretrained/sd")
    parser.add_argument("--variant", type=str, default="")
    # dataset
    parser.add_argument("--std_font_path", type=str, default="./assets/NotoSansSC-Regular.ttf")
    parser.add_argument("--anno_path", type=str, default="./assets/full_gen_cases.json")
    parser.add_argument("--num_sample_per_case", type=int, default=4)
    # inference
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--lamda_gate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--use_overshooting", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    test_full_gen(args)