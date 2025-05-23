import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "utdesign"))
sys.path.append(os.path.join(os.getcwd(), "tools", "lama"))
import gc
import argparse
from tqdm import tqdm

import json
import random
import numpy as np

import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import check_min_version
from openai import OpenAI
from ultralytics import YOLO

from tools.lama.bin.predict import load_lama_model, inpaint_image
from tools.llamafact.utils import get_layout_plan_stage2

from utdesign.custom_encoder import FusionModule
from utdesign.custom_model import CustomTransformer2DModel
from utdesign.custom_pipeline import CustomPipeline
from utdesign.custom_scheduler import StochasticRFOvershotDiscreteScheduler
from trans_vae.custom_model import TransAutoencoderKL

from custom_dataset.utils.helper import render_chs, composite_image, reorder_bboxs

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

    return pipeline

def detect_bbox(image, det_model):

    det_result = det_model.predict(
        image,
        imgsz=320,
        conf=0.1,
        iou=0.7,
        max_det=100,
        half=False,
    )[0]

    res_bboxs = det_result.boxes
    name_list = det_result.names
    pred_labels = [name_list[idx] for idx in res_bboxs.cls.to(torch.int32).tolist()]
    pred_bboxs = res_bboxs.xyxy.to(torch.int32).tolist()
    bbox_indices = reorder_bboxs(pred_bboxs)

    s_bboxs = []
    s_labels = []
    s_refs = []
    for b_idx in bbox_indices:
        pred_label = pred_labels[b_idx]
        pred_bbox = pred_bboxs[b_idx]
        s_x1, s_y1, s_x2, s_y2 = pred_bbox
        pred_bbox = [s_x1, s_y1, s_x2, s_y2]
        s_ref = image.crop((s_x1, s_y1, s_x2, s_y2)).resize((args.resolution, args.resolution))

        if pred_bbox not in s_bboxs:
            s_bboxs.append(pred_bbox)
            s_labels.append(pred_label)
            s_refs.append(s_ref)
            
    label = "".join(s_labels)
        
    return label, s_bboxs, s_refs

def test_worker(annos, det_model, lama_model, pl_client, pipeline, args, gpu_id=0):

    generator = torch.Generator(device=torch.device("cuda", index=gpu_id)).manual_seed(args.seed) if args.seed else None
    for case_idx in tqdm(range(len(annos))):
        anno = annos[case_idx]
        this_index = f"case_{case_idx}"
        os.makedirs(os.path.join(args.save_dir, this_index), exist_ok=True)

        ori_path = anno["image_path"]
        ori_image = Image.open(ori_path).convert("RGB")
        width, height = ori_image.size
        edited_text = anno["edited_text"]

        # step 1: detect glyphs
        _, s_bboxs, s_refs = detect_bbox(ori_image, det_model)
        _, c_refs_w_pil, _ = render_chs(list(edited_text), font_path=args.std_font_path, resolution=args.resolution)

        # step 2: remove original text
        mask = Image.new("L", (width, height), 0)
        for s_bbox in s_bboxs:
            x1, y1, x2, y2 = s_bbox
            mask.paste(255, (x1, y1, x2, y2))
        bg = inpaint_image(ori_image, mask, lama_model)
        bg.save(os.path.join(args.save_dir, this_index, "bg.png"))

        # step 3: plan fine-grained layout
        new_s_bboxs = get_layout_plan_stage2(pl_client, args.planner_model, list(edited_text), bg)

        # step 4: render text
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipeline(
                height=args.resolution,
                width=args.resolution,
                cfg_scale=args.cfg,
                lamda_gate=args.lamda_gate,
                c_refs_pil=[c_refs_w_pil],
                s_refs_pil=[s_refs],
                num_inference_steps=args.steps,
                generator=generator,
            )
        glyph_images = output.images[0]
        images_concat_pil = Image.new("RGBA", (args.resolution*len(glyph_images), args.resolution), (0, 0, 0, 0))
        for i, image in enumerate(glyph_images):
            x, y = i * args.resolution, 0
            images_concat_pil.paste(image, (x, y))
        images_concat_pil.save(os.path.join(args.save_dir, this_index, "glyphs.png"))
        final_output_pil = composite_image(bg.copy(), glyph_images, new_s_bboxs)
        final_output_pil.save(os.path.join(args.save_dir, this_index, "edited.png"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del det_model, lama_model, pipeline

def test_edit(args):

    if args.seed is not None:
        seed_everything(args.seed)

    args.save_dir = f"outputs/test_edit{args.variant}"
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize dataset
    annos = []
    with open(args.anno_path, "r") as f:
        annos = json.load(f)

    # Initialize planner
    pl_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")

    # Initialize tools
    det_model = YOLO(args.det_model_path).to(device=torch.device("cuda", index=args.gpu_id))
    lama_model = load_lama_model(args.lama_model_path).to(device=torch.device("cuda", index=args.gpu_id))

    # Initialize pipeline
    pipeline = initialize_pipeline(args, args.gpu_id)

    test_worker(annos, det_model, lama_model, pl_client, pipeline, args, gpu_id=args.gpu_id)

    print("\033[32mALL DONE\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--planner_model", type=str, default="vllm_layout_planner")
    parser.add_argument("--det_model_path", type=str, default="./checkpoints/tools/yolo.pt")
    parser.add_argument("--lama_model_path", type=str, default="./checkpoints/tools/big-lama")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/utdesign_l16+8_lora64_res256")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default="")
    # dataset
    parser.add_argument("--std_font_path", type=str, default="./assets/NotoSansSC-Regular.ttf")
    parser.add_argument("--anno_path", type=str, default="./assets/edit_cases.json")
    # inference
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--lamda_gate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--use_overshooting", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    test_edit(args)