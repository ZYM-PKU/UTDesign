import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "utdesign"))
sys.path.append(os.path.join(os.getcwd(), "tools", "lama"))
import gc
import argparse
import torch
import numpy as np

import random
from PIL import Image

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from diffusers.utils import check_min_version
from openai import OpenAI
from ultralytics import YOLO

from tools.lama.bin.predict import load_lama_model, inpaint_image
from tools.llamafact.utils import get_layout_plan_stage1, get_layout_plan_stage2

from utdesign.custom_encoder import FusionModule
from utdesign.custom_model import CustomTransformer2DModel
from utdesign.custom_pipeline import CustomPipeline
from utdesign.custom_scheduler import StochasticRFOvershotDiscreteScheduler
from trans_vae.custom_model import TransAutoencoderKL

from custom_dataset.utils.helper import (
    get_bbox_from_points, 
    reorder_bboxs, 
    expand_bbox, 
    get_bbox_from_mask,
    render_chs,
    composite_image,
    get_rescaled_bbox,
    composite_prompt
)

from paddleocr import PaddleOCR # MUST import paddleocr here to avoid Malignant BUGS!!!

import gradio as gr

check_min_version("0.31.0.dev0")

### global variables
edit_bboxs = []
gen_inputs = {}

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

def load_lora():
    global pipeline
    print("\033[38;5;208m[INFO] Loading LoRA...\033[0m")
    pipeline.load_lora_weights(args.ckpt_dir)

def drop_lora():
    global pipeline
    print("\033[38;5;208m[INFO] Dropping LoRA...\033[0m")
    pipeline.unload_lora_weights()
    gc.collect()
    torch.cuda.empty_cache()

def calc_biased_iou(box1, box2):
    
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0

    union = (box2[2]-box2[0])*(box2[3]-box2[1])

    return float(inter)/union

def arrange_bboxs(crop_bbox, num_s_bboxs):

    x1, y1, x2, y2 = crop_bbox
    total_width = x2 - x1
    total_height = y2 - y1
    
    if total_width > total_height:
        side = min(total_height, total_width // num_s_bboxs)
        total_boxes_width = side * num_s_bboxs

        start_x = x1 + (total_width - total_boxes_width) // 2
        start_y = y1 + (total_height - side) // 2

        s_bboxes = []
        for i in range(num_s_bboxs):
            box_x1 = start_x + i * side
            box_y1 = start_y
            box_x2 = box_x1 + side
            box_y2 = box_y1 + side
            s_bboxes.append([int(box_x1), int(box_y1), int(box_x2), int(box_y2)])

    elif total_width <= total_height:
        side = min(total_width, total_height // num_s_bboxs)
        total_boxes_height = side * num_s_bboxs

        start_x = x1 + (total_width - side) // 2
        start_y = y1 + (total_height - total_boxes_height) // 2

        s_bboxes = []
        for i in range(num_s_bboxs):
            box_x1 = start_x
            box_y1 = start_y + i * side
            box_x2 = box_x1 + side
            box_y2 = box_y1 + side
            s_bboxes.append([int(box_x1), int(box_y1), int(box_x2), int(box_y2)])

    return s_bboxes

def edit_image(input_image, text, cfg, lambda_gate, seed):
    global edit_bboxs

    image = input_image.get("background")
    mask = input_image.get("layers")[0]

    if mask is None:
        print("Please draw a mask on the image!")
        return None, None

    crop_bbox = get_bbox_from_mask(mask)
    if crop_bbox is None:
        print("Illegal mask: you forget to paint the mask!")
        return None, None

    # Step 1: Inpaint image to remove text
    print("\033[38;5;208m[INFO] Inpainting image...\033[0m")
    inpainted_image = inpaint_image(image, mask, lama_model)

    # Step 2: Generate new text
    print("\033[38;5;208m[INFO] Generating new text...\033[0m")
    valid = False
    target_s_bboxs = []
    target_s_refs = []
    for i, bbox in enumerate(edit_bboxs):
        s_bboxs = []
        for s_bbox in bbox["s_bboxs"]:
            iou = calc_biased_iou(crop_bbox, s_bbox)
            if iou > 0.6:
                s_bboxs.append(s_bbox)
        if len(s_bboxs) > 0:
            valid = True
            target_s_bboxs += s_bboxs
            target_s_refs += bbox["s_refs"]
    
    if not valid:
        return inpainted_image, None
    
    if len(text) != len(target_s_bboxs):
        target_s_bboxs = arrange_bboxs(crop_bbox, len(text))
    
    _, c_refs_w_pil, _ = render_chs(list(text), font_path=args.std_font_path, resolution=args.resolution)

    generator.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipeline(
            height=args.resolution,
            width=args.resolution,
            cfg_scale=cfg,
            lamda_gate=lambda_gate,
            c_refs_pil=[c_refs_w_pil],
            s_refs_pil=[target_s_refs],
            num_inference_steps=args.steps,
            generator=generator,
        )
    ch_images = output.images[0]

    cb_bg = Image.open("./assets/checkerboard.png").convert("RGBA").resize((args.resolution, args.resolution))
    cb_images = [Image.alpha_composite(cb_bg, img) for img in ch_images]

    output_image = composite_image(inpainted_image, ch_images, target_s_bboxs)

    print("\033[32mALL DONE\033[0m")

    return inpainted_image, output_image, cb_images

def detect_bbox(image):

    ocr_result = ocr_engine.ocr(np.array(image), cls=False)[0]
    if ocr_result is None: 
        print("No text found in image!")
        return []
    else:
        bboxs = []
        for item in ocr_result:
            points = item[0]
            label = item[1][0]

            x1, y1, x2, y2 = get_bbox_from_points(points)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < args.min_bbox_size:
                continue
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), image.height, image.width, 0.1, no_random=True)
            crop_img = image.crop((x1, y1, x2, y2))
            det_result = det_model.predict(
                crop_img,
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
                if pred_label not in label:
                    continue
                pred_bbox = pred_bboxs[b_idx]
                s_x1, s_y1, s_x2, s_y2 = pred_bbox
                pred_bbox = [s_x1+x1, s_y1+y1, s_x2+x1, s_y2+y1]
                s_ref = crop_img.crop((s_x1, s_y1, s_x2, s_y2)).resize((args.resolution, args.resolution))

                if pred_bbox not in s_bboxs:
                    s_bboxs.append(pred_bbox)
                    s_labels.append(pred_label)
                    s_refs.append(s_ref)
            
            label = "".join(s_labels)

            bbox = {
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "s_bboxs": s_bboxs,
                "s_refs": s_refs,
            }
            bboxs.append(bbox)
        
        return bboxs

def detect_and_render(input_image):
    global edit_bboxs

    print("\033[38;5;208m[INFO] Detecting Text Bboxes...\033[0m")
    
    # Ensure we're working with an RGB image
    image = input_image["background"].convert("RGB")
    bboxs = detect_bbox(image)
    edit_bboxs = bboxs

    annos = []
    for bbox in bboxs:
        label = bbox["label"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        annos.append(((x1, y1, x2, y2), label))
        for ch, s_bbox in zip(label, bbox["s_bboxs"]):
            x1, y1, x2, y2 = s_bbox
            annos.append(((x1, y1, x2, y2), ch))
    
    return (image, annos)

def generate_bg(input_text, height, width, seed):

    print("\033[38;5;208m[INFO] Generating background...\033[0m")

    # step 1: generate background using t2i
    instruct = "Create a raw graphic design (background and main foreground objects) with no text or typography included. The design image is about:"
    prompt = instruct + " " + input_text
    generator.manual_seed(seed)
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

    return bg, (bg, [])

def plan_layout(input_image, input_text, layout_table):

    bg_image = input_image.get("background")
    labels = []
    for label in layout_table.Text:
        if not label: break
        labels.append(label)
    if not labels:
        print("Please input text to render!")
        return None, None
    
    # step 2: generate layout
    print("\033[38;5;208m[INFO] Planning Layout...\033[0m")
    stage1_bboxs = get_layout_plan_stage1(pl_client, args.planner_model, labels, bg_image, input_text)

    annos = []
    for i, (label, bbox) in enumerate(zip(labels, stage1_bboxs)):
        x1, y1, x2, y2 = bbox
        annos.append(((x1, y1, x2, y2), label))
        layout_table.iloc[i] = [labels[i], x1, y1, x2, y2]
    
    return (bg_image, annos), layout_table

def update_layout_annotations(input_image, layout_table):

    bg_image = input_image.get("background")
    
    annos = []
    for line in layout_table.iloc:
        if not line[0]: continue
        label = line[0]
        try:
            x1, y1, x2, y2 = int(line[1]), int(line[2]), int(line[3]), int(line[4])
            annos.append(((x1, y1, x2, y2), label))
        except:
            continue

    return (bg_image, annos)

def plan_layout_fine_grained(input_image, input_text, layout_table):
    global gen_inputs
    
    bg_image = input_image.get("background")
    width, height = bg_image.size

    labels = []
    crop_bboxs = []
    for line in layout_table.iloc:
        if not line[0]: break
        labels.append(line[0])
        x1, y1, x2, y2 = int(line[1]), int(line[2]), int(line[3]), int(line[4])
        crop_bboxs.append([x1, y1, x2, y2])

    if not labels:
        print("Please input text to render!")
        return None, None
    
    n_lines = len(labels)
    
    # step 2: generate layout
    print("\033[38;5;208m[INFO] Planning Fine-grained Layout...\033[0m")
    case_r_crop_bboxs = []
    case_global_seg_bboxs = []
    case_c_refs_w_pil = []
    case_prompts = []
    for label, crop_bbox in zip(labels, crop_bboxs):
        x1, y1, x2, y2 = crop_bbox
        r_crop_bbox = get_rescaled_bbox([x1, y1, x2, y2], width, height)

        x_offset, y_offset = crop_bbox[:2]
        crop_bg = bg_image.crop(crop_bbox)
        seg_bboxs = get_layout_plan_stage2(pl_client, args.planner_model, list(label), crop_bg)
        global_seg_bboxs = [[bbox[0]+x_offset, bbox[1]+y_offset, bbox[2]+x_offset, bbox[3]+y_offset] for bbox in seg_bboxs]
    
        case_r_crop_bboxs.append(r_crop_bbox)
        case_global_seg_bboxs.append(global_seg_bboxs)

        instruction = "Design the style (font, color, texture) for the text rendered in the design image."
        prompt = composite_prompt(input_text, [label], instruction)
        case_prompts.append(prompt)

        _, c_refs_w_pil, _ = render_chs(list(label), font_path=args.std_font_path, resolution=args.resolution)
        case_c_refs_w_pil.append(c_refs_w_pil)

    case_bg_pil = [bg_image] * n_lines
    gen_inputs = dict(
        case_c_refs_w_pil=case_c_refs_w_pil,
        case_bg_pil=case_bg_pil,
        case_r_crop_bboxs=case_r_crop_bboxs,
        case_prompts=case_prompts,
        case_labels=labels,
        case_global_seg_bboxs=case_global_seg_bboxs,
    )

    annos = []
    for i, (label, crop_bbox, seg_bboxs) in enumerate(zip(labels, crop_bboxs, case_global_seg_bboxs)):
        x1, y1, x2, y2 = crop_bbox
        annos.append(((x1, y1, x2, y2), label))
        for ch, seg_bbox in zip(label, seg_bboxs):
            x1, y1, x2, y2 = seg_bbox
            annos.append(((x1, y1, x2, y2), ch))
    
    return (bg_image, annos)

def render_text(seed):
    global gen_inputs
    
    # Step 3: Render text
    print("\033[38;5;208m[INFO] Rendering text...\033[0m")
    generator.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipeline(
            height=args.resolution,
            width=args.resolution,
            cfg_scale=args.cfg,
            lamda_gate=args.lamda_gate,
            c_refs_pil=gen_inputs["case_c_refs_w_pil"],
            bg_pil=gen_inputs["case_bg_pil"],
            reg_bboxs=gen_inputs["case_r_crop_bboxs"],
            prompt=gen_inputs["case_prompts"],
            label=gen_inputs["case_labels"],
            num_inference_steps=args.steps,
            generator=generator,
        )
    ch_images = output.images

    bg_image = gen_inputs["case_bg_pil"][0]
    final_output_pil = bg_image.copy()
    cb_bg = Image.open("./assets/checkerboard.png").convert("RGBA").resize((args.resolution, args.resolution))
    cb_images = []
    for images, global_seg_bboxs in zip(ch_images, gen_inputs["case_global_seg_bboxs"]):
        final_output_pil = composite_image(final_output_pil, images, global_seg_bboxs)
        concatenated = Image.new("RGBA", (args.resolution*len(images), args.resolution), (0, 0, 0, 0))
        x_offset = 0
        for img in images:
            concatenated.paste(cb_bg, (x_offset, 0))
            concatenated.paste(img, (x_offset, 0))
            x_offset += args.resolution
        cb_images.append(concatenated)

    print("\033[32mALL DONE\033[0m")

    return (final_output_pil, []), cb_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meta
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--min_bbox_size", type=int, default=32, help="Minimum bounding box size.")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of labels.")
    parser.add_argument("--enable_cpu_offload", type=int, default=1, help="Enable CPU offload.")
    # tools
    parser.add_argument("--ocr_model_path", type=str, default="PP-OCRv4", help="OCR model path.")
    parser.add_argument("--det_model_path", type=str, default="./checkpoints/tools/yolo.pt", help="Detection model path.")
    parser.add_argument("--lama_model_path", type=str, default="./checkpoints/tools/big-lama", help="Path to the LAMA model.")
    parser.add_argument("--planner_model", type=str, default="vllm_layout_planner")
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev", help="Name of t2i pipeline.")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/utdesign_l16+8_lora64_res256")
    parser.add_argument("--cache_dir", type=str, default="/blob/pretrained/sd", help="Cache dir.")
    # pipeline
    parser.add_argument("--std_font_path", type=str, default="./assets/NotoSansSC-Regular.ttf", help="Path to the standard font.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--lamda_gate", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--use_overshooting", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # Initialize ALL
    print("\033[38;5;208m[INFO] Initializing Models...\033[0m")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', ocr_version=args.ocr_model_path)
    det_model = YOLO(args.det_model_path).to(device=torch.device("cuda", index=args.gpu_id))
    lama_model = load_lama_model(args.lama_model_path).to(device=torch.device("cuda", index=args.gpu_id))
    t2i_pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
    t2i_pipe = t2i_pipe.to(torch.device("cuda", index=args.gpu_id))
    if args.enable_cpu_offload:
        t2i_pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)
    generator = torch.Generator(device=torch.device("cuda", index=args.gpu_id)).manual_seed(args.seed)
    pl_client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    pipeline = initialize_pipeline(args, args.gpu_id)

    # Demo Page
    with gr.Blocks(title="UTDesign") as demo:
        gr.Markdown("## A UTDesign Demo")
        
        with gr.Tab("Design Image Editing") as tab_editing:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.ImageEditor(label="Input Image", type="pil", brush=gr.Brush(default_color="white"), format="png", height=args.height//2)
                        with gr.Column():
                            detect_image = gr.AnnotatedImage(label="Detect", format="png", height=args.height//2)
                    with gr.Row():
                        edit_text_input = gr.Textbox(label="Edit Text")
                    with gr.Row():
                        with gr.Column():
                            cfg_input = gr.Slider(label="CFG", minimum=1.0, maximum=10.0, value=args.cfg, step=0.1)
                        with gr.Column():
                            lambda_input = gr.Slider(label="Lambda", minimum=0.0, maximum=5.0, value=args.lamda_gate, step=0.1)
                    with gr.Row():
                        seed_input = gr.Slider(label="seed", minimum=0, maximum=999999, value=42, step=1)
                    with gr.Row():
                        edit_button = gr.Button("Edit", variant="primary")
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            inpainted_image = gr.ImageEditor(label="Inpainted", interactive=False, type="pil", format="png", height=args.height//2)
                        with gr.Column():
                            edit_output_image = gr.ImageEditor(label="Result", interactive=False, type="pil", format="png", height=args.height//2)
                    with gr.Row():
                        edit_output_rgbas = gr.Gallery(label="RGBA Outputs", format="png", height=args.resolution, preview=True)
        
        with gr.Tab("Design Image Generation") as tab_generation:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Background Generation / Upload")
                    bg_input_image = gr.ImageEditor(label="Background", type="pil", format="png", height=args.height//2)
                    bg_text_input = gr.Textbox(label="Prompt")
                    with gr.Row():
                        height_input = gr.Number(label="Height", value=args.height)
                        width_input = gr.Number(label="Width", value=args.width)
                    bg_generate_button = gr.Button("Generate")
                with gr.Column():
                    gr.Markdown("#### Step 2: Layout Planning")
                    bg_output_image = gr.AnnotatedImage(label="Layout", format="png", height=args.height//2)
                    layout_table = gr.Dataframe(
                        label="Text Instances",
                        headers=["Text", "x1", "y1", "x2", "y2"],
                        datatype=["str", "number", "number", "number", "number"],
                        row_count=(3, "dynamic"),
                        col_count=(5, "fixed"),
                        show_row_numbers=True,
                    )
                    with gr.Row():
                        layout_plan_button = gr.Button("Plan Coarse Layout")
                        modify_plan_button = gr.Button("Modify")
            gr.Markdown("---", elem_id="separator")  # separator to clearly divide the steps
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 3: Text Rendering")
                    fine_layout_image = gr.AnnotatedImage(label="Fine-grained Layout", format="png", height=args.height//2)
                    with gr.Row():
                        fine_layout_button = gr.Button("Plan Fine Layout")
                        text_render_button = gr.Button("Render Text", variant="primary")
                with gr.Column():
                    final_output_image = gr.AnnotatedImage(label="Result", format="png", height=args.height//2)
                    final_output_rgbas = gr.Gallery(label="RGBA Outputs", format="png", height=args.resolution, preview=True)

        # Define Callbacks
        tab_editing.select(fn=drop_lora)
        tab_generation.select(fn=load_lora)
        input_image.upload(fn=detect_and_render, inputs=input_image, outputs=detect_image)
        edit_button.click(fn=edit_image, inputs=[input_image, edit_text_input, cfg_input, lambda_input, seed_input], outputs=[inpainted_image, edit_output_image, edit_output_rgbas])
        bg_generate_button.click(fn=generate_bg, inputs=[bg_text_input, height_input, width_input, seed_input], outputs=[bg_input_image, bg_output_image])
        layout_plan_button.click(fn=plan_layout, inputs=[bg_input_image, bg_text_input, layout_table], outputs=[bg_output_image, layout_table])
        modify_plan_button.click(fn=update_layout_annotations, inputs=[bg_input_image, layout_table], outputs=bg_output_image)
        fine_layout_button.click(fn=plan_layout_fine_grained, inputs=[bg_input_image, bg_text_input, layout_table], outputs=[fine_layout_image])
        text_render_button.click(fn=render_text, inputs=[seed_input], outputs=[final_output_image, final_output_rgbas])

    demo.queue()
    demo.launch()