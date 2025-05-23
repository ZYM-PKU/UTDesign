import argparse
import os
import sys
sys.path.append(os.getcwd())
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from get_ocr_anno import get_ocr_anno, get_single_anno
from get_lama_bg import get_removal_bg
from get_sam_fg import extract_fg_images
from get_vllm_api_caption import anno_image_caption


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def data_split(args):

    image_root = os.path.join(args.target_data_root, "images")
    image_paths = os.listdir(image_root)
    random.shuffle(image_paths)

    n_train = int(len(image_paths) * args.train_ratio)
    train_anno, val_anno = [], []
    for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        real_image_path = os.path.join(image_root, image_path)
        img = Image.open(real_image_path).convert("RGB")
        width, height = img.size
        item = {
            "image_name": image_path,
            "height": height,
            "width": width,
        }
        if idx < n_train:
            train_anno.append(item)
        else:
            val_anno.append(item)
    
    with open(os.path.join(args.target_data_root, "train_anno.json"), "w") as f:
        json.dump(train_anno, f, indent=4, ensure_ascii=False)
        f.flush()
    
    with open(os.path.join(args.target_data_root, "val_anno.json"), "w") as f:
        json.dump(val_anno, f, indent=4, ensure_ascii=False)
        f.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_data_root", type=str, default="{data root}", help="Root directory of the target data.")
    parser.add_argument("--ocr_model_path", type=str, default="PP-OCRv4", help="OCR model path.")
    parser.add_argument("--det_model_path", type=str, default="{path to detection model}", help="Detection model path.")
    parser.add_argument("--lama_model_path", type=str, default="{path to lama model}", help="Path to the LAMA model.")
    parser.add_argument("--sam_model_path", type=str, default="{path to sam model}", help="Checkpoint directory.")
    parser.add_argument("--vllm_model_path", type=str, default="gpt-4o", help="Model to use for image captioning.")
    parser.add_argument("--image_base", type=str, default="images", help="Base directory of the images.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train ratio.")
    parser.add_argument("--min_box_size", type=int, default=32, help="Minimum size of the bounding box.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    seed_everything(args.seed)

    pipeline_stages = ["split", "ocr", "single", "bg", "fg", "caption"]

    ### split data
    if "split" in pipeline_stages:
        print("\033[38;5;208m[INFO] Stage 1: Splitting data...\033[0m")
        data_split(args)
        print("\033[32m[INFO] Data split done.\033[0m")

    ### get ocr
    if "ocr" in pipeline_stages:
        print("\033[38;5;208m[INFO] Stage 2: Getting OCR annotation...\033[0m")
        get_ocr_anno(args)
        print("\033[32m[INFO] OCR annotation done.\033[0m")

    ### get single ocr
    if "single" in pipeline_stages:
        print("\033[38;5;208m[INFO] Stage 3: Getting single OCR annotation...\033[0m")
        get_single_anno(args)
        print("\033[32m[INFO] Single OCR annotation done.\033[0m")

    ### get lama bg
    if "bg" in pipeline_stages:
        print("\033[38;5;208m[INFO] Stage 4: Getting LAMA background...\033[0m")
        get_removal_bg(args)
        print("\033[32m[INFO] LAMA background done.\033[0m")

    ### get sam fg
    if "fg" in pipeline_stages:
        print("\033[38;5;208m[INFO] Stage 5: Getting SAM foreground...\033[0m")
        extract_fg_images(args)
        print("\033[32m[INFO] SAM foreground done.\033[0m")

    ### get vllm caption
    if "caption" in pipeline_stages:
        print("\033[38;5;208m[INFO] Stage 6: Getting VLLM caption...\033[0m")
        anno_image_caption(args)
        print("\033[32m[INFO] VLLM caption done.\033[0m")

    print("\033[32m[INFO] ALL DONE!\033[0m")