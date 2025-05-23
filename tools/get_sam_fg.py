import os
import sys
sys.path.append(os.getcwd())
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import SamModel, SamProcessor
from custom_dataset.utils.helper import get_bbox_from_points, expand_bbox


def get_a_mask_sam(crop_img, seg_bboxs, processor, model):

    inputs = processor(crop_img, input_boxes=[seg_bboxs], return_tensors="pt").to(model.device)
    outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu(),
        binarize=False,
    )[0]
    # Min-max normalization
    masks = masks.clamp(min=0.0, max=1.0)

    mask = masks.max(dim=0)[0][0]
    mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype("uint8")

    return mask

def extract_fg_images(args):

    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_model = SamModel.from_pretrained(args.sam_model_path).to(torch.device("cuda", index=args.gpu_ids[0]))

    anno_json = []
    for datype in ["train", "val"]:
        anno_json_path = os.path.join(args.target_data_root, f"{datype}_anno.json")
        with open(anno_json_path, "r") as f:
            anno_json += json.load(f)

    gt_dir = os.path.join(args.target_data_root, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    for anno_idx, anno in tqdm(enumerate(anno_json), total=len(anno_json)):
        img_path = os.path.join(args.target_data_root, "images", anno["image_name"])
        img = Image.open(img_path).convert("RGB")

        img_rgba = Image.new("RGBA", img.size, (255, 255, 255, 0))
        bboxs = anno["bboxs"]
        for b_id, bbox in enumerate(bboxs):
            if bbox["label"] == "":
                continue
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), img.height, img.width, ratio=0.2)

            crop_img = img.crop((x1, y1, x2, y2))
            seg_bboxs = []
            for seg in bbox["items"]:
                seg_bbox = get_bbox_from_points(seg["points"], x1, y1)
                seg_bboxs.append(seg_bbox)

            mask = get_a_mask_sam(crop_img, seg_bboxs, sam_processor, sam_model)
            mask_pil = Image.fromarray(mask)
            crop_img_rgba = Image.merge("RGBA", (*crop_img.split(), mask_pil))
            img_rgba.paste(crop_img_rgba, (int(x1), int(y1)), mask_pil)
        
        img_rgba.save(os.path.join(gt_dir, anno["image_name"]))