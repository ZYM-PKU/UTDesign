import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm


def get_removal_bg(args):

    target_image_dir = os.path.join(args.target_data_root, "images")
    temp_data_root = os.path.join(args.target_data_root, "temp")
    cache_data_root = os.path.join(args.target_data_root, "cache")
    inpainted_data_root = os.path.join(args.target_data_root, "inpainted")
    os.makedirs(temp_data_root, exist_ok=True)
    os.makedirs(cache_data_root, exist_ok=True)
    os.makedirs(inpainted_data_root, exist_ok=True)

    anno_json = []
    for datype in ["train", "val"]:
        anno_json_path = os.path.join(args.target_data_root, f"{datype}_anno.json")
        with open(anno_json_path, "r") as f:
            anno_json += json.load(f)

    for anno in tqdm(anno_json):
        
        bboxs = anno["bboxs"]
        seg_bboxs = []
        for bbox in bboxs:
            seg_bboxs.extend([item["points"] for item in bbox["items"]])

        image_name = anno["image_name"]
        image_path = os.path.join(target_image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_path}")
            continue
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        all_points = np.concatenate([np.array(seg_bbox) for seg_bbox in seg_bboxs], axis=0)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        big_bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        
        # for seg_bbox in seg_bboxs:
        #     mask = cv2.fillConvexPoly(mask, np.array(seg_bbox, dtype=np.int32), color=255)
        mask = cv2.fillConvexPoly(mask, np.array(big_bbox, dtype=np.int32), color=255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.int8), iterations=3) # dilate

        temp_image_path = os.path.join(temp_data_root, image_name)
        temp_mask_path = os.path.join(temp_data_root, f"{image_name.split('.')[0]}_mask001.png")
        if os.path.exists(temp_image_path) and os.path.exists(temp_mask_path):
            continue
        shutil.copy(image_path, temp_image_path)
        cv2.imwrite(temp_mask_path, mask)

    os.environ["PYTORCH_WEIGHTS_ONLY"] = "0"
    os.system(f"python tools/lama/bin/predict.py model.path={args.lama_model_path} indir={temp_data_root} outdir={cache_data_root}")

    cache_image_paths = os.listdir(cache_data_root)
    for cache_image_path in tqdm(cache_image_paths):
        cache_image_path = os.path.join(cache_data_root, cache_image_path)
        inpainted_image_path = cache_image_path.replace("cache", "inpainted").replace("_mask001", "")
        shutil.copy(cache_image_path, inpainted_image_path)

    shutil.rmtree(temp_data_root)
    shutil.rmtree(cache_data_root)