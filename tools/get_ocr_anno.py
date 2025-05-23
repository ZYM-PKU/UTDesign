import gc
import os
import sys
sys.path.append(os.getcwd())
import json
import torch
from tqdm import tqdm
from PIL import Image

from paddleocr import PaddleOCR
from ultralytics import YOLO
from custom_dataset.utils.helper import get_bbox_from_points, reorder_bboxs, get_points_from_bbox
from concurrent.futures import ThreadPoolExecutor, as_completed


def ocr_worker(args, anno_json, ocr_engine, start_idx, end_idx):

    for anno_idx, anno in tqdm(enumerate(anno_json[start_idx:end_idx]), total=end_idx-start_idx):
        img_path = os.path.join(args.target_data_root, args.image_base, anno["image_name"])
        ocr_result = ocr_engine.ocr(img_path, cls=False)[0]

        if ocr_result is None: 
            print(f"No text found for image {img_path}")
            anno["bboxs"] = []
            continue
        else:
            bboxs = []
            for item in ocr_result:
                points = item[0]
                label = item[1][0]

                x1, y1, x2, y2 = get_bbox_from_points(points)
                if min(x2-x1, y2-y1) < args.min_box_size:
                    continue
                bbox = {
                    "label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                bboxs.append(bbox)
            anno["bboxs"] = bboxs

        gc.collect()

def get_ocr_anno(args):

    for datype in ["train", "val"]:
        anno_json_path = os.path.join(args.target_data_root, f"{datype}_anno.json")
        with open(anno_json_path, "r") as f:
            anno_json = json.load(f)

        num_samples = len(anno_json)
        splits = len(args.gpu_ids)
        split_len = num_samples // splits
        with ThreadPoolExecutor(max_workers=10) as executor:

            futures = []
            ocr_engines = [PaddleOCR(use_angle_cls=True, lang='ch', ocr_version=args.ocr_model_path) for _ in range(splits)]
            for wid, ocr_engine in enumerate(ocr_engines):
                start_idx = wid * split_len
                end_idx = (wid + 1) * split_len if wid < splits - 1 else num_samples
                futures.append(executor.submit(ocr_worker, args, anno_json, ocr_engine, start_idx, end_idx))

            try:
                for future in tqdm(as_completed(futures), total=len(futures)):
                    future.result()
            except KeyboardInterrupt:
                print("\033[1;31mKeyboard interrupt detected. Stopping all tasks...\033[0m")
                for future in futures:
                    future.cancel()
                executor.shutdown(cancel_futures=True)

        with open(anno_json_path, "w") as f:
            json.dump(anno_json, f, indent=4, ensure_ascii=False)
            f.flush()

def get_single_anno(args):

    det_model = YOLO(args.det_model_path).to(device=torch.device(f"cuda:{args.gpu_ids[0]}"))

    for datype in ["train", "val"]:
        anno_json_path = os.path.join(args.target_data_root, f"{datype}_anno.json")
        with open(anno_json_path, "r") as f:
            anno_json = json.load(f)

        for anno_idx, anno in tqdm(enumerate(anno_json), total=len(anno_json)):
            bboxs = anno["bboxs"]
            if len(bboxs) == 0:
                continue
            
            img_path = os.path.join(args.target_data_root, args.image_base, anno["image_name"])
            img = Image.open(img_path).convert("RGB")
            crop_imgs = []
            for bbox in bboxs:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                crop_img = img.crop((x1, y1, x2, y2))
                crop_imgs.append(crop_img)

            det_results = det_model.predict(
                crop_imgs,
                imgsz=320,
                conf=0.1,
                iou=0.7,
                max_det=100,
                half=False,
            )

            for bbox, det_result in zip(bboxs, det_results):
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                label = bbox["label"]
                res_bboxs = det_result.boxes
                name_list = det_result.names
                pred_labels = [name_list[idx] for idx in res_bboxs.cls.to(torch.int32).tolist()]
                pred_bboxs = res_bboxs.xyxy.to(torch.int32).tolist()
                bbox_indices = reorder_bboxs(pred_bboxs)

                item_list = []
                for b_idx in bbox_indices:
                    pred_label = pred_labels[b_idx]
                    pred_bbox = pred_bboxs[b_idx]
                    if pred_label not in label:
                        continue
                    pred_points = get_points_from_bbox(pred_bbox, x_offset=x1, y_offset=y1)
                    item = dict(
                        label=pred_label,
                        points=pred_points,
                    )
                    item_list.append(item)
                label = "".join([item["label"] for item in item_list])

                bbox["label"] = label
                bbox["items"] = item_list

        with open(anno_json_path, "w") as f:
            json.dump(anno_json, f, indent=4, ensure_ascii=False)
            f.flush()