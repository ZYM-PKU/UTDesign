import os
import sys
sys.path.append(os.getcwd())
import json
import random
import torch
import torch.utils.data as data
from torchvision.utils import save_image
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

from custom_dataset.utils.helper import (
    get_char_dict,
    get_bbox_from_points,
    get_rescaled_bbox,
    get_regular_bbox,
    expand_bbox,
    calc_avg_size,
    render_chs,
    crop_chs,
    split_chs,
    general_collate_fn,
    composite_prompt,
)


class CollectedGlyphDataset(data.Dataset):

    def __init__(
        self,
        datype,
        resolution,
        data_root,
        asset_root,
        std_font_path,
        expand_ratio=0.0,
        min_box_size=None,
        max_box_items=None,
        min_image_size=None,
        max_image_size=None,
        force_valid=False,
        max_items_per_box=None,
        min_distance=None,
        min_alpha_ratio=None,
        shuffle_items=False,
        split_conds=False,
        edge_constraint=None,
    ):
        super().__init__() 

        self.datype = datype
        self.resolution = resolution
        self.std_font_path = std_font_path
        self.expand_ratio = expand_ratio
        self.min_alpha_ratio = min_alpha_ratio
        self.data_root = data_root
        self.image_root = os.path.join(data_root, "images")
        self.bg_root = os.path.join(data_root, "inpainted")
        self.gt_root = os.path.join(data_root, "gt")
        real_datype = "train" if datype not in ["train", "val", "test"] else datype
        self.anno_path = os.path.join(data_root, f"{real_datype}_anno.json")
        self.shuffle_items = shuffle_items
        self.split_conds = split_conds
        self.edge_constraint = edge_constraint

        char_root = os.path.join(asset_root, "chars")
        self.char_dict = get_char_dict(char_root)
        valid_chars = self.char_dict["valid"]

        with open(self.anno_path, "r") as f:
            self.anno_list = json.load(f)
            
        for anno_idx, anno in enumerate(self.anno_list):
            if "caption" not in anno:
                anno["caption"] = ""
            if "trans_labels" not in anno or len(anno["trans_labels"]) != len(anno["bboxs"]):
                anno["trans_labels"] = [""] * len(anno["bboxs"])

        # filter samples
        self.annos = []
        self.index_map = []
        for anno_idx, anno in enumerate(self.anno_list):
            if not os.path.exists(os.path.join(self.image_root, anno["image_name"])): 
                continue
            if not os.path.exists(os.path.join(self.bg_root, anno["image_name"])): 
                continue
            if not os.path.exists(os.path.join(self.gt_root, anno["image_name"])):
                continue
            if max_image_size is not None and max(anno["height"], anno["width"]) > max_image_size:
                continue
            if min_image_size is not None and min(anno["height"], anno["width"]) < min_image_size:
                continue

            anno_bboxs = []
            curr_box_items = 0
            for bbox_idx, bbox in enumerate(anno["bboxs"]):
                x1, y1, x2, y2, label = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], bbox["label"]
                if min_box_size is not None and min(x2 - x1, y2 - y1) < min_box_size:
                    continue
                if label=="" or len(label) != len(bbox["items"]):
                    continue
                if force_valid and not all([l in valid_chars for l in label]):
                    continue
                if max_items_per_box is not None and len(bbox["items"]) > max_items_per_box:
                    continue
                if min_distance is not None and "distance" in bbox and bbox["distance"] < min_distance:
                    continue
                anno_bboxs.append([anno_idx, bbox_idx])
                curr_box_items = max(curr_box_items, len(bbox["items"]))

            self.index_map += anno_bboxs
            if len(anno_bboxs) == 0:
                continue
            if datype != "train" and len(anno_bboxs) != len(anno["bboxs"]): 
                continue
            if max_box_items is not None and curr_box_items*len(anno_bboxs) > max_box_items: 
                continue
            self.annos.append(anno)

        self.vae_processor = VaeImageProcessor()
    
    def __len__(self):

        return len(self.index_map)
    
    def __getitem__(self, index):

        try:
            # get annotation and bbox
            anno_idx, bbox_idx = self.index_map[index]
            anno = self.anno_list[anno_idx]
            bboxs = anno["bboxs"]
            bbox = bboxs[bbox_idx]

            # read image and ensure files are closed
            image_name = anno["image_name"]
            img_path = os.path.join(self.image_root, image_name)
            bg_path = os.path.join(self.bg_root, image_name)
            gt_path = os.path.join(self.gt_root, image_name)
            with Image.open(img_path) as im:
                img = im.convert("RGBA").copy()
            with Image.open(bg_path) as im:
                bg = im.convert("RGBA").copy()
            with Image.open(gt_path) as im:
                gt = im.convert("RGBA").copy()

            # crop region
            w, h = img.size
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), img.height, img.width, ratio=self.expand_ratio)
            crop_bbox = [x1, y1, x2, y2]
            r_crop_bbox = get_rescaled_bbox(crop_bbox, w, h)
            crop_img = img.crop((x1, y1, x2, y2))
            crop_bg = bg.crop((x1, y1, x2, y2))
            crop_gt = gt.crop((x1, y1, x2, y2))

            # get region bboxs
            crop_w, crop_h = crop_img.size
            labels = []
            seg_bboxs = []
            r_seg_bboxs = []
            global_seg_bboxs = []

            items = bbox["items"]
            if self.datype == "train" and self.shuffle_items:
                random.shuffle(items)
            for seg in items:
                labels.append(seg["label"])
                seg_bbox = get_bbox_from_points(seg["points"], x1, y1)
                r_seg_bbox = get_regular_bbox(seg_bbox, crop_w, crop_h)
                global_seg_bbox = get_bbox_from_points(seg["points"])
                seg_bboxs.append(seg_bbox)
                r_seg_bboxs.append(r_seg_bbox)
                global_seg_bboxs.append(global_seg_bbox)
            avg_bbox_size = calc_avg_size(seg_bboxs)

            # get content reference and gt
            s_seg_bboxs = seg_bboxs[:]
            if self.datype == "train" and self.split_conds:
                split_ratio = random.uniform(0.3, 0.7)
                gt_len = max(1, int(len(labels)*split_ratio))
                cond_len = max(1, len(labels) - gt_len)
                labels = labels[:gt_len]
                s_seg_bboxs = seg_bboxs[-cond_len:]
                seg_bboxs = seg_bboxs[:gt_len]
            label = "".join(labels)

            # get prompt
            caption = anno["caption"]
            instruction = "Design the style (font, color, texture) for the text rendered in the design image."
            prompt = composite_prompt(caption, [label], instruction)

            _, c_refs_w_pil, _ = render_chs(labels, font_path=self.std_font_path, resolution=self.resolution)
            _, s_refs_w_pil, _ = crop_chs(crop_img, s_seg_bboxs, resolution=self.resolution)
            gt_pil, gt_w_pil, gt_a_mask = crop_chs(crop_gt, seg_bboxs, resolution=self.resolution)

            # filter based on constraints
            if self.min_alpha_ratio is not None:
                n = len(labels)
                alpha_ratio = gt_a_mask.reshape(n, -1).mean(dim=-1).min().item()
                if alpha_ratio < self.min_alpha_ratio:
                    raise ValueError(f"Alpha ratio {alpha_ratio} is less than {self.min_alpha_ratio}, skipping this sample.")
            
            if self.edge_constraint is not None:
                border_width = 4
                mask = torch.squeeze(gt_a_mask, dim=1)  # shape: (n, 256, 256)

                border_mask = torch.zeros_like(mask, dtype=torch.bool)
                border_mask[:, :border_width, :] = True        # top border
                border_mask[:, -border_width:, :] = True       # bottom border
                border_mask[:, :, :border_width] = True        # left border
                border_mask[:, :, -border_width:] = True       # right border

                border_one_ratio = mask[border_mask].float().mean().item()
                if border_one_ratio > self.edge_constraint:
                    raise ValueError(f"Edge constraint violated: {border_one_ratio:.4f} > {self.edge_constraint}")

            gt_tensor = self.vae_processor.preprocess(image=gt_w_pil)
            gt_tensor *= gt_a_mask  # make it gray!

            batch = {
                # gt
                "gt": gt_tensor,
                "gt_pil": gt_pil,
                # conditions
                "c_refs_pil": c_refs_w_pil,
                "s_refs_pil": s_refs_w_pil,
                "prompt": prompt,
                "label": label,
                "labels": labels,
                "img": img,
                "bg": bg,
                "crop_img": crop_img,
                "crop_bg": crop_bg,
                # bboxs
                "crop_bbox": crop_bbox,
                "r_crop_bbox": r_crop_bbox,
                "seg_bboxs": seg_bboxs,
                "r_seg_bboxs": r_seg_bboxs,
                "global_seg_bboxs": global_seg_bboxs,
                "avg_bbox_size": avg_bbox_size,
            }

            return batch
        
        except Exception as e:
            # print(f"Data Error in index {index}: {e}")
            return self.__getitem__((index + 1) % len(self.index_map))
        

class CollectedGlyphDPODataset(data.Dataset):

    def __init__(
        self,
        datype,
        resolution,
        data_root,
        dpo_data_root,
        asset_root,
        std_font_path,
        expand_ratio=0.0,
        max_items_per_box=None,

    ):
        super().__init__() 

        self.datype = datype
        self.resolution = resolution
        self.std_font_path = std_font_path
        self.expand_ratio = expand_ratio
        self.data_root = data_root
        self.image_root = os.path.join(data_root, "images")
        self.bg_root = os.path.join(data_root, "inpainted")
        real_datype = "train" if datype not in ["train", "val", "test"] else datype
        self.anno_path = os.path.join(data_root, f"{real_datype}_anno.json")

        char_root = os.path.join(asset_root, "chars")
        self.char_dict = get_char_dict(char_root)

        with open(self.anno_path, "r") as f:
            self.anno_list = json.load(f)

        self.dpo_image_root = os.path.join(dpo_data_root, "images")
        dpo_anno_path = os.path.join(dpo_data_root, f"dpo_anno.json")
        with open(dpo_anno_path, "r") as f:
            self.dpo_anno_list = json.load(f)
        crop_keys = list(self.dpo_anno_list.keys())
        image_keys = list(set([key.split("_crop_")[0] for key in crop_keys]))
            
        for anno_idx, anno in enumerate(self.anno_list):
            if "caption" not in anno:
                anno["caption"] = ""
            if "trans_labels" not in anno or len(anno["trans_labels"]) != len(anno["bboxs"]):
                anno["trans_labels"] = [""] * len(anno["bboxs"])

        # filter samples
        self.index_map = []
        for anno_idx, anno in enumerate(self.anno_list):
            if anno["image_name"].split(".")[0] not in image_keys:
                continue
            if not os.path.exists(os.path.join(self.image_root, anno["image_name"])): 
                continue
            if not os.path.exists(os.path.join(self.bg_root, anno["image_name"])): 
                continue

            anno_bboxs = []
            for bbox_idx, bbox in enumerate(anno["bboxs"]):
                if max_items_per_box is not None and len(bbox["items"]) > max_items_per_box:
                    continue
                if f"{anno['image_name'].split('.')[0]}_crop_{bbox_idx}" not in crop_keys:
                    continue
                anno_bboxs.append([anno_idx, bbox_idx])

            if len(anno_bboxs) == 0:
                continue
            self.index_map += anno_bboxs

        self.vae_processor = VaeImageProcessor()
    
    def __len__(self):

        return len(self.index_map)
    
    def __getitem__(self, index):

        try:
            # get annotation and bbox
            anno_idx, bbox_idx = self.index_map[index]
            anno = self.anno_list[anno_idx]
            bboxs = anno["bboxs"]
            bbox = bboxs[bbox_idx]

            # read image and ensure files are closed
            image_name = anno["image_name"]
            img_path = os.path.join(self.image_root, image_name)
            bg_path = os.path.join(self.bg_root, image_name)
            with Image.open(img_path) as im:
                img = im.convert("RGBA").copy()
            with Image.open(bg_path) as im:
                bg = im.convert("RGBA").copy()
            
            # get win-lose pairs
            image_index = anno["image_name"].split(".")[0]
            dpo_item = self.dpo_anno_list[f"{image_index}_crop_{bbox_idx}"]
            win_value, lose_value = dpo_item["win"], dpo_item["lose"]
            win_gt_path = os.path.join(self.dpo_image_root, f"{image_index}_crop_{bbox_idx}_score_{win_value:.5f}_win.png")
            lose_gt_path = os.path.join(self.dpo_image_root, f"{image_index}_crop_{bbox_idx}_score_{lose_value:.5f}_lose.png")
            with Image.open(win_gt_path) as im:
                win_gt_pil = im.convert("RGBA").copy()
            with Image.open(lose_gt_path) as im:
                lose_gt_pil = im.convert("RGBA").copy()

            # crop region
            w, h = img.size
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), img.height, img.width, ratio=self.expand_ratio)
            crop_bbox = [x1, y1, x2, y2]
            r_crop_bbox = get_rescaled_bbox(crop_bbox, w, h)
            crop_img = img.crop((x1, y1, x2, y2))
            crop_bg = bg.crop((x1, y1, x2, y2))

            # get region bboxs
            crop_w, crop_h = crop_img.size
            labels = []
            seg_bboxs = []
            r_seg_bboxs = []
            global_seg_bboxs = []
            for seg in bbox["items"]:
                labels.append(seg["label"])
                seg_bbox = get_bbox_from_points(seg["points"], x1, y1)
                r_seg_bbox = get_regular_bbox(seg_bbox, crop_w, crop_h)
                global_seg_bbox = get_bbox_from_points(seg["points"])
                seg_bboxs.append(seg_bbox)
                r_seg_bboxs.append(r_seg_bbox)
                global_seg_bboxs.append(global_seg_bbox)
            avg_bbox_size = calc_avg_size(seg_bboxs)

            # get content reference and gt
            label = "".join(labels)
            # get prompt
            caption = anno["caption"]
            instruction = "Design the style (font, color, texture) for the text rendered in the design image."
            prompt = composite_prompt(caption, [label], instruction)
            
            _, c_refs_w_pil, _ = render_chs(labels, font_path=self.std_font_path, resolution=self.resolution)
            _, s_refs_w_pil, _ = crop_chs(crop_img, seg_bboxs, resolution=self.resolution)

            win_gt_pil, win_gt_w_pil, win_gt_a_mask = split_chs(win_gt_pil, resolution=self.resolution)
            lose_gt_pil, lose_gt_w_pil, lose_gt_a_mask = split_chs(lose_gt_pil, resolution=self.resolution)

            win_gt_tensor = self.vae_processor.preprocess(image=win_gt_w_pil)
            win_gt_tensor *= win_gt_a_mask  # make it gray!
            lose_gt_tensor = self.vae_processor.preprocess(image=lose_gt_w_pil)
            lose_gt_tensor *= lose_gt_a_mask  # make it gray!

            batch = {
                # gt
                "win_gt": win_gt_tensor,
                "lose_gt": lose_gt_tensor,
                "win_gt_pil": win_gt_pil,
                "lose_gt_pil": lose_gt_pil,
                # conditions
                "c_refs_pil": c_refs_w_pil,
                "s_refs_pil": s_refs_w_pil,
                "prompt": prompt,
                "label": label,
                "labels": labels,
                "img": img,
                "bg": bg,
                "crop_img": crop_img,
                "crop_bg": crop_bg,
                # bboxs
                "crop_bbox": crop_bbox,
                "r_crop_bbox": r_crop_bbox,
                "seg_bboxs": seg_bboxs,
                "r_seg_bboxs": r_seg_bboxs,
                "global_seg_bboxs": global_seg_bboxs,
                "avg_bbox_size": avg_bbox_size,
            }

            return batch
        
        except Exception as e:
            print(f"Data Error in index {index}: {e}")
            return self.__getitem__((index + 1) % len(self.index_map))