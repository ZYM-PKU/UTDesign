import os
import sys
sys.path.append(os.getcwd())
import json
import random
import torch
import torch.utils.data as data
from torchvision.utils import save_image
from random import randint, choice, choices
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

from custom_dataset.utils.augment import aug_chs, disturb_chs
from custom_dataset.utils.helper import (
    get_char_dict,
    get_font_paths,
    render_chs,
    composite_image, 
    composite_mask,
    get_bbox_from_points,
    get_regular_bbox,
    expand_bbox,
    general_collate_fn,
)


class SynthGlyphDataset(data.Dataset):

    def __init__(
        self, datype, resolution, \
        asset_root, font_root, std_font_path, char_set="common", \
        samples_per_style=1000, interval_c_refs=[2,10], interval_s_refs=[2,10], \
        augment_color_texture=False, disturb_style=False,
    ):
        super().__init__()

        self.std_font_path = std_font_path
        self.resolution = resolution
        self.samples_per_style = samples_per_style
        self.interval_c_refs = interval_c_refs
        self.interval_s_refs = interval_s_refs
        self.augment_color_texture = augment_color_texture
        self.disturb_style = disturb_style

        char_root = os.path.join(asset_root, "chars")
        font_list_path = os.path.join(asset_root, f"{datype}_font_list.json")
        font_char_dict_path = os.path.join(asset_root, f"font_char_dict_{char_set}.json")
        
        self.char_dict = get_char_dict(char_root)
        with open(font_list_path, "r") as f:
            font_list = json.load(f)
        self.font_paths = [os.path.join(font_root, font_base) for font_base in font_list]
        self.num_fonts = len(self.font_paths)

        with open(font_char_dict_path, "r") as f:
            self.font_char_dict = json.load(f)

        self.vae_processor = VaeImageProcessor()
            
    def __len__(self):

        return self.num_fonts * self.samples_per_style

    def __getitem__(self, index):

        font_idx = index // self.samples_per_style
        font_path = self.font_paths[font_idx]
        font_name = os.path.basename(font_path).split(".")[0]
        font_char_list = self.font_char_dict[font_name]

        n_c_refs = randint(*self.interval_c_refs)
        c_chs = choices(font_char_list, k=n_c_refs)
        n_s_refs = randint(*self.interval_s_refs)
        s_chs = choices(font_char_list, k=n_s_refs)

        gt_pil, gt_w_pil, gt_a_mask = render_chs(c_chs, font_path=font_path, resolution=self.resolution)
        _, c_refs_w_pil, _ = render_chs(c_chs, font_path=self.std_font_path, resolution=self.resolution)
        s_refs_pil, s_refs_w_pil, _ = render_chs(s_chs, font_path=font_path, resolution=self.resolution)

        if self.augment_color_texture:
            aug_seed = randint(0, 1e9)
            gt_pil, gt_w_pil, gt_a_mask = aug_chs(gt_pil, aug_seed)
            s_refs_pil, s_refs_w_pil, _ = aug_chs(s_refs_pil, aug_seed, aug_bg=True)

        gt = self.vae_processor.preprocess(image=gt_w_pil)
        gt *= gt_a_mask # make it gray!

        if self.disturb_style:
            dist_seed = randint(0, 1e9)
            s_refs_w_pil = disturb_chs(s_refs_w_pil, dist_seed)

        batch = {
            "gt": gt,
            "c_text": "".join(c_chs),
            "s_text": "".join(s_chs),
            "gt_pil": gt_pil,
            "c_refs_pil": c_refs_w_pil,
            "s_refs_pil": s_refs_w_pil,
        }

        return batch
    

class SynthGlyphSingleDataset(data.Dataset):

    def __init__(
        self, datype, resolution, \
        asset_root, font_root, char_set="common", \
        samples_per_style=5000, augment_color_p=0.7,
    ):
        super().__init__()

        self.resolution = resolution
        self.samples_per_style = samples_per_style
        self.augment_color_p = augment_color_p

        char_root = os.path.join(asset_root, "chars")
        font_list_path = os.path.join(asset_root, f"{datype}_font_list.json")
        font_char_dict_path = os.path.join(asset_root, f"font_char_dict_{char_set}.json")
        
        self.char_dict = get_char_dict(char_root)
        with open(font_list_path, "r") as f:
            font_list = json.load(f)
        self.font_paths = [os.path.join(font_root, font_base) for font_base in font_list]
        self.num_fonts = len(self.font_paths)

        with open(font_char_dict_path, "r") as f:
            self.font_char_dict = json.load(f)

        self.vae_processor = VaeImageProcessor()
            
    def __len__(self):

        return self.num_fonts * self.samples_per_style

    def __getitem__(self, index):

        font_idx = index // self.samples_per_style
        font_path = self.font_paths[font_idx]
        font_name = os.path.basename(font_path).split(".")[0]
        font_char_list = self.font_char_dict[font_name]

        ch = choice(font_char_list)
        gt_pil, gt_w_pil, gt_a_mask = render_chs([ch], font_path=font_path, resolution=self.resolution)

        if random.random() < self.augment_color_p:
            aug_seed = randint(0, 1e9)
            gt_pil, gt_w_pil, gt_a_mask = aug_chs(gt_pil, aug_seed)

        gt = self.vae_processor.preprocess(image=gt_w_pil[0])[0]
        gt_a_mask = gt_a_mask[0]
        gt *= gt_a_mask # make it gray!
        
        gt_a_mask = (gt_a_mask * 2) - 1
        gt_rgba = torch.cat([gt, gt_a_mask], dim=0)

        batch = {
            "gt": gt,
            "gt_rgba": gt_rgba,
            "a_mask": gt_a_mask,
        }

        return batch


class SynthGlyphRegionDataset(data.Dataset):

    def __init__(
        self, datype, data_root, \
        asset_root, font_root, char_set="common",
        augments_per_region=10, expand_ratio=0.2,
    ):
        super().__init__() 

        self.image_root = os.path.join(data_root, "inpainted")
        self.anno_path = os.path.join(data_root, f"{datype}_anno.json")
        self.augments_per_region = augments_per_region
        self.expand_ratio = expand_ratio

        with open(self.anno_path, "r") as f:
            self.anno_list = json.load(f)

        self.index_map = []
        for anno_idx, anno in enumerate(self.anno_list):
            for bbox_idx in range(len(anno["bboxs"])):
                self.index_map.append((anno_idx, bbox_idx))

        char_root = os.path.join(asset_root, "chars")
        font_char_dict_path = os.path.join(asset_root, f"font_char_dict_{char_set}.json")

        self.char_dict = get_char_dict(char_root)
        self.font_paths = get_font_paths(font_root)

        with open(font_char_dict_path, "r") as f:
            self.font_char_dict = json.load(f)
    
    def __len__(self):

        return len(self.index_map)*self.augments_per_region
    
    def __getitem__(self, index):

        # get annotation and bbox
        idx = index // self.augments_per_region
        anno_idx, bbox_idx = self.index_map[idx]
        anno = self.anno_list[anno_idx]
        bboxs = anno["bboxs"]
        bbox = bboxs[bbox_idx]

        # read image and crop
        img_path = os.path.join(self.image_root, anno["image_name"])
        with Image.open(img_path) as im:
            img = im.convert("RGBA").copy()
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img)

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), img.height, img.width, ratio=self.expand_ratio)
        crop_img = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop_img.size

        # get region bboxs
        seg_bboxs = []
        for seg in bbox["items"]:
            seg_bbox = get_bbox_from_points(seg["points"], x1, y1)
            seg_bboxs.append(seg_bbox)
    
        r_seg_bboxs = []
        for seg_bbox in seg_bboxs:
            r_seg_bbox = get_regular_bbox(seg_bbox, crop_w, crop_h)
            r_seg_bboxs.append(r_seg_bbox)

        # render chars
        aug_seed = randint(0, 1e9)
        font_path = choice(self.font_paths)
        font_name = os.path.basename(font_path).split(".")[0]
        font_char_list = self.font_char_dict[font_name]

        n_chs = len(seg_bboxs)
        chs = choices(font_char_list, k=n_chs)
        chs_pil, _, chs_a_mask = render_chs(chs, font_path=font_path, resolution=512)
        chs_pil, _, chs_a_mask = aug_chs(chs_pil, aug_seed)

        composite_img = composite_image(crop_img, chs_pil, seg_bboxs)
        composite_a_mask = composite_mask(crop_h, crop_w, chs_a_mask, seg_bboxs)

        batch = dict(
            composite_img=composite_img,
            composite_a_mask=composite_a_mask,
            labels=chs,
            seg_bboxs=seg_bboxs,
            r_seg_bboxs=r_seg_bboxs,
        )
            
        return batch