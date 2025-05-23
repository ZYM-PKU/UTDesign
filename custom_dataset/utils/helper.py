import os
import string
import torch
import numpy as np
import torch.nn.functional as F
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageOps
from random import randint


def composite_prompt(caption, labels, instruction="Create a design image."):

    prompt = instruction + " " + caption
    prompt += " Text to render: "
    for i, label in enumerate(labels):
        prompt += f"\"{label}\""
        if i < len(labels) - 1:
            prompt += ", "
        else:
            prompt += "."

    return prompt

### read files ###
def get_font_paths(root):

    def traversal_files(dir):

        file_paths = []
        for item in os.scandir(dir):
            if item.is_file():
                file_paths.append(item.path)
            else:
                file_paths += traversal_files(item)
        
        return file_paths

    font_paths = traversal_files(root)
    font_paths = [path for path in font_paths if path[-3:] in ('ttf', 'otf', 'TTF', 'OTF')]

    return font_paths

def get_char_dict(root):

    # Common Chinese characters
    with open(os.path.join(root, "GB2312_1.txt"), "r") as fp:
        common_cn_chars = list(fp.readline()) # 3755
    # Complex Chinese characters
    with open(os.path.join(root, "GB2312_2.txt"), "r") as fp:
        complex_cn_chars = list(fp.readline()) # 3008

    num_chars = list(string.printable)[:10]
    en_chars = list(string.printable)[10:62]
    hf_chars = list(string.printable)[62:94] # Halfwidth Forms
    all_en_chars = num_chars + en_chars + hf_chars

    # cn_chars = [chr(i) for i in range(0x4e00, 0x9fa5 + 1)] # All Chinese characters
    cn_chars = common_cn_chars + complex_cn_chars
    ff_chars = [chr(i) for i in range(0xff01, 0xffef + 1)][:101] # Fullwidth Forms
    ff_chars += ["·", "¥", "￥", "“", "”", "「", "」", "–", "▪", "•", "﹩", "○", "『", "』", "﹛", "﹜", "—"]
    all_cn_chars = cn_chars + ff_chars
    valid_chars = num_chars + en_chars + cn_chars
    all_chars = all_en_chars + all_cn_chars

    char_dict = {
        "num": num_chars,
        "en": en_chars,
        "hf": hf_chars,
        "all_en": all_en_chars,
        "common": common_cn_chars,
        "complex": complex_cn_chars,
        "cn": cn_chars,
        "ff": ff_chars,
        "all_cn": all_cn_chars,
        "valid": valid_chars,
        "all": all_chars,
    }

    return char_dict

def get_char_list_from_ttf(font_file):

    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()
    
    char_list = []
    for key, _ in m_dict.items():
        try:
            char_list.append(chr(key))
        except:
            continue

    return char_list

### render characters ###
def render_single_ch(ch, font_path, resolution=256):

    bg_color = (255, 255, 255, 0)
    fg_color = (0, 0, 0, 255)

    # find the best font_size to render characters
    font_size = resolution * 2
    while True:
        try:
            font = ImageFont.truetype(font_path, font_size)
            std_l, std_t, std_r, std_b = font.getbbox(ch)
            std_h, std_w = std_b - std_t, std_r - std_l

            img = Image.new(mode='RGBA', size=(std_w, std_h), color=bg_color)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), ch, fill=fg_color, font=font, anchor="lt")
            break
        except OSError:
            font_size = font_size // 2
            if font_size < 16:
                img = Image.new(mode='RGBA', size=(resolution, resolution), color=bg_color)
                break
    
    # padding and resize
    d_top, d_bottom, d_left, d_right = 0, 0, 0, 0
    if std_h >= std_w:
        d_left = d_right = (std_h - std_w) // 2
    else:
        d_top = d_bottom = (std_w - std_h) // 2
    img = ImageOps.expand(img, border=(d_left, d_top, d_right, d_bottom), fill=bg_color)
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)

    # fill white bg
    img_w = Image.new(mode='RGBA', size=(resolution, resolution), color=(255, 255, 255, 255))
    img_w.paste(img, (0, 0), img)
    img_w = img_w.convert("RGB")

    # get alpha mask
    _, _, _, alpha = img.split()
    a_mask = np.array(alpha, dtype=np.float32) / 255.0
    a_mask = torch.from_numpy(a_mask).unsqueeze(0)
    
    return img, img_w, a_mask

def render_chs(chs, font_path, resolution=256):

    imgs = []
    img_ws = []
    a_masks = []
    for ch in chs:
        img, img_w, a_mask = render_single_ch(ch, font_path, resolution)
        imgs.append(img)
        img_ws.append(img_w)
        a_masks.append(a_mask)

    a_mask = torch.stack(a_masks, dim=0)

    return imgs, img_ws, a_mask

### composite image ###
def concat_images(imgs):
    
    concat_img = np.concatenate([np.array(img) for img in imgs], axis=1)
    concat_img_pil = Image.fromarray(concat_img)

    return concat_img_pil

def composite_image(bg_pil, chs_pils, bboxs):

    composited = bg_pil.convert("RGBA")
    for ch_pil, bbox in zip(chs_pils, bboxs):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w == 0 or h == 0:
            continue
        ch_pil = ch_pil.convert("RGBA")
        ch_pil = ch_pil.resize((w, h), Image.Resampling.LANCZOS)
        composited.paste(ch_pil, bbox, ch_pil)

    composited = composited.convert("RGB")

    return composited

def composite_image_pt(bg_pil, ch_pts, bboxs):
    """
    Paste foreground patches onto background images.
    bg_pil: list of PIL RGB images.
    ch_pts: list of list of torch.Tensor, each tensor shape (4, H, W) with values in [-1, 1]
            (first 3 channels for color, 4th channel for alpha).
    bboxs: list of list of bbox [x1, y1, x2, y2] for each foreground in corresponding bg.
    Returns:
        A tensor of shape (N, 3, H, W) with values in [-1, 1].
    """
    out_images = []
    for bg, pts_list, bbox_list in zip(bg_pil, ch_pts, bboxs):
        # Convert background to tensor, scale from [0,255] to [-1,1]
        bg_np = (np.array(bg.convert("RGB")).astype(np.float32) / 255.0) * 2 - 1  # shape (H, W, 3)
        bg_tensor = torch.from_numpy(bg_np).permute(2, 0, 1).to(ch_pts.device)  # (3,H,W)
        
        result_tensor = bg_tensor.clone()

        for pt, bbox in zip(pts_list, bbox_list):
            x1, y1, x2, y2 = bbox
            target_h = y2 - y1
            target_w = x2 - x1

            # Resize foreground patch to bbox size.
            # pt shape: (4, H, W), add batch dim.
            resized = F.interpolate(pt.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)[0]
            fg_color = resized[:3]  # first three channels.
            fg_alpha = resized[3:4]  # alpha channel.

            # Convert alpha from [-1,1] to [0,1]
            alpha = (fg_alpha + 1) / 2.0

            # Extract corresponding region from background.
            bg_region = result_tensor[:, y1:y2, x1:x2]

            # Composite: alpha * fg + (1 - alpha) * bg
            composite_region = fg_color * alpha + bg_region * (1 - alpha)

            # Replace the region in the background with the composite.
            new_result = result_tensor.clone()
            new_result[:, y1:y2, x1:x2] = composite_region
            result_tensor = new_result

        out_images.append(result_tensor)

    return torch.stack(out_images)


def composite_mask(height, width, a_masks, bboxs):

    composite_mask = torch.zeros(a_masks.shape[0], height, width)
    for idx, (a_mask, bbox) in enumerate(zip(a_masks, bboxs)):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        a_mask = F.interpolate(a_mask[None], size=(h, w))[0, 0]
        composite_mask[idx, y1:y2, x1:x2] = a_mask
    
    return composite_mask

### bbox calculation ###
def check_bbox(bbox, height, width):
    if len(bbox) != 4:
        return False
    if any([not isinstance(val, int) for val in bbox]):
        return False
    x1, y1, x2, y2 = bbox
    if x1 < 0 or x1 >= width or x2 <= 0 or x2 > width:
        return False
    if y1 < 0 or y1 >= height or y2 <= 0 or y2 > height:
        return False
    if x1 >= x2 or y1 >= y2:
        return False
    
    return True

def calc_avg_size(bboxs):
    avg_size = 0
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        avg_size += (w + h) / 2
    avg_size /= len(bboxs)

    return avg_size

def reorder_bboxs(bboxs):

    def overlap(bbox1, bbox2, thres=0.5):
        if bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
            return False
        
        # Calculate vertical overlap
        overlap_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
        overlap_ratio = overlap_height / (min(bbox1[3]-bbox1[1], bbox2[3]-bbox2[1]) + 1e-6)
  
        return overlap_ratio > thres

    def group_bboxes(bboxs):
        groups = []
        for bbox in bboxs:
            added = False
            for group in groups:
                if any(overlap(bbox, g) for g in group):
                    group.append(bbox)
                    added = True
                    break
            if not added:
                groups.append([bbox])
        return groups

    def sort_group(group):
        return sorted(group, key=lambda x: x[0])

    groups = group_bboxes(bboxs)
    sorted_indices = []
    for group in groups:
        sorted_group = sort_group(group)
        for bbox in sorted_group:
            sorted_indices.append(bboxs.index(bbox))
    
    return sorted_indices

def expand_bbox(bbox, height, width, ratio=0.2, no_random=False):

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    add_left = int(w * ratio) if no_random else randint(0, int(w * ratio))
    add_right = int(w * ratio) if no_random else randint(0, int(w * ratio))
    add_top = int(h * ratio) if no_random else randint(0, int(h * ratio))
    add_bottom = int(h * ratio) if no_random else randint(0, int(h * ratio))

    new_x1 = max(0, x1 - add_left)
    new_y1 = max(0, y1 - add_top)
    new_x2 = min(width, x2 + add_right)
    new_y2 = min(height, y2 + add_bottom)

    return [new_x1, new_y1, new_x2, new_y2]

def get_bbox_from_points(points, x_offset=0, y_offset=0):

    seg_x1 = int(min(point[0] for point in points) - x_offset)
    seg_y1 = int(min(point[1] for point in points) - y_offset)
    seg_x2 = int(max(point[0] for point in points) - x_offset)
    seg_y2 = int(max(point[1] for point in points) - y_offset)

    return [seg_x1, seg_y1, seg_x2, seg_y2]

def get_points_from_bbox(bbox, x_offset=0, y_offset=0):

    x1, y1, x2, y2 = bbox
    points = [
        [x1+x_offset, y1+y_offset],  # top-left
        [x2+x_offset, y1+y_offset],  # top-right
        [x2+x_offset, y2+y_offset],  # bottom-right
        [x1+x_offset, y2+y_offset]   # bottom-left
    ]
    return points

def get_regular_bbox(bbox, w, h):

    # regular bbox format for COCO
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    r_x_center = x_center / w
    r_y_center = y_center / h
    r_width = width / w
    r_height = height / h
    r_x_center, r_y_center, r_width, r_height
    
    return [r_x_center, r_y_center, r_width, r_height]

def get_rescaled_bbox(bbox, w, h):

    x1, y1, x2, y2 = bbox
    r_x1, r_x2 = x1 / w, x2 / w
    r_y1, r_y2 = y1 / h, y2 / h

    return [r_x1, r_y1, r_x2, r_y2]

def get_bbox_from_mask(mask):

    # Convert mask to grayscale for processing.
    mask_array = np.array(mask.convert("L"))
    # Identify white pixels (using a threshold in case of anti-aliasing).
    white_pixels = np.where(mask_array > 250)
    if white_pixels[0].size == 0 or white_pixels[1].size == 0:
        return None
    else:
        # Get the bounding box coordinates of the white area.
        min_y, max_y = int(np.min(white_pixels[0])), int(np.max(white_pixels[0]))
        min_x, max_x = int(np.min(white_pixels[1])), int(np.max(white_pixels[1]))
        
        return [min_x, min_y, max_x, max_y]

### others ###
def crop_chs(image, bboxs, resolution=256):
    
    chs = []
    ch_ws = []
    a_masks = []
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        ch = image.crop((x1, y1, x2, y2))
        ch = ch.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # fill white bg
        ch_w = Image.new(mode='RGBA', size=(resolution, resolution), color=(255, 255, 255, 255))
        ch_w.paste(ch, (0, 0), ch)
        ch_w = ch_w.convert("RGB")

        # get alpha mask
        _, _, _, alpha = ch.split()
        a_mask = np.array(alpha, dtype=np.float32) / 255.0
        a_mask = torch.from_numpy(a_mask).unsqueeze(0)

        chs.append(ch)
        ch_ws.append(ch_w)
        a_masks.append(a_mask)
    
    a_mask = torch.stack(a_masks, dim=0)

    return chs, ch_ws, a_mask

def split_chs(image, resolution=256):

    width, height = image.size
    step = height
    n = width // step

    chs = []
    ch_ws = []
    a_masks = []
    for i in range(n):
        left = int(round(i * step))
        right = int(round((i + 1) * step))
        ch = image.crop((left, 0, right, height))
        ch = ch.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # fill white bg
        ch_w = Image.new(mode='RGBA', size=(resolution, resolution), color=(255, 255, 255, 255))
        ch_w.paste(ch, (0, 0), ch)
        ch_w = ch_w.convert("RGB")

        # get alpha mask
        _, _, _, alpha = ch.split()
        a_mask = np.array(alpha, dtype=np.float32) / 255.0
        a_mask = torch.from_numpy(a_mask).unsqueeze(0)

        chs.append(ch)
        ch_ws.append(ch_w)
        a_masks.append(a_mask)

    a_mask = torch.stack(a_masks, dim=0)
        
    return chs, ch_ws, a_mask

def convert_gray_bg_to_white(images):
    
    new_images = []
    for i, img in enumerate(images):
        data = np.array(img)
        # Define range for gray detection
        delta = 10
        # Compute the mean across the color channels
        mean = np.mean(data, axis=2)
        # Create mask where all channels are within delta of the mean
        mask = np.abs(mean - 127.5) < delta
        # Replace gray pixels with white
        data[mask] = [255, 255, 255]
        new_images.append(Image.fromarray(data))
    
    return new_images

### collate_fn ###
def pad_and_stack(tensors, dim=0):

    max_size = max(t.shape[dim] for t in tensors)
    padded = []
    for t in tensors:
        pad_size = max_size - t.shape[dim]
        if pad_size > 0:
            pad = []
            for i in reversed(range(t.dim())):
                if i == dim:
                    pad.extend([0, pad_size])
                else:
                    pad.extend([0, 0])
            t = F.pad(t, pad)
        padded.append(t)
        
    return torch.stack(padded, dim=0), max_size

def pad_and_concat(tensors, dim=0):

    max_size = max(t.shape[dim] for t in tensors)
    padded = []
    for t in tensors:
        pad_size = max_size - t.shape[dim]
        if pad_size > 0:
            pad = []
            for i in reversed(range(t.dim())):
                if i == dim:
                    pad.extend([0, pad_size])
                else:
                    pad.extend([0, 0])
            t = F.pad(t, pad)
        padded.append(t)
        
    return torch.cat(padded, dim=0), max_size

def general_collate_fn(batched_data):

    batch = {}

    for key in batched_data[0]:
        if isinstance(batched_data[0][key], int) or isinstance(batched_data[0][key], float):
            batch[key] = torch.tensor([data[key] for data in batched_data])
        elif isinstance(batched_data[0][key], torch.Tensor):
            batch[key], _ = pad_and_stack([data[key] for data in batched_data], dim=0)
        elif isinstance(batched_data[0][key], list):
            if len(batched_data[0][key]) > 0 and isinstance(batched_data[0][key][0], torch.Tensor):
                batch[key] = [torch.stack([data[key][idx] for data in batched_data], dim=0) for idx in range(len(batched_data[0][key]))]
            else:
                batch[key] = [data[key] for data in batched_data]
        else:
            batch[key] = [data[key] for data in batched_data]
    
    return batch