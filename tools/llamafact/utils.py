import io
import json
import base64
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from math import ceil, floor
from typing import List
from pydantic import BaseModel, Field
from utdesign.utils import qwen_smart_resize


def refine_bbox(bboxs, height, width, min_size=16):
    new_bboxs = []
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        if x1 >= x2:
            x1, x2 = x2, x1
        if y1 >= y2:
            y1, y2 = y2, y1
        if x2 - x1 < min_size:
            x1 -= (min_size - (x2 - x1)) // 2
            x2 += (min_size - (x2 - x1) + 1) // 2
        if y2 - y1 < min_size:
            y1 -= (min_size - (y2 - y1)) // 2
            y2 += (min_size - (y2 - y1) + 1) // 2
        x1 = max(0, min(x1, width - 1))
        x2 = max(1, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(1, min(y2, height))

        new_bbox = [x1, y1, x2, y2]
        if new_bbox != bbox:
            print(f"[BBOX INFO] Refined bbox from {bbox} to {new_bbox}.")
        new_bboxs.append(new_bbox)

    return new_bboxs

def show_bboxs(bboxs, ax):
    colormap = plt.get_cmap("tab10")
    for i, box in enumerate(bboxs):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        color = colormap(i % 10)
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=1))

def draw_bboxs(image, font_path, labels, bboxs, s_bboxs=None):

    draw = ImageDraw.Draw(image, "RGBA")

    cmap = cm.get_cmap("tab10", len(bboxs)) if bboxs else None
    
    for i, (label, box) in enumerate(zip(labels, bboxs)):
        if cmap:
            # Get a unique color for this bbox (RGBA values in 0-1)
            r_f, g_f, b_f, a_f = cmap(i)
            color = (int(r_f * 255), int(g_f * 255), int(b_f * 255), 255)
        else:
            color = (255, 0, 0, 255)
        
        # Create a semi-transparent overlay color from the picked color
        overlay_color = color[:3] + (70,)
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        
        # Draw the main bounding box with a colored outline and overlay fill
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.rectangle((x1, y1, x2, y2), fill=overlay_color)

        if s_bboxs is not None:
            # Optionally, draw secondary bbox with a complementary color
            comp_color = tuple(255 - c for c in color[:3]) + (255,)
            for s_box in s_bboxs:
                draw.rectangle(s_box, outline=comp_color, width=2)
        
        # Determine optimal text and background label color based on bbox color brightness
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        if brightness > 127:
            text_color = (0, 0, 0)
            label_bg = (255, 255, 255, 150)
        else:
            text_color = (255, 255, 255)
            label_bg = (0, 0, 0, 150)

        # Calc a proper font size
        font_size_factor = 0.16
        min_font_size = 12
        max_font_size = 32
        dynamic_font_size = int(box_height * font_size_factor)
        dynamic_font_size = max(min_font_size, min(max_font_size, dynamic_font_size))
        font = ImageFont.truetype(font_path, dynamic_font_size)

        # Prepare and draw label with background for readability
        std_l, std_t, std_r, std_b = font.getbbox(label)
        text_width, text_height = std_r - std_l, std_b - std_t
        padding = 8
        label_x = x1
        label_y = max(y1 - text_height - padding, 0)
        draw.rectangle(
            [label_x, label_y, label_x + text_width + padding, label_y + text_height + padding],
            fill=label_bg
        )
        draw.text((label_x, label_y), label, fill=text_color, font=font)
                
    return image

def render_bboxs(image, bboxs, seg_bbox_list=None, ch_ref_list=None):
    if seg_bbox_list is None or ch_ref_list is None:
        seg_bbox_list = [None] * len(bboxs)
        ch_ref_list = [None] * len(bboxs)
    image = image.copy()
    draw = ImageDraw.Draw(image)
    colormap = plt.get_cmap("tab10")
    thickness = max(1, int(min(image.size) * 0.005))
    for i, (bbox, seg_bboxs, ch_refs) in enumerate(zip(bboxs, seg_bbox_list, ch_ref_list)):
        color = tuple(int(255 * channel) for channel in colormap(i % 10)[:3])
        draw.rectangle(bbox, outline=color, width=thickness)

        if seg_bboxs is not None and ch_refs is not None:
            for seg_bbox, ch_ref in zip(seg_bboxs, ch_refs):
                draw.rectangle(seg_bbox, fill=color)
                w, h = seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1]
                ch_ref = ch_ref.resize((w, h))
                # Choose text color based on brightness of the background color.
                brightness = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                ch_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                # Set ch_ref's RGB channels to ch_color while preserving the alpha channel.
                data = list(ch_ref.getdata())
                new_data = [ch_color + (alpha,) for (_, _, _, alpha) in data]
                ch_ref.putdata(new_data)
                image.paste(ch_ref, seg_bbox, ch_ref)
            
    return image

def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            image = Image.open(image_file)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_api(client, model, prompt, image=None, json_schema=None):
    content = [
        {
            "type": "text",
            "text": prompt,
        },
    ]
    if image is not None:
        base64_image = encode_image(image)
        content.insert(
            0,
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            }
        )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content,
            }
        ],
        extra_body={"guided_json": json_schema},
        max_completion_tokens=512,
        temperature=0.5,
    ) 

    return completion.choices[0].message.content

def get_layout_plan_stage1(client, model, labels, bg, caption, min_pixels=4*28*28, max_pixels=2500*28*28, max_trials=5):

    print(f"Getting layout plan for stage 1. Labels:{labels}")

    len_labels = len(labels)
    bg_w, bg_h = bg.size
    new_h, new_w = qwen_smart_resize(bg_h, bg_w, min_pixels=min_pixels, max_pixels=max_pixels)
    bg = bg.resize((new_w, new_h), Image.LANCZOS)

    query = dict(
        height=bg_h,
        width=bg_w,
        items=[
            dict(
                label=label,
                bbox=[],
            )
            for label in labels
        ],
    )
        
    prompt = f"Please help me design a layout to place {len(labels)} foreground text items over the background of original size w={bg_w}, h={bg_h} (resized to w={new_w}, h={new_h}). \
{caption} The foreground text items are {labels}. Place the items carefully to avoid unbalance, overlap, and out-of-bounds. \
The layout should contain all the text items in given order, in which each item has a bounding box described as [left, top, right, bottom] (all the values are integer numbers). \
Return the result by filling in the initial JSON file while keeping the label of items unchanged and do not return any extra explaination. The initial JSON is defined as: {query}"
    
    # Structured Outputs
    class Item(BaseModel):
        label: str
        bbox: List[int] = Field(..., min_items=4, max_items=4)

    class Answer(BaseModel):
        height: int
        width: int
        items: List[Item] = Field(..., min_items=len_labels, max_items=len_labels)
    
    json_schema = Answer.model_json_schema()
    
    answer = None
    while max_trials > 0:
        try:
            response = call_api(client, model, prompt, bg, json_schema)
            response = response[response.find("{"):response.rfind("}")+1]
            answer = json.loads(response)
            break
        except Exception as e:
            print(f"Error: {e} Retrying...")
            time.sleep(1)
            max_trials -= 1
            continue

    if not answer:
        raise ValueError("Failed to get layout plan")
    
    seg_bboxs = [item["bbox"] for item in answer["items"]]
    seg_bboxs = refine_bbox(seg_bboxs, bg_h, bg_w)

    print(f"Layout plan for stage 1 obtained: {seg_bboxs}")

    return seg_bboxs

def get_layout_plan_stage2(client, model, labels, crop_image, min_pixels=4*28*28, max_pixels=2500*28*28, max_trials=5):

    print(f"Getting layout plan for stage 2. Labels: {labels}")

    len_labels = len(labels)
    crop_w, crop_h = crop_image.size
    new_h, new_w = qwen_smart_resize(crop_h, crop_w, min_pixels=min_pixels, max_pixels=max_pixels)
    crop_image = crop_image.resize((new_w, new_h), Image.LANCZOS)

    query = dict(
        height=crop_h,
        width=crop_w,
        items=[
            dict(
                label=label,
                bbox=[],
            )
            for label in labels
        ],
    )
        
    prompt = f"Please help me design a layout to place {len(labels)} foreground glyph items in reading order over the background region of size w={crop_w}, h={crop_h} (resized to w={new_w}, h={new_h}). \
The foreground glyph items are {labels}. Place the items carefully to avoid unbalance, overlap, and out-of-bounds. \
The layout should contain all the glyph items in given order, in which each item has a bounding box described as [left, top, right, bottom] (all the values are integer numbers). \
Return the result by filling in the initial JSON file while keeping the label of items unchanged and do not return any extra explaination. The initial JSON is defined as: {query}"
    
    # Structured Outputs
    class Item(BaseModel):
        label: str
        bbox: List[int] = Field(..., min_items=4, max_items=4)

    class Answer(BaseModel):
        height: int
        width: int
        items: List[Item] = Field(..., min_items=len_labels, max_items=len_labels)
    
    json_schema = Answer.model_json_schema()
    
    answer = None
    while max_trials > 0:
        try:
            response = call_api(client, model, prompt, crop_image, json_schema)
            response = response[response.find("{"):response.rfind("}")+1]
            answer = json.loads(response)
            break
        except Exception as e:
            print(f"Error: {e} Retrying...")
            time.sleep(1)
            max_trials -= 1
            continue

    if not answer:
        raise ValueError("Failed to get layout plan")
    
    seg_bboxs = [item["bbox"] for item in answer["items"]]
    seg_bboxs = refine_bbox(seg_bboxs, crop_h, crop_w)

    print(f"Layout plan for stage 2 obtained: {seg_bboxs}")

    return seg_bboxs