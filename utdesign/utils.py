import math

def compose_vlm_prompt(vlm_processor, prompt, bbox, label):

    instruction = f"{prompt} Please describe a proper style (font, colors, etc.) of the foreground glyphs: {label}, \
          which will be rendered on the image at the position: {bbox}."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preparation for inference
    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    return text


### qwen-2.5-vl utils
QWEN_IMAGE_FACTOR = 28
QWEN_MIN_PIXELS = 4 * 28 * 28
QWEN_MAX_PIXELS = 2048 * 28 * 28 # 16384 * 28 * 28
QWEN_MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def qwen_smart_resize(
    height: int, width: int, factor: int = QWEN_IMAGE_FACTOR, min_pixels: int = QWEN_MIN_PIXELS, max_pixels: int = QWEN_MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > QWEN_MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {QWEN_MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar