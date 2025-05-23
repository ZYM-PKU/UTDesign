import openai
import base64
import time
import io
import os
import json
from PIL import Image
from transformers import CLIPTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def encode_image(image_path, max_size=512):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        if max(image.size) > max_size:
            ratio = max_size / float(max(image.size))
            new_size = tuple([int(x * ratio) for x in image.size])
            image = image.resize(new_size)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_img_caption(client, model, image_path, max_trials=5):
    base64_image = encode_image(image_path)
    prompt = "Please first describe the main motivation (theme) of the design image and then describe the visual appearance, focusing on color, texture, patterns, and overall style. \
        Keep the description concise and avoid describing any text present in the image. The description should be in English and not exceed 100 words. Don't use line breaks."

    while max_trials > 0:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in describing the main theme and visual appearance of design images including posters, banners, ads, etc."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
            ) 

            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e} Retrying...")
            time.sleep(5)
            max_trials -= 1
            continue

    raise ValueError("Max trials reached. Unable to get image caption.")

def process_annotation(client, model, target_data_root, anno_idx, anno, tokenizer):

    image_path = os.path.join(target_data_root, "images", anno["image_name"])
    if "caption" in anno and anno["caption"] != "":
        caption = anno["caption"]
        print("Image already captioned.")
    else:
        try:
            caption = get_img_caption(client=client, model=model, image_path=image_path)
            print("Successfully captioned image.")
        except ValueError:
            caption = ""
        anno["caption"] = caption
    inputs = tokenizer(caption, return_tensors="pt")
    caption_length = len(inputs["input_ids"][0])
    print(f"Image {anno_idx}: {caption}\nCaption length (in tokens): {caption_length}")

    return caption_length

def anno_image_caption(args):

    gpt_client = openai.OpenAI(
        base_url="{base_url}",
        api_key="{api_key}",
    )

    for datype in ["train", "val"]:

        anno_json_path = os.path.join(args.target_data_root, f"{datype}_anno.json")
        with open(anno_json_path, "r") as f:
            anno_json = json.load(f)

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        success_trials = 0
        caption_lengths = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_annotation, gpt_client, args.vllm_model_path, \
                                        args.target_data_root, anno_idx, anno, tokenizer) for anno_idx, anno in enumerate(anno_json)]
            try:
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        caption_length = future.result()
                        caption_lengths.append(caption_length)
                        success_trials += 1
                    except Exception as e:
                        print(f"Error processing annotation: {e}")
            except KeyboardInterrupt:
                print("\033[1;31mKeyboard interrupt detected. Stopping all tasks...\033[0m")
                for future in futures:
                    future.cancel()
                executor.shutdown(cancel_futures=True)
                caption_lengths = None

        print(f"Successfully annotated {success_trials} out of {len(anno_json)} images.")
        with open(anno_json_path, "w") as f:
            json.dump(anno_json, f, indent=4, ensure_ascii=False)
            f.flush()