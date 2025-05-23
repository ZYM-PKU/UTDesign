# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer.grpo_trainer_planner import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    dataset_name: Optional[str] = None

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "data json path"},
    )
    max_pixels: Optional[int] = field(
        default=2500 * 28 * 28,
        metadata={"help": "max image size"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "overlay", "consistency"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'overlay', 'consistency'."},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    cache_dir: str = None
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LayoutPlanningDataset(Dataset):
    def __init__(self, data_path: str, max_pixels: int = None):
        super(LayoutPlanningDataset, self).__init__()

        self.max_pixels = max_pixels
        self.list_data_dict = []
        with open(data_path, "r", encoding='utf-8') as f:
            self.list_data_dict = json.load(f)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index):

        try:
            data_dict = self.list_data_dict[index]
            problem = data_dict["messages"][0]["content"]
            answer = data_dict["messages"][1]["content"]
            image_path = data_dict["images"][0]
            stage = data_dict["stage"]

            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            if self.max_pixels is not None and w * h > self.max_pixels:
                raise ValueError(
                    f"Image size {max(w, h)} exceeds max pixels: {self.max_pixels}. Skiping."
                )
            

            prompt = [
                # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": problem.replace("<image>", "")},
                    ],
                },
            ]

            return {
                'stage': stage,
                'image': image,
                'prompt': prompt,
                'answer': answer
            }
        
        except:
            return self.__getitem__((index+1)%len(self))



def calc_iou(box1, box2):
    
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter

    return float(inter)/union

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.8, the reward is 1.0, otherwise 0.0 .
'''
def iou_reward(completions, answer, **kwargs):
    """
    iou_reward: Computes reward based on Intersection over Union (IoU).

    This function compares the predicted bounding boxes from the model with the ground truth bounding boxes.
    For each sample:
    1. It attempts to parse the predicted content and ground truth answer as JSON.
    2. If parsing fails, or if the predicted JSON does not contain the key "items", or if the number of items
        in the prediction does not match the number in the ground truth, a reward of 0.0 is assigned.
    3. For each corresponding pair of bounding boxes, the IoU is computed using the calc_iou function.
    4. The average IoU over all pairs is calculated.
    5. If the average IoU exceeds 0.8, a hard reward of 1.0 is assigned; otherwise, the average IoU itself is used as the reward.

    The resulting reward for each sample is a float in the range [0.0, 1.0].
    """

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, ans in zip(contents, answer):
        reward = 0.0
        ans = json.loads(ans)
        try: 
            pred = json.loads(content)
        except: 
            rewards.append(reward)
            continue

        if "items" not in pred:
            rewards.append(reward)
            continue
        
        if len(pred["items"]) != len(ans["items"]):
            rewards.append(reward)
            continue

        ious = []
        for pred_item, ans_item in zip(pred["items"], ans["items"]):
            try: 
                pred_bbox = pred_item["bbox"]
                ans_bbox = ans_item["bbox"]
                iou = calc_iou(pred_bbox, ans_bbox)
                ious.append(iou)
            except:
                ious.append(0.0)
        
        reward = sum(ious) / len(ious)
        reward = 1.0 if reward > 0.8 else reward # hard reward
        rewards.append(reward)

    return rewards


def overlay_reward(completions, answer, **kwargs):
    """
    Reward function that penalizes bbox overlap.
    For each prediction in completions, this function computes the average pairwise IoU
    among predicted bounding boxes. If there is only one bbox, reward is 1.0.
    The final reward is calculated as 1 - avg_iou (clipped between 0.0 and 1.0).
    """

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, ans in zip(contents, answer):
        ans = json.loads(ans)
        try:
            pred = json.loads(content)
        except Exception:
            rewards.append(0.0)
            continue

        if "items" not in pred:
            rewards.append(0.0)
            continue

        if len(pred["items"]) != len(ans["items"]):
            rewards.append(0.0)
            continue

        items = pred["items"]
        # If there is only one bbox, we assume no overlap, so max reward.
        if len(items) <= 1:
            rewards.append(1.0)
            continue

        pair_ious = []
        # Compute IoU for all unique bbox pairs.
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                try:
                    bbox1 = items[i]["bbox"]
                    bbox2 = items[j]["bbox"]
                    iou = calc_iou(bbox1, bbox2)
                    pair_ious.append(iou)
                except Exception:
                    # On error, treat as high overlap penalty for this pair.
                    pair_ious.append(1.0)

        if len(pair_ious) != len(items)*(len(items)-1)//2:
            reward = 0.0
        else:
            avg_iou = max(pair_ious)
            # Less overlap gives higher reward.
            reward = max(0.0, min(1.0, 1.0 - avg_iou))

        rewards.append(reward)

    # weighting
    for i, reward in enumerate(rewards):
        rewards[i] = reward * 0.5

    return rewards

def consistency_reward(completions, answer, stage, **kwargs):
    """
    Reward function that penalizes size variation among predicted bounding boxes.
    For each prediction in completions, this function computes the area of each bbox,
    calculates the coefficient of variation (std/mean), and converts it into a reward in [0.0, 1.0].
    Less variation (i.e. more consistent sizes) results in a higher reward.
    If there is only one bbox, the reward is maximal (1.0).
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, ans, st in zip(contents, answer, stage):
        if st == 1:
            rewards.append(1.0)
            continue

        ans = json.loads(ans)
        try:
            pred = json.loads(content)
        except Exception:
            rewards.append(0.0)
            continue

        if "items" not in pred:
            rewards.append(0.0)
            continue

        if len(pred["items"]) != len(ans["items"]):
            rewards.append(0.0)
            continue

        items = pred["items"]
        if len(items) <= 1:
            rewards.append(1.0)
            continue

        sizes = []
        for item in items:
            try:
                bbox = item["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                sizes.append(width * height)
            except Exception:
                continue

        if len(sizes) != len(items):
            rewards.append(0.0)
            continue

        mean_size = sum(sizes) / len(sizes)
        std_size = math.sqrt(sum((s - mean_size) ** 2 for s in sizes) / len(sizes))
        variation = std_size / mean_size if mean_size != 0 else 1.0
        # A lower variation yields a higher reward, clamped between 0 and 1
        reward = max(0.0, min(1.0, 1.0 - variation))
        rewards.append(reward)

    # weighting
    for i, reward in enumerate(rewards):
        rewards[i] = reward * 0.5

    return rewards


reward_funcs_registry = {
    "accuracy": iou_reward,
    "overlay": overlay_reward,
    "consistency": consistency_reward,
}


def main(script_args, training_args, model_args):

    # Prepare reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LayoutPlanningDataset(script_args.data_path, script_args.max_pixels)

    training_args.model_init_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    # Initialize the GRPO trainer
    trainer = Qwen2VLGRPOTrainer(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
