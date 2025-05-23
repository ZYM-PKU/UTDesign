import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union

from transformers import AutoImageProcessor, CLIPVisionModel, Dinov2Model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.controlnet import zero_module
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.import_utils import is_xformers_available
from utils import qwen_smart_resize, compose_vlm_prompt

if is_xformers_available():
    import xformers
    import xformers.ops


### text embeddings
def _get_clip_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: str,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=text_encoder.device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return pooled_prompt_embeds, prompt_embeds

### image embeddings
def get_image_embeds(
    image_encoder,
    image_inputs: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or image_encoder.device
    dtype = dtype or image_encoder.dtype

    # Get image embeddings
    image_inputs = image_inputs.to(dtype=dtype, device=device)
    image_embeds = image_encoder(image_inputs, output_hidden_states=True)

    # Use pooled output of CLIPImageModel
    pooled_image_embeds = image_embeds.pooler_output
    pooled_image_embeds = pooled_image_embeds.to(dtype=dtype, device=device)

    # Use last hidden state of CLIPImageModel
    image_embeds = image_embeds.last_hidden_state[:,1:,:]
    image_embeds = image_embeds.to(dtype=dtype, device=device)

    return pooled_image_embeds, image_embeds

def get_vlm_embeds(
    vlm_name,
    vlm_processor,
    vlm_model,
    image: Image,
    bbox: List[float],
    prompt: str,
    label: str,
):
    if "Qwen" in vlm_name:
        new_h, new_w = qwen_smart_resize(image.height, image.width)
        image = image.resize((new_w, new_h))

        text = compose_vlm_prompt(vlm_processor, prompt=prompt, bbox=bbox, label=label)
        inputs = vlm_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(vlm_model.device)

        outputs = vlm_model(**inputs, output_hidden_states=True)
        vlm_embeds = outputs.hidden_states[-1]

        return vlm_embeds

    else:
        raise ValueError(f"Invalid VLM model name: {vlm_name}")


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


class EmbeddingDiscriminator(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        num_mid_layers: int,
    ):
        super().__init__()
        self.proj_in = FeedForward(input_dim, hidden_dim)
        self.mid_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=16,
                batch_first=True,
            ),
            num_layers=num_mid_layers,
        )
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):

        out = self.proj_in(x)
        out = self.mid_block(out)
        out = self.proj_out(out)

        return out
    

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.0,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_q = nn.RMSNorm(dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-6)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)])

    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states,
        attention_mask = None,
    ):

        encoder_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        batch_size = query.shape[0]
        if is_xformers_available():
            query = query.view(batch_size, -1, self.heads, self.dim_head).contiguous()
            key = key.view(batch_size, -1, self.heads, self.dim_head).contiguous()
            value = value.view(batch_size, -1, self.heads, self.dim_head).contiguous()
            
            query = self.norm_q(query)
            key = self.norm_k(key)

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, p=0.0, scale=None, op=None
            )
            hidden_states = hidden_states.reshape(batch_size, -1, self.heads*self.dim_head)
            hidden_states = hidden_states.to(query.dtype)
        else:
            query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2).contiguous()
            key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2).contiguous()
            value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2).contiguous()

            query = self.norm_q(query)
            key = self.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads*self.dim_head)
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

class PerceiverResampler(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        *,
        dim,
        dim_out,
        depth=8,
        dim_head=64,
        heads=8,
        num_query_embeds=256,
        ff_mult=2,
    ):
        super().__init__()

        self.query_embeds = nn.Parameter(torch.randn(1, num_query_embeds, dim))
        self.global_query_embeds = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                    FeedForward(dim = dim, mult = ff_mult)
                ])
            )

        self.to_out = nn.Sequential(
            FeedForward(dim, dim_out),
            nn.LayerNorm(dim_out, elementwise_affine=False, eps=1e-6),
        )

        self.gradient_checkpointing = False

    def forward(self, encoder_hidden_states):
        with torch.autocast("cuda", dtype=torch.float32):
            hidden_states = torch.cat([self.global_query_embeds, self.query_embeds], dim=1)

            for attn, ff in self.layers:
                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(attn, hidden_states, encoder_hidden_states, use_reentrant=False) + hidden_states
                    hidden_states = torch.utils.checkpoint.checkpoint(ff, hidden_states, use_reentrant=False) + hidden_states
                else:
                    hidden_states = attn(hidden_states, encoder_hidden_states) + hidden_states
                    hidden_states = ff(hidden_states) + hidden_states

            hidden_states = self.to_out(hidden_states)
            global_embeds, embeds = hidden_states[0][0], hidden_states[0][1:]

            return global_embeds, embeds

class FusionModule(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        attn_heads=16,
        dim_head=256,
        pr_depth=8,
        pr_seq_len=256,
        clip_model_name="openai/clip-vit-base-patch16",
        dino_model_name="facebook/dinov2-large",
        vlm_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        clip_cache_dir=None,
        vlm_cache_dir=None,
    ):
        super().__init__()
        
        self.pretrained_module_names = ["clip_image_encoder", "dino_image_encoder", "vlm_model"]
        self.clip_image_processor = AutoImageProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.dino_image_processor = AutoImageProcessor.from_pretrained(dino_model_name, use_fast=True)

        self.clip_image_encoder = CLIPVisionModel.from_pretrained(clip_model_name, cache_dir=clip_cache_dir)
        self.dino_image_encoder = Dinov2Model.from_pretrained(dino_model_name, cache_dir=clip_cache_dir)

        self.vlm_model_name = vlm_model_name
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model_name, 
            cache_dir=vlm_cache_dir, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    
        dino_hidden_dim = self.dino_image_encoder.config.hidden_size
        clip_hidden_dim = self.clip_image_encoder.config.hidden_size
        vlm_hidden_dim = self.vlm_model.config.hidden_size
        assert dino_hidden_dim == clip_hidden_dim

        self.content_mapper = nn.Sequential(
            BasicTransformerBlock(
                dim=dino_hidden_dim,
                num_attention_heads=attn_heads,
                attention_head_dim=dim_head,
            ),
            nn.LayerNorm(dino_hidden_dim, elementwise_affine=False, eps=1e-6),
        )
        self.style_mapper = nn.Sequential(
            BasicTransformerBlock(
                dim=clip_hidden_dim,
                num_attention_heads=attn_heads,
                attention_head_dim=dim_head,
            ),
            nn.LayerNorm(clip_hidden_dim, elementwise_affine=False, eps=1e-6),
        )
        self.pooled_style_mapper = nn.Sequential(
            FeedForward(dim=clip_hidden_dim),
            nn.LayerNorm(clip_hidden_dim, elementwise_affine=False, eps=1e-6),
        )

        self.pr_seq_len = pr_seq_len
        self.perceiver_resampler = PerceiverResampler(
            dim=vlm_hidden_dim,
            dim_out=clip_hidden_dim,
            depth=pr_depth,
            dim_head=dim_head,
            heads=attn_heads,
            num_query_embeds=pr_seq_len,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def enable_gradient_checkpointing(self) -> None:
        self.perceiver_resampler.gradient_checkpointing = True

    def state_dict(self, *args, **kwargs):

        sd = super().state_dict(*args, **kwargs)
        for key in list(sd.keys()):
            if any([key.startswith(module_name) for module_name in self.pretrained_module_names]):
                sd.pop(key)
                
        return sd

    def get_cond_ids(self, n_glyphs, l, type="content"):
        
        if type == "content":
            scale = int(l**0.5)
            cond_ids_list = []
            for seq_idx in range(n_glyphs):
                cond_ids = torch.zeros(scale, scale, 3) # [s, s, 3]
                cond_ids[..., 0] = seq_idx # use seq_idx to divide glyphs
                cond_ids[..., 1] = cond_ids[..., 1] + torch.arange(scale)[:, None]
                cond_ids[..., 2] = cond_ids[..., 2] + torch.arange(scale)[None, :]

                cond_ids = cond_ids.view(-1, 3) # (l, 3)
                cond_ids_list.append(cond_ids)
            cond_ids = torch.cat(cond_ids_list, dim=0) # (n*l, 3)
        elif type == "style":
            cond_ids = torch.zeros(n_glyphs*l, 3) # (m*l, 3)
        else:
            raise ValueError(f"Invalid type: {type}")

        return cond_ids

    def forward(
        self,
        content_images: List[List[Image.Image]] = None,
        style_images: List[List[Image.Image]] = None,
        bg_images: List[Image.Image] = None,
        reg_bboxs: List[List[float]] = None,
        prompts: List[str] = None,
        labels: List[str] = None,
    ):
        if content_images is not None:
            if style_images is not None:
                return self._editing_forward(content_images, style_images)
            elif bg_images is not None and reg_bboxs is not None and prompts is not None and labels is not None:
                return self._generation_forward(content_images, bg_images, reg_bboxs, prompts, labels)
            else:
                raise ValueError("[Fusion Module] Either style_images or prompt with bg_image must be provided.")
        elif style_images is not None and bg_images is not None and reg_bboxs is not None and prompts is not None and labels is not None:
            return self._fusion_forward(style_images, bg_images, reg_bboxs, prompts, labels)
        else:
            raise ValueError("[Fusion Module] Illegal input.")

    def _editing_forward(
        self,
        content_images: List[List[Image.Image]],
        style_images: List[List[Image.Image]],
    ):
        content_embeds_list = []
        style_embeds_list = []
        pooled_style_embeds_list = []
        for content_image, style_image in zip(content_images, style_images):
            content_image_inputs = self.dino_image_processor(images=content_image, return_tensors="pt").pixel_values
            style_image_inputs = self.clip_image_processor(images=style_image, return_tensors="pt").pixel_values

            # Get content embeddings
            _, content_embeds = get_image_embeds(
                image_encoder=self.dino_image_encoder,
                image_inputs=content_image_inputs,
            )
            content_embeds = self.content_mapper(content_embeds) # (n, l, d)

            # Get style embeddings
            pooled_style_embeds, style_embeds = get_image_embeds(
                image_encoder=self.clip_image_encoder,
                image_inputs=style_image_inputs,
            )
            pooled_style_embeds = self.pooled_style_mapper(pooled_style_embeds) # (m, d)
            pooled_style_embeds = pooled_style_embeds.mean(dim=0, keepdim=False) # (d)
            style_embeds = self.style_mapper(style_embeds) # (m, l, d)

            content_embeds_list.append(content_embeds)
            style_embeds_list.append(style_embeds)
            pooled_style_embeds_list.append(pooled_style_embeds)

        b, l, d = len(content_embeds_list), content_embeds_list[0].shape[1], content_embeds_list[0].shape[2]

        content_embeds, c_max_size = pad_and_stack(content_embeds_list, dim=0) # (b, N, l, d)
        style_embeds, s_max_size = pad_and_stack(style_embeds_list, dim=0) # (b, M, l, d)
        content_embeds = content_embeds.view(b, -1, d) # (b, N*l, d)
        style_embeds = style_embeds.view(b, -1, d) # (b, M*l, d)
        pooled_style_embeds = torch.stack(pooled_style_embeds_list, dim=0) # (b, d)

        # Get RoPE embeddings
        c_cond_ids = self.get_cond_ids(c_max_size, l, type="content").to(content_embeds)
        s_cond_ids = self.get_cond_ids(s_max_size, l, type="style").to(style_embeds)
        cond_ids = (c_cond_ids, s_cond_ids)

        return content_embeds, style_embeds, pooled_style_embeds, cond_ids
    
    def _generation_forward(
        self,
        content_images: List[List[Image.Image]],
        bg_images: List[Image.Image],
        reg_bboxs: List[List[float]],
        prompts: List[str],
        labels: List[str],
    ):
        content_embeds_list = []
        style_embeds_list = []
        pooled_style_embeds_list = []
        for content_image, bg_image, reg_bbox, prompt, label in zip(content_images, bg_images, reg_bboxs, prompts, labels):
            content_image_inputs = self.dino_image_processor(images=content_image, return_tensors="pt").pixel_values

            # Get content embeddings
            _, content_embeds = get_image_embeds(
                image_encoder=self.dino_image_encoder,
                image_inputs=content_image_inputs,
            )
            content_embeds = self.content_mapper(content_embeds) # (n, l, d)

            # Get style embeddings
            vlm_embeds = get_vlm_embeds(
                vlm_name=self.vlm_model_name,
                vlm_processor=self.vlm_processor,
                vlm_model=self.vlm_model,
                image=bg_image,
                bbox=reg_bbox,
                prompt=prompt,
                label=label,
            )
            pooled_style_embeds, style_embeds = self.perceiver_resampler(vlm_embeds) # (d) (k*l, d)
            pooled_style_embeds = pooled_style_embeds.to(content_embeds.dtype)
            style_embeds = style_embeds.to(content_embeds.dtype)

            content_embeds_list.append(content_embeds)
            style_embeds_list.append(style_embeds)
            pooled_style_embeds_list.append(pooled_style_embeds)

        b, l, d = len(content_embeds_list), content_embeds_list[0].shape[1], content_embeds_list[0].shape[2]

        content_embeds, c_max_size = pad_and_stack(content_embeds_list, dim=0) # (b, N, l, d)
        content_embeds = content_embeds.view(b, -1, d) # (b, N*l, d)
        style_embeds = torch.stack(style_embeds_list, dim=0) # (b, k*l, d)
        pooled_style_embeds = torch.stack(pooled_style_embeds_list, dim=0) # (b, d)

        # Get RoPE embeddings
        c_cond_ids = self.get_cond_ids(c_max_size, l, type="content").to(content_embeds)
        s_cond_ids = self.get_cond_ids(self.pr_seq_len//l, l, type="style").to(style_embeds)
        cond_ids = (c_cond_ids, s_cond_ids)

        return content_embeds, style_embeds, pooled_style_embeds, cond_ids
    
    def _fusion_forward(
        self,
        style_images: List[List[Image.Image]],
        bg_images: List[Image.Image],
        reg_bboxs: List[List[float]],
        prompts: List[str],
        labels: List[str],
    ):
        style_embeds_list_1 = []
        style_embeds_list_2 = []
        pooled_style_embeds_list_1 = []
        pooled_style_embeds_list_2 = []
        for style_image, bg_image, reg_bbox, prompt, label in zip(style_images, bg_images, reg_bboxs, prompts, labels):
            style_image_inputs = self.clip_image_processor(images=style_image, return_tensors="pt").pixel_values
            
            # Get style embeddings 1
            pooled_style_embeds_1, style_embeds_1 = get_image_embeds(
                image_encoder=self.clip_image_encoder,
                image_inputs=style_image_inputs,
            )
            pooled_style_embeds_1 = self.pooled_style_mapper(pooled_style_embeds_1) # (m, d)
            pooled_style_embeds_1 = pooled_style_embeds_1.mean(dim=0, keepdim=False) # (d)
            style_embeds_1 = self.style_mapper(style_embeds_1) # (m, l, d)
            style_embeds_1 = style_embeds_1.reshape(-1, style_embeds_1.shape[-1]) # (m*l, d)
            style_embeds_1 = style_embeds_1[:self.pr_seq_len, :]
            style_embeds_1 = F.pad(style_embeds_1, (0, 0, 0, self.pr_seq_len - style_embeds_1.shape[0])) # (k*l, d)

            # Get style embeddings 2
            vlm_embeds = get_vlm_embeds(
                vlm_name=self.vlm_model_name,
                vlm_processor=self.vlm_processor,
                vlm_model=self.vlm_model,
                image=bg_image,
                bbox=reg_bbox,
                prompt=prompt,
                label=label,
            )
            pooled_style_embeds_2, style_embeds_2 = self.perceiver_resampler(vlm_embeds.float()) # (d) (k*l, d)

            style_embeds_list_1.append(style_embeds_1)
            style_embeds_list_2.append(style_embeds_2)
            pooled_style_embeds_list_1.append(pooled_style_embeds_1)
            pooled_style_embeds_list_2.append(pooled_style_embeds_2)
        
        style_embeds_1 = torch.stack(style_embeds_list_1, dim=0) # (b, m*l, d)
        style_embeds_2 = torch.stack(style_embeds_list_2, dim=0) # (b, k*l, d)
        pooled_style_embeds_1 = torch.stack(pooled_style_embeds_list_1, dim=0) # (b, d)
        pooled_style_embeds_2 = torch.stack(pooled_style_embeds_list_2, dim=0) # (b, d)

        return style_embeds_1, style_embeds_2, pooled_style_embeds_1, pooled_style_embeds_2