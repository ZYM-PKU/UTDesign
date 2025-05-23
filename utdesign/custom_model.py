import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxSingleTransformerBlock
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_xformers_available
from custom_attention import FluxAttnProcessor_xformers

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomTransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_content = AdaLayerNormZero(dim)
        self.norm1_style = AdaLayerNormZero(dim)

        processor = FluxAttnProcessor2_0()

        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_content = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_style = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_content = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.ff_style = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        content_hidden_states: torch.FloatTensor,
        style_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        xc_image_rotary_emb=None,
        xs_image_rotary_emb=None,
        joint_attention_kwargs=None,
        lamda_gate: float = 1.0,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_content_hidden_states, content_gate_msa, content_shift_mlp, content_scale_mlp, content_gate_mlp = self.norm1_content(
            content_hidden_states, emb=temb,
        )
        norm_style_hidden_states, style_gate_msa, style_shift_mlp, style_scale_mlp, style_gate_mlp = self.norm1_style(
            style_hidden_states, emb=temb,
        )
        
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attn_output, content_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_content_hidden_states,
            image_rotary_emb=xc_image_rotary_emb, # use latents and content rope
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        attn_output, style_attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_style_hidden_states,
            image_rotary_emb=xs_image_rotary_emb, # use latents and style rope
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = lamda_gate * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `content_hidden_states`.
        content_attn_output = content_gate_msa.unsqueeze(1) * content_attn_output
        content_hidden_states = content_hidden_states + content_attn_output

        norm_content_hidden_states = self.norm2_content(content_hidden_states)
        norm_content_hidden_states = norm_content_hidden_states * (1 + content_scale_mlp[:, None]) + content_shift_mlp[:, None]

        content_ff_output = self.ff_content(norm_content_hidden_states)
        content_hidden_states = content_hidden_states + content_gate_mlp.unsqueeze(1) * content_ff_output
        if content_hidden_states.dtype == torch.float16:
            content_hidden_states = content_hidden_states.clip(-65504, 65504)

        # Process attention outputs for the `style_hidden_states`.
        style_attn_output = style_gate_msa.unsqueeze(1) * style_attn_output
        style_hidden_states = style_hidden_states + style_attn_output

        norm_style_hidden_states = self.norm2_style(style_hidden_states)
        norm_style_hidden_states = norm_style_hidden_states * (1 + style_scale_mlp[:, None]) + style_shift_mlp[:, None]

        style_ff_output = self.ff_style(norm_style_hidden_states)
        style_hidden_states = style_hidden_states + style_gate_mlp.unsqueeze(1) * style_ff_output
        if style_hidden_states.dtype == torch.float16:
            style_hidden_states = style_hidden_states.clip(-65504, 65504)

        return hidden_states, content_hidden_states, style_hidden_states


class CustomTransformer2DModel(FluxTransformer2DModel):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super(FluxTransformer2DModel, self).__init__()
        
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)
        self.content_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.style_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                CustomTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

        if is_xformers_available():
            logger.info("xformers is available. Using xformers attention processor.")
            self.set_attn_processor(FluxAttnProcessor_xformers())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, AdaLayerNormZero, AdaLayerNormContinuous)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(
        self,
        hidden_states: torch.Tensor,
        content_hidden_states: torch.Tensor = None,
        style_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        cond_ids: Tuple[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        lamda_gate: float = 1.0,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        timestep = timestep.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep, pooled_projections)

        hidden_states = self.x_embedder(hidden_states)
        content_hidden_states = self.content_embedder(content_hidden_states)
        style_hidden_states = self.style_embedder(style_hidden_states)

        l_x, l_c, l_s = hidden_states.shape[1], content_hidden_states.shape[1], style_hidden_states.shape[1]

        c_cond_ids, s_cond_ids = cond_ids
        xc_ids = torch.cat((img_ids, c_cond_ids), dim=0)
        xs_ids = torch.cat((img_ids, s_cond_ids), dim=0)
        ids = torch.cat((img_ids, c_cond_ids, s_cond_ids), dim=0)
        xc_image_rotary_emb = self.pos_embed(xc_ids)
        xs_image_rotary_emb = self.pos_embed(xs_ids)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, content_hidden_states, style_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    content_hidden_states,
                    style_hidden_states,
                    temb,
                    xc_image_rotary_emb,
                    xs_image_rotary_emb,
                    joint_attention_kwargs,
                    lamda_gate,
                    **ckpt_kwargs,
                )

            else:
                hidden_states, content_hidden_states, style_hidden_states = block(
                    hidden_states=hidden_states,
                    content_hidden_states=content_hidden_states,
                    style_hidden_states=style_hidden_states,
                    temb=temb,
                    xc_image_rotary_emb=xc_image_rotary_emb,
                    xs_image_rotary_emb=xs_image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    lamda_gate=lamda_gate,
                )

        hidden_states = torch.cat([hidden_states, content_hidden_states, style_hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = hidden_states[:, :l_x, :]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)