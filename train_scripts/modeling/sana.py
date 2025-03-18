from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel,SanaTransformerBlock,SanaModulatedNorm,GLUMBConv

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AttnProcessor2_0,
    SanaLinearAttnProcessor2_0,
)
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from .encoder import pad_reflection
from diffusers.utils.deprecation_utils import deprecate
from diffusers.models.activations import  FP32SiLU
from .context_transformer import RotaryEmbedding
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def copy_weights(module_from,module_to,mapping = lambda k,v:v):
    state_dict = module_from.state_dict()
    state_dict = {k:mapping(k,v) for k,v in state_dict.items()}
    module_to.load_state_dict(state_dict)
    del state_dict
    return
    
class QwenMLP(nn.Module):
    def __init__(self, hidden_size,intermediate_size,act_fn="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if act_fn == "gelu_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_fn = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_fn = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def cap_bias(k,v):
    if 'to_k.bias' in k:
        max_abs = v.abs().max().item()
        if max_abs > 5:
            v = v / max_abs  * 5
    return v

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)




# attention with rope
class AttnProcessor2_0_Rope:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query,key = apply_rotary_pos_emb(query, key, cos, sin,unsqueeze_dim=1)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SanaTransformerBlockReflectionContext(nn.Module):
    r"""
    Transformer block introduced in [Sana](https://huggingface.co/papers/2410.10629).
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
        qk_norm=None,
    ) -> None:
                # super().__init__(dim=dim,num_attention_heads=num_attention_heads
                #          ,attention_head_dim=attention_head_dim,
                #          dropout=dropout,
                #          num_cross_attention_heads=num_cross_attention_heads,
                #          cross_attention_head_dim=cross_attention_head_dim,
                #          cross_attention_dim=cross_attention_dim,
                #          attention_bias=attention_bias,
                #          norm_elementwise_affine=norm_elementwise_affine,
                #          norm_eps=norm_eps,
                #          attention_out_bias=attention_out_bias,
                #          mlp_ratio=mlp_ratio
                #          )
        # 1. Self Attention
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            qk_norm=qk_norm,
            processor=AttnProcessor2_0_Rope(),
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.attn2 = None
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = QwenMLP(dim,int(dim*mlp_ratio),'silu')
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: int = None,
        width: int = None,
        position_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states,position_embeddings=position_embeddings)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.attn2 is not None:
            raise NotImplementedError()
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # N L D
        #norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states)
        #ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        # N L D
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


def init_weights(module,std=0.02):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    else:
        print(module)



class SanaTransformer2DModelWithContext(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A 2D Transformer model introduced in [Sana](https://huggingface.co/papers/2410.10629) family of models.

    Args:
        in_channels (`int`, defaults to `32`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `32`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `70`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `32`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SanaTransformerBlock", "PatchEmbed", "SanaModulatedNorm"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        reflection: bool = False,
        vision_feature_dim: int = 1024,
        num_reflection_transformer_layers: int = 0,
        reflection_mode: str = 'vanilla', # [vanilla, decoupled]
        context_qk_norm: str = None,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_type="sincos" if interpolation_scale is not None else None,
        )

        # 2. Additional condition embeddings
        self.time_embed = AdaLayerNormSingle(inner_dim)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
        
        # CLIP Embedding
        self.reflection = reflection
        self.reflection_mode = reflection_mode
        block_cls = SanaTransformerBlock
        if self.reflection:
            self.clip_projection = PixArtAlphaTextProjection(in_features=vision_feature_dim, hidden_size=inner_dim)
            self.clip_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
            self.reflection_text_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            self.clip_projection.apply(init_weights)
            self.reflection_text_projection.apply(init_weights)
            
            
            self.reflection_text_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
            self.reflection_transformer =nn.ModuleList(
                [
                SanaTransformerBlockReflectionContext(
                    inner_dim,
                    cross_attention_dim // cross_attention_head_dim,
                    cross_attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=0,
                    cross_attention_head_dim=0,
                    cross_attention_dim=None,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=context_qk_norm
                )
                    for _ in range(num_reflection_transformer_layers)
                ]
            )
            self.reflection_transformer.apply(init_weights)
            if num_reflection_transformer_layers > 0:
                self.reflection_rope = RotaryEmbedding(2048, cross_attention_head_dim)
                self.reflection_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
                self.reflection_norm.weight =  nn.Parameter(torch.ones(inner_dim) * 0.0)
            else:
                self.reflection_rope = None
                self.reflection_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
                self.reflection_norm.weight =  nn.Parameter(torch.ones(inner_dim) * 0.0)
            if reflection_mode == 'decoupled':
                raise NotImplementedError()
        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                block_cls(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)


    @staticmethod
    def convert_context_batch_ids(context_batch_ids):
        groups = []
        if not context_batch_ids:
            return groups

        current_group = []
        current_id = context_batch_ids[0]

        for idx, batch_id in enumerate(context_batch_ids):
            # When the batch id changes, start a new group
            if batch_id != current_id:
                groups.append(current_group)
                current_group = []
                current_id = batch_id
            current_group.append(idx)

        # Append the last group
        groups.append(current_group)
        return groups
    
    def init_weights_copy(self):
        assert self.reflection_mode in ['decoupled']
        if self.reflection_mode == 'decoupled':
            for transformer_layer in self.transformer_blocks:
                transformer_layer.init_weights_copy()
    
    def get_decoupled_trainable_layers(self):
        all_layers = []
        all_layers.append(self.clip_projection)
        all_layers.append(self.clip_norm)
        all_layers.append(self.reflection_text_projection)
        all_layers.append(self.reflection_text_norm)
        all_layers.append(self.reflection_transformer)
        all_layers.append(self.reflection_norm)
        for transformer_layer in self.transformer_blocks:
            all_layers.append(transformer_layer.attn3)
            all_layers.append(transformer_layer.attn3_proj)
            all_layers.append(transformer_layer.norm3)
        return all_layers
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        context_clip_encoding: Optional[torch.Tensor] = None, # N_REFS X 64 X D
        context_text_encoding: Optional[torch.Tensor] = None, # N_REFS X L_FEEDBACK X D
        context_text_encoding_mask: Optional[torch.Tensor] = None, # N_REFS X L_FEEDBACK 
        context_batch_ids: Optional[list] = None, # [ [indices]  [indices]]
        reflection_context_mask_dropout: Optional[torch.Tensor] = None,
        convert_batch_id:bool = True
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        # prepare input size
        batch_size, num_channels, height, width = hidden_states.shape
        # prepare timestamp
        timestep, embedded_timestep = self.time_embed(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )
        # preprocess extra context
        do_reflection = self.reflection and context_clip_encoding is not None
        drop_method = 'embed'              
        if do_reflection:
            reflection_input = []
            context_clip_encoding = self.clip_projection(context_clip_encoding)
            context_clip_encoding = self.clip_norm(context_clip_encoding)
            
            context_text_encoding = self.reflection_text_projection(context_text_encoding)
            context_text_encoding = self.reflection_text_norm(context_text_encoding)
            reflection_context = []
            reflection_context_length = []
            raw_context_batch_ids = context_batch_ids
            if convert_batch_id:
                context_batch_ids = self.convert_context_batch_ids(context_batch_ids)
            for idx,reflection_indices in enumerate(context_batch_ids):
                images_reflection = context_clip_encoding[reflection_indices] # N_REFS X L_FEEDBACK X D
                text_reflection = context_text_encoding[reflection_indices] # N_REFS X L_FEEDBACK X D
                text_lengthes = context_text_encoding_mask[reflection_indices].sum(-1) # N_REFS 
                if reflection_context_mask_dropout is not None and drop_method == 'embed':
                    if reflection_context_mask_dropout[idx] == 0:
                        images_reflection = images_reflection[:1] * 0 # 1 X  L_FEEDBACK X D
                        text_reflection = text_reflection[:1] * 0
                        text_lengthes = [4]
                current_payload = []
                current_length = 0
                for img,text,length in zip(images_reflection,text_reflection,text_lengthes):
                    current_payload.append(img)
                    current_payload.append(text[:length])
                    current_length += img.shape[0]
                    current_length += length
                reflection_context.append(torch.cat(current_payload))
                reflection_context_length.append(current_length)
            reflection_context,reflection_context_mask = pad_reflection(reflection_context,reflection_context_length)
            l_reflection = reflection_context_mask.shape[1]
            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if reflection_context_mask is not None and reflection_context_mask.ndim == 2:
                reflection_context_mask_processed = (1 - reflection_context_mask.to(hidden_states.dtype)) * -10000.0
                reflection_context_mask_processed = reflection_context_mask_processed.unsqueeze(1)
            # 2. Transformer blocks
            position_ids = torch.arange(
                0,  reflection_context.shape[1], device=reflection_context.device
            ).unsqueeze(0) # 1 L 
            if self.reflection_rope is not None:
                position_embeddings = (self.reflection_rope(hidden_states, position_ids),)
                # set up as a tuple
            else:
                position_embeddings = ()
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for block in self.reflection_transformer:
                    reflection_context = self._gradient_checkpointing_func(
                        block,
                        reflection_context,
                        reflection_context_mask_processed,
                        None,
                        None,
                        timestep, # hack way to get rid of timestamp
                        1,
                        l_reflection,
                        *position_embeddings
                    )

            else:
                for block in self.reflection_transformer:
                    reflection_context = block(
                        reflection_context,
                        reflection_context_mask_processed,
                        None,
                        None,
                        timestep,
                        1, # height
                        l_reflection, # wdith 
                        *position_embeddings
                    )
            reflection_context = self.reflection_norm(reflection_context)
            # now fix all context
            
            
                    
            # encoder_attention_mask
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask

        # 1. Input
       
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)



        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        prompt_token_len = encoder_attention_mask.sum(-1) # N
        max_prompt_len = prompt_token_len.max().item()
        encoder_hidden_states_raw = encoder_hidden_states
        encoder_attention_mask_raw = encoder_attention_mask
        # breakpoint()
        # encoder_attention_mask =   encoder_attention_mask[:,:max_prompt_len+4]
        # encoder_hidden_states = encoder_hidden_states[:,:max_prompt_len+4]
        extra_args = ()
        if do_reflection:
            if self.reflection_mode == 'decoupled':
                reflection_context_mask = reflection_context_mask.bool()
                reflection_context_mask = reflection_context_mask.unsqueeze(1)
                if drop_method == 'embed':
                    reflection_context_mask_dropout = torch.ones_like(reflection_context_mask_dropout).bool()
                #if drop_method == 'mask':
                #     reflection_context_mask = reflection_context_mask.bool()
                #     reflection_context_mask = reflection_context_mask.unsqueeze(1)
                # else:
                #     reflection_context_mask = reflection_context_mask.bool()
                #     reflection_context_mask = reflection_context_mask.unsqueeze(1)
                #     reflection_context_mask_dropout = torch.ones_like(reflection_context_mask_dropout).bool()
                if drop_method == 'mask':
                    reflection_context_mask = reflection_context_mask.bool()
                    reflection_context_mask = reflection_context_mask.unsqueeze(1)
                else:
                    reflection_context_mask = torch.ones_like(reflection_context_mask).bool()
                    reflection_context_mask = reflection_context_mask.unsqueeze(1)
                # breakpoint()
                extra_args = (
                    reflection_context,
                    reflection_context_mask,
                    reflection_context_mask_dropout
                )
            else:
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states,reflection_context
                    ],dim=1
                )
                #print(self.caption_norm.weight.max().item(),self.caption_norm.weight.min().item(),self.caption_norm.weight.std().item())
                #print(encoder_hidden_states.mean(),encoder_hidden_states.std(),reflection_context.mean(),reflection_context.std())
                 
                if reflection_context_mask_dropout is not None:
                    # print(reflection_context_mask_dropout)
                    if drop_method == 'mask':
                        reflection_context_mask = reflection_context_mask * reflection_context_mask_dropout.view(batch_size,1)
                    else:
                        pass
                    #print(encoder_attention_mask,reflection_context_mask)
                # print(1,encoder_attention_mask.shape,encoder_attention_mask.dtype)
                encoder_attention_mask =torch.cat(
                    [encoder_attention_mask,reflection_context_mask
                    ],dim=1
                ).long()
                # encoder_hidden_states = encoder_hidden_states[:,:max_prompt_len+1]
                # encoder_attention_mask = encoder_attention_mask[:,:max_prompt_len+1]
                # print(2,encoder_attention_mask.shape,encoder_attention_mask.dtype)
                assert encoder_attention_mask.shape == encoder_hidden_states.shape[:2]
                encoder_attention_mask[:,-1] = 0
        
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            #encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.bool()
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        
        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    *extra_args
                )

        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    *extra_args
                )

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
