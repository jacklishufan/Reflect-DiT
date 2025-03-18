from torch import nn
import torch
from typing import Optional
DEFAULT_CONFIG ="Qwen/Qwen2-VL-2B-Instruct" 
import math
from typing import Tuple

def _compute_default_rope_parameters(
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    rope_theta=500000.0,
    head_dim=128,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = rope_theta
    partial_rotary_factor = 1.0
    head_dim = head_dim
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def _compute_llama3_parameters(
     device: "torch.device", seq_len: Optional[int] = None,
     factor=8,
     low_freq_factor=1,
     high_freq_factor=4,
     old_context_len=2048,
     head_dim=128,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters( device, seq_len,head_dim=head_dim)

    factor = factor  # `8` in the original implementation
    low_freq_factor =low_freq_factor  # `1` in the original implementation
    high_freq_factor = high_freq_factor  # `4` in the original implementation
    old_context_len = old_context_len   # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

class RotaryEmbedding(nn.Module):
    
    def __init__(self, max_position_embeddings=2048,head_dim=128, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"

        self.rope_type = "default"
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        inv_freq, self.attention_scaling = _compute_llama3_parameters(device,head_dim=head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


if __name__ == '__main__':
    # model = ContextTransformerReflection()
    # from transformers import AutoConfig,LlamaForCausalLM
    # config = 'meta-llama/Llama-3.1-8B-Instruct'
    # config = AutoConfig.from_pretrained(config)
    # config.num_hidden_layers = 2
    # model = LlamaForCausalLM._from_config(config)
    # breakpoint()
    model = RotaryEmbedding(2048,32)
    x = torch.rand(2,1024,128*6)
    y = model(x,torch.arange(x.shape[1])[None])
    y[0].shape
    breakpoint()