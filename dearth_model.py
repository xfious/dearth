import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

import logging
import copy

from dearth_config import DearthConfig

_USE_FAST_ROPE = False

class RMSNorm(torch.nn.Module): # a variant of LayerNorm that is faster and more memory efficient
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # set the weight to be 1 initially
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

        self.register_buffer("default_pos_ids", 
                             torch.arange(0, self.max_position_embeddings, dtype=torch.long).view(-1, self.max_position_embeddings), 
                             persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False) # shape: (max_seq_len_cached, dim // 2)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed




class FastRope(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        cis = precompute_freqs_cis(dim, max_position_embeddings, theta=base)
        self.register_buffer("cis", cis, persistent=False)

    def forward(self, start_idx, seq_len):
        return self.cis[start_idx:start_idx+seq_len, :]

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    with torch.no_grad():
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis      

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis.shape: {freqs_cis.shape}, x.shape: {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)




class AttentionMask(nn.Module):
    attn_mask: torch.Tensor = None
    def __init__(self, config: DearthConfig):
        super().__init__()
        self.config = config
        self.sliding_window_size = config.sliding_window_size
        self.front_window_size = config.front_window_size
        if self.attn_mask is None: 
            tmp_attn_mask = self.build_causal_and_window_mask(config.max_token_len, config.sliding_window_size, config.front_window_size)
            self.attn_mask = tmp_attn_mask.requires_grad_(False) # shape: (max_token_len, max_token_len)
        #self.register_buffer("attn_mask", self.build_causal_and_window_mask(config.max_token_len, config.sliding_window_size, config.front_window_size).requires_grad_(False), persistent=False)

    def forward(self, bz, n_head, q_seq_len, kv_seq_len, q_start_idx: int, device, dtype) -> torch.Tensor:
        if self.attn_mask.device != device or self.attn_mask.dtype != dtype:
            self.attn_mask = self.attn_mask.to(device=device, dtype=dtype).requires_grad_(False)
        end_idx = q_start_idx + q_seq_len
        q_k_diff_len = kv_seq_len - q_seq_len # it should be >= 0, because it is meaningless to attend future tokens
        top = q_start_idx
        bottom = end_idx
        if q_start_idx == 0 and q_k_diff_len == 0:
            # assume: sliding window size = 100, front window size = 50
            # case 1: training: q_start_idx = 0, q_seq_len = 1000, kv_seq_len = 1000
            mask = self.attn_mask[:end_idx, :end_idx]
        elif q_k_diff_len > 0 and q_start_idx > 0 and end_idx >= kv_seq_len:
            # TODO: not allow in training; remove this line after testing
            raise RuntimeError(f"NOT FOR TRAINING: q_start_idx = {q_start_idx}, q_seq_len = {q_seq_len}, kv_seq_len = {kv_seq_len}")
            if end_idx > self.front_window_size + self.sliding_window_size:
                # case 2: qsl < kvsl: q_start_idx = 190, q_seq_len = 10, kv_seq_len = 150, end_idx = 200
                # mask = self.attn_mask[top:bottom, :self.front_window_size] + \
                #     self.attn_mask[q_start_idx:end_idx, end_idx - (kv_seq_len - self.front_window_size):end_idx]
                mask = torch.cat([self.attn_mask[top:bottom, :self.front_window_size], self.attn_mask[top:bottom, end_idx - (kv_seq_len - self.front_window_size):end_idx]], dim=-1)
            elif end_idx <= self.front_window_size + self.sliding_window_size:
                # case 3: qsl < kvsl: q_start_idx = 140, q_seq_len = 10, kv_seq_len = 150, end_idx = 150
                mask = self.attn_mask[top:bottom, :end_idx]
        else:
            raise RuntimeError(f"q_start_idx = {q_start_idx}, q_seq_len = {q_seq_len}, kv_seq_len = {kv_seq_len}")
        return mask.expand(bz, n_head, q_seq_len, kv_seq_len).detach()
        
    
    @staticmethod
    def build_causal_and_window_mask(seq_len, sliding_window_size, front_window_size) -> torch.Tensor:
        mask = torch.ones(seq_len, seq_len)
        if seq_len > sliding_window_size: # need to apply sliding window mask, beacause the sequence is too long
            mask = torch.triu(mask, diagonal=-sliding_window_size+1)
            if front_window_size > 0:
                tmp_front_mask = torch.cat([torch.ones(seq_len, front_window_size), torch.zeros(seq_len, seq_len-front_window_size)], dim=-1)
                tmp_front_mask = torch.tril(tmp_front_mask, diagonal=-sliding_window_size)
                mask = mask + tmp_front_mask
        # apply causal mask
        mask = mask.tril(diagonal=0)
        mask = mask.log() # map 0 to -inf, 1 to 0
        # print(f"mask.shape: {mask.shape}, and mask")
        # print(mask)
        return mask
    

class SharedAttentionMask(nn.Module):
    def __init__(self, config: DearthConfig):
        super().__init__()
        self.config = config
        self.sliding_window_size = config.sliding_window_size
        self.front_window_size = config.front_window_size
        tmp_attn_mask = self.build_causal_and_window_mask(config.max_token_len, config.sliding_window_size, config.front_window_size)
        self.register_buffer("attn_mask", tmp_attn_mask, persistent=False)

    def forward(self, q_seq_len, kv_seq_len, q_start_idx: int) -> torch.Tensor:
        end_idx = q_start_idx + q_seq_len
        q_k_diff_len = kv_seq_len - q_seq_len # it should be >= 0, because it is meaningless to attend future tokens
        top = q_start_idx
        bottom = end_idx
        if q_start_idx == 0 and q_k_diff_len == 0:
            # assume: sliding window size = 100, front window size = 50
            # case 1: training: q_start_idx = 0, q_seq_len = 1000, kv_seq_len = 1000
            mask = self.attn_mask[:end_idx, :end_idx]
        elif q_k_diff_len > 0 and q_start_idx > 0 and end_idx >= kv_seq_len:
            # TODO: not allow in training; remove this line after testing
            raise RuntimeError(f"NOT FOR TRAINING: q_start_idx = {q_start_idx}, q_seq_len = {q_seq_len}, kv_seq_len = {kv_seq_len}")
            if end_idx > self.front_window_size + self.sliding_window_size:
                # case 2: qsl < kvsl: q_start_idx = 190, q_seq_len = 10, kv_seq_len = 150, end_idx = 200
                # mask = self.attn_mask[top:bottom, :self.front_window_size] + \
                #     self.attn_mask[q_start_idx:end_idx, end_idx - (kv_seq_len - self.front_window_size):end_idx]
                mask = torch.cat([self.attn_mask[top:bottom, :self.front_window_size], self.attn_mask[top:bottom, end_idx - (kv_seq_len - self.front_window_size):end_idx]], dim=-1)
            elif end_idx <= self.front_window_size + self.sliding_window_size:
                # case 3: qsl < kvsl: q_start_idx = 140, q_seq_len = 10, kv_seq_len = 150, end_idx = 150
                mask = self.attn_mask[top:bottom, :end_idx]
        else:
            raise RuntimeError(f"q_start_idx = {q_start_idx}, q_seq_len = {q_seq_len}, kv_seq_len = {kv_seq_len}")
        return mask.detach() # shape: (1, 1, seqlen, seqlen)
        
    
    @staticmethod
    def build_causal_and_window_mask(seq_len, sliding_window_size, front_window_size) -> torch.Tensor:
        mask = torch.ones(seq_len, seq_len)
        if seq_len > sliding_window_size: # need to apply sliding window mask, beacause the sequence is too long
            mask = torch.triu(mask, diagonal=-sliding_window_size+1)
            if front_window_size > 0:
                tmp_front_mask = torch.cat([torch.ones(seq_len, front_window_size), torch.zeros(seq_len, seq_len-front_window_size)], dim=-1)
                tmp_front_mask = torch.tril(tmp_front_mask, diagonal=-sliding_window_size)
                mask = mask + tmp_front_mask
        # apply causal mask
        mask = mask.tril(diagonal=0)
        mask = mask.log() # map 0 to -inf, 1 to 0
        # print(f"mask.shape: {mask.shape}, and mask")
        # print(mask)
        return mask



def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8, device=None):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292

    retrun shape: (1, num_heads, 1, sequence_length)
    """
    alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32, device=device).view(1, 1, 1, sequence_length)
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.float32, device=device)
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:num_heads]

    alibi = alibi * slopes
    return alibi


# def build_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8, device=None):
#     r"""
#     Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
#     relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
#     the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
#     https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292

#     retrun shape: (1, num_heads, 1, sequence_length)
#     """
#     slope = []
#     m_power = (-8/num_heads)
#     m_increace = -8/num_heads
#     for i in range(num_heads):
#         slope.append(m_power)
#         m_power += m_increace
#     slope = torch.tensor(slope, device=device)
#     alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32, device=device).view(1, 1, 1, sequence_length)
#     alibi = alibi * slope.view(1, num_heads, 1, 1)
#     return alibi

def compute_alibi(num_heads, sequence_length, alibi_bias_max=8, device=None):
    r"""
    Link to paper: https://arxiv.org/abs/2108.12409 - Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation. This implementation has been copied from
    the alibi implementation of MPT source code that led to slightly different results than the Bloom alibi:
    https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L292

    retrun shape: (1, num_heads, 1, sequence_length)
    """
    slope = []
    m_power = (-8/num_heads)
    m_increace = -8/num_heads
    for i in range(num_heads):
        slope.append(2 ** m_power)
        m_power += m_increace
    slope = torch.tensor(slope, device=device)
    alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32, device=device).view(1, 1, 1, sequence_length)
    alibi = alibi * slope.view(1, num_heads, 1, 1)
    return alibi


class Attention(nn.Module):
    def __init__(self, config: DearthConfig):
        super().__init__()
        assert config.dim % config.n_head == 0

        # regularization
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.dim = config.dim
        assert config.dim % config.n_head == 0
        self.dim_qk_head = config.dim_qk_head if config.dim_qk_head is not None else config.dim // config.n_head
        self.dim_v_head = config.dim // config.n_head
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        self.n_kv_group = config.n_head // config.n_kv_head
        self.dropout_rate = config.dropout_rate

        self.alibi_emb = None
        self.pos_emb = None

        self.sliding_window_size = config.sliding_window_size

        def _fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)
        
        # neg_inf_mask = _fill_with_neg_inf(torch.ones_like(torch.empty(config.max_token_len, config.max_token_len)))
        # window_size_mask = torch.triu(neg_inf_mask, diagonal=1)
        # if config.sliding_window_size is not None and config.max_token_len > config.sliding_window_size:
        #     window_size_mask = window_size_mask + torch.tril(neg_inf_mask, diagonal=-config.sliding_window_size)
        # self.register_buffer("window_size_mask", window_size_mask, persistent=False)
        # if config.use_alibi:
        #     alibi_emb = compute_alibi(config.n_head, config.max_token_len) # shape: (1, n_head, 1, seqlen)
        #     #self.alibi_emb = self.alibi_emb.expand(1, config.n_head, config.max_token_len, config.max_token_len) # shape: (1, n_head, seqlen, seqlen)
        #     self.register_buffer("alibi_emb", alibi_emb, persistent=False)

        self.window_size_mask = AttentionMask(config)

        if config.use_rotary:
            if not _USE_FAST_ROPE:
                self.pos_emb = RotaryEmbedding(
                    self.dim_qk_head,
                    max_position_embeddings=config.max_token_len,
                    base=config.rope_theta,
                )
            if _USE_FAST_ROPE:
                self.pos_emb = FastRope(
                    self.dim_qk_head,
                    max_position_embeddings=config.max_token_len,
                    base=config.rope_theta,
                )

        # query, key, values projections for all heads
        self.wq = nn.Linear(self.dim, self.n_head * self.dim_qk_head, bias=True)
        self.wk = nn.Linear(self.dim, self.n_kv_head * self.dim_qk_head, bias=True)
        self.wv = nn.Linear(self.dim, self.dim // self.n_kv_group, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        
    def forward(self, x: Tensor, attn_mask: Tensor, start_idx: Optional[int] = 0):
        batch_size, seqlen, emb_dim = x.size() # batch size, sequence length, embedding dimensionality (dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # split embedding dim into number of heads
        xq = xq.view(batch_size, seqlen, self.n_head, self.dim_qk_head)
        xk = xk.view(batch_size, seqlen, self.n_kv_head, self.dim_qk_head)
        xv = xv.view(batch_size, seqlen, self.n_kv_head, self.dim_v_head)

        if self.pos_emb is not None and _USE_FAST_ROPE:
            xq, xk = apply_rotary_emb(xq, xk, self.pos_emb(start_idx, seqlen))

        # transpose to get dimensions batch_size * n_head * seqlen * emb_dim
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        kv_seqlen = xk.size(2)

        # apply positional embeddings
        if self.pos_emb is not None and not _USE_FAST_ROPE:
            # self.pos_emb = self.pos_emb.to(x.device, dtype=x.dtype)
            # xq, xk = apply_rotary_pos_emb(xq, xk, self.pos_emb[start_idx:start_idx+seqlen])
            cos, sin = self.pos_emb(xv, seq_len=kv_seqlen)
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, self.pos_emb.default_pos_ids[:, :kv_seqlen])

        # TODO: add cache for fast inference


        # grouped query
        xk = repeat_kv(xk, self.n_kv_group)
        xv = repeat_kv(xv, self.n_kv_group)

        # self.window_size_mask = self.window_size_mask.to(x.device, dtype=x.dtype)
        # attn_mask = self.window_size_mask[start_idx:start_idx+seqlen, start_idx:start_idx+kv_seqlen]
        # attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # shape: (1, 1, seqlen, seqlen)
        # attn_mask = attn_mask.expand(batch_size, self.n_head, seqlen, kv_seqlen) # shape: (batch_size, n_head, seqlen, seqlen)
        # if self.alibi_emb is not None:
        #     self.alibi_emb = self.alibi_emb.to(x.device, dtype=x.dtype)
        #     attn_mask = attn_mask + self.alibi_emb[:,:,:,:kv_seqlen]

        #attn_mask = self.window_size_mask(batch_size, self.n_head, seqlen, kv_seqlen, start_idx, x.device, x.dtype) # -inf or 0
        
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout_rate if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, emb_dim) # merge heads

        # output projection
        return self.wo(y)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    hidden_states.shape = (batch, n_kv_head, seqlen, head_dim)
    """
    # if n_rep == 1:
    #     return hidden_states
    # return torch.repeat_interleave(hidden_states, n_rep, dim=1)

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
#     bs, slen, n_kv_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
#     return (
#         x[:, :, :, None, :]
#         .expand(bs, slen, n_kv_heads, n_rep, head_dim)
#         .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
#     )

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.dim
        hidden_dim = config.dim * 4 if config.hidden_dim is None else config.hidden_dim
        multiple_of = 64 if config.multiple_of is None else config.multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # round up to nearest multiple of multiple_of

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Mimic_Attn(Attention):
    def __init__(self, config):
        new_config = copy.deepcopy(config)
        new_config.n_head = config.mimic_n_head if config.mimic_n_head is not None else config.n_head
        new_config.n_kv_head = config.mimic_n_kv_head if config.mimic_n_kv_head is not None else config.n_kv_head
        new_config.dim_qk_head = config.mimic_dim_qk_head if config.mimic_dim_qk_head is not None else config.dim_qk_head
        new_config.dropout_rate = config.mimic_attn_dropout if config.mimic_attn_dropout is not None else 0.0
        new_config.use_rotary = config.mimic_use_rotary if config.mimic_use_rotary is not None else config.use_rotary
        new_config.use_alibi = config.mimic_use_alibi if config.mimic_use_alibi is not None else config.use_alibi

        super().__init__(new_config)
        self.saved_q = None
        self.saved_k = None
        self.saved_v = None
        self.saved_attn_map = None

    def forward(self, x: Tensor, attn_mask: Tensor, start_idx: Optional[int] = 0): # shape of attn_mask: (bz, n_head, q_seq_len, kv_seq_len)
        batch_size, seqlen, emb_dim = x.size() # batch size, sequence length, embedding dimensionality (dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        self.saved_v = xv

        # split embedding dim into number of heads
        xq = xq.view(batch_size, seqlen, self.n_head, self.dim_qk_head)
        xk = xk.view(batch_size, seqlen, self.n_kv_head, self.dim_qk_head)
        xv = xv.view(batch_size, seqlen, self.n_kv_head, self.dim_v_head)

        if self.pos_emb is not None and _USE_FAST_ROPE:
            xq, xk = apply_rotary_emb(xq, xk, self.pos_emb(start_idx, seqlen))

        # transpose to get dimensions batch_size * n_head * seqlen * emb_dim
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        kv_seqlen = xk.size(2)

        # # apply positional embeddings
        # if self.pos_emb is not None:
        #     self.pos_emb = self.pos_emb.to(x.device)
        #     xq, xk = apply_pos_emb(xq, xk, self.pos_emb[start_idx:start_idx+seqlen])
        if self.pos_emb is not None and not _USE_FAST_ROPE:
            cos, sin = self.pos_emb(xv, seq_len=kv_seqlen)
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, self.pos_emb.default_pos_ids[:, :kv_seqlen])

        # TODO: add cache for fast inference

        # grouped query
        xk = repeat_kv(xk, self.n_kv_group)
        xv = repeat_kv(xv, self.n_kv_group)

        # self.window_size_mask = self.window_size_mask.to(x.device)
        # kv_seqlen = xk.size(2)
        # attn_mask = self.window_size_mask[start_idx:start_idx+seqlen, start_idx:start_idx+kv_seqlen]
        # attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # shape: (1, 1, seqlen, seqlen)
        # attn_mask = attn_mask.expand(batch_size, self.n_head, seqlen, kv_seqlen) # shape: (batch_size, n_head, seqlen, seqlen)
        # if self.alibi_emb is not None:
        #     self.alibi_emb = self.alibi_emb.to(x.device)
        #     attn_mask = attn_mask + self.alibi_emb[:,:,:,:kv_seqlen]

        #attn_mask = self.window_size_mask(batch_size, self.n_head, seqlen, kv_seqlen, start_idx, x.device, x.dtype) # -inf or 0

        attn_weights = torch.matmul(xq, xk.transpose(2, 3)) * (1 / math.sqrt(self.dim_qk_head)) # shape: (batch_size, n_head, seqlen, seqlen)
        attn_weights = attn_weights + attn_mask.expand(batch_size, self.n_head, seqlen, kv_seqlen) # shape: (batch_size, n_head, seqlen, seqlen
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(xq.dtype) # shape: (batch_size, n_head, seqlen, seqlen)
        # use log_softmax to avoid overflow
        #attn_weights = F.log_softmax(attn_weights, dim=-1).exp() # shape: (batch_size, n_head, seqlen, seqlen)
        self.saved_attn_map = attn_weights

        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)

        y = torch.matmul(attn_weights, xv) # shape: (batch_size, n_head, seqlen, head_dim)

        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, emb_dim) # merge heads

        # output projection
        return self.wo(y)

    def get_intermediate_attn_v(self):
        return self.saved_attn_map, self.saved_v


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.dim)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.dim)
        self.mlp = MLP(config)

        self.residual_factor = config.residual_factor

    def forward(self, x: Tensor, attn_mask: Tensor, start_idx: int):
        # post-LN
        residual = x
        x = self.attn(x, attn_mask, start_idx=start_idx)
        x = self.ln_1(self.residual_connection(x, residual))

        residual = x
        x = self.mlp(x)
        x = self.ln_2(self.residual_connection(x, residual))

        return x
    
    def residual_connection(self, x, residual):
        # residual factor should > 1.0
        return residual * self.residual_factor + x



class DearthModel(nn.Module):
    def __init__(self, config: DearthConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_token_len is not None

        self.layer_init_factor = config.layer_init_factor if config.layer_init_factor is not None else float(config.n_layer * 8) ** (-1/2)
        self.residual_factor = config.residual_factor if config.residual_factor is not None else float(config.n_layer * 2) ** (1/4)
        if config.residual_factor is None:
            config.residual_factor = self.residual_factor
            logging.warning(f"residual_factor is not set, using default value {self.residual_factor} = (2 * n_layer) ** 1/4")
        if config.layer_init_factor is None:
            config.layer_init_factor = self.layer_init_factor
            logging.warning(f"layer_init_factor is not set, using default value {self.layer_init_factor} = (n_layer * 8) ** -1/2")
        
        self.config = config

        layers = []
        for i in range(config.n_layer):
            if config.mimic_attn_layer is not None and i+1 == config.mimic_attn_layer:
                new_layer = TransformerBlock(config)
                new_layer.attn = Mimic_Attn(config)
                layers.append(new_layer)
            else:
                layers.append(TransformerBlock(config))

        self.layers = nn.ModuleList(layers)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.ln_before = RMSNorm(config.dim)
        self.shared_attn_mask = SharedAttentionMask(config)

        if config.mimic_attn_layer is not None and config.mimic_attn_layer > 0 and config.mimic_attn_layer <= config.n_layer:
            self.mimic_attn = self.layers[config.mimic_attn_layer-1].attn
        else:
            self.mimic_attn = None

        # initialize weights
        _init_weight(self, self.layer_init_factor)

    def get_input_device(self):
        return self.embed_tokens.weight.device

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.xavier_normal_(module.weight, gain=self.layer_init_factor)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.xavier_normal_(module.weight, gain=1)
    #     elif isinstance(module, RMSNorm):
    #         module.weight.data.fill_(1.0)
            


    def forward(self, tokens, start_idx=0): # return all logits
        batch_size, seqlen = tokens.size()
        if seqlen > self.config.max_token_len:
            raise ValueError(f"input sequence length {seqlen} exceeds maximum sequence length {self.config.max_token_len}")

        # create token embeddings from token table; x.shape = (batch_size, seqlen, dim)
        h = self.embed_tokens(tokens)
        assert h.size() == (batch_size, seqlen, self.config.dim)

        h = self.ln_before(h)

        # transformer layers
        attn_mask = self.shared_attn_mask(seqlen, seqlen, q_start_idx=start_idx) # TODO: it will not work if q_seq_len != kv_seq_len
        for layer in self.layers:
            h = layer(h, attn_mask, start_idx=start_idx) # h.shape = (batch_size, seqlen, dim)

        return h, None


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        #n_params = sum(p.numel() for p in self.parameters())
        n_params = sum(p.numel() for p in self.transformer.layers[0].parameters() if p.requires_grad)
        return int(n_params)
    
    
    def get_intermediate_attn_v(self):
        if self.mimic_attn is None:
            return torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
        return self.mimic_attn.get_intermediate_attn_v()


class DearthForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DearthConfig):
        super().__init__()
        self.model = DearthModel(config)
        self.dearth_config = config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        torch.nn.init.xavier_normal_(self.lm_head.weight, gain=1)

        self.front_window_size = config.front_window_size
        self.sliding_window_size = config.sliding_window_size

    def get_input_device(self):
        return self.model.get_input_device()
    
    def get_intermediate_attn_v(self):
        return self.model.get_intermediate_attn_v()
    
    def print_all_params(self):
        for name, param in self.named_parameters():
            print(f"name: {name}, param.shape: {param.shape}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        use_cache: Optional[bool] = False,
    ) ->Tuple: #-> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs = self.model(
            tokens=input_ids
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        output = (logits,) + outputs[1:]
        return output


def _init_weight(model, weight_init_factor): # TODO: fix this part if change any model structure
    small_list = {'wv', 'wo', 'w1', 'w2', 'w3'}
    norm_list = {'ln_before', 'ln_2', 'ln_1'}
    for name, p in model.named_parameters():
        percise_name = name.split(".")[-2]
        if "bias" in name:
            logging.debug(f"the parameter {name} is initialized with 0.0")
            p.data.fill_(0.0)
        elif percise_name in small_list:
            logging.debug(f"the parameter {name} is initialized with gain={weight_init_factor}")
            torch.nn.init.xavier_normal_(p, gain=weight_init_factor)
        elif percise_name in norm_list:
            logging.debug(f"the parameter {name} is initialized with 1.0")
            p.data.fill_(1.0)
        else:
            logging.debug(f"the parameter {name} is initialized with gain=1.0")
            torch.nn.init.xavier_normal_(p, gain=1)
            