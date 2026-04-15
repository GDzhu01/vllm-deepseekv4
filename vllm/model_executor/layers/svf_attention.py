# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SVF MLA Attention Layer
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
    SVFFlashMLASparseBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import SVFSWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


@dataclass
class SVFMLAModules:
    """Modules used in SVF MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("svf_multi_head_latent_attention")
class SVFMultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: SVFMLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # SVF-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a
        self.wo_b = mla_modules.wo_b

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer
        # FlashMLA sparse prefill kernel requires 64 or 128 heads
        min_heads = 64
        if self.n_local_heads < min_heads:
            pad_size = min_heads - self.n_local_heads
            self.attn_sink_padded = F.pad(
                mla_modules.attn_sink,
                (0, pad_size),
                value=-float("inf"),
            )
        else:
            self.attn_sink_padded = mla_modules.attn_sink

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        self.swa_cache_layer = SVFSWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = SVFMLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=self.attn_sink_padded,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the SVFMLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
            )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".svf_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        # Q projection: wq_a -> q_norm -> wq_b -> reshape
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(qr)
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        # RMS normalization on q (per-head, no learnable weights)
        q = self.q_head_norm(q)

        kv = self.kv_norm(kv)

        # RoPE
        q, kv = self.rotary_emb(positions, q, kv.unsqueeze(1))
        kv = kv.squeeze(dim=1)

        # Only wrap attention computation
        output = torch.ops.vllm.svf_attention(
            hidden_states, q, kv, qr, positions, self.layer_name
        )

        return output

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Indexer (if sparse attention with compress_ratio == 4)
        if self.indexer is not None:
            _topk_indices = self.indexer(
                hidden_states, qr, positions, self.indexer_rotary_emb
            )

        compressed_kv = None
        if self.compressor is not None:
            compressed_kv = self.compressor(hidden_states, positions, self.rotary_emb)

        # FlashMLA sparse prefill kernel requires 64 or 128 heads
        min_heads = 64
        if self.n_local_heads < min_heads:
            pad_size = min_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention
        # o shape: [num_tokens, padded_heads, head_dim]
        o = self.mla_attn(q, kv, compressed_kv, positions)

        # Slice back to original head count after attention
        if self.n_local_heads < min_heads:
            o = o[:, : self.n_local_heads, :]
        # Apply inverse RoPE on output (rope portion only)
        o, _ = self.rotary_emb(positions, o, inverse=True)
        # NOTE(yifan): not sure if we have a better way for o proj.
        # Output projection: wo_a (einsum per group) + wo_b
        # Reshape: [num_tokens, n_local_heads, head_dim] ->
        #          [num_tokens, n_local_groups, heads_per_group * head_dim]
        # Ensure contiguous before view after cat/inverse RoPE operations
        num_tokens = hidden_states.shape[0]
        o = o.view(num_tokens, self.n_local_groups, -1)
        # wo_a weight: [n_local_groups * o_lora_rank, heads_per_group * head_dim]
        # Reshape to [n_local_groups, o_lora_rank, heads_per_group * head_dim]
        wo_a_weight = self.wo_a.weight
        wo_a_weight = wo_a_weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # einsum: [num_tokens, groups, dim] x [groups, lora_rank, dim] ->
        #         [num_tokens, groups, lora_rank]
        o = torch.einsum("tgd,grd->tgr", o, wo_a_weight)
        # Flatten groups and lora_rank: [num_tokens, n_local_groups * o_lora_rank]
        # Then apply wo_b: [num_tokens, hidden_size]
        output = self.wo_b(o.flatten(1))
        return output


def svf_attention(
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    return self.attention_impl(hidden_states, q, kv, qr, positions)


def svf_attention_fake(
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    return torch.empty(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )


direct_register_custom_op(
    op_name="svf_attention",
    op_func=svf_attention,
    mutates_args=["q"],
    fake_impl=svf_attention_fake,
)


@triton.jit
def quantize_and_insert_k_kernel(
    # Input tensors
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    # Output tensor
    k_cache_ptr,  # [num_blocks, block_bytes] as uint8 (flattened view)
    # Dimensions
    num_tokens,
    input_dim: tl.constexpr,  # 512
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded)
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 8 (7 real + 1 padding)
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - [0, 64*576): Token data, each token has 448 fp8 + 128 bf16
    - [64*576, 64*576 + 64*8): Scales, each token has 8 uint8 scales
    - [64*576 + 64*8, block_stride): Padding

    One program per token.
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Get slot mapping
    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Input pointer for this token
    input_row_ptr = k_ptr + pid * input_dim

    # Calculate pointers into the cache block
    # Block base pointer
    cache_block_ptr = k_cache_ptr + block_idx * block_stride

    # Token data pointer: token data is stored contiguously at start of block
    # Each token's data is at offset pos_in_block * token_data_size
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

    # Scale pointer: scales are stored after ALL token data in the block
    # Scale for this token is at offset (64 * 576) + pos_in_block * 8
    token_scale_ptr = (
        cache_block_ptr + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    # Token data layout: [0:448] fp8, [448:576] bf16
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    # ========== Quantize and store FP8 portion (first 448 elements) ==========
    # Using UE8M0 quantization strategy (scale is power of 2, stored as uint8 exponent)
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block

        if qblock_start < fp8_dim:
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim

            # Load bf16 input
            x = tl.load(input_row_ptr + offsets, mask=mask, other=0.0)

            # Compute absmax scale (same as CUDA kernel)
            abs_x = tl.abs(x)
            block_max = tl.max(abs_x, axis=0)
            block_max = tl.maximum(block_max, 1e-4)  # Match CUDA: fmaxf(amax, 1e-4)

            # UE8M0: Round scale UP to next power of 2
            # scale = 2^ceil(log2(block_max / fp8_max))
            raw_scale = block_max / fp8_max
            log_scale = tl.log2(raw_scale)
            exponent = tl.ceil(log_scale)  # Round UP to next integer exponent
            scale = tl.exp2(exponent)  # scale = 2^exponent (power of 2)

            # Quantize to fp8: fp8_value = bf16_value / scale
            x_scaled = x / scale
            x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)

            # Convert to fp8, then bitcast to uint8 for storage
            x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

            # Store as uint8 (1 byte each)
            tl.store(token_fp8_ptr + offsets, x_uint8, mask=mask)

            # UE8M0 scale encoding: stored_value = exponent + 127 (bias)
            # During dequant: scale = 2^(stored_value - 127)
            encoded_scale = exponent + 127.0
            encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
            tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # Padding scale at index 7
    tl.store(token_scale_ptr + 7, tl.zeros((), dtype=tl.uint8))

    # ========== Store BF16 portion (last 64 elements, no quantization) ==========
    bf16_input_offset = fp8_dim

    # Process bf16 in chunks of 16
    bf16_out_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        chunk_offsets = i * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(input_row_ptr + bf16_input_offset + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


def quantize_and_insert_k_cache(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    block_size: int = 64,
    is_ue8m0: bool = True,
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576
    """
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert is_ue8m0, "Only support ue8m0 quantization."

    # NOTE: When using DP, slot_mapping.shape[0] can be less than k.shape[0] due to
    # padding. Always use slot_mapping.shape[0] as the token count.
    num_tokens = slot_mapping.shape[0]
    block_stride = k_cache.stride(0)  # bytes per block

    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    grid = (num_tokens,)

    quantize_and_insert_k_kernel[grid](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        input_dim=512,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=block_stride,
        fp8_max=FP8_MAX,
        n_quant_blocks=8,
    )


@triton.jit
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    # Constants
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 or 128 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded) int32
    output_dim: tl.constexpr,  # 512
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 7 real blocks
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        # Gather all tokens
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        # Calculate the actual token index in the sequence
        pos = start_pos + i

        # Calculate which block and position within block
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # Calculate pointers into the cache block
        # Cast to int64 to avoid int32 overflow in the multiplication
        # (physical_block_idx * block_stride can exceed 2^31 when there
        # are many KV-cache blocks, e.g. >= 57 K with block_stride ~37 K).
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        # Token data pointer
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        # Scale pointer: after all token data
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        # Token data layout: [0:448] fp8, [448:576] bf16
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        # Output pointer for this token (flattened)
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        # ========== Dequantize FP8 portion using UE8M0 ==========
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block

            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim

                # Load quantized fp8 values (stored as uint8)
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)

                # Bitcast uint8 back to fp8
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)

                # Convert fp8 to float32 for computation
                x_float = x_fp8.to(tl.float32)

                # Load and decode UE8M0 scale
                # UE8M0: scale = 2^(stored_value - 127)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)

                # Dequantize: bf16_value = fp8_value * scale
                x_dequant = x_float * scale

                # Store as bf16
                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

        # ========== Copy BF16 portion directly ==========
        bf16_output_offset = fp8_dim  # After 448 elements in output

        # Read bf16 from cache
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))

        # Process in chunks of 16
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_and_gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=FP8_MAX,
        n_quant_blocks=7,
    )


class SVFMLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: SVFSWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"SVFMLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: SVFSWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        # SVF only supports fp8 kv-cache format for now
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"SVF only supports fp8 kv-cache format for now, got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for SVF for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once(
                "Using DeepSeek's fp8_ds_mla KV cache format. To use standard "
                "fp8 kv-cache format, please set `--attention-backend "
                "FLASHINFER_MLA_SPARSE`"
            )

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return SVFFlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if self.compress_ratio <= 1:  # SWA part. Allocated separately as SVFSWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            model_version="svf",
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        compressed_kv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Handle dummy run (no metadata)
        if not isinstance(attn_metadata, dict):
            return torch.zeros_like(q)
        flashmla_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = (
            self.kv_cache if not swa_only else None
        )
        swa_kv_cache = self.swa_cache_layer.kv_cache
        # flatten the last two dims
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # swa cache insertion
        quantize_and_insert_k_cache(
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            block_size=swa_metadata.block_size,
        )

        if compressed_kv is not None:
            assert flashmla_metadata
            assert self_kv_cache is not None
            block_size = flashmla_metadata.block_size // self.compress_ratio
            # flatten the last two dims
            self_kv_cache_2d = self_kv_cache.view(self_kv_cache.shape[0], -1)
            quantize_and_insert_k_cache(
                compressed_kv,
                self_kv_cache_2d,
                flashmla_metadata.slot_mapping,
                block_size=block_size,
            )

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Allocate output
        output = torch.zeros_like(q)

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            output[:num_decode_tokens] = self._forward_decode(
                q=q[:num_decode_tokens],
                positions=positions[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
            )

        return output

    def _forward_decode(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
    ):
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        block_size = None
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[:num_decode_tokens]
            else:
                topk_indices = generate_topk_indices_for_c128a(
                    positions, compress_ratio=128
                )
            topk_indices = compute_global_topk_indices(
                topk_indices,
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
            )
            topk_lens = (topk_indices >= 0).sum(dim=-1).int()
            topk_indices = topk_indices.view(num_decode_tokens, 1, -1)

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # Guard: zero out lengths for padding requests so FlashMLA skips them.
        # During CUDA graph replay, padding requests have stale positions and
        # block_table entries, producing invalid KV cache indices.  Setting
        # their lengths to 0 prevents the kernel from reading those indices.
        if topk_lens is not None:
            topk_lens = topk_lens * swa_metadata.is_valid_token[:num_decode_tokens]

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices
        q = q.unsqueeze(1)
        # pad head_size to 64 or 128
        actual_num_heads = q.size(2)
        padded_num_heads = self.padded_heads

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # Get fresh metadata for each call. The FlashMLA kernel allocates
        # tile_scheduler_metadata and num_splits internally. During CUDA graph
        # capture, these allocations go to the graph's private memory pool and
        # are automatically reused at the same addresses during replay.
        tile_metadata = get_mla_metadata()[0]

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
        )

        # Slice output back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out.squeeze(1)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.seq_lens[num_decodes:]
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu

        # Compute query_lens and query_start_loc_cpu for prefill
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        query_lens_cpu = query_lens_cpu[num_decodes:]
        query_start_loc_cpu = torch.empty(
            num_prefills + 1, dtype=torch.int32, device="cpu"
        )
        query_start_loc_cpu[0] = 0
        query_start_loc_cpu[1:] = torch.cumsum(query_lens_cpu, dim=0)

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        query_lens = query_lens[num_decodes:]
        prefix_lens = seq_lens - query_lens
        gather_lens = query_lens + torch.clamp(prefix_lens, max=self.window_size - 1)

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                topk_indices = generate_topk_indices_for_c128a(
                    positions[:num_prefill_tokens], compress_ratio=128
                )
        else:
            # HACK
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            topk_indices = topk_indices[:num_prefill_tokens].fill_(-1)

        PREFILL_CHUNK_SIZE = 4  # TODO
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        max_compressed_tokens = topk_indices.shape[-1]
        M = max_compressed_tokens + self.window_size + self.max_num_batched_tokens
        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=max_compressed_tokens,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = query_start_loc_cpu[chunk_start]
            query_end = query_start_loc_cpu[chunk_end]

            chunk_query_lens = query_lens[chunk_start:chunk_end]
            chunk_query_start_loc = torch.empty(
                chunk_size + 1, dtype=torch.int32, device=q.device
            )
            chunk_query_start_loc[:1] = 0
            torch.cumsum(chunk_query_lens, dim=0, out=chunk_query_start_loc[1:])

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                chunk_query_start_loc,
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                M,
                max_compressed_tokens,
            )

            output_chunk, _, _ = flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )

            # Write to output (use query_slice since output is prefill-local)
            output[query_start:query_end] = output_chunk


def compute_global_topk_indices(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    _compute_global_topk_indices_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices


@triton.jit
def _compute_global_topk_indices_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset, mask=mask
        )
        is_valid = topk_indices >= 0

        block_indices = topk_indices // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & is_valid,
        )
        block_offsets = topk_indices % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, topk = topk_indices.shape
    num_reqs = seq_lens.shape[0]
    combined_indices = torch.full(
        (num_tokens, topk + window_size),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    topk_len = (topk_indices >= 0).sum(dim=-1)
    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_len,
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=triton.next_power_of_2(topk),
    )
    return combined_indices, combined_lens


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk_len_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    # The SWA portion of the gathered buffer starts from position
    # (seq_len - gather_len), not position 0. We need this offset
    # to correctly index into the gathered buffer.
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        topk_len = tl.load(topk_len_ptr + token_idx)
        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len

        # Top-k indices from compressed KV
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset, mask=mask
        )
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + offset,
            topk_indices + M * batch_idx,
            mask=mask,
        )

        # SWA indices
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)
        offset = tl.arange(0, WINDOW_SIZE)
        # Index into gathered buffer: N + (position - gather_start)
        # For positions [pos - swa_len + 1, pos], the buffer indices are:
        # [N + pos - swa_len + 1 - gather_start, N + pos - gather_start]
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + offset,
            M * batch_idx + N + offset + pos - swa_len + 1 - gather_start,
            mask=offset < swa_len,
        )

        combined_len = topk_len + swa_len
        tl.store(combined_lens_ptr + token_idx, combined_len)


@triton.jit
def _c128a_topk_indices_kernel(
    topk_indices_ptr,
    topk_indices_stride,
    positions_ptr,
    compress_ratio,
    max_compressed_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx)
    num_compressed = (position + 1) // compress_ratio
    num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
    for i in tl.range(0, max_compressed_tokens, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        topk_indices = tl.where(offset < num_compressed, offset, -1)
        tl.store(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            topk_indices,
            mask=offset < max_compressed_tokens,
        )


def generate_topk_indices_for_c128a(
    positions: torch.Tensor,
    compress_ratio: int = 128,
    max_compressed_tokens: int = 8192,
) -> torch.Tensor:
    num_tokens = positions.shape[0]
    topk_indices = torch.empty(
        (num_tokens, max_compressed_tokens), dtype=torch.int32, device=positions.device
    )
    if num_tokens == 0:
        return topk_indices

    _c128a_topk_indices_kernel[(num_tokens,)](
        topk_indices,
        topk_indices.stride(0),
        positions,
        compress_ratio,
        max_compressed_tokens,
        BLOCK_SIZE=1024,
    )
    return topk_indices
