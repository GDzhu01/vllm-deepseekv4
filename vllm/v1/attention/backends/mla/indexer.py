# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
    split_prefill_chunks,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


class DeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1 if current_platform.is_rocm() else 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


class DeepseekSVFIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SVF_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    use_large_context_topk: bool
    offsets: torch.Tensor | None  # Precomputed offsets for speculative decoding


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor

    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int = 1
    natively_supported_next_n: list[int] = [1, 2]
    # TODO (matt): integrate kernel with next_n = 4 support

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM is not available. Disabling CUDA graph support "
                "for sparse attention indexer. This may reduce performance.",
            )
            return AttentionCGSupport.NEVER
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        next_n = self.num_speculative_tokens + 1
        self.reorder_batch_threshold += self.num_speculative_tokens
        self.use_flattening = next_n not in self.natively_supported_next_n

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.decode_lens_buffer = torch.empty(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.arange_buffer = torch.arange(
            scheduler_config.max_num_seqs * next_n,
            dtype=torch.int32,
            device=self.device,
        )
        self.expanded_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )
        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for SVF support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (SVF)
            # Compressed slot mapping output buffer
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )


    def build_one_prefill_chunk(
        self, reqs_start, reqs_end, query_start_loc_cpu, seq_lens_cpu, block_table
    ):
        prefill_query_start_loc = (
            query_start_loc_cpu[reqs_start : reqs_end + 1]
            - query_start_loc_cpu[reqs_start]
        )
        cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(
            prefill_query_start_loc, seq_lens_cpu[reqs_start:reqs_end], self.device
        )
        token_start = query_start_loc_cpu[reqs_start].item()
        token_end = query_start_loc_cpu[reqs_end].item()
        total_seq_lens = seq_lens_cpu[reqs_start:reqs_end].sum()
        seq_idx = torch.arange(0, reqs_end - reqs_start, dtype=torch.int32)
        token_to_seq = torch.repeat_interleave(
            seq_idx, seq_lens_cpu[reqs_start:reqs_end]
        ).to(self.device)
        assert total_seq_lens <= self.max_prefill_buffer_size
        cu_seq_lens = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    seq_lens_cpu[reqs_start:reqs_end].cumsum(dim=0),
                ]
            )
            .to(torch.int32)
            .to(self.device)
        )
        return DeepseekV32IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            cu_seq_lens=cu_seq_lens,
            token_to_seq=token_to_seq,
            total_seq_lens=total_seq_lens,
            block_table=block_table[reqs_start:reqs_end],
            token_start=token_start,
            token_end=token_end,
            num_reqs=reqs_end - reqs_start,
        )


    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        compressed_seq_lens_cpu = seq_lens_cpu
        if self.compress_ratio > 1:
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
            compressed_seq_lens = seq_lens // self.compress_ratio
            compressed_seq_lens_cpu = seq_lens_cpu // self.compress_ratio

        prefill_metadata = None
        # TODO(cmq): There are bugs due to the conflict, 
        # but this should no releated to ascend
        if num_prefills > 0:
            chunk_seq_ids = split_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                self.max_prefill_buffer_size,
                request_offset=num_decodes,
            )

            chunks = [
                self.build_one_prefill_chunk(
                    reqs_start,
                    reqs_end,
                    query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                )
                for reqs_start, reqs_end in chunk_seq_ids
            ]
            # for req_slice, query_slice in chunk_specs:
            #     # metadata = build_prefill_chunk_metadata(
            #     #     req_slice.start,
            #     #     req_slice.stop,
            #     #     query_start_loc,
            #     #     query_start_loc_cpu,
            #     #     seq_lens,
            #     #     compressed_seq_lens,
            #     #     compressed_seq_lens_cpu,
            #     #     block_table,
            #     #     self.compress_ratio,
            #     #     query_slice=query_slice,
            #     #     skip_kv_gather=query_slice.start > 0,
            #     # )
            #     # Skip when total_seq_lens is 0 (i.e., no compressed token).
            #     if metadata is not None:
            #         chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            # Padded CUDA graph requests have block_table entries of -1.
            # Clamp to 0 to prevent OOB access in the DeepGEMM kernel.
            # This is safe because padded requests have seq_lens=0, so the
            # kernel produces no meaningful output for those rows.
            block_table.clamp_(min=0)

            max_decode_len = int(decode_lens_cpu.max().item())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len == next_n

            if use_native and next_n > 1:
                offsets = self.offsets_buffer
                batch_size = num_decodes
            elif max_decode_len > 1:
                # Flatten multi-token decode requests into single-token
                # batch entries, expanding seq_lens and block tables so
                # the kernel always sees next_n=1.

                # Also handles the edge case where use_flattening=False
                # but max_decode_len != next_n (e.g. a batch containing some
                # short prefills (q_len < next_n) and no true decodes).

                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # [7, 6, 8, 0] -> [7, 7, 7, 6, 8, 8, 8, 8]
                expanded_base = torch.repeat_interleave(
                    seq_lens - decode_lens, decode_lens, output_size=actual_expanded
                )

                # [0, 3, 4, 8] -> [0, 0, 0, 3, 4, 4, 4, 4]
                expanded_starts = torch.repeat_interleave(
                    common_attn_metadata.query_start_loc[:num_decodes],
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [0, 1, 2, 0, 0, 1, 2, 3]
                positions_within = (
                    self.arange_buffer[:actual_expanded] - expanded_starts
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.expanded_seq_lens_buffer[:actual_expanded] = (
                    expanded_base + positions_within + 1
                )
                self.expanded_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                offsets = None
                batch_size = num_decode_tokens
            else:
                offsets = None
                batch_size = num_decodes

            # For SVF (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if max_decode_len > 1:
                    # MTP flattening: seq_lens is already in
                    # expanded_seq_lens_buffer, compress in-place.
                    seq_lens //= self.compress_ratio
                else:
                    # Non-MTP: seq_lens is a view of common_attn_metadata,
                    # copy to pre-allocated buffer to avoid modifying shared
                    # state and to ensure CUDA graph address stability.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # DeepGEMM is required for the paged MQA logits on CUDA devices
            if current_platform.is_cuda() and is_deep_gemm_supported():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )

            # Decide which top-k kernel to use based on batch size and sequence length
            # Decision logic based on micro-benchmark results:
            # - large_context_topk wins for batch <= 128 and seq_len > 8K
            # - top_k_per_row_decode wins for batch > 128 or seq_len <= 8K
            _is_large_context = common_attn_metadata.max_seq_len > 8192
            use_large_context_topk = batch_size <= 128 and _is_large_context

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=False,
                schedule_metadata=self.scheduler_metadata_buffer,
                use_large_context_topk=use_large_context_topk,
                offsets=offsets,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        # if get_tensor_model_parallel_rank() == 0:
        #     logger.info(f"attn_metadata: {attn_metadata}")
        return attn_metadata


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
    )


@triton.jit
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # Outputs
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (
            (offset < query_len)
            & (abs_pos >= query_slice_start)
            & (abs_pos < query_slice_stop)
        )
        out_pos = abs_pos - query_slice_start

        # Compute cu_seq_len_ks
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, seq_start, mask=mask)

        # Compute cu_seq_len_ke
        seq_len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            seq_start + seq_len_per_token,
            mask=mask,
        )

    # Compute token_to_seq
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)
