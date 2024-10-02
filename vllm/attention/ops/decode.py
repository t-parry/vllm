from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

@triton.jit
def _triton_paged_attention_v1(
    output, #[num_seq, num_head_q, head_size]
    q,#[num_seq, num_head_q, head_size]
    k_cache, #[num_blocks, num_head_kv, block_size, head_size]
    v_cache, #[num_blocks, num_head_kv, block_size, head_size]
    head_q_to_head_kv, #[num_head_q], GQA, mapping from query head to kv_head
    scale, 
    block_tables, #[num_seq, max_num_blocks_per_seq]
    seq_lens, #[num_seq], length of each sequence
    max_num_blocks_per_seq, #Maximum blocks per sequence
    stride_q_seq,
    stride_q_head,
    stride_o_seq,
    stride_o_head,
    stride_k_blocks,
    stride_k_head,
    stride_k_blk,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)
    kv_head_idx = tl.load(head_q_to_head_kv + head_idx)

    #Load q
    offs_q = seq_idx * stride_q_seq + head_idx * stride_q_head + tl.arange(0, HEAD_SIZE)
    q = tl.load(q + offs_q)
    q = (q * scale).to(tl.float16)

    #Read seq length for this sequence
    seq_len = tl.load(seq_lens + seq_idx)

    #Initialize qkv, m_prev, d_prev
    qkv = tl.zeros([BLOCK_SIZE, HEAD_SIZE], dtype=tl.float32)
    m_prev = tl.zeros([1,1], tl.float32) - float("inf")
    d_prev = tl.zeros([1,1], tl.float32)

    #Offsets for block, head_size, 
    block_offs = tl.arange(0, BLOCK_SIZE)
    head_size_offs = tl.arange(0, HEAD_SIZE)
    
    #Block base_ptr for this sequence
    block_base_ptrs = block_tables + seq_idx * max_num_blocks_per_seq

    #KV base offs
    kv_base_offs = (kv_head_idx * stride_k_head + block_offs[:, None]*stride_k_blk) + head_size_offs[None, :]

    #loop over all blocks for this sequence
    for b in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        #read block index from block table
        block_idx = tl.load(block_base_ptrs + b)

        #mask for k and v cache.
        mask = (block_offs[:, None] < (seq_len - b * BLOCK_SIZE)) &(
            head_size_offs[None, :] < HEAD_SIZE
         )
        #kv offsets
        kv_offs = block_idx * stride_k_blocks + kv_base_offs
        #load k
        k = tl.load(k_cache + kv_offs, mask=mask, other=0.0)
        #load v
        v = tl.load(v_cache + kv_offs, mask=mask, other=0.0)

        #qk
        x_i = tl.sum(q[None, :] * k, axis=1)[:,None]

        #apply masking for causal attention
        x_i = tl.where(block_offs[:, None] < (seq_len - b * BLOCK_SIZE), x_i, float("-inf"))

        #m_i, maximum compared with m_prev
        m_i = tl.maximum(m_prev, tl.max(x_i, axis=0))
        
        #d_i = scale d_prev and add new sum
        d_i = d_prev * tl.exp(m_prev - m_i) + tl.sum(tl.exp(x_i - m_i), axis=0)
        #qkv for this block
        qkv = (
            qkv * (d_prev * tl.exp(m_prev - m_i) / d_i) + (tl.exp(x_i - m_i) / d_i) * v
        )
        
        #update m_prev and d_prev
        m_prev = m_i
        d_prev = d_i

    #store output
    #offset for output
    offs_q = seq_idx * stride_o_seq + head_idx * stride_o_head + tl.arange(0, HEAD_SIZE)
    tl.store(output + offs_q, tl.sum(qkv, axis=0))



@torch.inference_mode()
def triton_paged_attention_v1(
    output : torch.Tensor, #[num_seq, num_head_q, head_size]
    query : torch.Tensor, #[num_seq, num_head_q, head_size]
    key_cache : torch.Tensor, #[num_blocks, num_head_kv, block_size, head_size]
    value_cache : torch.Tensor, #[num_blocks, num_head_kv, block_size, head_size]
    num_heads_kv : torch.Tensor, 
    scale: torch.Tensor,
    block_tables : torch.Tensor, #[num_seq, max_num_blocks_per_seq]
    seq_lens : torch.Tensor, #[num_seq]
    block_size : int,
    max_seq_len : int, 
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype : str, #"auto"
    k_scale: float,
    v_scale : float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int =0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):

    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    max_num_blocks_per_seq = block_tables.shape[1]

    #GQA
    num_queries_per_kv = num_heads // num_heads_kv
    head_q_to_head_kv = torch.repeat_interleave(torch.arange(num_heads_kv, dtype=torch.int32, device="cuda"), num_queries_per_kv)

    #Call Triton kernel
    grid = (num_heads, num_seqs, 1)
    _triton_paged_attention_v1[grid](
        output,
        query,
        key_cache,
        value_cache,
        head_q_to_head_kv,
        scale,
        block_tables,
        seq_lens,
        max_num_blocks_per_seq,
        query.stride(0),
        query.stride(1),
        output.stride(0),
        output.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
    )

    return output



@triton.jit
def _triton_paged_attention_v2(
    exp_sums,
    max_logits,
    out,
    q,
    k_cache,
    v_cache,
    head_q_to_head_kv,
    attn_scale,
    block_tables,
    seq_lens,
    partition_size,
    max_num_blocks_per_seq,
    alibi_slope,
    stride_q_seq,
    stride_q_head,
    stride_o_seq,
    stride_o_head,
    stride_o_partition,
    stride_k_blocks,
    stride_k_head,
    stride_k_blk,
    stride_exp_seq,
    stride_exp_head
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(axis=1)
    par_idx = tl.program_id(axis=2)

    seq_len = tl.load(seq_lens + seq_idx)
    if par_idx * partition_size >= seq_len:
        return
    
    num_context_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    num_blocks_per_par = partition_size // BLOCK_SIZE
    
    head_idx = tl.program_id(axis=0)
    kv_head_idx = tl.load(head_q_to_head_kv + head_idx)

    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)
    
    block_offs = tl.arange(0, BLOCK_SIZE)
    head_size_offs = tl.arange(0, HEAD_SIZE)
    q = tl.load(q + seq_idx * stride_q_seq + head_idx * stride_q_head + head_size_offs)
    q = (q * scale).to(tl.float16)

    qkv = tl.zeros([BLOCK_SIZE, HEAD_SIZE], dtype=tl.float32)
    qk_max = float("-inf")
    exp_sum = 0.0
    fp16_0 = tl.zeros([1,1], dtype=kv_cache_dtype.element_ty)
    base_offs_kv = (
        kv_head_idx * stride_k_head
        + block_offs[:, None] * stride_blk
        + head_size_offs[None, :]
    )

    for block_idx in range(start_block_idx, end_block_idx):
        physical_block_idx = tl.load(
            block_tables + seq_idx * max_num_blocks_per_seq + block_idx
        )
        mask = (block_offs[:, None] < (seq_len - block_idx * BLOCK_SIZE)) & (
            head_size_offs[None, :] < HEAD_SIZE
        )
        offs_kv = physical_block_idx * stride_k_blocks + base_offs_kv

        k = tl.load(k_cache + offs_kv, mask, other=0.0)
        v = tl.load(k_cache + offs_kv, mask, other=0.0)

        _qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
        _qk += alibi_slope * (block_idx * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_max = tl.maximum(tl.max(_qk, axis=0), qk_max)
        qk = tl.where(
            block_offs[:, None] < (seq_len - block_idx * BLOCK_SIZE),
            _qk[:, None],
            float("-inf"),
        )

        _exp_sum = (exp_sum * tl.exp(qk_max - _qk_max) / _exp_sum) + (tl.exp(qk - _qk_max) / _exp_sum) * v
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)/ _exp_sum)
                + (tl.exp(qk - _qk_max) / _exp_sum) * v
        )
        qk_max = _qk_max
        exp_sum = _exp_sum
    
    #Store exp_sum and max logits
    offs_exp = seq_idx * stride_exp_seq + head_idx * stride_exp_head + par_idx
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, qk_max)

    #Store output
    offs_out = (
        seq_idx * stride_o_seq
        + head_idx * stride_o_head
        + par_idx * stride_o_partition
        + head_size_offs
    )
    tl.store(out + offs_out, tl.sum(qkv, axis=0))


@triton.jit
def _triton_paged_attention_v2_unroll4(
    exp_sums, #[num_seq, num_heads, max_num_partitions]
    max_logits, #[num_seq, num_heads, max_num_partitions]
    out, #[num_seq, num_heads, max_num_partitions, head_size]
    q, #[num_seqs, num_heads, head_size]
    k_cache, #[num_blocks, num_heads_kv, block_size, head_size]
    v_cache,#[num_blocks, num_heads_kv, block_size, head_size]
    head_q_to_head_kv, #[num_heads]
    attn_scale,
    block_tables, #[num_seq, max_num_blocks_per_seq]
    seq_lens,
    partition_size,
    max_num_blocks_per_seq,
    alibi_slopes,
    stride_q_seq,
    stride_q_head,
    stride_o_seq,
    stride_o_head,
    stride_o_partition,
    stride_k_blocks,
    stride_k_head,
    stride_k_blk,
    stride_exp_seq,
    stride_exp_head,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr
):
    seq_idx = tl.program_id(axis=1)
    par_idx = tl.program_id(axis=2)
    seq_len = tl.load(seq_lens + seq_idx)

    if par_idx * partition_size >= seq_len:
        return
    
    num_context_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    num_blocks_per_par = partition_size // BLOCK_SIZE

    start_block_idx = par_idx * num_blocks_per_par
    end_block_idx = tl.minimum(start_block_idx + num_blocks_per_par, num_context_blocks)

    head_idx = tl.program_id(axis=0)
    kv_head_idx = tl.load(head_mapping + head_idx)

    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)

    qkv = tl.zeros([BLOCK_SIZE, HEAD_SIZE], dtype=tl.float32)
    qk_max = float("-inf")
    exp_sum = 0.0
    fp16_0 = tl.zeros([1,1], dtype=k_cache.dtype.element_ty)
    base_offs_kv = (
        kv_head_idx *stride_kn
        + block_offs[:, None] * stride_k_blk
        + head_size_offs[None, :]
    )
    block_base_ptrs = block_tables + seq_idx * max_blocks_per_seq

    for block_idx in range(start_block_idx, end_block_idx, 4):
        mask_0 = block_offs[:, None] < (seq_len - (block_idx + 0) * BLOCK_SIZE)
        mask_1 = block_offs[:, None] < (seq_len - (block_idx + 1) * BLOCK_SIZE)
        mask_2 = block_offs[:, None] < (seq_len - (block_idx + 2) * BLOCK_SIZE)
        mask_3 = block_offs[:, None] < (seq_len - (block_idx + 3) * BLOCK_SIZE)

        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0) * stride_k_blocks + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1) * stride_k_blocks + base_offs_kv
        offs_kv_2 = tl.load(block_base_ptrs + block_idx + 2) * stride_k_blocks + base_offs_kv
        offs_kv_3 = tl.load(block_base_ptrs + block_idx + 3) * stride_k_blocks + base_offs_kv

        k_0 = tl.load(k_cache + offs_kv_0, mask=mask_0, other=fp16_0)
        k_1 = tl.load(k_cache + offs_kv_1, mask=mask_0, other=fp16_0)
        k_2 = tl.load(k_cache + offs_kv_2, mask=mask_0, other=fp16_0)
        k_3 = tl.load(k_cache + offs_kv_3, mask=mask_0, other=fp16_0)
        
        v_0 = tl.load(k_cache + offs_kv_0, mask=mask_0, other=fp16_0)
        v_1 = tl.load(k_cache + offs_kv_1, mask=mask_0, other=fp16_0)
        v_2 = tl.load(k_cache + offs_kv_2, mask=mask_0, other=fp16_0)
        v_3 = tl.load(k_cache + offs_kv_3, mask=mask_0, other=fp16_0)

        _qk_0 = tl.sum((q[None, :] * k_0).to(tl.float32), axis=1)
        _qk_1 = tl.sum((q[None, :] * k_1).to(tl.float32), axis=1)
        _qk_2 = tl.sum((q[None, :] * k_2).to(tl.float32), axis=1)
        _qk_3 = tl.sum((q[None, :] * k_3).to(tl.float32), axis=1)

        _qk_0 += alibi_slope * ((block_idx + 0) * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_1 += alibi_slope * ((block_idx + 1) * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_2 += alibi_slope * ((block_idx + 2) * BLOCK_SIZE + block_offs - seq_len + 1)
        _qk_3 += alibi_slope * ((block_idx + 3) * BLOCK_SIZE + block_offs - seq_len + 1)

        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_2, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_3, axis=0), qk_max)

        _exp_sum = (
            exp_sum * tl.exp(qk_max - _qk_max)
            + tl.sum(tl.exp(_qk_0 - _qk_max), axis=0)
            + tl.sum(tl.exp(_qk_1 - _qk_max), axis=0)
            + tl.sum(tl.exp(_qk_2 - _qk_max), axis=0)
            + tl.sum(tl.exp(_qk_3 - _qk_max), axis=0)
        )

        qkv = (
            qkv * (exp_sum * tl.exp(qk_max - _qk_max)/ _exp_sum)
            + (tl.exp(qk_0 - _qk_max) / _exp_sum) * v_0
        )
        qk_max = _qk_max
        exp_sum = _exp_sum
    
    offs_exp = seq_idx * stride_exp_seq + head_idx * stride_exp_head + par_idx
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, qk_max)

    offs_out = (
        seq_idx * stride_o_seq
        + head_idx * stride_o_head
        + par_idx * stride_o_partition
        + head_size_offs
    )
    tl.store(out + offs_out, tl.sum(qkv, axis=0))
        

@triton.jit
def _paged_attention_v2_reduce(
    out, #[num_seqs, num_heads, head_size]
    exp_sums, #[num_seq, num_heads, max_num_partitions]
    max_logits,
    tmp_out,
    context_lens,
    stride_exp_seq,
    stride_exp_head,
    stride_out_seq,
    stride_out_head,
    stride_tmp_seq,
    stride_tmp_head,
    stride_tmp_partition,
    HEAD_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,    
):
    seq_idx = tl.program_id(axis=1)
    head_idx = tl.program_id(axis=0)
    context_len = tl.load(context_lens + seq_idx)

    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)

    exp_sum = 0.0
    max_logit = float("-inf")
    offs_logit = seq_idx * stride_exp_seq + head_idx * stride_exp_head

    head_size_offs = tl.arange(0, HEAD_SIZE)
    tmp_out_ptr = seq_idx * stride_tmp_seq + head_idx * stride_tmp_head
    out_ptr = seq_idx * stride_out_seq + head_idx * stride_out_head + head_size_offs

    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    logits = tl.load(
        max_logits + offs_logit + tl.arange(0, NUM_PARTITIONS),
        mask = tl.arange(0, NUM_PARTITIONS) < num_partitions,
        other=float("-inf")
    )
    max_logit = tl.max(logits, axis=0)

    exp_sum = tl.load(
        exp_sums + offs_logit + tl.arange(0, NUM_PARTITIONS),
        mask=tl.arange(0, NUM_PARTITIONS) < num_partitions,
        other=0.0,
    )
    rescaled_exp_sum = exp_sum * tl.exp(logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)

    tmp = tl.load(
        tmp_out
        + tmp_out_ptr
        + tl.arange(0, NUM_PARTITIONS)[:, None] * stride_tmp_k
        + head_size_offs
    )
    acc += tl.sum(tmp * rescaled_exp_sum[:, None], axis=0)

    inv_sum = 1.0 / (global_exp_sum + 1e-6)
    tl.store(out + out_ptr, acc * inv_sum)

@torch.inference_mode()
@torch.inference_mode()
def triton_paged_attention_v2(
    output,
    exp_sums,
    max_logits,
    tmp_output,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens,
    block_size,
    max_seq_len,
    alibi_slopes,
    kv_cache_dtype,
    k_scale,
    v_scale,
    tp_rank,
    blocksparse_local_blocks,
    blocksparse_vert_stride,
    blocksparse_block_size,
    blocksparse_head_sliding_step,
):

    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    max_num_blocks_per_seq = block_tables.shape[1]

    #GQA
    num_queries_per_kv = num_heads // num_head_kv
    head_q_to_head_kv = torch.repeat_interleave(torch.arange(num_heads_kv, dtype=torch.int32, device="cuda"), num_queries_per_kv)

    max_num_partitions = triton.cdiv(max_seq_len, PARTITION_SIZE)

    grid = (num_heads, num_seqs, max_num_partitions)
    _triton_paged_attention_v2_unroll4[grid](
        exp_sums,
        max_logits,
        tmp_output,
        query,
        k_cache,
        v_cache,
        head_q_to_head_kv,
        scale,
        block_tables,
        seq_lens,
        _PARTITION_SIZE,
        max_num_blocks_per_seq,
        alibi_slopes,
        query.stride(0),
        query.stride(1),
        tmp_output.stride(0),
        tmp_output.stride(1),
        tmp_output.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        exp_sums.stride(0),
        exp_sums.stride(1),
        block_size,
        head_size
    )

    num_partitions = triton.next_power_of_2(max_num_partitions)
    grid = (num_heads, num_seqs, 1)
    _paged_attention_v2_reduce[grid](
        output,
        exp_sums,
        tmp_output,
        seq_lens,
        exp_sums.stride(0),
        exp_sums.stride(1),
        output.stride(0),
        output.stride(1),
        tmp_output(0),
        tmp_output(1),
        tmp_output(2),
        head_size,
        num_partitions
    )