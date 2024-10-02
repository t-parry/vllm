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


