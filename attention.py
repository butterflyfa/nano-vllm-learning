import torch


@torch.inference_mode()
def decode_attention(query, cache, block_manager, layer_id, request_id, num_tokens):
    """单条请求在 Decode 阶段的分组查询注意力（GQA）计算。
    - query: 当前新 token 的 Q，形状 [num_q_heads, head_dim]
    - num_tokens: 参与注意力的总词数（历史词 + 当前词 = pos + 1），调用前当前词 K/V 须已写入缓存
    """
    if num_tokens <= 0:
        raise ValueError("参与 attention 的 token 数必须大于 0")

    # 从分页缓存聚集历史与当前 K/V: [num_tokens, num_kv_heads, head_dim]
    keys, values = cache.read(layer_id, request_id, num_tokens, block_manager)

    # 转为按头优先布局: [num_kv_heads, num_tokens, head_dim]
    keys = keys.transpose(0, 1)
    values = values.transpose(0, 1)

    num_q_heads, head_dim = query.shape
    num_kv_heads = keys.shape[0]

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Q 头数必须是 KV 头数的整数倍")

    # GQA 扩展：每组 repeats 个 Q 头共享同一组 KV 头 -> [num_q_heads, num_tokens, head_dim]
    repeats = num_q_heads // num_kv_heads
    keys = keys.repeat_interleave(repeats, dim=0)
    values = values.repeat_interleave(repeats, dim=0)

    q = query.unsqueeze(1)  # [num_q_heads, 1, head_dim]

    # 使用 FP32 计算缩放点积与 Softmax，避免 BF16 下累加精度损失
    # [num_q_heads, 1, head_dim] @ [num_q_heads, head_dim, num_tokens] -> [num_q_heads, 1, num_tokens]
    scores = torch.matmul(q.float(), keys.float().transpose(-2, -1)) / (head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)

    # [num_q_heads, 1, num_tokens] @ [num_q_heads, num_tokens, head_dim] -> [num_q_heads, head_dim]
    output = torch.matmul(weights, values.float())
    return output.squeeze(1).to(query.dtype)


@torch.inference_mode()
def project_qkv(attn, hidden_states):
    """
    - hidden_states: [B, seq_len, hidden_size]
    - 返回 query: [B, seq_len, num_q_heads, head_dim]
           key, value: [B, seq_len, num_kv_heads, head_dim]
    """
    batch_size, seq_len, _ = hidden_states.shape
    head_dim = attn.head_dim

    query = attn.q_proj(hidden_states).reshape(batch_size, seq_len, -1, head_dim)
    key = attn.k_proj(hidden_states).reshape(batch_size, seq_len, -1, head_dim)
    value = attn.v_proj(hidden_states).reshape(batch_size, seq_len, -1, head_dim)

    # Qwen3 架构在施加 RoPE 前需对每个 Q/K 头独立做 RMSNorm（V 无此操作）
    query = attn.q_norm(query)
    key = attn.k_norm(key)

    return query, key, value


def rotate_half(x):
    """RoPE 辅助变换：将最后一维切成前后两半并旋转为 [-x2, x1]。"""
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return torch.cat([-second, first], dim=-1)


@torch.inference_mode()
def apply_rope(query, key, cos, sin):
    """对 Q 和 K 施加旋转位置编码（RoPE），保持张量形状不变。"""
    # cos, sin: [B, seq_len, head_dim] -> [B, seq_len, 1, head_dim] 以广播至所有注意力头
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    query_rotated = query * cos + rotate_half(query) * sin
    key_rotated = key * cos + rotate_half(key) * sin
    return query_rotated, key_rotated


@torch.inference_mode()
def get_rope_cos_sin(model, hidden_states, positions):
    """按每条请求各自的序列位置下标计算对应的 RoPE 正余弦表。
    - positions: 长度为 B 的整数列表，允许同一 Batch 内各请求处于不同上下文长度
    - 返回 cos, sin: [B, 1, head_dim]
    """
    position_ids = torch.tensor(
        positions, dtype=torch.long, device=hidden_states.device
    ).unsqueeze(1)  # [B, 1]

    cos, sin = model.model.rotary_emb(hidden_states, position_ids)
    return cos, sin


@torch.inference_mode()
def decode_attention_layer(model, hidden_states, layer_id, request_ids, positions, cache, block_manager):
    """多请求 Batched Decode 注意力层：合并稠密权重矩阵乘，按页表分别寻址变长历史缓存。
    - hidden_states: [B, 1, hidden_size]
    - 返回 output: [B, 1, hidden_size]
    """
    batch_size = len(request_ids)
    attn = model.model.layers[layer_id].self_attn

    # 1. B 条请求合并为一次批量 GEMM 计算 QKV，并按各自位置施加 RoPE
    query, key, value = project_qkv(attn, hidden_states)
    cos, sin = get_rope_cos_sin(model, hidden_states, positions)
    query, key = apply_rope(query, key, cos, sin)

    # 2. 为 B 条请求的新 token 定位物理槽位，并一次性批量写入分页缓存
    block_ids = []
    offsets = []
    for request_id, pos in zip(request_ids, positions):
        block_manager.ensure_capacity(request_id, pos + 1)
        b_id, off = block_manager.get_slot(request_id, pos)
        block_ids.append(b_id)
        offsets.append(off)

    block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=hidden_states.device)
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=hidden_states.device)
    # 左右两侧形状均为 [B, num_kv_heads, head_dim]
    cache.key_cache[layer_id, block_ids_t, offsets_t] = key[:, 0]
    cache.value_cache[layer_id, block_ids_t, offsets_t] = value[:, 0]

    # 3. 各请求根据自身总词数 (pos + 1) 从页表读取缓存完成注意力，再拼回 Batch 统一过 o_proj
    outputs = []
    for row, (request_id, pos) in enumerate(zip(request_ids, positions)):
        out_i = decode_attention(
            query[row, 0],  # [num_q_heads, head_dim]
            cache,
            block_manager,
            layer_id,
            request_id,
            pos + 1,
        )
        outputs.append(out_i)

    output = torch.stack(outputs, dim=0).reshape(batch_size, 1, -1)  # [B, 1, num_q_heads * head_dim]
    output = attn.o_proj(output)                                     # [B, 1, hidden_size]
    return output

@torch.inference_mode()
def prefill_attention(query, key, value):
    """
    输入 query 是 [seq_len, Q头数, 头维度]
    key / value 是 [seq_len, KV头数, 头维度]
    """
    seq_len, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    repeats = num_q_heads // num_kv_heads

    # 转为按头优先: [头数, seq_len, head_dim]
    q = query.transpose(0, 1)
    k = key.transpose(0, 1).repeat_interleave(repeats, dim=0)
    v = value.transpose(0, 1).repeat_interleave(repeats, dim=0)

    scores = torch.matmul(q.float(), k.float.transpose(-2, -1)) / (head_dim ** 0.5)

    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, v.float()).to(query.dtype)

    # 转回 [seq_len, Q头数, 头维度]
    return output.transpose(0, 1)

@torch.inference_mode()
def prefill_attention_layer(model, hidden_states, layer_id, request_ids, seq_lens, cache, block_manager):
    attn = model.model.layers[layer_id].self_attn
    query, key, value = project_qkv(attn, hidden_states)

    position_ids = torch.cat([
        torch.arange(L, dtype = torch.long, device = hidden_states.device)
        for L in seq_lens
    ]).unsqueeze(0)
    # 把各个prefill的请求按照prompt的长度，摆成一行

    outputs = []
    cursor = 0
    for request_id, seq_len in zip(request_ids, seq_lens):
        # cursor是一个光标，用于裁剪每个请求自己的参数
        q_i = query[0, cursor : cursor + seq_len]
        k_i = key[0, cursor : cursor + seq_len]
        v_i = value[0, cursor : cursor + seq_len]

        block_manager.ensure_capacity(request_id, seq_len)
        block_ids, offsets = block_manager.get_slots(
            request_id, seq_len, device = hidden_states.device
        )

        cache.key_cache[layer_id, block_ids, offsets] = k_i
        cache.value_cache[layer_id, block_ids, offsets] = v_i

        outputs.append(prefill_attention(q_i, k_i, v_i))
        cursor += seq_len

    total_tokens = hidden_states.shape[1]
    output = torch.cat(outputs, dim=0).reshape(1, total_tokens, -1)
    return attn.o_proj(output)