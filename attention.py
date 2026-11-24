import torch

@torch.inference_mode()
def decode_attention(query, cache, block_manager,layer_id, request_id, num_tokens):
    # query：当前 token 的 Q，形状为 [Q头数, 头维度]
    # cache：你写的 PagedKVCache 对象
    # num_tokens：本次参与 attention 的 token 数，包含当前 token
    # 调用前，当前 token 的 K/V 也必须已经写入缓存
    
    if num_tokens <= 0:
        raise ValueError("参与attention的token数必须大于0")

    keys, values = cache.read(
        layer_id, request_id, num_tokens, block_manager
    )

    keys = keys.transpose(0, 1)
    values = values.transpose(0, 1)

    num_q_heads, head_dim = query.shape  # 取出 Q 的头数和头维度
    num_kv_heads = keys.shape[0]        # 取出 KV 的头数

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Q 头数必须是 KV 头数的整数倍")

    repeats = num_q_heads // num_kv_heads  # 每个 KV 头对应几个 Q 头

    # 为对应的 Q 头准备 K/V，形状变为 [Q头数, token数, 头维度]
    keys = keys.repeat_interleave(repeats, dim=0)
    values = values.repeat_interleave(repeats, dim=0)

    # Q 变为 [Q头数, 1, 头维度]，这里的 1 表示只有一个当前 token
    q = query.unsqueeze(1)

    # Q 与每个 token 的 K 做点积，得到关注分数
    # [Q头数, 1, 头维度] @ [Q头数, 头维度, token数]
    # 结果：[Q头数, 1, token数]
    scores = torch.matmul(q.float(), keys.float().transpose(-2, -1))

    scores = scores / (head_dim ** 0.5)

    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, values.float())

    return output.squeeze(1).to(query.dtype)

@torch.inference_mode()
def project_qkv(attn, hidden_states):
    # attn：Qwen 某一层的 attention 模块，里面有训练好的权重
    # hidden_states：进入该 attention 模块的特征
    # 形状：[请求数, 本轮输入token数, 模型隐藏维度]

    batch_size, seq_len, _ = hidden_states.shape
    head_dim = attn.head_dim

    query = attn.q_proj(hidden_states)
    # 使用 Q 的投影权重，计算 Q

    key = attn.k_proj(hidden_states)
    # 使用 K 的投影权重，计算 K

    value = attn.v_proj(hidden_states)
    # 使用 V 的投影权重，计算 V

    query = query.reshape(batch_size, seq_len, -1, head_dim)
    # 把 Q 拆成多个头：[请求数, token数, Q头数, 头维度]
    # -1 表示让 PyTorch 根据元素总数自动算出头数

    key = key.reshape(batch_size, seq_len, -1, head_dim)
    # K：[请求数, token数, KV头数, 头维度]

    value = value.reshape(batch_size, seq_len, -1, head_dim)
    # V：[请求数, token数, KV头数, 头维度]

    query = attn.q_norm(query)
    # Qwen3 会对每个 Q 头做归一化

    key = attn.k_norm(key)
    # Qwen3 也对每个 K 头做归一化，V 没有这一步

    return query, key, value
    # 返回计算好的 Q、K、V

def rotate_half(x):
    half = x.shape[-1] // 2

    first = x[..., :half]   # 前半部分，... 表示保留前面的所有维度
    second = x[..., half:] # 后半部分

    return torch.cat([-second, first], dim=-1)

@torch.inference_mode()
def apply_rope(query, key, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    query_rotated = query * cos + rotate_half(query) * sin


    key_rotated = key * cos + rotate_half(key) * sin
    # 对 K 加入位置信息，形状不变

    return query_rotated, key_rotated

@torch.inference_mode()
def get_rope_cos_sin(model, hidden_states, start_position):
    # start_position：本轮第一个 token 在整条序列中的位置，从 0 开始

    seq_len = hidden_states.shape[1]

    position_ids = torch.arange(
        start_position,
        start_position + seq_len,
        device=hidden_states.device,
    )

    position_ids = position_ids.unsqueeze(0)

    cos, sin = model.model.rotary_emb(hidden_states, position_ids)
    return cos, sin

@torch.inference_mode()
def decode_attention_layer(model, hidden_states, layer_id, request_id, position, cache, block_manager):
    # hidden_states 是这个函数的输入，但它是 token 的特征向量，不是原始文字或 token 编号
    attn = model.model.layers[layer_id].self_attn
    # attn 是 Qwen 某一层的 attention 模块对象，里面包含这一层计算 attention 所需的权重和配置


    query, key, value = project_qkv(attn, hidden_states)
    cos, sin = get_rope_cos_sin(model, hidden_states, position)

    query, key = apply_rope(query, key, cos, sin)

    num_tokens = position + 1

    block_manager.ensure_capacity(request_id, num_tokens)
    # 确保这个请求有足够的缓存块

    block_id, offset = block_manager.get_slot(request_id, position)
    # 找到当前 token 对应的物理块和块内位置

    cache.write(
        layer_id,
        block_id,
        offset,
        key[0, 0],    # [KV头数, 头维度]，已经做过 RoPE
        value[0, 0],  # [KV头数, 头维度]
    )

    output = decode_attention(
        query[0, 0],  # [Q头数, 头维度]
        cache,
        block_manager,
        layer_id,
        request_id,
        num_tokens,
    )    

    output = output.reshape(1, 1, -1)
    output = attn.o_proj(output)
    # output 是token在这一层经过atten后得到的张量,还不是最终的next_tensor

    return output
