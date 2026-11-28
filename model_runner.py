import torch
from attention import decode_attention_layer, prefill_attention_layer


@torch.inference_mode()
def decode_layer(model, hidden_states, layer_id, request_ids, positions, cache, block_manager):
    """单层 Pre-Norm Transformer Decoder 前向传播
    结构 x = x + SelfAttn(RMSNorm(x)) -> x = x + SwiGLU_MLP(RMSNorm(x))
    - hidden_states: [B, 1, hidden_size]
    """
    layer = model.model.layers[layer_id]

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = decode_attention_layer(
        model=model,
        hidden_states=hidden_states,
        layer_id=layer_id,
        request_ids=request_ids,
        positions=positions,
        cache=cache,
        block_manager=block_manager,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


@torch.inference_mode()
def decode_batch(model, input_ids, request_ids, positions, cache, block_manager):
    """多请求批量解码一步：共享一次模型权重访存，同时推进 B 条不同长度的序列。
    - input_ids: [B, 1]（每条请求本轮待编码的最新 1 个 token）
    - 返回 logits: [B, 1, vocab_size]
    """
    hidden_states = model.model.embed_tokens(input_ids)  # [B, 1, hidden_size]
    num_layers = len(model.model.layers)

    for layer_id in range(num_layers):
        hidden_states = decode_layer(
            model=model,
            hidden_states=hidden_states,
            layer_id=layer_id,
            request_ids=request_ids,
            positions=positions,
            cache=cache,
            block_manager=block_manager,
        )

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)  # [B, 1, vocab_size]
    return logits



@torch.inference_mode()
def prefill_layer(model, hidden_states, layer_id, request_ids, seq_lens, cache, block_manager):
    layer = model.model.layers[layer_id]

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = decode_attention_layer(
        model=model,
        hidden_states=hidden_states,
        layer_id=layer_id,
        request_ids=request_ids,
        seq_lens=seq_lens,
        cache=cache,
        block_manager=block_manager,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states

@torch.inference_mode()
def prefill_batch(model, input_ids, request_ids, seq_lens, cache, block_manager):
    hidden_states = model.model.embed_tokens(input_ids)
    for layer_id in range(model.model.layers):
        hidden_states = prefill_layer(
            model=model,
            hidden_states=hidden_states,
            layer_id=layer_id,
            request_ids=request_ids,
            seq_lens=seq_lens,
            cache=cache,
            block_manager=block_manager,
        )

    hidden_states = model.model.norm(hidden_states)

    last_indices = []
    cursor = 0
    for seq_len in seq_lens:
        cursor += seq_len
        last_indices.append(cursor - 1)

    last_hidden = hidden_states[0, last_indices]
    logits = model.lm_head(last_hidden).unsqueeze(1)
    return logits
