import torch
from attention import decode_attention_layer

@torch.inference_mode()
def decode_layer(model, hidden_states, layer_id, request_ids, positions, cache, block_manager):

    layer = model.model.layers[layer_id]
    residual = hidden_states

    hidden_states = layer.input_layernorm(hidden_states) # attention之前做一次norm

    hidden_states = decode_attention_layer(
        model=model,
        hidden_states=hidden_states,
        layer_id=layer_id,
        request_id=request_ids,
        position=positions,
        cache=cache,
        block_manager=block_manager,
    )

    hidden_states = residual + hidden_states
    residual = hidden_states

    hidden_states = layer.post_attention_layernorm(hidden_states)
    # 做ffn之前先做一次norm

    hidden_states = layer.mlp(hidden_states)
    # 做前馈网络，即mlp

    hidden_states = residual + hidden_states

    return hidden_states

@torch.inference_mode()
def decode_one_token(model, input_ids, request_ids, positions, cache, block_manager):

    hidden_states = model.model.embed_tokens(input_ids)
    num_layers = len(model.model.layers)


    for layer_id in range(num_layers):

        hidden_states = decode_layer(
            model=model,
            hidden_states=hidden_states,
            layer_id=layer_id,
            request_id=request_ids,
            position=positions,
            cache=cache,
            block_manager=block_manager,
        )

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    # 改变logits的形状，使得让hidden_states的维数能够match上词表中的token数
    # logits：[1, 1, 词表中的token数]

    return logits

