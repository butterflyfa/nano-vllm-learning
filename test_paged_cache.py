from ast import Return

import torch
from engine import model, tokenizer
from block_manager import BlockManager, PagedKVCache

with torch.inference_mode():
    inputs = tokenizer(
        "The captial of France is",
        return_tensors = "pt"
    ).to(model.device)

    out = model(**inputs, use_cache=True) # **inputs的意思是把inputs里的键和值展开，分别作为参数传给模型
    source = out.past_key_values
    # [请求，kv头，历史kv，头内的维度]

    first_key = source.key_cache[0]  # 第 0 层的 K
    _, num_kv_heads, num_tokens, head_dim = first_key.shape
    num_layers = len(source.key_cache)


    block_size = 4
    num_blocks = (num_tokens + block_size - 1) // block_size


    bm = BlockManager(num_blocks, block_size)
    cache = PagedKVCache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=first_key.device,
        dtype=first_key.dtype,
    )

    cache.write_from_cache(7, source, bm)
    print("请求 7 的块表：", bm.block_tables[7])

    for layer_id in range(num_layers):
        keys, values = cache.read(layer_id, 7, num_tokens, bm)
        expected_keys = source.key_cache[layer_id][0].transpose(0, 1)
        expected_values = source.value_cache[layer_id][0].transpose(0, 1)
        assert torch.equal(keys, expected_keys), f"第 {layer_id} 层 K 不一致"
        assert torch.equal(values, expected_values), f"第 {layer_id} 层 V 不一致"

    print(f"通过：全部 {num_layers} 层的 K/V 写入、读回后完全一致")