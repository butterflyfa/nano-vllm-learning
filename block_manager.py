from click import launch
import torch
from transformers import DynamicCache


class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size # 每块能存多少token
        self.free_blocks = list(range(num_blocks)) # 当前空闲的块编号,是个列表，存的是块的编号
        self.block_tables = {} # 键：请求编号 值：按顺序分到的块编号列表

    def ensure_capacity(self, request_id, num_tokens):
        needed = (num_tokens + self.block_size -1) // self.block_size # 向上取整

        blocks = self.block_tables.get(request_id, [])
        # 找到这个请求已经分配了的块

        missing = needed - len(blocks)
        # 算还缺了几块
        if missing <= 0:
            return 

        if missing > len(self.free_blocks):
            raise RuntimeError("空间KV缓存不足")

        for _ in range(missing):
            block_id = self.free_blocks.pop()
            blocks.append(block_id)

        self.block_tables[request_id] = blocks # 记下这个请求现在拥有的块

    def release(self, request_id):
        blocks = self.block_tables.pop(request_id, [])

        self.free_blocks.extend(blocks)

    def get_slot(self, request_id, position):
        if position < 0:
            raise ValueError("Token的位置不能小于0")

        blocks = self.block_tables[request_id] 

        logical_block = position // self.block_size
        offset = position % self.block_size

        if logical_block >= len(blocks): # len是块的数量，下标从0开始
            raise ValueError("这个token的位置还没有分配缓存块")

        block_id = blocks[logical_block]
        return block_id, offset

    def get_slots(self, request_id, num_tokens, device): # 减少每次查找所有词的计算
        if num_tokens < 0:
            raise ValueError("Token 的数量不能小于 0")
        if num_tokens == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        blocks = self.block_tables.get(request_id, [])
        needed = (num_tokens + self.block_size - 1) // self.block_size
        if needed > len(blocks):
            raise ValueError("这个请求还没有分配足够的缓存块")

        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        block_table = torch.tensor(blocks, dtype=torch.long, device=device)

        block_ids = block_table[positions // self.block_size]
        offsets = positions % self.block_size
        return block_ids, offsets    
    
class PagedKVCache:
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype=torch.float32):

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)

        self.key_cache = torch.empty(shape, device=device, dtype=dtype)
        self.value_cache = torch.empty(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def write(self, layer_id, block_id, offset, key, value): # 把模型计算出来的KV放到缓存中去
        key_slot = self.key_cache[layer_id, block_id, offset] # 找到这个token的k存放在什么地方
        value_slot = self.value_cache[layer_id, block_id, offset]

        if key.shape != key_slot.shape or value.shape != value_slot.shape:
            raise ValueError("传入的 K/V 形状与缓存位置不一致")

        key_slot.copy_(key)
        value_slot.copy_(value)

    def read(self, layer_id, request_id, num_tokens, block_manager):
        block_ids, offsets = block_manager.get_slots(request_id, num_tokens, device=self.key_cache.device)

        keys = self.key_cache[layer_id, block_ids, offsets]
        values = self.value_cache[layer_id, block_ids, offsets]

        return keys, values
                
    @torch.no_grad()
    def write_from_cache(self, request_id, source_cache, block_manager):
        num_tokens = source_cache.get_seq_length()

        block_manager.ensure_capacity(request_id, num_tokens)

        block_ids, offsets = block_manager.get_slots(
            request_id, num_tokens, device=self.key_cache.device
        )

        for layer_id in range(len(source_cache.key_cache)):
            keys = source_cache.key_cache[layer_id]
            # source_cache.key_cache  列表，每项是一层的 K 张量
            values = source_cache.value_cache[layer_id]

            if keys.shape[0] != 1:
                raise ValueError("这个方法每次接收一条请求的缓存")

            self.key_cache[layer_id, block_ids, offsets] = keys[0].transpose(0, 1)
            self.value_cache[layer_id, block_ids, offsets] = values[0].transpose(0, 1)

    @torch.no_grad()
    def to_dynamic_cache(self, request_id, num_tokens, block_manager):
        result = DynamicCache()
        num_layers = self.key_cache.shape[0]

        for layer_id in range(num_layers):
            keys, values = self.read(
                layer_id, request_id, num_tokens, block_manager
            )

            keys = keys.transpose(0,1).unsqueeze(0)
            values = values.transpose(0, 1).unsqueeze(0)
            # [请求数, KV头数, token数, 头维度]

            result.update(keys, values, layer_id)
        return result
