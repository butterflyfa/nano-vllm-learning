import torch


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

        blocks = self.block_tables[request_id] # paged attention，逻辑块和物理块之间的映射

        logical_block = position // self.block_size
        offset = position % self.block_size

        if logical_block >= len(blocks): # len是块的数量，下标从0开始
            raise ValueError("这个token的位置还没有分配缓存块")

        block_id = blocks[logical_block]
        return block_id, offset


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
        if num_tokens < 0:
            raise ValueError("读取的 token 数不能小于 0")
        
        num_kv_heads, head_dim = self.key_cache.shape[-2:] # 表示从倒数第二项开始，一直取到末尾

        shape = (num_tokens, num_kv_heads, head_dim)
        keys = self.key_cache.new_empty(shape) # 使用与缓存相同的设备和数据类型
        values = self.value_cache.new_empty(shape)

        for position in range(num_tokens):
            block_id, offset = block_manager.get_slot(request_id, position)

            keys[position].copy_(self.key_cache[layer_id, block_id, offset])
            values[position].copy_(self.value_cache[layer_id, block_id, offset])

        return keys, values
                

        



