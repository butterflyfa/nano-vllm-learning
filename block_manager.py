
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size # 每块能存多少token
        self.free_blocks = list(range(num_blocks)) # 当前空闲的块编号,是个列表，存的是块的编号
        self.block_tables = {} # 键：请求编号 值：按顺序分到的块编号列表,做了paged attention

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
        