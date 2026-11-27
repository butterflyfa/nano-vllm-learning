from turtle import position

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
from block_manager import BlockManager, PagedKVCache
import block_manager
from model_runner import decode_batch


model_id = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = dtype).to(device).eval()
# 加载模型权重，移到设备上，切换到评估模式

class Engine:
    def __init__(self, model, tokenizer, max_running = 2,
                 num_blocks = 64, block_size = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.requests = {} # 按编号保存请求
        # requests是一个字典，key是请求编号，value是请求的内容（一个字典）
        self.next_request_id = 0
        if max_running < 1:
            raise ValueError("最大值至少为1")
        self.max_running = max_running
        self.waiting = [] # 已提交，等空位的请求编号,不管是decode还是prefill
        self.running = [] # 已获准运行的请求编号

        self.block_manager = BlockManager(num_blocks, block_size)
        # 记录具体请求用哪些块
        first_attn = model.model.layers[0].self_attn

        self.paged_cache = PagedKVCache(
            num_layers=len(model.model.layers),
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=first_attn.head_dim,
            device=model.device,
            dtype=first_attn.k_proj.weight.dtype,
            )
        # 记录块中实际存放的KV数值


    def add_request(self, text, max_new_tokens): # 作用是把每个请求存起来
        request_id = self.next_request_id
        self.next_request_id += 1

        inputs = self.tokenizer(text, return_tensors = "pt").to(self.model.device)
        # 将输入文本编码为token ID张量，并移到模型所在的设备上
        self.requests[request_id] = {
            "ids" : inputs.input_ids,
            # inputs是一个字典，里面有input_ids、attention_mask等键值对
            "max_new_tokens" : max_new_tokens,
            "generated_tokens" : 0,
            "cached_tokens" : 0,
        }
        if max_new_tokens > 0:
            self.waiting.append(request_id)
        return request_id

    @torch.inference_mode() # 关闭梯度记录，节省内存
    def step(self): # 用于生成每个请求的下一个token
        while self.waiting and len(self.running) < self.max_running:
            request_id = self.waiting.pop(0) # 取出并删除列表中位置为0的元素
            self.running.append(request_id)
        update = {}
        eos = self.model.generation_config.eos_token_id
        stop_ids = {eos} if isinstance(eos, int) else set(eos or [])
        # active = []
        prefill_ids = []
        decode_ids = []

        for request_id in self.running: # 遍历的是请求
            req = self.requests[request_id]
            # 如果加了item()能保证每次都能取到一对键和值,也就是对应第18行的结构
            # 不加item()的话，request_id就是一个编号
            if req["generated_tokens"] >= req["max_new_tokens"] or req.get("finished", False):
                continue # 跳过这次请求，去生成下一个请求的token

            # active.append(request_id) # 记录本轮需要生成token的请求编号
            # 用prefill和decode区分的更加清晰

            if req["cached_tokens"] == 0 :
                prefill_ids.append(request_id) # prefill阶段，第一次做计算，cache为空，输入为整个ids
            else:
                decode_ids.append(request_id) # decode阶段，后续做计算，cache不为空，输入为ids的最后一个token

# ----------------- prefill ------------------
        prefill_groups = {}

        for request_id in prefill_ids:
            req = self.requests[request_id]
            length = req["ids"].shape[1]
            if length not in prefill_groups:
                prefill_groups[length] = []
            prefill_groups[length].append(request_id)

        for group in prefill_groups.values():
            prompts = []

            for request_id in group:
                ids = self.requests[request_id]["ids"]
                prompts.append(ids)
            input_ids = torch.cat(prompts,dim = 0)
            # 把所有请求都拼成一个input
            with torch.profiler.record_function("model_inference"):
                out = self.model(
                    input_ids = input_ids,
                    use_cache = True
                )

            next_ids = out.logits[:,-1,:].argmax(dim = -1, keepdim = True)
            new_caches = out.past_key_values.batch_split(len(group), 1)

            for row, request_id in enumerate(group):
                req = self.requests[request_id]
                prompt_len = req["ids"].shape[1]
                self.paged_cache.write_from_cache(request_id, new_caches[row], self.block_manager)
                req["cached_tokens"] = prompt_len

                next_id = next_ids[row:row+1]
                token_id = next_id.item()
                
                req["ids"] = torch.cat([req["ids"], next_id], dim = 1) # 要形状相同的向量进行拼接
                req["generated_tokens"] += 1
                req["finished"] = token_id in stop_ids
                update[request_id] = token_id


# ----------------- decode ------------------
        if decode_ids:  # 注意一定要加 if 判断，因为第 1 轮 decode_ids 是空列表！
            tokens = []
            positions = []
            for request_id in decode_ids:
                req = self.requests[request_id]
                tokens.append(req["ids"][:, -1:]) # 只取最后一个还没算过kv的token
                positions.append(["cached_tokens"])

            input_ids = torch.cat(tokens, dim = 0)

            with torch.profiler.record_function("model_inference"):
                logits = decode_batch(
                    model = self.model,
                    input_ids=input_ids,
                    request_ids=request_ids,
                    positions=positions,
                    cache = self.paged_cache,
                    block_manager=self.block_manager,
                )
            next_ids = logits[:,-1,:].argmax(dim=-1, keepdim = True)

            for row, request_id in enumerate(decode_ids):
                req = self.requests[request_id]
                next_id = next_ids[row : row + 1]
                token_id = next_id.item()
                req["ids"] = torch.cat([req["ids"], next_id], dim=1)
                req["cached_tokens"] += 1
                req["generated_tokens"] += 1
                req["finished"] = token_id in stop_ids
                update[request_id] = token_id


# ----------------- judege ------------------
        still_running = []
        for request_id in self.running:
            req = self.requests[request_id]
            reached_limit = req["generated_tokens"] >= req["max_new_tokens"] # 判断是否要把这个请求移除
            if not reached_limit and not req.get("finished",False):
                still_running.append(request_id)
            else:
                self.block_manager.release(request_id)

        self.running = still_running

        return update

if __name__ == "__main__":
    engine = Engine(model, tokenizer)
    a = engine.add_request("The capital of France is", max_new_tokens=2)
    b = engine.add_request("The capital of Germany is", max_new_tokens=3)

    # 模型每次运行前，打印这次输入的形状：[请求数, 每条请求输入的 token 数]
    def show_input(module, args, kwargs):
        print("  模型输入形状：", tuple(kwargs["input_ids"].shape))

    hook = model.register_forward_pre_hook(show_input, with_kwargs=True)

    for step_num in range(1, 5):
        print(f"第 {step_num} 轮")
        updates = engine.step()
        print("  本轮输出：", updates)