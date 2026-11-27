import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
from block_manager import BlockManager, PagedKVCache
from model_runner import decode_batch


def load_model(model_id="Qwen/Qwen3-0.6B"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, tokenizer


class Engine:
    """支持 Continuous Batching 与 PagedKVCache 的轻量级推理调度引擎。"""

    def __init__(self, model, tokenizer, max_running=2, num_blocks=64, block_size=16):
        if max_running < 1:
            raise ValueError("max_running 至少为 1")

        self.model = model
        self.tokenizer = tokenizer
        self.max_running = max_running

        self.requests = {}  # {request_id: 请求状态字典}
        self.next_request_id = 0
        self.waiting = []   # 等待调度的请求 ID 队列
        self.running = []   # 当前并发执行中的请求 ID 列表

        self.block_manager = BlockManager(num_blocks, block_size)
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

    def add_request(self, text, max_new_tokens):
        """编码提示词并注册新请求，加入 waiting 等待队列。"""
        request_id = self.next_request_id
        self.next_request_id += 1

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        self.requests[request_id] = {
            "ids": inputs.input_ids,          # [1, seq_len]，包含提示词与后续生成的 token
            "max_new_tokens": max_new_tokens,
            "generated_tokens": 0,            # 已生成的续写 token 数（用于停止条件判断）
            "cached_tokens": 0,               # 已写入 PagedKVCache 的 token 数（用于计算 position）
        }
        if max_new_tokens > 0:
            self.waiting.append(request_id)
        return request_id

    @torch.inference_mode()
    def step(self):
        """推进一轮调度与推理：准入 -> Prefill / Decode 分流执行 -> 完成回收。"""
        while self.waiting and len(self.running) < self.max_running:
            request_id = self.waiting.pop(0)
            self.running.append(request_id)

        update = {}
        eos = self.model.generation_config.eos_token_id
        stop_ids = {eos} if isinstance(eos, int) else set(eos or [])

        prefill_ids = []
        decode_ids = []
        for request_id in self.running:
            req = self.requests[request_id]
            if req["generated_tokens"] >= req["max_new_tokens"] or req.get("finished", False):
                continue

            if req["cached_tokens"] == 0:
                prefill_ids.append(request_id)
            else:
                decode_ids.append(request_id)

        # ----------------- 1. Prefill 阶段 ------------------
        prefill_groups = {}
        for request_id in prefill_ids:
            length = self.requests[request_id]["ids"].shape[1]
            prefill_groups.setdefault(length, []).append(request_id)

        for group in prefill_groups.values():
            prompts = [self.requests[rid]["ids"] for rid in group]
            input_ids = torch.cat(prompts, dim=0)  # [group_size, prompt_len]

            with torch.profiler.record_function("model_prefill"):
                out = self.model(input_ids=input_ids, use_cache=True)

            next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [group_size, 1]
            new_caches = out.past_key_values.batch_split(len(group), 1)

            for row, request_id in enumerate(group):
                req = self.requests[request_id]
                # 在追加新预测词前记录提示词长度：此时仅提示词的 K/V 已完成计算并写入分页缓存
                prompt_len = req["ids"].shape[1]
                self.paged_cache.write_from_cache(request_id, new_caches[row], self.block_manager)
                req["cached_tokens"] = prompt_len

                next_id = next_ids[row : row + 1]
                token_id = next_id.item()
                req["ids"] = torch.cat([req["ids"], next_id], dim=1)
                req["generated_tokens"] += 1
                req["finished"] = token_id in stop_ids
                update[request_id] = token_id

        # ----------------- 2. Decode 阶段 ------------------
        if decode_ids:
            tokens = []
            positions = []
            for request_id in decode_ids:
                req = self.requests[request_id]
                tokens.append(req["ids"][:, -1:])       # 取最新待编码的 1 个 token，保持 [1, 1]
                positions.append(req["cached_tokens"])  # 当前 token 在整条序列中的下标位置

            input_ids = torch.cat(tokens, dim=0)        # [B, 1]，不同长度的请求合并为一个 Batch

            with torch.profiler.record_function("model_decode"):
                logits = decode_batch(
                    model=self.model,
                    input_ids=input_ids,
                    request_ids=decode_ids,
                    positions=positions,
                    cache=self.paged_cache,
                    block_manager=self.block_manager,
                )
            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

            for row, request_id in enumerate(decode_ids):
                req = self.requests[request_id]
                next_id = next_ids[row : row + 1]
                token_id = next_id.item()

                req["ids"] = torch.cat([req["ids"], next_id], dim=1)
                req["cached_tokens"] += 1
                req["generated_tokens"] += 1
                req["finished"] = token_id in stop_ids
                update[request_id] = token_id

        # ----------------- 3. Postprocess 回收阶段 ------------------
        still_running = []
        for request_id in self.running:
            req = self.requests[request_id]
            reached_limit = req["generated_tokens"] >= req["max_new_tokens"]
            if not reached_limit and not req.get("finished", False):
                still_running.append(request_id)
            else:
                self.block_manager.release(request_id)

        self.running = still_running
        return update


if __name__ == "__main__":
    model, tokenizer = load_model()
    engine = Engine(model, tokenizer)
    a = engine.add_request("The capital of France is", max_new_tokens=3)
    b = engine.add_request("In Europe, the capital city of Germany is", max_new_tokens=4)

    def show_input(module, args, kwargs):
        print("  Prefill 模型输入形状：", tuple(kwargs["input_ids"].shape))

    hook = model.register_forward_pre_hook(show_input, with_kwargs=True)

    for step_num in range(1, 6):
        print(f"第 {step_num} 轮")
        updates = engine.step()
        print("  本轮输出：", updates)

    hook.remove()
    print("请求 a 生成结果：", tokenizer.decode(engine.requests[a]["ids"][0], skip_special_tokens=True))
    print("请求 b 生成结果：", tokenizer.decode(engine.requests[b]["ids"][0], skip_special_tokens=True))
    print("结束后空闲物理块数：", len(engine.block_manager.free_blocks))
