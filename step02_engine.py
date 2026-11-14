import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
# 加载模型权重，移到设备上，切换到评估模式

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.requests = {} # 按编号保存请求
        # requests是一个字典，key是请求编号，value是请求的内容（一个字典）
        self.next_request_id = 0

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
            "cache" : None,
        }
        return request_id

    @torch.inference_mode() # 关闭梯度记录，节省内存
    def step(self): # 用于生成每个请求的下一个token
        update = {}
        eos = self.model.generation_config.eos_token_id
        stop_ids = {eos} if isinstance(eos, int) else set(eos or [])

        for request_id, req in self.requests.items():
            # item()能保证每次都能取到一对键和值,也就是对应第18行的结构
            if req["generated_tokens"] >= req["max_new_tokens"] or req.get("finished", False):
                continue # 跳过这次请求，去生成下一个请求的token

            ids = req["ids"]
            if req["cache"] is not None:
                input_ids = ids
            else:
                input_ids = ids[:, -1:] # 只取最后一个token ID
            print("请求", request_id, "本轮输入长度", input_ids.shape[1])

            out = self.model(input_ids = input_ids, use_cache=True, past_key_values=req["cache"])
            req["cache"] = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            req["ids"] = torch.cat([next_id, ids], dim = 1);
            req["generated_tokens"] += 1

            token_id = next_id.item()
            req["finished"] = token_id in stop_ids
            # 创建了finished键，表示是否生成了结束符
            update[request_id] = token_id
        return update
    
engine = Engine(model, tokenizer)
a = engine.add_request("The capital of France is", max_new_tokens=1)
b = engine.add_request("Count from one to ten:", max_new_tokens=2)

for i in range(3):
    updates = engine.step()          # 推进一轮
    print("第", i + 1, "轮：", updates)

print("A 生成了", engine.requests[a]["generated_tokens"], "个 token")
print("B 生成了", engine.requests[b]["generated_tokens"], "个 token")