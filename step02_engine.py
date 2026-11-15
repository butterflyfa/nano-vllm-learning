import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
from transformers import DynamicCache


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
        # active = []
        prefill_ids = []
        decode_ids = []

        for request_id in self.requests: # 遍历的是请求
            req = self.requests[request_id]
            # 如果加了item()能保证每次都能取到一对键和值,也就是对应第18行的结构
            # 不加item()的话，request_id就是一个编号
            if req["generated_tokens"] >= req["max_new_tokens"] or req.get("finished", False):
                continue # 跳过这次请求，去生成下一个请求的token

            # active.append(request_id) # 记录本轮需要生成token的请求编号
            # 用prefill和decode区分的更加清晰

            if req["cache"] is None:
                prefill_ids.append(request_id) # prefill阶段，第一次做计算，cache为空，输入为整个ids
            else:
                decode_ids.append(request_id) # decode阶段，后续做计算，cache不为空，输入为ids的最后一个token

# ----------------- prefill ------------------
        for request_id in prefill_ids:
            req = self.requests[request_id]
            ids = req["ids"]

            """
             if req["cache"] is None:
                input_ids = ids # prefill阶段，第一次做计算，cache为空，输入为整个ids
            else:
                input_ids = ids[:, -1:]  # decode阶段，后续做计算，cache不为空，输入为ids的最后一个token
            """

            input_ids = ids
            print("请求", request_id, "本轮输入长度", input_ids.shape[1])
            with torch.profiler.record_function("model_inference"):
                out = self.model(input_ids = input_ids, use_cache=True, past_key_values=req["cache"])
            req["cache"] = out.past_key_values # 每轮都把刚刚计算好的kv加入缓存中
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            req["ids"] = torch.cat([ids, next_id], dim = 1)
            req["generated_tokens"] += 1

            token_id = next_id.item()
            req["finished"] = token_id in stop_ids
                # 创建了finished键，表示是否生成了结束符
            update[request_id] = token_id


# ----------------- decode ------------------
        groups = {} # 键：缓存长度  值：request_id,请求id
        for request_id in decode_ids:
            cache = self.requests[request_id]["cache"]
            length = cache.get_seq_length()
            if length not in groups:
                groups[length] = [] # 第一次遇到这个长度时，为它新建一个列表
            groups[length].append(request_id)

        for group in groups.values():
            tokens = [] # 存每个请求最新的一个token，也就是每个请求新一轮的输入
            caches = []

            for request_id in group:
                req = self.requests[request_id]
                tokens.append(req["ids"][:,-1:]) # 只取最后一个token，要取出这个token作为下一轮的输入
                caches.append(req["cache"])

            input_ids = torch.cat(tokens, dim = 0) # 按行拼
            # input_ids的形状为：[请求数, 每个请求本轮输入的 token 数]，所以要按行来拼接

            batch_cache = DynamicCache.from_batch_splits(caches)
            # 按同样的行顺序合并KV缓存

            with torch.profiler.recode_function("model_inference"):
                out = self.model(
                    input_ids = input_ids, # 形状[请求数，1]，每行对应一个请求，每行输入一个token
                    use_cache = True, # 保留本次输入的KV，存到past_key_values中
                    past_key_values = batch_cache, 
                )

            next_ids = out.logits[:,-1,:].argmax(dim = -1, keepdim = True)
            new_caches = out.past_key_values.batch_split(len(group), 1)

            for row, request_id in enumerate(group): 
                # row是批次行号，request_id是请求编号
                # enumerate能返回两个值：元素位置和元素本身，位置默认从0开始
                req = self.requests[request_id]
                next_id = next_ids[row:row+1]
                # 意思是只取第row行，同时保留二维形状
                token_id = next_id.item()

                req["cache"] = new_caches[row]
                req["ids"] = torch.cat([req["ids"], next_id], dim = 1)

                req["generated_tokens"] += 1
                req["finished"] = token_id in stop_ids
                update[request_id] = token_id
                
        return update

def profile_decode(model, tokenizer):
    engine = Engine(model, tokenizer)
    engine.add_request("The capital of France is", max_new_tokens=16)
    engine.add_request("The capital of Germany is", max_new_tokens=16)

    engine.step() # 第一轮，读入提示词，保存缓存，生成第一个token

    activities = [ProfilerActivity.CPU] # 记录cpu上的操作
    if model.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA) # 同时记录GPU上的操作
        torch.cuda.synchronize(model.device) # 等第一轮GPU计算完成，避免后续的计算被第一轮的计算影响

    with profile(activities=activities, record_shapes=True) as prof:
        updates = engine.step()

    print("本轮生成结果：", updates)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
