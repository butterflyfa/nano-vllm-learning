import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
# 加载模型权重，移到设备上，切换到评估模式


prompt = tokenizer("The capital of France is", return_tensors="pt").to(device)
# 每个token经过tokenizer会转换成token ID，后将输入文本编码返回为张量，并移到设备上
ids = prompt.input_ids
# 取出token ID张量，作为模型的输入
eos = model.generation_config.eos_token_id
stop_ids = {eos} if isinstance(eos, int) else set(eos)
# 统一转换为集合类型，方便后续判断是否生成了结束符
# 里面存的均为数值

with torch.inference_mode():
    for _ in range(16):
        logits = model(input_ids=ids,use_cache=False).logits
        # 传入token ID张量，得到logits张量
        # 传入ids后，model会自动加载embedding层，将token ID转换为向量
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # logits张量的形状为[哪条输入, 输入中的哪个位置, 词表中的哪个token]
        # logits[:, -1, :]，-1指的是取出最后一个token的logits，形状为[哪条输入, 词表中的哪个token]，说的是最后一个这个输入最后一个token的logits
        # dim = -1的意思是沿着行方向去找
        # 取出最后一个token的logits，计算最大值的索引，即为下一个token的ID
        print(f"输入长度: {ids.shape[1]}, 下一个token ID: {next_token.item()}")
        ids = torch.cat([ids,next_token], dim = 1)
        # 将下一个token ID拼接到输入的token ID张量中
        # dim = 1的意思是在列的方向上拼接
        if next_token.item() in stop_ids: # item()是将张量转换为python的数值类型
            break

    reference = model.generate(
        **prompt,max_new_tokens=16, do_sample=False, eos_token_id=eos
    )
assert torch.equal(ids, reference)
print(tokenizer.decode(ids[0], skip_special_tokens=True))  # 把第一条完整序列转回文字并打印