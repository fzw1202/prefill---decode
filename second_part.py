import os
GPUs = [0,1]
num_gpus = len(GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUs))

import torch
import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本文件给出一个可能的大致框架，同学们可以基于其实现，也可以基于自己的想法修改

app = FastAPI()

model_name = "/root/autodl-fs/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device_prefill = torch.device('cuda:0')
device_decode = torch.device('cuda:1')

model_prefill = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device_prefill)
model_prefill.eval()

model_decode = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device_decode)
model_decode.eval()

prefill_queue = asyncio.Queue()
request_queue = []
kv_caches = None
max_batch_size = 4

class InferenceRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    max_length: int = 8192
    min_length: int = 10

@app.post('/generate')
async def generate(request: InferenceRequest):
    messages = [
        {'role': 'user', 'content': request.prompt}
    ]
    request.prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response_future = asyncio.Future()
    request_item = {'request': request, 'response': response_future}
    await prefill_queue.put(request_item)

    result = await response_future
    return result

async def prefill_worker():
    # 可以在这个函数里在prefill设备上运行模型，获取kv_caches
    # 然后将kv_caches移动到decode设备，并整理decode阶段需要的信息进行转发
    request_item = await prefill_queue.get()
    decode_request_item = {
            'request': request,
            'kv_cache': kv_cache,
            ...
        }
    request_queue.append(decode_request_item)

async def generate_one_step():
    global kv_caches, request_queue, model_decode
    if kv_caches==None:
        return
    # 如果想要实现可以动态插入删除当前request的队列，就不能按batch来处理，而是要每一步decode分开做
    # 可以在这个函数里实现单步的decode


async def check_and_update_batch():
    global kv_caches, request_queue, model_decode
    # 可以在这个函数里处理需要插入or删除request时对于kvcache，attention_mask的对齐修改等操作
    
async def decode_worker():
    while True:
        if not request_queue:
            await asyncio.sleep(0.1)
            continue
        # 这里的逻辑也可以基于同学们的想法进行修改
        await check_and_update_batch()
        await generate_one_step()
        await asyncio.sleep(0.00001)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(prefill_worker())
    asyncio.create_task(decode_worker())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
