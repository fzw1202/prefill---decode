import os

GPUs = [0, 1]
num_gpus = len(GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUs))

import torch
import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入必要的库

app = FastAPI()

model_name = "/root/autodl-fs/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if num_gpus >= 1:
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device_map
    )
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to("cpu")

model.eval()

request_queue = []
max_batch_size = 4


class InferenceRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    max_length: int = 8192
    min_length: int = 10


@app.post("/generate")
async def generate(request: InferenceRequest):
    messages = [{"role": "user", "content": request.prompt}]
    request.prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response_queue = asyncio.Queue()
    request_item = {"request": request, "response": response_queue}
    request_queue.append(request_item)

    if len(request_queue) >= max_batch_size:
        await process_batch()

    result = await response_queue.get()
    return result


async def process_batch():
    global request_queue
    if not request_queue:
        return
    # 你需要在这里填充代码，使得你的推理系统可以正确响应每个不同请求的参数要求。
    # hint 可以考虑直接调用model(即model.forward)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_scheduler())


async def batch_scheduler():
    while True:
        if request_queue:
            await process_batch()
        await asyncio.sleep(3)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
