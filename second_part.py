import os
import time

GPUs = [0, 1]
num_gpus = len(GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUs))

import torch
import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI()

model_name = "/root/autodl-fs/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device_prefill = torch.device("cuda:0")
device_decode = torch.device("cuda:1")

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
max_batch_size = 4
cnt = 0


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

    global cnt
    print(f"request {cnt}: min: {request.min_length}, max: {request.max_length}")
    cnt += 1

    response_future = asyncio.Future()
    request_item = {"request": request, "response": response_future}
    await prefill_queue.put(request_item)

    result = await response_future
    return result


async def prefill_worker():
    while True:
        request_item = await prefill_queue.get()
        inputs = tokenizer(
            request_item["request"].prompt, return_tensors="pt", padding=True
        ).to(device_prefill)

        with torch.no_grad():
            outputs = model_prefill(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones(
                        (next_token.size(0), 1),
                        dtype=torch.long,
                        device=inputs["input_ids"].device,
                    ),
                ],
                dim=-1,
            )

        past_key_values = tuple(
            (key.to(device_decode), value.to(device_decode))
            for key, value in past_key_values
        )

        decode_request_item = {
            "request": inputs.to(device_decode),
            "response": request_item["response"],
            "kv_cache": past_key_values,
            "done_flag": False,
            "max": request_item["request"].max_length,
            "min": request_item["request"].min_length,
            "new_flag": True,
        }
        request_queue.append(decode_request_item)


# 如果想要实现可以动态插入删除当前request的队列，就不能按batch来处理，而是要每一步decode分开做
# 可以在这个函数里实现单步的decode
async def generate_one_step():
    global request_queue, model_decode

    with torch.no_grad():
        for index, item in enumerate(request_queue):
            next_input_ids = item["request"]["input_ids"][:, -1:]
            next_attention_mask = item["request"]["attention_mask"][:, -1:]
            outputs = model_decode.forward(
                input_ids=next_input_ids,
                attention_mask=next_attention_mask,
                past_key_values=item["kv_cache"],
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            item["request"]["input_ids"] = torch.cat(
                [item["request"]["input_ids"], next_token], dim=-1
            )
            item["request"]["attention_mask"] = torch.cat(
                [
                    item["request"]["attention_mask"],
                    torch.ones(
                        (next_token.size(0), 1),
                        dtype=torch.long,
                        device=item["request"]["input_ids"].device,
                    ),
                ],
                dim=-1,
            )

            if item["request"]["input_ids"][0].shape[0] >= item["max"]:
                item["done_flag"] = True
            elif (
                item["request"]["input_ids"][0].shape[0] >= item["min"]
                and item["request"]["input_ids"][0][-1].item() == tokenizer.eos_token_id
            ):
                item["done_flag"] = True

            item["kv_cache"] = outputs.past_key_values


decode_cnt = 0


# 可以在这个函数里处理需要插入or删除request时对于kvcache，attention_mask的对齐修改等操作
async def check_and_update_batch():
    global request_queue, model_decode, decode_cnt

    for item in request_queue:
        if item["new_flag"]:
            item["new_flag"] = False

            item["id"] = decode_cnt
            print(f"decode: {decode_cnt}")
            decode_cnt += 1

    for index, item in enumerate(request_queue):
        if item["done_flag"]:
            print(f"finish: {item["id"]}")

            decoded_output = tokenizer.decode(
                item["request"]["input_ids"][0], skip_special_tokens=True
            )
            item["response"].set_result({"generated_text": decoded_output})
            request_queue.pop(index)


async def decode_worker():
    while True:
        if not request_queue:
            await asyncio.sleep(0.1)
            continue
        await check_and_update_batch()
        await generate_one_step()
        await asyncio.sleep(0.00001)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(prefill_worker())
    asyncio.create_task(decode_worker())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
