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


cnt = 0


@app.post("/generate")
async def generate(request: InferenceRequest):
    messages = [{"role": "user", "content": request.prompt}]
    request.prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    global cnt
    print(f"request {cnt}: min: {request.min_length}, max: {request.max_length}")
    cnt += 1

    response_queue = asyncio.Queue()
    request_item = {"request": request, "response": response_queue}
    request_queue.append(request_item)

    if len(request_queue) >= 2 * max_batch_size:
        await process_batch()

    result = await response_queue.get()
    return result


async def process_batch():
    global request_queue
    if not request_queue:
        return
    
    request_queue.sort(key=lambda item: item["request"].min_length)
    current_batch = request_queue[:max_batch_size]
    request_queue = request_queue[max_batch_size:]

    prompts = [item["request"].prompt for item in current_batch]
    max_lengths = [item["request"].max_length for item in current_batch]
    min_lengths = [item["request"].min_length for item in current_batch]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        done_flags = [False] * len(current_batch)
        outputs = model(**inputs, use_cache=True)
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

        for step in range(max(max_lengths)):
            if step % 100 == 0:
                print(f"step: {step}")

            if all(done_flags):
                break

            next_input_ids = inputs["input_ids"][:, -1:].contiguous()
            next_attention_mask = inputs["attention_mask"][:, -1:].contiguous()
            outputs = model.forward(
                input_ids=next_input_ids,
                attention_mask=next_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]

            new_tokens = []
            for i, logits in enumerate(next_token_logits):
                if not done_flags[i] and inputs["input_ids"].shape[1] < max_lengths[i]:
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                    new_tokens.append(next_token)
                else:
                    padding_token = torch.tensor(
                        [tokenizer.pad_token_id], dtype=torch.long, device=logits.device
                    )
                    new_tokens.append(padding_token)

            new_tokens = torch.stack(new_tokens).squeeze(1)
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], new_tokens.unsqueeze(1)], dim=-1
            )
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones(
                        (new_tokens.size(0), 1),
                        dtype=torch.long,
                        device=new_tokens.device,
                    ),
                ],
                dim=-1,
            )

            for i, seq in enumerate(inputs["input_ids"]):
                if not done_flags[i]:
                    if seq.shape[0] >= max_lengths[i]:
                        done_flags[i] = True
                    elif (
                        seq.shape[0] >= min_lengths[i]
                        and seq[-1].item() == tokenizer.eos_token_id
                    ):
                        done_flags[i] = True
            past_key_values = outputs.past_key_values

    print(f"step: {step}")

    decoded_outputs = tokenizer.batch_decode(
        inputs["input_ids"], skip_special_tokens=True
    )

    for item, output in zip(current_batch, decoded_outputs):
        await item["response"].put({"generated_text": output})


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_scheduler())


async def batch_scheduler():
    while True:
        if request_queue:
            await process_batch()
        await asyncio.sleep(10)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
