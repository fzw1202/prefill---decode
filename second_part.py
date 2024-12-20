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
kvcache = None
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

        global cnt
        print(
            f"request {cnt}: min: {request_item["request"].min_length}, max: {request_item["request"].max_length}"
        )

        decode_request_item = {
            "request": inputs.to(device_decode),
            "response": request_item["response"],
            "kv_cache": past_key_values,
            "done_flag": False,
            "max": request_item["request"].max_length,
            "min": request_item["request"].min_length,
            "id": cnt,
            "padding": 0,
        }
        request_queue.append(decode_request_item)
        cnt += 1


check_time = 0
generate_time = 0


async def check_and_update_batch() -> int:
    global kvcache, request_queue, model_decode, check_time, generate_time

    done_requests = [item for item in request_queue if item["done_flag"]]
    for item in done_requests:
        print(
            f"finish: {item['id']} check_time: {check_time} generate_time: {generate_time}"
        )
        decoded_output = tokenizer.decode(
            item["request"]["input_ids"][0], skip_special_tokens=True
        )
        item["response"].set_result({"generated_text": decoded_output})

    remaining_requests = [item for item in request_queue if not item["done_flag"]]
    request_queue = sorted(
        remaining_requests,
        key=lambda x: (
            x.get("min", float("inf"))
            - (x["request"]["input_ids"].shape[1] - x.get("padding", 0))
        ),
    )

    batch = request_queue[:max_batch_size]
    # print("Batch IDs:", " ".join(str(item["id"]) for item in batch))

    if len(batch) == 0:
        return 0

    length = max([item["request"]["input_ids"].shape[1] for item in batch])
    for item in batch:
        padding_length = length - item["request"]["input_ids"].shape[1]

        if padding_length > 0:
            item["padding"] += padding_length
            padding_input = torch.full(
                (1, padding_length),
                tokenizer.pad_token_id,
                dtype=item["request"]["input_ids"].dtype,
                device=device_decode,
            )
            padding_mask = torch.zeros(
                (item["request"]["attention_mask"].shape[0], padding_length),
                dtype=item["request"]["input_ids"].dtype,
                device=device_decode,
            )

            item["request"]["input_ids"] = torch.cat(
                (padding_input, item["request"]["input_ids"]), dim=1
            )
            item["request"]["attention_mask"] = torch.cat(
                (padding_mask, item["request"]["attention_mask"]), dim=1
            )

            kv = item["kv_cache"]
            padded_kv = []
            for layer in kv:
                padded_layer_cache = []
                for tensor in layer:
                    cache_padding = torch.zeros(
                        (
                            tensor.shape[0],
                            tensor.shape[1],
                            padding_length,
                            tensor.shape[3],
                        ),
                        dtype=tensor.dtype,
                        device=device_decode,
                    )
                    padded_tensor = torch.cat(
                        (cache_padding, tensor), dim=2
                    ).contiguous()
                    padded_layer_cache.append(padded_tensor)

                padded_kv.append(tuple(padded_layer_cache))
            item["kv_cache"] = tuple(padded_kv)

    kvcache = []
    for layer_idx in range(32):
        layer_keys = []
        layer_values = []

        for item in batch:
            key, value = item["kv_cache"][layer_idx]
            layer_keys.append(key.squeeze(0))
            layer_values.append(value.squeeze(0))

        layer_key = torch.stack(layer_keys, dim=0)
        layer_value = torch.stack(layer_values, dim=0)
        kvcache.append((layer_key, layer_value))

    kvcache = tuple(kvcache)
    return len(batch)


async def generate_one_step(bsz):
    global kvcache, request_queue, model_decode, check_time, generate_time

    with torch.no_grad():
        current_batch = request_queue[:bsz]
        input_ids = torch.stack(
            [item["request"]["input_ids"].squeeze(0) for item in current_batch], dim=0
        )
        attention_masks = torch.stack(
            [item["request"]["attention_mask"].squeeze(0) for item in current_batch],
            dim=0,
        )
        inputs = {"input_ids": input_ids, "attention_mask": attention_masks}
        next_input_id = inputs["input_ids"][:, -1:].contiguous()
        outputs = model_decode.forward(
            input_ids=next_input_id,
            attention_mask=attention_masks,
            past_key_values=kvcache,
            use_cache=True,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
        inputs["attention_mask"] = torch.cat(
            [
                inputs["attention_mask"],
                torch.ones(
                    (next_token.size(0), 1),
                    dtype=torch.long,
                    device=inputs["attention_mask"].device,
                ),
            ],
            dim=-1,
        )

        for index, item in enumerate(current_batch):
            item["request"]["input_ids"] = inputs["input_ids"][index].unsqueeze(0)
            item["request"]["attention_mask"] = inputs["attention_mask"][
                index
            ].unsqueeze(0)

        for item, seq in zip(current_batch, inputs["input_ids"]):
            if not item["done_flag"]:
                if seq.shape[0] - item["padding"] >= item["max"]:
                    item["done_flag"] = True
                elif (
                    seq.shape[0] - item["padding"] >= item["min"]
                    and seq[-1].item() == tokenizer.eos_token_id
                ):
                    item["done_flag"] = True

        for item in current_batch:
            item["kv_cache"] = []

        for layer_idx in range(32):
            layer_key, layer_value = outputs.past_key_values[layer_idx]

            for index, item in enumerate(current_batch):
                key = layer_key[index].unsqueeze(0)
                value = layer_value[index].unsqueeze(0)
                item["kv_cache"].append((key, value))

        for item in current_batch:
            item["kv_cache"] = tuple(item["kv_cache"])


async def decode_worker():
    global check_time, generate_time

    while True:
        if not request_queue:
            await asyncio.sleep(0.1)
            continue

        start_time = time.time()
        bsz = await check_and_update_batch()
        end_time = time.time()
        check_time += end_time - start_time

        if bsz == 0:
            await asyncio.sleep(0.1)
            continue

        start_time = time.time()
        await generate_one_step(bsz)
        end_time = time.time()
        generate_time += end_time - start_time

        await asyncio.sleep(0.00001)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(prefill_worker())
    asyncio.create_task(decode_worker())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
