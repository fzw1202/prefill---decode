import os  # 导入操作系统相关的库

GPUs = [0, 1]  # 定义可用的GPU列表
num_gpus = len(GPUs)  # 获取GPU数量
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUs))  # 设置可见的GPU设备

import torch  # 导入PyTorch库
import asyncio  # 导入异步IO库
import uvicorn  # 导入Uvicorn服务器库，用于运行FastAPI应用
from fastapi import FastAPI  # 从FastAPI库中导入FastAPI类
from pydantic import BaseModel  # 导入Pydantic的数据验证模型
from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入用于加载语言模型和分词器的模块

app = FastAPI()  # 创建FastAPI应用实例

model_name = "/root/autodl-fs/Meta-Llama-3.1-8B-Instruct"  # 定义模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 从指定路径加载分词器
if tokenizer.pad_token is None:  # 如果没有定义填充符
    tokenizer.pad_token = tokenizer.eos_token  # 将填充符设为结束符

if num_gpus >= 1:  # 如果GPU数量大于等于1
    device_map = "auto"  # 设定设备映射为自动
    model = AutoModelForCausalLM.from_pretrained(  # 从指定路径加载模型
        model_name,
        torch_dtype=torch.float16,  # 使用半精度浮点数
        device_map=device_map  # 使用自动映射的设备
    )
else:  # 如果没有GPU可用
    model = AutoModelForCausalLM.from_pretrained(  # 加载模型到CPU
        model_name,
        torch_dtype=torch.float16
    )
    model.to('cpu')  # 将模型移动到CPU上

model.eval()  # 将模型设置为评估模式

request_queue = []  # 创建请求队列
max_batch_size = 4  # 定义批处理的最大请求数量

class InferenceRequest(BaseModel):  # 定义推理请求数据模型
    prompt: str  # 请求中的提示词
    temperature: float = 1.0  # 生成文本时的随机性参数，默认为1.0
    max_length: int = 8192  # 生成文本的最大长度
    min_length: int = 10  # 生成文本的最小长度

@app.post('/generate')  # 定义POST接口用于生成文本
async def generate(request: InferenceRequest):
    response_queue = asyncio.Queue()  # 创建异步队列用于存放响应
    request_item = {'request': request, 'response': response_queue}  # 将请求和响应队列打包
    request_queue.append(request_item)  # 将请求添加到队列

    if len(request_queue) >= max_batch_size:  # 如果请求数量达到批处理上限
        await process_batch()  # 处理当前批次请求

    result = await response_queue.get()  # 等待获取响应结果
    return result  # 返回生成的文本

async def process_batch():  # 批处理函数
    global request_queue  # 使用全局请求队列
    if not request_queue:  # 如果请求队列为空，直接返回
        return

    current_batch = request_queue[:max_batch_size]  # 获取当前批次请求
    request_queue = request_queue[max_batch_size:]  # 更新请求队列

    # 提取批次中的各类请求参数
    prompts = [item['request'].prompt for item in current_batch]
    temperatures = [item['request'].temperature for item in current_batch]
    max_lengths = [item['request'].max_length for item in current_batch]
    min_lengths = [item['request'].min_length for item in current_batch]

    inputs = tokenizer(prompts, return_tensors='pt', padding=True)  # 对输入进行分词和填充

    with torch.no_grad():  # 在不计算梯度的上下文中进行推理
        outputs = model.generate(  # 使用模型生成文本
            **inputs,
            max_length=max(max_lengths),  # 设置最大长度
            min_length=min(min_lengths),  # 设置最小长度
            temperature=max(temperatures),  # 设置温度参数
            do_sample=True,  # 进行随机采样生成
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)  # 解码生成的文本

    # 将解码后的输出放入响应队列中
    for item, output in zip(current_batch, decoded_outputs):
        await item['response'].put({'generated_text': output})

@app.on_event("startup")  # 在应用启动时触发的事件
async def startup_event():
    asyncio.create_task(batch_scheduler())  # 启动批处理调度器

async def batch_scheduler():  # 批处理调度器
    while True:
        if request_queue:  # 如果请求队列不为空
            await process_batch()  # 处理批次请求
        await asyncio.sleep(3)  # 每3秒检查一次队列

if __name__ == "__main__":  # 主程序入口
    uvicorn.run(app, host="127.0.0.1", port=8000)  # 启动Uvicorn服务器，在指定地址和端口运行应用
