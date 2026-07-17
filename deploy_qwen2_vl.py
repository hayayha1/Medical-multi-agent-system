import os
import torch
import base64
import uvicorn
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 初始化 FastAPI 应用
app = FastAPI(
    title="Qwen2-VL-7B-Instruct Local API Server",
    description="本地多模态医学影像描述服务",
    version="1.0"
)

# 全局变量，用于存储加载的模型和处理器
model = None
processor = None

# 默认的模型 ID 或本地路径
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")


class PredictRequest(BaseModel):
    image_path: str = Field(None, description="医学影像的本地绝对路径（如 D:/images/lung.png）")
    image_base64: str = Field(None, description="图像的 Base64 编码字符串（如果不想通过路径读取）")
    prompt: str = Field("请详细描述该医学影像中的视觉发现，包括解剖部位、异常密度影、占位病变及其特征，暂不做出最终临床诊断。", description="指导模型进行影像分析的 Prompt")
    max_new_tokens: int = Field(512, description="生成的最大 Token 数量")
    temperature: float = Field(0.1, description="采样温度，医疗影像建议设低（0.1）以确保描述稳定客观")


class PredictResponse(BaseModel):
    findings: str = Field(..., description="模型生成的影像视觉发现描述")


@app.on_event("startup")
def load_model():
    """服务器启动时加载 Qwen2-VL 模型，针对单张显卡优化"""
    global model, processor
    print(f"正在从 {MODEL_PATH} 加载 Qwen2-VL 模型...")
    
    # 自动检测本地 GPU，若无 GPU 则使用 CPU (极慢，仅做备用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到的运行设备: {device}")
    
    # 针对 Windows 单张显卡（如 RTX 3090/4090）进行半精度加载优化
    # 如果显存较小（如 8G/12G），可将 torch_dtype 改为 torch.float16，并开启 load_in_4bit=True（需安装 bitsandbytes）
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("模型与处理器加载成功！")
    except Exception as e:
        print(f"模型加载失败，请检查路径或显存。错误信息: {e}")
        raise e


def process_image(request: PredictRequest) -> Image.Image:
    """根据请求，从本地路径或 Base64 加载图像"""
    if request.image_path:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail=f"本地图像路径不存在: {request.image_path}")
        try:
            return Image.open(request.image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法解析本地图像文件: {e}")
            
    elif request.image_base64:
        try:
            # 去除可能包含 of data:image/png;base64, 前缀
            base64_data = request.image_base64.split(",")[-1]
            image_data = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 图像数据解析失败: {e}")
            
    else:
        raise HTTPException(status_code=400, detail="必须提供 image_path 或 image_base64 其中之一")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """单次影像特征提取接口"""
    global model, processor
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="模型未加载成功，请检查服务端日志")

    # 1. 加载和预处理图像
    image = process_image(request)

    # 2. 构造 Qwen2-VL 标准的 Chat Prompt 模板
    # 医疗场景下，设定 System Prompt 让其充当客观的影像描述医生
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.prompt}
            ]
        }
    ]

    # 3. 准备输入数据
    # 使用 processor 进行数据转化
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=image,
        padding=True,
        return_tensors="pt"

        
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 执行推理并生成文本
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False
            )
            
        # 裁剪掉输入部分的 tokens，仅保留生成的描述
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return PredictResponse(findings=output_text.strip())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理执行失败: {e}")


@app.get("/health")
def health_check():
    """健康检查接口"""
    if model is not None and processor is not None:
        return {"status": "healthy", "device": str(model.device)}
    return {"status": "unhealthy", "message": "Model not loaded yet"}


if __name__ == "__main__":
    print("======================================================================")
    print(" 运行本服务前请确保已安装以下依赖：")
    print(" pip install fastapi uvicorn transformers accelerate pillow")
    print(" ======================================================================")
    # 默认在本地 8000 端口启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)
