import base64
import logging
import os
from typing import Optional
import httpx
import requests
from PIL import Image
import io
import re


# Configure logger
logger = logging.getLogger(__name__)

# Environment variables with default values
VLLM_API_URL = (
    os.getenv("VLLM_API_URL")
    or os.getenv("VISION_MODEL_SERVER")
    or "http://localhost:8005/v1/chat/completions"
)
VLLM_MODEL_NAME = os.getenv(
    "LLM_MODEL",
    "5CD-AI/Vintern-1B-v3_5"
)


def resize_image(image_path: str, ratio: int) -> bytes:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            new_size = (width // ratio, height // ratio)
            resized_img = img.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG")
            return buffer.getvalue()
    except IOError as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        raise


def encode_image_to_base64_resize(image_path: str, resize_ratio) -> str:
    try:
        image_bytes = resize_image(image_path, ratio=resize_ratio)
        return base64.b64encode(image_bytes).decode("utf-8")
    except IOError as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise

def encode_image_to_base64(image_path: str) -> str:
    try:
        image_bytes = resize_image(image_path, ratio=1)
        return base64.b64encode(image_bytes).decode("utf-8")
    except IOError as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise

def call_vllm(prompt: str, image_path: Optional[str] = None, image_base64: Optional[str] = None, max_tokens: int = 1300) -> str:
    headers = {"Content-Type": "application/json"}
    content = [{"type": "text", "text": prompt}]

    if image_path:
        image_base64 = encode_image_to_base64(image_path)
    if image_base64:
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    data = {
        "model": VLLM_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }

    response = requests.post(VLLM_API_URL, headers=headers, json=data, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# Alias để không phải sửa callers cũ
call_vllm_base64 = call_vllm


async def call_vllm_async(client: httpx.AsyncClient, prompt: str, image_path: str) -> str:
    headers = {"Content-Type": "application/json"}
    base64_image = encode_image_to_base64(image_path)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        {"type": "text", "text": prompt}
    ]

    data = {
        "model": VLLM_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1300,
        "temperature": 0.0
    }

    try:
        response = await client.post(VLLM_API_URL, headers=headers, json=data, timeout=120.0)
        response.raise_for_status()
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].strip()
    except (httpx.RequestError, KeyError, IndexError) as e:
        logger.error(f"Async VLLM call failed for {image_path}: {e}")
        return ""

def call_llm_for_profile_name(info: str) -> str:
    organization_name = os.getenv("ORG_NAME_1")
    organization_name_2 = os.getenv("ORG_NAME_4")
    prompt = f"""Thông tin:
<info>
{info}
</info>
Yêu cầu:
Hãy sử dụng thông tin đã cho ở trên đây đặt tên cho bộ hồ sơ chứa các văn bản đó trong khoảng 50 từ (Không cần giải thích, không cần đưa ra lý do, chỉ trả về tên hồ sơ). Ví dụ như sau:
Thông tin ví dụ:
<thông tin ví dụ>
Ngày tháng\tTên loại và trích yếu\tTác giả
02/11/2020\t"THÔNG BÁO\nVề việc thay đổi Lịch tiếp công dân của Bí thư Tỉnh ủy tháng 11 năm 2020"\t VĂN PHÒNG {organization_name_2}
21/12/2020\t"THÔNG BÁO\nVề việc thay đổi Lịch tiếp dân của Bí thư Tỉnh ủy Tháng 12 năm 2020"\t VĂN PHÒNG {organization_name_2}
</thông tin ví dụ>
Kết quả sinh tên của ví dụ:
Thông báo của Văn phòng {organization_name_2} về lịch và thay đổi lịch tiếp dân của Bí thư Tỉnh uỷ. Tháng 11/2020 - 2024"""

    try:
        result = call_vllm(prompt=prompt, max_tokens=200)
        return re.sub(r"\n\s*\n+", "\n\n", result).strip()
    except Exception as e:
        logger.error(f"call_llm_for_profile_name failed: {e}")
        return ""
