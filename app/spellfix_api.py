"""Dịch vụ sửa lỗi chính tả bằng model ProtonX Legal Text Correction (T5 seq2seq).

Chạy riêng (giống detectron2_api) để model chỉ nạp một lần trên GPU:
    uvicorn app.spellfix_api:app --host 0.0.0.0 --port 8006

Model chỉ nhận tối đa ~160 token nên văn bản dài được cắt theo câu thành nhiều đoạn,
sửa theo lô rồi ghép lại. Đoạn nào có kết quả lệch độ dài bất thường (dấu hiệu model
sinh thêm/bớt nội dung) thì giữ nguyên bản gốc.
"""
import logging
import os
import re
import threading
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("SPELLFIX_MODEL", "protonx-models/protonx-legal-tc")
NUM_BEAMS = int(os.getenv("SPELLFIX_NUM_BEAMS", "4"))
MAX_TOKENS = int(os.getenv("SPELLFIX_MAX_TOKENS", "160"))
BATCH_SIZE = int(os.getenv("SPELLFIX_BATCH_SIZE", "16"))
DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
    os.getenv("SPELLFIX_DTYPE", "float32")]
MIN_LEN_RATIO = float(os.getenv("SPELLFIX_MIN_LEN_RATIO", "0.8"))
MAX_LEN_RATIO = float(os.getenv("SPELLFIX_MAX_LEN_RATIO", "1.25"))

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.;:!?\n])\s+')

_model = None
_tokenizer = None
_device = None
_lock = threading.Lock()


class CorrectRequest(BaseModel):
    texts: List[str]


class CorrectResponse(BaseModel):
    texts: List[str]


def get_model():
    global _model, _tokenizer, _device
    with _lock:
        if _model is None:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading spellfix model {MODEL_PATH} on {_device} ({DTYPE})")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE).to(_device).eval()
    return _model, _tokenizer, _device


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def _chunk(tokenizer, text: str, limit: int) -> List[str]:
    """Cắt theo câu, gộp các câu liền nhau sao cho mỗi đoạn không vượt quá limit token."""
    if _n_tokens(tokenizer, text) <= limit:
        return [text]
    pieces = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if _n_tokens(tokenizer, sentence) <= limit:
            pieces.append(sentence)
            continue
        words, current = sentence.split(), []          
        for word in words:
            if current and _n_tokens(tokenizer, " ".join(current + [word])) > limit:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and _n_tokens(tokenizer, candidate) > limit:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _accept(original: str, corrected: str) -> str:
    if not corrected:
        return original
    ratio = len(corrected) / max(len(original), 1)
    return corrected if MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO else original


def correct_texts(texts: List[str]) -> List[str]:
    model, tokenizer, device = get_model()
    limit = MAX_TOKENS - 10                    
    chunk_map, flat = [], []
    for i, text in enumerate(texts):
        stripped = (text or "").strip()
        if not stripped:
            continue
        for part in _chunk(tokenizer, stripped, limit):
            chunk_map.append(i)
            flat.append(part)

    corrected = []
    for start in range(0, len(flat), BATCH_SIZE):
        batch = flat[start:start + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_TOKENS).to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=NUM_BEAMS, max_new_tokens=MAX_TOKENS, early_stopping=True)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        corrected.extend(_accept(o, c.strip()) for o, c in zip(batch, decoded))

    results = [[] for _ in texts]
    for i, part in zip(chunk_map, corrected):
        results[i].append(part)
    return [" ".join(parts) if parts else (texts[i] or "") for i, parts in enumerate(results)]


app = FastAPI(title="ProtonX spellfix service")


@app.on_event("startup")
def _warmup():
    get_model()


@app.post("/spellfix/correct", response_model=CorrectResponse)
def correct(req: CorrectRequest):
    return CorrectResponse(texts=correct_texts(req.texts))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "device": _device}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
