# utils.py
import os
import re
import torch

def tokenize_function(tokenizer, example, max_length=3072):
    """토크나이저를 이용해 prompt를 토큰화합니다."""
    return tokenizer(example["prompt"], truncation=True, max_length=max_length)

def clean_output(text: str) -> str:
    """생성된 텍스트에서 불필요한 부분 제거"""
    if "--- End ---" in text:
        text = text.split("--- End ---")[0]
    text = re.sub(r'\s*\d+(\s+\d+)+\s*$', '', text)
    return text

def get_device():
    """
    분산 학습(DDP) 환경에서 LOCAL_RANK 환경변수를 확인하여 디바이스를 설정합니다.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device

def print_available_gpus():
    print(f"Available GPUs: {torch.cuda.device_count()}")
