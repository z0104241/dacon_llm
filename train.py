#!/usr/bin/env python
import os
import sys
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# 데이터 처리 및 유틸 함수 임포트
from data_processing import load_train_dataset
from utils import tokenize_function, get_device

# 설정 값 불러오기 (config.py)
from config import (
    TRAIN_CSV,
    RESULTS_DIR,
    MODEL_ID,
    NUM_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_BIAS,
    LORA_USE_GRADIENT_CHECKPOINTING,
    LORA_RANDOM_STATE,
    LORA_USE_RSLORA,
    LORA_LOFTQ_CONFIG,
)

def main():
    # 현재 디렉터리를 모듈 검색 경로에 추가 (필요한 경우)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # 디바이스 설정 (예: LOCAL_RANK 환경변수 활용)
    device = get_device()
    print(f"[Setup] Using device: {device}")
    
    # 학습 데이터 로드 및 프롬프트 생성
    train_dataset = load_train_dataset(TRAIN_CSV)
    print(f"[Preprocessing] Loaded training dataset with {len(train_dataset)} examples.")
    
    # 모델 및 토크나이저 불러오기 (4-bit 양자화 옵션 사용)
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        device_map="balanced"
    )
    model.to(device)
    
    # LoRA 적용 (config.py에 정의된 설정값 사용)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing=LORA_USE_GRADIENT_CHECKPOINTING,
        random_state=LORA_RANDOM_STATE,
        use_rslora=LORA_USE_RSLORA,
        loftq_config=LORA_LOFTQ_CONFIG,
    )
    
    # 토크나이즈: 데이터셋의 각 예제에 대해 토크나이저 적용
    tokenized_dataset = train_dataset.map(lambda ex: tokenize_function(tokenizer, ex), batched=True)
    print(f"[Fine-tuning] Tokenization complete: {len(tokenized_dataset)} examples.")
    
    # 학습/검증 데이터셋 분할 (예: 90% 학습, 10% 검증)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_split = split_dataset["train"]
    eval_split = split_dataset["test"]
    print(f"[Preprocessing] Training dataset: {len(train_split)} examples, Validation dataset: {len(eval_split)} examples")
    
    # 학습 인자 설정 (각 GPU 기준 배치 사이즈 등)
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_steps=500,
        report_to="none",
        warmup_steps=100,
    )
    
    if is_bfloat16_supported():
        training_args.bf16 = True
        print("[TrainingArgs] Using bf16 precision")
    else:
        training_args.fp16 = True
        print("[TrainingArgs] Using fp16 precision")
    
    # Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        tokenizer=tokenizer,
    )
    
    print("[Fine-tuning] Starting fine-tuning...")
    trainer.train()
    print("[Fine-tuning] Fine-tuning complete!")
    
    # 파인튜닝 완료 후 모델과 토크나이저 저장
    finetuned_model_path = os.path.join(RESULTS_DIR, "finetuned_model")
    model.save_pretrained(finetuned_model_path)
    tokenizer.save_pretrained(finetuned_model_path)
    print(f"[Fine-tuning] Model and tokenizer saved to: {finetuned_model_path}")

if __name__ == "__main__":
    main()
