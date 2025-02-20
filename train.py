# train.py
import os
import sys
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from data_processing import load_train_dataset
from utils import tokenize_function  # get_device 제거 가능
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
    
    # 단일 프로세스 실행이므로 device 관련 코드는 생략
    print("[Setup] Using all available GPUs for model parallelism")
    
    
    # 모델 및 토크나이저 로드 (모델 병렬 분산 적용)
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        device_map="balanced"  # 여러 GPU에 균등 분산
    )
    # model.to(device) 호출 제거
    train_dataset = load_train_dataset('train.csv')
    
    # LoRA 적용
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
    special_tokens = {
        "additional_special_tokens": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    
    # 토크나이즈
    tokenized_dataset = train_dataset.map(lambda ex: tokenize_function(tokenizer, ex), batched=True)
    print(f"[Fine-tuning] Tokenization complete: {len(tokenized_dataset)} examples.")
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_split = split_dataset["train"]
    eval_split = split_dataset["test"]
    print(f"[Preprocessing] Training dataset: {len(train_split)} examples, Validation dataset: {len(eval_split)} examples")
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_steps=500,
        report_to="none",
        warmup_steps=30,
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
    
    finetuned_model_path = os.path.join(RESULTS_DIR, "finetuned_model")
    model.save_pretrained(finetuned_model_path)
    tokenizer.save_pretrained(finetuned_model_path)
    print(f"[Fine-tuning] Model and tokenizer saved to: {finetuned_model_path}")


if __name__ == "__main__":
    main()
