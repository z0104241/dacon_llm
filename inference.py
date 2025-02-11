# inference.py
import os
import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from data_processing import load_test_dataframe
from utils import clean_output, get_device
from config import TEST_CSV, RESULTS_DIR

def main():
    # 디바이스 설정
    device = get_device()
    print(f"[Inference] Using device: {device}")
    
    # 테스트 데이터 로드 및 프롬프트 생성
    test_df = load_test_dataframe(TEST_CSV)
    print("[Inference] Test prompts created.")
    
    # 파인튜닝된 모델 및 토크나이저 로드
    finetuned_model_path = os.path.join(RESULTS_DIR, "finetuned_model")
    model, tokenizer = FastLanguageModel.from_pretrained(finetuned_model_path)
    model.to(device)
    
    # 다중 GPU 사용 시 DataParallel 적용
    if torch.cuda.device_count() > 1:
        print(f"[Inference] Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    
    model = FastLanguageModel.for_inference(model)
    print("[Inference] Fine-tuned model loaded and set to inference mode")
    
    # 각 프롬프트에 대해 예측 수행 (beam search 사용)
    predictions = []
    for prompt in tqdm(test_df["prompt"], desc="Generating predictions"):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 128,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        raw_output = generated_text.split("Output:")[-1].strip()
        output = clean_output(raw_output)
        predictions.append(output)
    
    test_df["output"] = predictions
    output_csv_path = os.path.join(RESULTS_DIR, "test_with_predictions.csv")
    test_df.to_csv(output_csv_path, index=False)
    print(f"[Inference] Predictions saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
