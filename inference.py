import os
import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from data_processing import load_test_dataframe
from utils import clean_output  # get_device 제거 가능
from config import TEST_CSV, RESULTS_DIR

def main():
    print("[Inference] Loading fine-tuned model using model parallelism")
    
    # 테스트 데이터 로드 및 프롬프트 생성
    test_df = load_test_dataframe(TEST_CSV)
    print("[Inference] Test prompts created.")
    
    # 모델과 토크나이저 로드 (모델 병렬 분산 적용)
    finetuned_model_path = os.path.join(RESULTS_DIR, "finetuned_model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        finetuned_model_path,
        device_map="balanced"  # 여러 GPU에 분산
    )
    
    # model.to(device)나 DataParallel 감싸기는 제거합니다.
    model = FastLanguageModel.for_inference(model)
    print("[Inference] Fine-tuned model loaded and set to inference mode")
    
    # 각 프롬프트에 대해 예측 수행 (beam search 사용)
    predictions = []
    for prompt in tqdm(test_df["prompt"], desc="Generating predictions"):
        inputs = tokenizer(prompt, return_tensors="pt")
        # 입력 데이터를 GPU에 올립니다. (여기서는 단일 GPU로 올리더라도 모델은 분산되어 있으므로 문제 없음)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 128,
            num_beams=1,
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
