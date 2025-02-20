# inference.py
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
    
    # 모델과 토크나이저 로드 (모델 병렬 분산 적용)
    finetuned_model_path = os.path.join(RESULTS_DIR, "finetuned_model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        finetuned_model_path,
        device_map="balanced"  # 여러 GPU에 분산
    )
    
    # 특수 토큰 등 필요한 경우 추가 (이미 적용되어 있다면 생략)
    # special_tokens = {"additional_special_tokens": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]}
    # tokenizer.add_special_tokens(special_tokens)
    # model.resize_token_embeddings(len(tokenizer))
    
    # 테스트 데이터 로드 및 프롬프트, token 컬럼 생성
    test_df = load_test_dataframe(TEST_CSV, tokenizer)
    print("[Inference] Test prompts and token counts created.")
    
    # 모델을 추론 모드로 설정
    model = FastLanguageModel.for_inference(model)
    print("[Inference] Fine-tuned model loaded and set to inference mode")
    
    predictions = []
    # DataFrame의 각 행을 순회하면서, 해당 행의 token 값에 따라 max_length 설정
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
        prompt = row["prompt"]
        token_val = row["token"]
        inputs = tokenizer(prompt, return_tensors="pt")
        # 입력 데이터를 GPU로 이동
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        # prompt의 토큰 수 계산 (add_special_tokens=False)
        prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        max_length = prompt_token_count + token_val 
        
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=1,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        raw_output = generated_text.split("Output:")[-1].strip()
        output = clean_output(raw_output)
        predictions.append(output)
    
    test_df["output"] = predictions
    output_csv_path = os.path.join(RESULTS_DIR, "test_output_0217.csv")
    test_df.to_csv(output_csv_path, index=False)
    print(f"[Inference] Predictions saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
