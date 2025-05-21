"""간소화된 추론 모듈"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm
import config

class Predictor:
    def __init__(self, model_path=config.OUTPUT_DIR):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """모델 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, 
            trust_remote_code=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
    def predict_single(self, sentences, use_thinking=False):
        """단일 예측"""
        messages = [
            {"role": "system", "content": "당신은 문장 순서를 논리적으로 배열하는 전문가입니다."},
            {"role": "user", "content": f"""다음 4개의 문장을 올바른 순서로 배열하세요.

문장들:
0: {sentences[0]}
1: {sentences[1]}
2: {sentences[2]}
3: {sentences[3]}

올바른 순서를 쉼표로 구분된 숫자로 답해주세요."""}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=use_thinking
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(text):].strip()
        
        # thinking mode 처리
        if use_thinking and "</think>" in generated:
            generated = generated.split("</think>")[-1].strip()
        
        # 답안 파싱
        numbers = re.findall(r'\d', generated.split('\n')[0])
        if len(numbers) >= 4:
            result = [int(n) for n in numbers[:4]]
            if set(result) == {0, 1, 2, 3}:
                return result
        
        return [0, 1, 2, 3]  # 기본값
    
    def predict_csv(self, test_file=config.TEST_FILE, output_file=config.PREDICTIONS_FILE):
        """CSV 파일 예측"""
        if not self.model:
            self.load_model()
            
        df = pd.read_csv(test_file)
        predictions = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sentences = [row[f'sentence_{i}'] for i in range(4)]
            pred = self.predict_single(sentences)
            
            predictions.append({
                'ID': row['ID'],
                'answer_0': pred[0],
                'answer_1': pred[1],
                'answer_2': pred[2],
                'answer_3': pred[3]
            })
        
        result_df = pd.DataFrame(predictions)
        result_df.to_csv(output_file, index=False)
        print(f"예측 완료: {output_file}")
        return result_df

def main():
    """메인 함수"""
    predictor = Predictor()
    predictor.predict_csv()

if __name__ == "__main__":
    main()