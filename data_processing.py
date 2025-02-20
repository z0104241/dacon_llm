# data_processing.py
from typing import List, Tuple
import pandas as pd
from datasets import Dataset

def build_prompt(input_text: str) -> str:
    """
    주어진 입력 텍스트에 대해 통일된 프롬프트 템플릿을 생성합니다.
    (학습 시에는 모델의 출력(target)과 결합하여 사용하고, 추론 시에는 입력 프롬프트로 사용됩니다.)
    """
    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "[IMPORTANT NOTICE]\n"
        "For this task, **the following rules must be followed without any violation.**\n"
        "Rules:\n"
        "1) **Whitespace**: All whitespace in the input text must be preserved exactly as is.\n"
        "2) **Special Characters**: The length and arrangement of all special characters must be identical to the input and must not be altered.\n"
        "3) **Word and Sentence Length**: The length of each word and the overall sentence structure must be maintained exactly. No additions or deletions are allowed.\n"
        "You must strictly comply with these rules. Restore the text exactly as provided.\n"
        "<|eot_id|>\n\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Input: {input_text}\n"
        "<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "Output:\n\n"
    )
    return prompt_template

def augment_pair_with_ngrams(input_text: str, output_text: str, 
                             ngram_min: int = 6, ngram_max: int = 7) -> List[Tuple[str, str]]:
    """
    input_text와 output_text를 각각 띄어쓰기를 기준으로 단어 분리한 후,
    길이가 1인 단어는 제외하고, 지정된 ngram 범위(n=ngram_min ~ ngram_max)로 
    두 텍스트를 동시에 슬라이딩 윈도우(슬라이드 간격 1) 방식으로 잘라서
    (input_ngram, output_ngram) 쌍의 리스트를 반환합니다.
    
    예시:
        input_text = "1 2 3 4 5"
        output_text = "a b c d e"
        ngram_min=4, ngram_max=4 인 경우:
         - 인덱스 0: ("1 2 3 4", "a b c d")
         - 인덱스 1: ("2 3 4 5", "b c d e")
    
    :param input_text: 원본 입력 텍스트
    :param output_text: 원본 출력 텍스트
    :param ngram_min: 최소 ngram 길이 (기본값 6)
    :param ngram_max: 최대 ngram 길이 (기본값 7)
    :return: (input_ngram, output_ngram) 쌍의 리스트
    """
    input_words = [w for w in input_text.split() if len(w) > 1]
    output_words = [w for w in output_text.split() if len(w) > 1]
    
    # 두 텍스트가 정렬되어 있다고 가정하고, 최소 길이를 기준으로 진행
    L = min(len(input_words), len(output_words))
    augmented_pairs = []
    
    if L < ngram_min:
        return augmented_pairs
    
    # ngram 길이 범위 내에서, 두 텍스트를 동시에 슬라이딩 윈도우(슬라이드 간격 1)로 잘라서 생성
    for n in range(ngram_min, min(ngram_max, L) + 1):
        for i in range(0, L - n + 1, 5):
            in_ngram = " ".join(input_words[i:i+n])
            out_ngram = " ".join(output_words[i:i+n])
            augmented_pairs.append((in_ngram, out_ngram))
    return augmented_pairs

def load_train_dataset(train_csv_path: str) -> Dataset:
    """
    train.csv 파일을 읽어, 각 행의 input과 output에 대해
    ngram 증강을 진행합니다. (원본 포함)
    증강된 각 input과 output 쌍에 대해 프롬프트 템플릿을 적용하여 최종 프롬프트를 생성합니다.
    
    최종적으로 증강된 프롬프트들을 Hugging Face Dataset으로 변환합니다.
    """
    df = pd.read_csv(train_csv_path)
    augmented_data = []
    
    for _, row in df.iterrows():
        original_input = row["input"]
        original_output = row["output"]
        
        # 원본 쌍 포함
        pair_list = [(original_input, original_output)]
        # input과 output을 동시에 증강
        pair_list.extend(augment_pair_with_ngrams(original_input, original_output, ngram_min=6, ngram_max=7))
        
        for aug_in, aug_out in pair_list:
            # 프롬프트 생성: 증강된 input에 대해 템플릿을 적용한 뒤, 그 뒤에 증강된 output을 이어붙임
            prompt = build_prompt(aug_in) + aug_out
            augmented_data.append({
                "prompt": prompt
            })
    
    print(f"원본 {len(df)}행 -> 증강 후 {len(augmented_data)}행")
    
    augmented_df = pd.DataFrame(augmented_data)
    dataset = Dataset.from_pandas(augmented_df)
    return dataset



def load_test_dataframe(test_csv_path: str, tokenizer) -> pd.DataFrame:
    """
    test.csv 파일을 읽어 DataFrame으로 변환한 후,
    각 행의 input 컬럼의 토큰 수를 계산하여 'token' 컬럼을 추가하고,
    build_prompt 함수를 이용해 prompt 컬럼을 생성합니다.
    """
    df = pd.read_csv(test_csv_path)
    # input 텍스트의 토큰 수 계산 (add_special_tokens=False로 불필요한 토큰 제외)
    df["token"] = df["input"].apply(lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    df["prompt"] = df["input"].apply(build_prompt)
    return df
