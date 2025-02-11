# data_processing.py
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

def load_train_dataset(train_csv_path: str) -> Dataset:
    """
    train.csv 파일을 읽어 "input"과 "output" 컬럼을 결합하여,
    프롬프트 템플릿에 따라 'prompt' 컬럼을 생성한 후,
    Hugging Face Dataset으로 변환합니다.
    """
    df = pd.read_csv(train_csv_path)
    # 학습 데이터에서는 프롬프트에 target(출력)까지 포함시킵니다.
    df["prompt"] = df["input"].apply(build_prompt) + df["output"]
    dataset = Dataset.from_pandas(df[["prompt"]])
    return dataset

def load_test_dataframe(test_csv_path: str) -> pd.DataFrame:
    """
    test.csv 파일을 읽어 DataFrame으로 반환합니다.
    (추론 시에는 입력 텍스트에 대해 프롬프트 템플릿만 적용합니다.)
    """
    df = pd.read_csv(test_csv_path)
    df["prompt"] = df["input"].apply(build_prompt)
    return df
