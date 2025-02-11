# config.py
import os

# config.py

# 모델 설정
MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

# 학습 하이퍼파라미터
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5

# LoRA 설정 (파라미터 수를 줄이기 위한 설정)
LORA_R = 8  # LoRA rank 값 (기존 16에서 4로 낮춤)
LORA_ALPHA = 8  # 보통 LORA_R에 비례하도록 설정
LORA_DROPOUT = 0  # dropout 사용하지 않음
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 필요한 모듈만 지정
LORA_BIAS = "none"
LORA_USE_GRADIENT_CHECKPOINTING = "unsloth"
LORA_RANDOM_STATE = 3407
LORA_USE_RSLORA = False
LORA_LOFTQ_CONFIG = None

# 경로 설정
import os
BASE_PATH = ""  # 데이터 및 결과 폴더의 베이스 경로
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
RESULTS_DIR = os.path.join(BASE_PATH, "results")
FINETUNED_MODEL_PATH = os.path.join(RESULTS_DIR, "finetuned_model")

#torchrun --nproc_per_node=2 train.py && torchrun --nproc_per_node=2 inference.py
