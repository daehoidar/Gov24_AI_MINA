import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    WhisperProcessor
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ------------------------------------------------------------------
# 1. 설정 (Configuration)
# ------------------------------------------------------------------
MODEL_ID = "openai/whisper-large-v3-turbo"
DATASET_PATH = {
    "train":'/home/data/bootcamp/team/final_dataset/train_pp',
    "valid":'/home/data/bootcamp/team/final_dataset/valid_pp',
}
OUTPUT_DIR = "./whisper-turbo-finetuned"

# A6000 4장 활용을 위한 하이퍼파라미터
# Global Batch = 32(device) * 4(gpu) * 2(accum) = 256 (매우 강력함)
PER_DEVICE_BATCH_SIZE = 16 
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3  # 80시간 데이터 기준 2~3 에폭이면 충분

# ------------------------------------------------------------------
# 2. 데이터 콜레이터 (Data Collator)
# 이미 전처리가 되어 있어도, 배치를 만들 때 Padding은 동적으로 해야 합니다.
# ------------------------------------------------------------------
# @dataclass
# class DataCollatorSpeechSeq2SeqWithPadding:
#     processor: Any

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         in_feats = [{"input_features": f["input_features"]} for f in features]
#         batch_input = self.processor.feature_extractor.pad(in_feats, return_tensors="pt")

#         # labels padding
#         label_feats = [{"input_ids": f["labels"]} for f in features]
#         labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
#         labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

#         return {
#             "input_features": batch_input["input_features"], # 오디오
#             "labels": labels                           # 정답 텍스트
#         }
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ------------------------------------------------------------------
# 3. 메인 학습 로직
# ------------------------------------------------------------------
def main():
    dist.init_process_group("nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 프로세서 로드 (Collator용)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="korean")
    
    # 데이터셋 로드 (Arrow 형식 등으로 저장된 디스크 데이터)
    train_dataset = load_from_disk(DATASET_PATH["train"])
    valid_dataset = load_from_disk(DATASET_PATH["valid"])

    # ------------------------------------------------------------------
    # 모델 로드 및 양자화 (QLoRA)
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16  # A6000은 BF16 지원
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": local_rank},  # 현재 프로세스의 GPU에 매핑
        attn_implementation="flash_attention_2", # A6000 속도 가속 핵심
        use_cache=False # 학습 중에는 캐시 사용 안 함
    )

    # 학습 전처리 (Gradient Checkpointing 등 활성화)
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # LoRA 설정
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    
    # 메인 프로세스에서만 출력
    if torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 학습 Arguments 설정
    # ------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=300,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,  # A6000 필수 옵션
        
        # 평가 및 저장 전략
        eval_strategy="steps",
        eval_steps=200,      # 80시간 데이터가 크므로 자주 하지 않음
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,  # 용량 관리 (최근 3개만 저장)
        
        # 로깅
        logging_steps=25,
        report_to="tensorboard", # 지금 여기 오류 해결
        
        # DDP 관련 필수 설정
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8, # 데이터 로딩 속도 확보
        
        # 기타 최적화
        predict_with_generate=False, # 학습 속도 저하 방지 (평가는 따로 수행 추천)
        gradient_checkpointing_kwargs={"use_reentrant": False}, # 최신 Torch 경고 방지
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # Trainer 실행
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    
    # LoRA 어댑터 저장
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    main()