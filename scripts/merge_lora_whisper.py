import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# 1. 경로 설정
base_model_id = "openai/whisper-large-v3-turbo"
lora_adapter_path = "/home/data/bootcamp/team/whisper-turbo-finetuned/checkpoint-1200"  # 학습된 어댑터가 저장된 곳
save_path = "/home/data/bootcamp/team/whisper-turbo-final"       # 최종 병합 모델을 저장할 곳

# 2. 베이스 모델 및 프로세서 로드
processor = WhisperProcessor.from_pretrained(base_model_id)
base_model = WhisperForConditionalGeneration.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 3. LoRA 어댑터 연결
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

# 4. 중요: 어댑터와 베이스 모델 병합
# 이 과정을 거쳐야 오프라인에서 Peft 라이브러리 의존성 없이 로드 가능합니다.
merged_model = model.merge_and_unload()

# 5. 최종 로컬 저장
merged_model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"✅ 모델이 {save_path}에 성공적으로 저장되었습니다.")