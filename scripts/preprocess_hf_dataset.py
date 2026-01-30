import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from datasets import load_from_disk, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)

# =================================================================
# 데이터 전처리
# =================================================================
# train_ds = load_from_disk("./train",)
# valid_ds = load_from_disk("./valid",)
test_ds = load_from_disk("../test",)
test_ds = test_ds.cast_column('audio', Audio(decode=False))

MODEL_ID = 'openai/whisper-base'
LANGUAGE = "Korean"
TASK = "transcribe"

device = "cuda:0"
print("device:", device)

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
print("Loaded:", MODEL_ID)

def prepare_batch(batch):
    audio = batch["audio"]
    raw_audio = audio.get_all_samples().data # torch.Tensor

    if raw_audio.shape[0] > 1:
        raw_audio = torch.mean(raw_audio, dim=0)
    raw_audio = raw_audio.squeeze()

    inputs = feature_extractor(raw_audio, sampling_rate=16000)
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

def prepare_batch_test(batch):
    audio_info = batch["audio"]  # decode=False 상태이므로 딕셔너리가 나옴
    file_path = audio_info.get("path", "")
    audio_bytes = audio_info.get("bytes")
    
    waveform = None
    sampling_rate = None

    # 1. PCM 파일 수동 디코딩
    if file_path.endswith(".pcm"):
        # PCM은 보통 int16 (2바이트) 형식입니다.
        # 바이트 데이터를 numpy int16 배열로 변환
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        pcm_data = np.frombuffer(audio_bytes, dtype=np.int16)
        # int16(-32768 ~ 32767)을 float32(-1.0 ~ 1.0)로 정규화
        waveform = torch.from_numpy(pcm_data).float() / 32768.0
        waveform = waveform.unsqueeze(0) # (Channels, Time) 형태로 만듦 -> (1, N)
        
        # 중요: PCM은 파일에 SR 정보가 없으므로 사용자가 알고 있는 값을 지정해야 함
        sampling_rate = 16000 
        
    # 2. 일반 파일(WAV, MP3) 디코딩 (torchaudio 사용 추천)
    else:
        # bytes 데이터를 메모리 파일처럼 만들어서 로드
        import io
        waveform, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))

    # --- 이후 로직은 기존과 동일 ---
    
    # 모노 변환
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 리샘플링 (16000Hz가 아니라면)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 차원 정리 (squeeze)
    waveform = waveform.squeeze() # (N,)

    # Feature Extractor 입력
    inputs = feature_extractor(waveform, sampling_rate=16000)
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    
    return batch

# print("🚀 시작: Train 데이터셋 전처리...")
# train_prep = train_ds.map(
#     prepare_batch,
#     remove_columns=train_ds.column_names,
#     num_proc=32,
#     desc="Extracting Mel features (Train)"
# )
# train_prep.save_to_disk("./train_pp")

# print("🚀 시작: Valid 데이터셋 전처리...")
# valid_prep = valid_ds.map(
#     prepare_batch,
#     remove_columns=valid_ds.column_names,
#     num_proc=32,
#     desc="Extracting Mel features (Vaild)"
# )
# valid_prep.save_to_disk("./valid_pp")

print("🚀 시작: Test 데이터셋 전처리...")
test_prep = test_ds.map(
    prepare_batch_test,
    remove_columns=test_ds.column_names,
    num_proc=16,
    desc="Extracting Mel features (Test)"
)
test_prep.save_to_disk("./test_pp_base")