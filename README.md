# 정부24 인공지능 서비스: 미나(MINA)

ASR (Whisper) → RAG (ThreeStepRAG) → TTS (Qwen3-TTS) 통합 파이프라인

## 📋 시스템 구성

### 1. 주요 기능
- **음성 인식 (STT)**: 파인튜닝된 Whisper 모델을 사용한 한국어 음성 인식
- **지능형 응답 (RAG)**: ThreeStepRAG를 활용한 민원 안내 시스템
- **음성 합성 (TTS)**: Qwen3-TTS를 사용한 자연스러운 음성 출력
- **웹 인터페이스**: FastAPI 기반의 사용하기 쉬운 웹 GUI

### 2. 파이프라인 흐름
```
사용자 음성 입력
    ↓
[Whisper STT] 음성 → 텍스트 변환
    ↓
[ThreeStepRAG] 민원 정보 검색 및 응답 생성
    ↓
[Qwen3-TTS] 텍스트 → 음성 변환
    ↓
응답 음성 출력
```

## 🚀 설치 및 실행

### 1. 가상환경 구성
- environment.yaml
### 2. whisper-large-v3-turbo & LoRA 파인튜닝
- 데이터셋 정보
  - Train: 72.4시간 / 48291 샘플(문장, 발화)
  - Valid: 10.2시간 / 6930  샘플
  - Test:  11.5시간 / 9575 샘플
- 데이터셋 구성
  - Train: 복지분야 콜센터 상담 데이터(AIHUB) 62.5% + 상담 음성 데이터(AIHUB)25% + Zeroth Korean 12.5%
  - Valid: Train과 같은 구성 
  - Test: 복지분야 콜센터 상담 데이터(AIHUB) 25% + 상담 음성 데이터(AIHUB)25% + 한국어 음성데이터(AIHUB) 50%
### 3. 모델 불러오기
### 4. 실행
### 5. Web 접속

## 📚 참고 자료

- [Whisper Documentation](https://github.com/openai/whisper)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qwen-TTS](https://github.com/QwenLM/Qwen-TTS)

## 📄 라이선스

이 프로젝트는 각 모델의 라이선스를 따릅니다.
