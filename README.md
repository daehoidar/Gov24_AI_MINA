# 정부24 인공지능 서비스: 미나(MINA)
> 2025 서울과학기술대학교 동계 인공지능 SCI 부트캠프 우수상(사업단장상)

Whisper(STT) → ThreeStepRAG → Qwen3-TTS 통합 음성 민원 상담 파이프라인

---


## 시스템 구성

### 주요 기능
- **음성 인식 (STT)**: 파인튜닝된 Whisper 모델 기반 한국어 음성 인식
- **지능형 응답 (RAG)**: ThreeStepRAG 기반 민원 안내
- **음성 합성 (TTS)**: Qwen3-TTS 음성 출력
- **웹 인터페이스**: FastAPI + WebSocket 실시간 대화

### 파이프라인
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

### ThreeStepRAG
SentenceTransformer 임베딩 기반 3단계 계층적 검색으로, 단일 유사도 검색에서 발생하는 발급/조회/신고 등의 세부 구분 실패를 해결.

- **Step 1 — 주제 대분류** (유사도 ≥ 0.80): 질문의 주제 카테고리 매칭
- **Step 2 — 제공 서비스** (유사도 ≥ 0.78): 발급/조회/신고 등 서비스 유형 구분
- **Step 3 — 처리 대상** (유사도 ≥ 0.83): 구체적 대상 확인

각 단계에서 임계값 미달 시 즉시 중단(Early Exit)하고 고정 안내문을 반환하여, LLM이 답을 지어내는 환각을 구조적으로 차단. 응답 생성에는 Qwen3-1.7B를 사용하며, 시스템 프롬프트와 대화 히스토리를 함께 전달.

## 설치 및 실행

### 1. 가상환경 구성
- environment.yaml

### 2. Whisper LoRA 파인튜닝
- 베이스 모델: whisper-large-v3-turbo
- 학습 방식: QLoRA (4bit + LoRA r=32)
- 데이터셋 정보
  - Train: 72.4시간 / 48,291 샘플
  - Valid: 10.2시간 / 6,930 샘플
  - Test: 11.5시간 / 9,575 샘플
- 데이터셋 구성
  - Train: 복지분야 콜센터 상담 데이터(AIHub) 62.5% + 상담 음성 데이터(AIHub) 25% + Zeroth Korean 12.5%
  - Valid: Train과 같은 구성
  - Test: 복지분야 콜센터 상담 데이터(AIHub) 25% + 상담 음성 데이터(AIHub) 25% + 한국어 음성데이터(AIHub) 50%
- 성능: CER 14.96%, 기본 Turbo 대비 오류율 9.5% 감소

### 3. 모델 불러오기
### 4. 실행
### 5. Web 접속

## 참고 자료
- [Whisper](https://github.com/openai/whisper)
- [Transformers](https://huggingface.co/docs/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Qwen-TTS](https://github.com/QwenLM/Qwen-TTS)

## 라이선스
각 모델의 라이선스를 따릅니다.
