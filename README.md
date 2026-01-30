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

### 1. 필수 패키지 설치
```bash
pip install torch transformers sentence-transformers
pip install fastapi uvicorn python-multipart
pip install soundfile numpy pandas openpyxl scipy
pip install qwen-tts
```

### 2. 모델 준비
- Whisper 모델: `/home/data/bootcamp/team/whisper-turbo-merged`
- RAG 데이터: `rag.xlsx` (같은 디렉토리에 위치)

### 3. 서버 실행
```bash
python integrated_voice_rag_system.py
```

### 4. 웹 접속
브라우저에서 `http://localhost:8000` 접속

## 💻 사용 방법

### 웹 인터페이스 사용

#### 1. 음성 녹음 방식
1. "녹음 시작" 버튼 클릭
2. 마이크에 대고 질문 (예: "주민등록증 재발급 방법 알려주세요")
3. "녹음 중지" 버튼 클릭
4. 자동으로 처리되어 응답 음성 재생

#### 2. 파일 업로드 방식
1. "음성 파일 선택" 버튼 클릭
2. 음성 파일 선택 (.wav, .mp3 등)
3. 자동으로 처리되어 응답 표시

#### 3. 대화 관리
- **대화 초기화**: 대화 히스토리를 모두 삭제
- **대화 내역**: 이전 대화를 기억하여 후속 질문 처리 가능

### Python 코드로 직접 사용

```python
from integrated_voice_rag_system import VoiceRAGPipeline
import soundfile as sf

# 파이프라인 초기화
pipeline = VoiceRAGPipeline()
pipeline.load_models()

# 음성 파일 로드
audio_data, sample_rate = sf.read("question.wav")

# 전체 파이프라인 실행
result = pipeline.process_voice(
    audio_data, 
    sample_rate,
    output_audio_path="response.wav"
)

# 결과 확인
print(f"인식된 텍스트: {result['transcribed_text']}")
print(f"응답: {result['response_text']}")
print(f"참고 서비스: {result['referenced_service']}")
print(f"총 처리 시간: {result['timings']['total']:.2f}초")
```

## 🎯 주요 클래스 및 메서드

### VoiceRAGPipeline 클래스

```python
class VoiceRAGPipeline:
    """ASR → RAG → TTS 통합 파이프라인"""
    
    def load_models(self):
        """모든 모델 로드 (초기화 시 1회 실행)"""
        
    def transcribe(self, audio_data, sample_rate=16000) -> str:
        """음성을 텍스트로 변환"""
        
    def get_rag_response(self, text: str) -> Tuple[str, Optional[str]]:
        """RAG를 통해 응답 생성"""
        
    def synthesize(self, text: str, output_path=None) -> np.ndarray:
        """텍스트를 음성으로 변환"""
        
    def process_voice(self, audio_data, sample_rate=16000, output_audio_path=None) -> Dict:
        """전체 파이프라인 실행"""
        
    def reset_conversation(self):
        """대화 히스토리 초기화"""
```

### ThreeStepRAG 클래스

```python
class ThreeStepRAG:
    """대화 히스토리를 기억하는 RAG 시스템"""
    
    def load(self):
        """모델 및 데이터 로드"""
        
    def chat(self, query: str) -> Tuple[str, Optional[str]]:
        """대화 처리 (히스토리 자동 저장)"""
        
    def reset_history(self):
        """대화 히스토리 초기화"""
```

## 📊 성능 및 리소스

### 모델 로딩 시간 (예상)
- STT 모델: ~5초
- RAG 모델 (LLM + 임베딩): ~10초
- TTS 모델: ~2초
- **총 초기화 시간**: ~17초

### 처리 시간 (예상, GPU 사용 시)
- STT: ~0.5초 (5초 음성 기준)
- RAG: ~1-2초
- TTS: ~0.5초
- **총 응답 시간**: ~2-3초

### GPU 메모리 사용량 (예상)
- STT: ~2GB (cuda:0)
- RAG (LLM): ~4GB (cuda:3)
- TTS: ~2GB (cuda:0)
- **총 메모리**: ~8GB

## 🔧 설정 커스터마이징

### GPU 설정 변경
```python
# integrated_voice_rag_system.py 파일에서

STT_DEVICE = 'cuda:0'  # STT 모델 GPU
RAG_DEVICE = 'cuda:3'  # RAG 모델 GPU
TTS_DEVICE = 'cuda:0'  # TTS 모델 GPU
```

### 모델 경로 변경
```python
MODEL_ID = {
    "STT": '/your/path/to/whisper-model',
    "LLM": "Qwen/Qwen3-1.7B",
    "TTS": 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    "EMBEDDING": "intfloat/multilingual-e5-large"
}

RAG_EXCEL_PATH = "/your/path/to/rag.xlsx"
```

### 포트 변경
```python
# 메인 실행 부분에서
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,  # 원하는 포트로 변경
    log_level="info"
)
```

## 📝 API 엔드포인트

### POST /process_audio
음성 파일을 처리하여 응답 생성

**Request:**
- `audio`: 음성 파일 (multipart/form-data)

**Response:**
```json
{
    "transcribed_text": "주민등록증 재발급 방법 알려주세요",
    "response_text": "6개월 이내에 촬영한 증명사진과...",
    "referenced_service": "주민등록증 재발급",
    "audio_base64": "UklGRi4HAABXQVZF...",
    "timings": {
        "stt": 0.52,
        "rag": 1.34,
        "tts": 0.48,
        "total": 2.34
    }
}
```

### POST /reset
대화 히스토리 초기화

**Response:**
```json
{
    "message": "대화 히스토리가 초기화되었습니다."
}
```

## 🎨 웹 인터페이스 기능

### 실시간 상태 표시
- 🎤 녹음 중 상태 표시 (애니메이션)
- ⏳ 처리 중 스피너 표시
- ✅ 성공/실패 메시지 표시

### 대화 내역 관리
- 사용자와 AI의 대화 내역 시각적 구분
- 스크롤 자동 이동
- 응답 음성 자동 재생

### 반응형 디자인
- 모바일 기기 지원
- 다양한 화면 크기 대응

## ⚠️ 주의사항

1. **GPU 메모리**: 충분한 GPU 메모리 확보 필요 (최소 8GB 권장)
2. **모델 경로**: 모델 파일이 올바른 위치에 있는지 확인
3. **RAG 데이터**: `rag.xlsx` 파일이 프로젝트 루트에 위치해야 함
4. **마이크 권한**: 브라우저에서 마이크 접근 권한 허용 필요
5. **HTTPS**: 일부 브라우저는 HTTPS에서만 마이크 사용 가능

## 🐛 문제 해결

### 모델 로딩 실패
```bash
# 모델 경로 확인
ls /home/data/bootcamp/team/whisper-turbo-merged

# GPU 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족
- GPU 설정에서 다른 GPU로 분산
- 배치 크기 조정
- 사용하지 않는 프로세스 종료

### 오디오 처리 오류
- 오디오 파일 형식 확인 (WAV, MP3 권장)
- 샘플레이트 확인 (16kHz 권장)
- scipy 패키지 설치 확인

## 📚 참고 자료

- [Whisper Documentation](https://github.com/openai/whisper)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qwen-TTS](https://github.com/QwenLM/Qwen-TTS)

## 📄 라이선스

이 프로젝트는 각 모델의 라이선스를 따릅니다.
