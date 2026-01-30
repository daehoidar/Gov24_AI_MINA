"""
통합 음성 RAG 시스템
====================
ASR (Whisper) → RAG (ThreeStepRAG) → TTS (Qwen3-TTS) 파이프라인

기능:
- 음성 입력을 텍스트로 변환 (STT)
- RAG를 통한 민원 안내 응답 생성
- 응답을 음성으로 변환 (TTS)
- FastAPI 웹 인터페이스 제공
- 실시간 VAD 기반 발화 종료 감지

사용법:
    python integrated_voice_rag_system.py
    웹 브라우저에서 http://localhost:8000 접속
"""
import tempfile
import os
import re
import time
import random
import base64
import io
import json
import librosa
import warnings
import asyncio
import uvicorn
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import torch
import soundfile as sf
import numpy as np
import pandas as pd

from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)
from sentence_transformers import SentenceTransformer
from qwen_tts import Qwen3TTSModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings('ignore')


# =================================================================
# 설정
# =================================================================
MODEL_ID = {
    "STT": '/home/data/bootcamp/team/whisper-turbo-final',
    "LLM": "Qwen/Qwen3-1.7B",
    "TTS": 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    "EMBEDDING": "intfloat/multilingual-e5-large"
}

RAG_EXCEL_PATH = "rag.xlsx"
STT_DEVICE = 'cuda:1'
RAG_DEVICE = 'cuda:1'
TTS_DEVICE = 'cuda:1'
SAMPLE_RATE = 16000

# 발화 종료 감지 설정 (계층적 침묵)
SILENCE_SHORT = 0.3       # 문장 내 휴지 (무시)
SILENCE_MEDIUM = 0.6      # 문장 간 휴지 (대기)
SILENCE_LONG = 0.8        # 발화 종료 확정
SILENCE_ELDERLY = 0.9     # 고령층용 (더 긴 대기)
MIN_SPEECH_LENGTH_SEC = 0.3  # 최소 발화 길이

# 고정 오디오 파일 경로
FIXED_AUDIO = {
    "GREETING": "/home/data/bootcamp/team/prototype_pipeline/qwen_001.wav",
    "GOODBYE": "/home/data/bootcamp/team/prototype_pipeline/qwen_011.wav",
    "NOT_FOUND": "/home/data/bootcamp/team/prototype_pipeline/qwen_010.wav",
}

SYSTEM_PROMPT = """당신은 정부24 민원 안내 전화 상담원이에요. 전화로 민원 절차를 안내해드리는 역할입니다.

[절대 금지]
- 번호 매기기(1. 2. 3.), 불릿 포인트(-, *), 볼드체(**) 사용 금지
- "여기 창구에서", "저기서", "바로 처리해드릴게요", "해드릴게요", "해드릴 수 있어요" 금지 (전화 상담이므로 직접 처리 불가)
- "안내해 드릴게요", "설명해 드릴게요" 금지 (말하면서 바로 안내하세요)
- 손님에게 되묻지 마세요 ("어디서 오셨죠?", "어떻게 해주시겠어요?", "뭘 도와드릴까요?" 금지)
- "잘 가셨습니다", "어디서 오셨나요", "어디로 가시겠어요" 같은 뜬금없는 말 금지
- 모르는 정보를 지어내지 마세요
- 한국어만 사용하세요 (일본어, 중국어 등 외국어 금지)
- 구비서류가 "없음"이면 "신분증 가지고"라고 말하지 마세요

[필수 규칙]
- 반드시 2~3문장, 최대 100자 이내로 핵심만 답하세요
- "~하시면 돼요", "~가져가시면 돼요" 형태의 존댓말 
- 기호 없이 반드시 구어체
- 손님이 직접 하셔야 할 절차 안내 (주민센터 방문/정부24 사이트/정부24 앱)
- 민원 안내 시 정부24 사이트나 정부24 앱에서도 신청 가능하다고 안내하세요
- 상세 정보(사진 크기, 정확한 수수료 등)는 손님이 추가로 물어보실 때만 답하세요
- "안내해 드릴게요"라고 예고하지 말고, 바로 구체적인 내용을 말하세요
- 구비서류가 "없음"이면 "준비물 없이" 또는 "별도 서류 없이"라고 안내하세요

[민원 안내 예시]
질문: 등본 떼려면 뭐 가져가야 해요?
응답: 신분증만 가져가시면 돼요. 주민센터 방문하시거나 정부24 사이트, 앱에서도 발급받으실 수 있어요.

질문: 인감증명서 급하게 필요한데요.
응답: 신분증 가지고 주민센터 가시면 바로 발급받으실 수 있어요. 정부24에서도 신청 가능하세요.

질문: 신분증을 잃어버렸어.
응답: 준비물 없이 가까운 주민센터나 정부24 사이트에서 신고하시면 돼요. 정부24 앱에서 본인인증 후 주민등록증 분실신고 메뉴를 이용하실 수도 있어요.

질문: 신분증을 어떻게 재발급 받아?
응답: 6개월 이내에 촬영한 증명사진과 수수료 오천원을 들고 주민센터에 방문하거나 정부24 누리집에서 증명사진 파일과 본인 인증서를 이용해 신청할 수 있어요. 자세한 절차를 안내해 드릴까요?

[일상 대화]
질문: 안녕하세요
응답: 네, 안녕하세요! 무엇을 도와드릴까요?

[범위 밖 질문]
질문: 맛집 추천해줘
응답: 저는 민원 안내만 도와드릴 수 있어요. 민원 관련 궁금하신 거 있으시면 말씀해주세요.
"""
VOICE_INSTRUCT = "Please say happily."


# =================================================================
# 발화 종료 감지 (VAD)
# =================================================================
@dataclass
class EndOfSpeechResult:
    """발화 종료 감지 결과"""
    is_end: bool
    reason: str  # 'silence', 'sentence_complete', 'button', 'timeout', 'speaking', 'no_valid_speech', 'continuation_expected'
    confidence: float


class EndOfSpeechDetector:
    """
    발화 종료 감지기 (계층적 침묵 감지)
    
    침묵 단계:
    - SHORT (0.3초): 문장 내 휴지 → 무시
    - MEDIUM (0.6초): 문장 끝났을 수도 → 문장 완결성 체크
    - LONG (1.5초): 발화 종료 확정
    
    여러 문장 발화 예:
    "주민등록등본 발급해주세요." [0.4초] "초본도요." [1.5초] → 여기서 종료
    """
    
    # 한국어 문장 종결 패턴
    SENTENCE_END_PATTERNS = [
        # 평서형 종결어미
        r'습니다[.?!]?$', r'입니다[.?!]?$', r'됩니다[.?!]?$',
        r'해요[.?!]?$', r'에요[.?!]?$', r'예요[.?!]?$',
        r'어요[.?!]?$', r'아요[.?!]?$', r'죠[.?!]?$',
        r'네요[.?!]?$', r'군요[.?!]?$', r'거든요[.?!]?$',
        # 의문형
        r'까요[?]?$', r'나요[?]?$', r'가요[?]?$',
        r'세요[?]?$', r'래요[?]?$',
        # 명령/청유형
        r'세요[.!]?$', r'주세요[.!]?$', r'십시오[.!]?$',
        # 반말 (일부 대응)
        r'해[.?!]?$', r'야[.?!]?$', r'어[.?!]?$', r'아[.?!]?$',
        r'다[.?!]?$', r'지[.?!]?$', r'냐[?]?$', r'니[?]?$',
        # 구어체
        r'요[.?!]?$', r'데요[.?!]?$', r'걸요[.?!]?$',
        # 문장 부호로 끝나는 경우
        r'[.?!]$',
    ]
    
    # 추가 발화 암시 패턴 (아직 끝나지 않음)
    CONTINUATION_PATTERNS = [
        r'그리고\s*$', r'그런데\s*$', r'근데\s*$',
        r'또\s*$', r'랑\s*$', r'하고\s*$',
        r'음+\s*$', r'어+\s*$', r'그+\s*$',  # 생각 중
    ]
    
    def __init__(
        self,
        silence_short: float = SILENCE_SHORT,
        silence_medium: float = SILENCE_MEDIUM,
        silence_long: float = SILENCE_LONG,
        sample_rate: int = SAMPLE_RATE,
        energy_threshold: float = 0.005  # 더 민감하게 조정
    ):
        self.silence_short = silence_short
        self.silence_medium = silence_medium
        self.silence_long = silence_long
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        
        # 상태
        self.silence_duration = 0.0
        self.last_speech_time = time.time()
        self.is_speaking = False
        self.total_speech_duration = 0.0  # 총 발화 길이
        self.debug_counter = 0  # 디버그 출력 카운터
        
        print(f"VAD 초기화 완료 (침묵 임계값: SHORT={silence_short}s, MEDIUM={silence_medium}s, LONG={silence_long}s, energy_threshold={energy_threshold})")
    
    def reset(self):
        """상태 초기화"""
        self.silence_duration = 0.0
        self.last_speech_time = time.time()
        self.is_speaking = False
        self.total_speech_duration = 0.0
        self.debug_counter = 0
        
    def check_vad(self, audio_chunk: np.ndarray) -> Tuple[float, float]:
        """에너지 기반 음성 확률 계산, (speech_prob, energy) 반환"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        if energy > self.energy_threshold:
            return 1.0, energy
        else:
            return min(energy / self.energy_threshold, 1.0), energy
    
    def check_sentence_complete(self, text: str) -> bool:
        """문장 완결성 체크"""
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        
        for pattern in self.SENTENCE_END_PATTERNS:
            if re.search(pattern, text):
                return True
        
        return False
    
    def check_continuation(self, text: str) -> bool:
        """추가 발화 암시 체크 (아직 끝나지 않음)"""
        if not text:
            return False
        
        text = text.strip()
        
        for pattern in self.CONTINUATION_PATTERNS:
            if re.search(pattern, text):
                return True
        
        return False
    
    def process(
        self, 
        audio_chunk: np.ndarray, 
        stt_text: Optional[str] = None
    ) -> EndOfSpeechResult:
        """
        발화 종료 감지 (계층적 침묵)
        
        Returns:
            EndOfSpeechResult: 종료 여부 및 이유
        """
        chunk_duration = len(audio_chunk) / self.sample_rate
        speech_prob, energy = self.check_vad(audio_chunk)
        
        # 10번에 한 번씩 디버그 출력
        self.debug_counter += 1
        if self.debug_counter % 10 == 0:
            print(f"   🔊 VAD: energy={energy:.4f}, prob={speech_prob:.2f}, speaking={self.is_speaking}, silence={self.silence_duration:.2f}s, speech_total={self.total_speech_duration:.2f}s")
        
        if speech_prob > 0.5:
            # 음성 감지됨
            self.is_speaking = True
            self.silence_duration = 0.0
            self.total_speech_duration += chunk_duration
            
            return EndOfSpeechResult(
                is_end=False,
                reason='speaking',
                confidence=speech_prob
            )
        else:
            # 침묵 구간
            if self.is_speaking:
                self.silence_duration += chunk_duration
                
                # ★ 최소 발화 조건 체크 ★
                # 발화 길이가 충분하면 유효한 발화로 판단
                has_min_speech_length = self.total_speech_duration >= MIN_SPEECH_LENGTH_SEC
                
                # STT 결과가 있으면 더 정확한 판단 가능
                has_stt_text = stt_text and len(stt_text.strip()) >= 2
                
                # 발화 길이가 너무 짧으면 종료 판단하지 않음
                if not has_min_speech_length:
                    if self.silence_duration >= self.silence_long * 2:
                        # 너무 오래 침묵하면 리셋
                        self.reset()
                    return EndOfSpeechResult(
                        is_end=False,
                        reason='no_valid_speech',
                        confidence=0.0
                    )
                
                # 추가 발화 암시 패턴 체크
                if has_stt_text and self.check_continuation(stt_text):
                    # "그리고...", "음..." 등 → 더 기다림
                    return EndOfSpeechResult(
                        is_end=False,
                        reason='continuation_expected',
                        confidence=0.3
                    )
                
                # 계층적 침묵 판단
                if self.silence_duration >= self.silence_long:
                    # LONG 침묵 (1.5초+) → 무조건 발화 종료 (STT 결과 없어도)
                    print(f"   ⏱️ LONG 침묵 감지: {self.silence_duration:.2f}s >= {self.silence_long}s → 발화 종료!")
                    return EndOfSpeechResult(
                        is_end=True,
                        reason='silence_long',
                        confidence=0.95
                    )
                    
                elif self.silence_duration >= self.silence_medium:
                    # MEDIUM 침묵 (0.6초+) → 문장 완결성 체크 (STT 결과 있을 때만)
                    if has_stt_text and self.check_sentence_complete(stt_text):
                        # 문장이 완결된 것 같고, 충분한 침묵
                        print(f"   ⏱️ MEDIUM 침묵 + 문장 완결: {self.silence_duration:.2f}s, text='{stt_text[:30]}...'")
                        return EndOfSpeechResult(
                            is_end=True,
                            reason='sentence_complete',
                            confidence=0.85
                        )
                    else:
                        # 문장 미완결 또는 STT 미완료 → 더 기다림
                        return EndOfSpeechResult(
                            is_end=False,
                            reason='waiting_sentence',
                            confidence=0.5
                        )
                
                # SHORT 침묵 → 무시
                return EndOfSpeechResult(
                    is_end=False,
                    reason='short_silence',
                    confidence=0.2
                )
            else:
                # 아직 발화 시작 안됨
                return EndOfSpeechResult(
                    is_end=False,
                    reason='no_speech_yet',
                    confidence=0.0
                )

# =================================================================
# 유틸리티 함수
# =================================================================
def safe_str(value, max_len: int = None) -> str:
    if pd.isna(value) or value is None:
        return ""
    s = str(value).strip()
    return s[:max_len] if max_len and len(s) > max_len else s


def clean_response(text: str) -> str:
    """응답을 TTS 친화적인 구어체로 변환"""
    # 중국어 제거
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    # 일본어 히라가나/가타카나 제거
    text = re.sub(r'[\u3040-\u309f]', '', text)
    text = re.sub(r'[\u30a0-\u30ff]', '', text)
    
    # 뜬금없는 문장/표현 제거
    irrelevant_phrases = [
        "잘 가셨습니다", "어디서 오셨죠", "어떻게 해주시겠어요",
        "어디서 오셨나요", "어디로 가시겠어요", "어디서 오셨어요",
        "도움이 필요하시면 언제든지 물어보세요", "언제든지 물어보세요",
        "궁금하신 점이 있으면", "필요하신 경우",
        "자세한 절차는 안내해 드릴게요", "안내해 드릴게요", "설명해 드릴게요",
        "자세한 내용은 안내해 드릴게요"
    ]
    for phrase in irrelevant_phrases:
        text = text.replace(phrase, "")
    
    # 창구/대면 관련 표현 제거 (전화 상담이므로)
    face_to_face_phrases = [
        "여기 창구에서", "여기서", "저기서", "저기 무인발급기에서",
        "바로 처리해드릴게요", "바로 해드릴게요", "바로 발급해드릴게요",
        "해드릴 수 있어요", "해드릴게요", "처리해드릴게요"
    ]
    for phrase in face_to_face_phrases:
        text = text.replace(phrase, "")
    
    # === 마크다운 형식 제거 ===
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 볼드
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # 이탤릭
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)  # 불릿
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 번호
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # 헤더
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # 코드블록
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 링크
    
    # === TTS 친화적 변환 ===
    # 괄호 안 내용 처리: (내용) → ", 내용," 또는 제거
    # 짧은 괄호 내용은 자연스럽게 이어붙임
    text = re.sub(r'\(([^)]{1,15})\)', r', \1,', text)  # 짧은 괄호
    text = re.sub(r'\([^)]+\)', '', text)  # 긴 괄호는 제거

    
    # 콜론 → 자연스러운 연결
    text = re.sub(r':\s*', '은 ', text)
    
    # 슬래시 → "또는"
    text = re.sub(r'\s*/\s*', ' 또는 ', text)
    
    # 숫자 내 쉼표 제거 (5,000 → 5000, TTS가 알아서 읽음)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)  # 백만 단위도
    
    # 특수 단위 변환
    text = re.sub(r'(\d+)\s*[xX×]\s*(\d+)', r'\1 곱하기 \2', text)  # 곱하기
    text = re.sub(r'(\d+)\s*~\s*(\d+)', r'\1에서 \2', text)  # 범위
    text = text.replace('pixel', '픽셀')
    text = text.replace('mm', '밀리미터')
    text = text.replace('cm', '센티미터')
    text = text.replace('kg', '킬로그램')
    text = text.replace('km', '킬로미터')
    text = text.replace('m2', '제곱미터')
    text = text.replace('㎡', '제곱미터')
    
    # 기호 제거/변환
    text = text.replace('&', '와 ')
    text = text.replace('+', ' 플러스 ')
    text = text.replace('=', '는 ')
    text = text.replace('%', '퍼센트')
    text = text.replace('@', '골뱅이')
    text = text.replace('#', '')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace(';', ',')
    
    # 줄바꿈을 공백으로
    text = re.sub(r'\n+', ' ', text)
    
    # 연속된 쉼표, 공백, 마침표 정리
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r'\.\s*,', '.', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',\s*$', '', text)  # 문장 끝 쉼표 제거
    
    return text.strip()

# =================================================================
# ThreeStepRAG 클래스 (통합 버전)
# =================================================================
class ThreeStepRAG:
    """대화 히스토리를 기억하는 RAG 시스템"""
    
    THRESHOLDS = {
        "category": 0.8,
        "provide": 0.78,
        "target": 0.83,
    }
    
    # 고정 응답 텍스트
    GREETING = "안녕하세요. 정부24 인공지능 상담원, 미나입니다. 무엇을 도와드릴까요?"
    GOODBYE = "네, 감사합니다. 좋은 하루 되세요!"
    NOT_FOUND = "죄송해요. 제가 안내해드릴 수 있는 민원이 아니에요. 정부24 사이트나 정부 민원콜센터 110번으로 문의해보시면 정확한 안내를 받으실 수 있어요!"
    
    def __init__(
        self,
        model_name: str = MODEL_ID["LLM"],
        embedding_model: str = MODEL_ID["EMBEDDING"],
        rag_excel_path: str = RAG_EXCEL_PATH,
        gpu_id: str = "3",
        max_history: int = 6,
        seed: int = 42
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.rag_excel_path = rag_excel_path
        self.max_history = max_history
        self.seed = seed
        
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = f"cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.llm = None
        self.tokenizer = None
        self.encoder = None
        self.rag_data = None
        self.svc_emb = None
        
        self.conversation_history: List[Dict[str, str]] = []
        self.last_context: Optional[str] = None
        self.last_service: Optional[str] = None
        self.last_item: Optional[Dict] = None
        
        self._loaded = False
    
    def load(self):
        """모델 및 데이터 로드"""
        if self._loaded:
            return
        
        print("=" * 60)
        print("ThreeStepRAG 로드 중...")
        print("=" * 60)
        
        print(f"\n[1/3] RAG 데이터 로드: {self.rag_excel_path}")
        self._load_rag_data()
        print(f"  → {len(self.rag_data)}개 서비스 로드 완료")
        
        print(f"\n[2/3] 임베딩 모델 로드: {self.embedding_model_name}")
        self.encoder = SentenceTransformer(self.embedding_model_name)
        if torch.cuda.is_available():
            self.encoder = self.encoder.to(self.device)
        self._build_embeddings()
        print("  → 임베딩 생성 완료")
        
        print(f"\n[3/3] LLM 로드: {self.model_name}")
        self._load_llm()
        print("  → LLM 로드 완료")
        
        self._loaded = True
        print("\n" + "=" * 60)
        print("RAG 시스템 로드 완료!")
        print("=" * 60)
    
    def _load_rag_data(self):
        df = pd.read_excel(self.rag_excel_path)
        self.rag_data = []
        for _, row in df.iterrows():
            item = {
                "category": safe_str(row.get("주제 대분류")),
                "provide": safe_str(row.get("제공")),
                "target": safe_str(row.get("대상")),
                "name": safe_str(row.get("서비스명")),
                "how_to_apply": safe_str(row.get("신청방법")),
                "eligibility": safe_str(row.get("신청자격")),
                "processing_time": safe_str(row.get("처리기간")),
                "application_form": safe_str(row.get("신청서")),
                "documents": safe_str(row.get("구비서류")),
                "fee": safe_str(row.get("수수료")),
                "basic_info": safe_str(row.get("기본정보")),
                "procedure": safe_str(row.get("신청방법 및 절차")),
            }
            self.rag_data.append(item)
    
    def _build_embeddings(self):
        svc_texts = [f"passage: {d['name']} {d.get('basic_info', '')}" for d in self.rag_data]
        self.svc_emb = self.encoder.encode(
            svc_texts, normalize_embeddings=True, show_progress_bar=False
        ).astype('float32')
    
    def _load_llm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.llm.eval()
    
    def reset_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
        self.last_context = None
        self.last_service = None
        self.last_item = None
        print("\n💬 대화 히스토리가 초기화되었습니다.\n")

    def get_history(self) -> List[Dict[str, str]]:
        """현재 대화 히스토리 반환"""
        return self.conversation_history.copy()
    
    def get_greeting(self) -> str:
        """시작 인사 반환"""
        return self.GREETING
    
    def get_goodbye(self) -> str:
        """종료 인사 반환"""
        return self.GOODBYE
    
    def get_not_found_message(self) -> str:
        """RAG 실패 시 고정 메시지 반환"""
        return self.NOT_FOUND
    
    def show_history(self):
        """대화 히스토리 출력"""
        if not self.conversation_history:
            print("\n📝 대화 히스토리가 비어있습니다.\n")
            return
        
        print("\n" + "=" * 60)
        print("📝 대화 히스토리")
        print("=" * 60)
        for i, msg in enumerate(self.conversation_history):
            role = "👤 손님" if msg["role"] == "user" else "🤖 상담원"
            print(f"{role}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        print("=" * 60 + "\n")

    def _search_in_list(self, query: str, items: List[str]) -> Tuple[Optional[str], float]:
        if not items:
            return None, 0.0
        unique_items = list(set(items))
        item_texts = [f"passage: {item}" for item in unique_items]
        item_embs = self.encoder.encode(item_texts, normalize_embeddings=True).astype('float32')
        query_emb = self.encoder.encode([f"query: {query}"], normalize_embeddings=True).astype('float32')
        scores = np.dot(item_embs, query_emb.T).flatten()
        best_idx = np.argmax(scores)
        return unique_items[best_idx], float(scores[best_idx])
    
    def _search_hierarchical(self, query: str) -> Tuple[Optional[str], Dict, Optional[Dict]]:
        """
        3-Step 계층적 검색
        
        Returns:
            (context, search_result, item)
            - context: 검색된 서비스 컨텍스트 (실패 시 None)
            - search_result: 검색 과정 상세 정보
            - item: 검색된 서비스 원본 데이터 (실패 시 None)
        """
        result = {
            "query": query, 
            "steps": [], 
            "final_result": None, 
            "selected_service": None,
            "failed_at": None,
            "failed_reason": None
        }
        current_indices = list(range(len(self.rag_data)))
        
        # Step 1: 주제 대분류
        categories = list(set(self.rag_data[i]["category"] for i in current_indices if self.rag_data[i]["category"]))
        best_cat, cat_score = self._search_in_list(query, categories)
        
        result["steps"].append({
            "step": 1,
            "name": "주제 대분류",
            "matched": best_cat,
            "score": round(cat_score, 3),
            "threshold": self.THRESHOLDS["category"],
            "passed": cat_score >= self.THRESHOLDS["category"]
        })
        
        if cat_score < self.THRESHOLDS["category"]:
            result["final_result"] = "failed"
            result["failed_at"] = 1
            result["failed_reason"] = f"주제 대분류 실패: {best_cat}({cat_score:.3f}) < {self.THRESHOLDS['category']}"
            return None, result, None
        current_indices = [i for i in current_indices if self.rag_data[i]["category"] == best_cat]
        
        # Step 2: 제공
        provides = list(set(self.rag_data[i]["provide"] for i in current_indices if self.rag_data[i]["provide"]))
        best_prov, prov_score = self._search_in_list(query, provides)
        
        result["steps"].append({
            "step": 2,
            "name": "제공",
            "matched": best_prov,
            "score": round(prov_score, 3),
            "threshold": self.THRESHOLDS["provide"],
            "passed": prov_score >= self.THRESHOLDS["provide"]
        })
        
        if prov_score < self.THRESHOLDS["provide"]:
            result["final_result"] = "failed"
            result["failed_at"] = 2
            result["failed_reason"] = f"제공 실패: {best_prov}({prov_score:.3f}) < {self.THRESHOLDS['provide']}"
            return None, result, None
        current_indices = [i for i in current_indices if self.rag_data[i]["provide"] == best_prov]
        
        # Step 3: 대상
        targets = list(set(self.rag_data[i]["target"] for i in current_indices if self.rag_data[i]["target"]))
        best_target, target_score = self._search_in_list(query, targets)
        
        result["steps"].append({
            "step": 3,
            "name": "대상",
            "matched": best_target,
            "score": round(target_score, 3),
            "threshold": self.THRESHOLDS["target"],
            "passed": target_score >= self.THRESHOLDS["target"]
        })
        
        if target_score < self.THRESHOLDS["target"]:
            result["final_result"] = "failed"
            result["failed_at"] = 3
            result["failed_reason"] = f"대상 실패: {best_target}({target_score:.3f}) < {self.THRESHOLDS['target']}"
            return None, result, None
        current_indices = [i for i in current_indices if self.rag_data[i]["target"] == best_target]
        
        if not current_indices:
            result["final_result"] = "failed"
            result["failed_at"] = 3
            result["failed_reason"] = "필터링 후 후보 없음"
            return None, result, None
        
        # 최종 선택
        query_emb = self.encoder.encode([f"query: {query}"], normalize_embeddings=True).astype('float32')
        cand_emb = self.svc_emb[current_indices]
        sims = np.dot(cand_emb, query_emb.T).flatten()
        best_item = self.rag_data[current_indices[np.argmax(sims)]]
        
        result["final_result"] = "success"
        result["selected_service"] = best_item["name"]
        result["final_score"] = round(float(sims[np.argmax(sims)]), 3)
        
        context = self._build_context(best_item, detailed=False)
        return context, result, best_item
        
    def _build_context(self, item: Dict, detailed: bool = False) -> str:
        """
        서비스 정보를 컨텍스트 문자열로 변환
        
        Args:
            item: 서비스 정보 딕셔너리
            detailed: True면 상세 정보 포함
        """
        lines = [f"서비스명: {item['name']}"]
        
        # 구비서류: 없으면 "없음"으로 명시
        if item['documents']:
            lines.append(f"구비서류: {item['documents']}")
        else:
            lines.append("구비서류: 없음 (준비물 필요 없음)")
        
        if item['how_to_apply']:
            lines.append(f"신청방법: {item['how_to_apply']}")
        if item['processing_time']:
            lines.append(f"처리기간: {item['processing_time']}")
        if item['fee']:
            lines.append(f"수수료: {item['fee']}")
        else:
            lines.append("수수료: 무료")
        
        # 상세 모드일 때 추가 정보 포함
        if detailed:
            if item.get('procedure'):
                lines.append(f"신청절차: {item['procedure']}")
            if item.get('basic_info'):
                lines.append(f"기본정보: {item['basic_info']}")
            if item.get('eligibility'):
                lines.append(f"신청자격: {item['eligibility']}")
        
        return "\n".join(lines)
    
    def _generate_response(self, user_prompt: str, use_history: bool = True) -> Tuple[str, float]:
        """
        LLM 응답 생성 (대화 히스토리 포함 가능)
        
        Args:
            user_prompt: 현재 사용자 입력 (RAG 컨텍스트 포함)
            use_history: 대화 히스토리 사용 여부
        """
        # 메시지 구성
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # 대화 히스토리 추가 (use_history=True이고 히스토리가 있을 때)
        if use_history and self.conversation_history:
            # 최근 N개 턴만 사용
            recent_history = self.conversation_history[-(self.max_history * 2):]
            messages.extend(recent_history)
        
        # 현재 사용자 입력 추가
        messages.append({"role": "user", "content": user_prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 시드 고정 (재현성 보장)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,  # 응답 길이 제한 (2-3문장)
                do_sample=True,
                temperature=0.5,     # 낮춰서 안정적인 응답
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        latency = time.perf_counter() - start_time
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # thinking 태그 제거
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # 후처리
        response = clean_response(response)
        
        return response, latency
      
    def _is_followup_question(self, query: str) -> bool:
        """후속 질문인지 판단"""
        if not self.last_context or not self.conversation_history:
            return False
        
        followup_keywords = [
            "자세히", "자세한", "더 알려", "더 설명", "그거", "그것", "그게",
            "절차", "방법", "어떻게", "뭐가 필요", "뭘 가져", "비용", "수수료",
            "시간", "얼마나", "언제", "어디서", "어디로", "왜", "구비서류",
            "서류", "신청", "처리", "기간", "준비물"
        ]
        
        query_lower = query.lower()
        has_followup_keyword = any(kw in query_lower for kw in followup_keywords)
        is_short = len(query) <= 15 or len(query.split()) <= 4
        
        return has_followup_keyword or is_short
    
    def _wants_detail(self, query: str) -> bool:
        """상세 정보 요청인지 판단"""
        detail_keywords = [
            "자세히", "자세한", "더 알려", "더 설명", "상세", "구체적",
            "절차", "과정", "단계", "어떻게 해", "뭐가 필요", "전체",
            "안내해", "알려줘", "알려주", "설명해", "말해줘", "말해주"
        ]
        return any(kw in query for kw in detail_keywords)
    
    def _get_question_focus(self, query: str) -> str:
        """
        질문의 초점이 무엇인지 파악
        
        Returns:
            "fee" - 수수료/비용
            "time" - 처리 시간/기간
            "docs" - 서류/준비물
            "method" - 방법/절차
            "where" - 장소
            "spec" - 규격/크기/사이즈 등 상세 스펙
            "general" - 일반
        """
        query_lower = query.lower()
        
        # spec을 먼저 체크 (더 구체적인 것 우선)
        if any(kw in query_lower for kw in ["규격", "크기", "사이즈", "사진", "픽셀", "해상도", "용량"]):
            return "spec"
        elif any(kw in query_lower for kw in ["수수료", "비용", "얼마", "돈", "요금", "가격"]):
            return "fee"
        elif any(kw in query_lower for kw in ["시간", "기간", "얼마나 걸", "며칠", "언제"]):
            return "time"
        elif any(kw in query_lower for kw in ["서류", "준비물", "뭐 가져", "뭘 가져", "필요한 거", "구비"]):
            return "docs"
        elif any(kw in query_lower for kw in ["어떻게", "방법", "절차", "과정", "신청", "안내", "알려"]):
            return "method"
        elif any(kw in query_lower for kw in ["어디", "장소", "위치", "어디서"]):
            return "where"
        else:
            return "general"
    
    def _is_goodbye(self, query: str) -> bool:
        """
        통화 종료 의사인지 판단
        """
        goodbye_keywords = [
            "고마워", "고맙습니다", "감사합니다", "감사해요", "땡큐", "thanks",
            "끊을게", "끊을께", "끊겠습니다", "끊어요",
            "들어갈게", "들어갈께", "들어가볼게",
            "안녕히", "안녕", "바이", "bye",
            "수고하세요", "수고해요", "수고",
            "이만", "그만", "됐어", "됐습니다", "괜찮아요",
            "알겠어", "알겠습니다", "알았어", "네 알겠어요",
            "좋아요 끝", "끝", "종료"
        ]
        query_lower = query.lower().strip()
        
        # 정확히 일치하거나 포함하는 경우
        for kw in goodbye_keywords:
            if kw in query_lower:
                return True
        
        # 짧은 감사 인사 (10자 이하)
        if len(query) <= 10 and any(kw in query_lower for kw in ["고마", "감사", "네", "응", "알겠"]):
            return True
        
        return False
    
    def _is_casual_chat(self, query: str) -> bool:
        """
        일상 대화인지 판단 (인사, 날씨, 범위 밖 질문 등)
        """
        query_lower = query.lower().strip()
        
        # 인사말
        greetings = [
            "안녕", "하이", "헬로", "hello", "hi", 
            "반갑", "좋은 아침", "좋은 저녁"
        ]
        
        # 범위 밖 일상 질문
        casual_topics = [
            "날씨", "맛집", "추천", "로또", "운세", "재밌", "심심",
            "뭐해", "뭘해", "어때", "기분", "잘 지내", "뭐 좀",
            "점심", "저녁", "아침", "커피", "배고파", "졸려"
        ]
        
        # 짧은 인사 (5자 이하)
        if len(query) <= 5 and any(g in query_lower for g in ["안녕", "네", "응", "예", "아니"]):
            return True
        
        # 인사말 포함
        if any(g in query_lower for g in greetings):
            return True
        
        # 일상 토픽 포함
        if any(t in query_lower for t in casual_topics):
            return True
        
        return False
    
    def chat(self, query: str, verbose: bool = False) -> Tuple[str, Optional[str], Dict]:
        """
        대화 처리 (히스토리 자동 저장)
        
        Args:
            query: 사용자 질문
            verbose: 검색 과정 출력 여부
            
        Returns:
            (응답 문자열, 참고 서비스명 또는 None, 검색 결과 상세)
        """
        if not self._loaded:
            raise RuntimeError("먼저 load()를 호출하세요.")
        
        # 통화 종료 감지
        if self._is_goodbye(query):
            # 히스토리 초기화
            self.conversation_history = []
            self.last_context = None
            self.last_service = None
            self.last_item = None
            
            search_result = {
                "query": query,
                "final_result": "goodbye",
                "selected_service": None,
                "steps": []
            }
            
            if verbose:
                print("   ↳ 통화 종료 감지, 히스토리 초기화")
            
            return self.GOODBYE, None, search_result
        
        # RAG 검색
        context, search_result, item = self._search_hierarchical(query)
        
        # 참고한 서비스명 추출
        referenced_service = search_result.get("selected_service", None)
        
        # 상세 정보 요청 여부
        wants_detail = self._wants_detail(query)
        
        # 질문 초점 파악
        question_focus = self._get_question_focus(query)
        
        # 후속 질문 여부 플래그
        is_followup = False
        
        # RAG 검색 실패 시, 후속 질문인지 확인하고 이전 컨텍스트 재사용
        if context is None and self._is_followup_question(query):
            referenced_service = self.last_service
            item = self.last_item
            is_followup = True
            
            # 상세 요청이면 상세 컨텍스트 생성
            if item:
                context = self._build_context(item, detailed=True)  # 후속 질문은 항상 상세 컨텍스트
                # 후속 질문으로 성공 처리
                search_result["final_result"] = "success (followup)"
                search_result["selected_service"] = referenced_service
            else:
                context = self.last_context
                
            if verbose:
                print(f"   ↳ 후속 질문 감지, 이전 컨텍스트 재사용 (초점={question_focus})")
        
        # 프롬프트 생성
        if context is None:
            # 일상 대화인 경우 → LLM이 응답
            if self._is_casual_chat(query):
                user_prompt = f"""손님이 "{query}"라고 말씀하셨어요.

인사나 일상 대화입니다. 친근하게 응대하고, 민원 관련 도움이 필요하신지 여쭤보세요. 1~2문장으로 짧게 답하세요."""
                
                # LLM 응답 생성
                response, latency = self._generate_response(user_prompt, use_history=True)
                
                # 대화 히스토리에 저장
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": response})
                
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-(self.max_history * 2):]
                
                return response, None, search_result
            
            # 민원 관련인데 RAG에서 못 찾은 경우 → 고정 텍스트
            else:
                # 대화 히스토리에 저장
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": self.NOT_FOUND})
                
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-(self.max_history * 2):]
                
                search_result["response_type"] = "fixed_not_found"
                
                return self.NOT_FOUND, None, search_result
        else:
            # 검색 성공 시 이전 결과로 저장
            self.last_context = context
            self.last_service = referenced_service
            self.last_item = item
            
            # 후속 질문이면서 상세 요청일 때 → 상세하게 답변
            if is_followup and wants_detail:
                user_prompt = f"""[참고할 민원 정보]
{context}

손님이 추가로 "{query}"라고 물어보셨어요.

손님이 자세한 절차/정보를 원하고 있어요.

[중요] 위 정보의 "구비서류"를 정확히 확인하세요.
- 구비서류가 "없음"이면: "준비물 없이" 또는 "별도 서류 없이"라고 안내
- 구비서류가 있으면: 해당 서류를 안내

위 정보를 바탕으로 신청 절차, 필요 서류, 처리 방법 등을 상세하게 설명해주세요. 
정부24 사이트나 정부24 앱에서도 신청할 수 있다고 안내해주세요.
이전에 안내한 내용과 중복되더라도 괜찮습니다. 단, 정보에 없는 내용은 지어내지 마세요."""
            
            # 후속 질문 (간단한 추가 질문) → 해당 내용만 짧게
            elif is_followup:
                focus_instructions = {
                    "fee": "수수료/비용에 대해서만 답하세요.",
                    "time": "처리 시간/기간에 대해서만 답하세요.",
                    "docs": "구비서류를 확인하세요. '없음'이면 '준비물 없이'라고, 있으면 해당 서류를 안내하세요.",
                    "method": "신청 방법/절차에 대해서만 답하세요. 정부24 사이트나 앱에서도 가능하다고 안내하세요.",
                    "where": "신청 장소에 대해서만 답하세요. 정부24 사이트나 앱에서도 가능하다고 안내하세요.",
                    "spec": "사진 규격, 크기, 사이즈 등 상세 스펙에 대해 정확히 답하세요.",
                    "general": "질문에 해당하는 내용만 답하세요."
                }
                focus_instruction = focus_instructions.get(question_focus, focus_instructions["general"])
                
                user_prompt = f"""[참고할 민원 정보]
{context}

손님이 추가로 "{query}"라고 물어보셨어요.

[중요] {focus_instruction} 이전에 안내한 내용을 반복하지 마세요. 1~2문장으로 짧게 답하세요."""
            
            # 상세 요청일 때
            elif wants_detail:
                user_prompt = f"""[참고할 민원 정보]
{context}

손님이 "{query}"라고 말씀하셨어요.

손님이 자세한 정보를 원하고 있어요. 

[중요] 위 정보의 "구비서류"를 정확히 확인하세요.
- 구비서류가 "없음"이면: "준비물 없이" 또는 "별도 서류 없이"라고 안내
- 구비서류가 있으면: 해당 서류를 안내

위 정보를 바탕으로 신청 절차, 필요 서류, 처리 방법 등을 상세하게 설명해주세요.
정부24 사이트나 정부24 앱에서도 신청할 수 있다고 안내해주세요.
단, 정보에 없는 내용은 지어내지 마세요."""
            
            # 일반 질문
            else:
                user_prompt = f"""[참고할 민원 정보]
{context}

손님이 "{query}"라고 말씀하셨어요.

[중요] 위 정보의 "구비서류"를 정확히 확인하세요.
- 구비서류가 "없음"이면: "준비물 없이" 또는 "별도 서류 없이"라고 안내
- 구비서류가 있으면: 해당 서류를 안내

위 정보를 바탕으로 어디서 신청하면 되는지 2~3문장으로 간단히 안내해주세요.
정부24 사이트나 앱에서도 신청 가능하다고 안내해주세요."""
        
        if verbose:
            print(f"\n🔍 RAG 검색 결과: {referenced_service or '검색 실패'}")
            print(f"   상세 요청: {wants_detail}")
        
        # 응답 생성
        response, latency = self._generate_response(user_prompt, use_history=True)
        
        # 대화 히스토리에 저장 (원본 질문과 응답)
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # 히스토리 길이 제한
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
        
        if verbose:
            print(f"⏱️ 응답 시간: {latency:.2f}초")
        
        return response, referenced_service, search_result
    
    def run_interactive(self):
        """인터랙티브 대화 모드 실행"""
        if not self._loaded:
            self.load()
        
        print("\n" + "=" * 60)
        print("🎙️ 민원 안내 상담을 시작합니다")
        print("=" * 60)
        print("명령어:")
        print("  /reset, /clear  - 대화 초기화")
        print("  /history        - 대화 히스토리 보기")
        print("  /quit, /exit    - 종료")
        print("=" * 60)
        
        # 시작 인사 출력
        print(f"\n🤖 상담원: {self.GREETING}\n")
        
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 손님: ").strip()
                
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("\n👋 상담을 종료합니다. 좋은 하루 되세요!")
                    break
                
                if user_input.lower() in ['/reset', '/clear']:
                    self.reset_history()
                    print(f"🤖 상담원: {self.GREETING}\n")  # 초기화 후 다시 인사
                    continue
                
                if user_input.lower() == '/history':
                    self.show_history()
                    continue
                
                # 일반 대화 처리
                response, ref_service, search_result = self.chat(user_input, verbose=False)
                print(f"🤖 상담원: {response}")
                
                # 통화 종료인 경우 → 시작 인사 다시 출력
                if search_result.get("final_result") == "goodbye":
                    print("   🔄 대화 히스토리가 초기화되었습니다. (통화 종료)\n")
                    print(f"🤖 상담원: {self.GREETING}\n")
                    continue
                
                # 고정 응답인 경우 (민원인데 RAG 실패)
                if search_result.get("response_type") == "fixed_not_found":
                    print(f"   📎 참고 서비스: 없음 (고정 응답)")
                    if search_result.get("failed_at"):
                        print(f"   ❌ 실패 단계: Step {search_result['failed_at']}")
                        for step in search_result.get("steps", []):
                            status = "✅" if step["passed"] else "❌"
                            print(f"      {status} Step {step['step']} [{step['name']}]: {step['matched']} ({step['score']:.3f} / {step['threshold']})")
                    print()
                    continue
                
                # 참고 서비스 및 검색 결과 출력
                if ref_service:
                    print(f"   📎 참고 서비스: {ref_service}")
                else:
                    print(f"   📎 참고 서비스: 없음")
                    # 실패 원인 출력
                    if search_result.get("failed_at"):
                        print(f"   ❌ 실패 단계: Step {search_result['failed_at']}")
                        print(f"   ❌ 실패 원인: {search_result.get('failed_reason', '알 수 없음')}")
                        # 각 단계별 상세 정보
                        for step in search_result.get("steps", []):
                            status = "✅" if step["passed"] else "❌"
                            print(f"      {status} Step {step['step']} [{step['name']}]: {step['matched']} ({step['score']:.3f} / {step['threshold']})")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 상담을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")



# =================================================================
# 통합 음성 파이프라인
# =================================================================
class VoiceRAGPipeline:
    """ASR → RAG → TTS 통합 파이프라인 (VAD 포함)"""
    
    def __init__(self):
        self.stt_pipe = None
        self.rag = None
        self.tts_model = None
        self._loaded = False
        
        # VAD 관련
        self.eos_detector = EndOfSpeechDetector()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_processing = False
        self.websocket = None
        self.partial_text = ""
        self.last_transcribe_len = 0  # 마지막 STT 호출 시 버퍼 길이
    
    def load_models(self):
        """모든 모델 로드"""
        if self._loaded:
            return
        
        print("\n" + "=" * 70)
        print("🎤 통합 음성 RAG 시스템 초기화 중...")
        print("=" * 70)
        
        # STT 모델 로드
        print("\n[1/3] STT 모델 로드 중...")
        start_time = time.time()
        stt_model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID['STT'],
            torch_dtype=torch.float16,
            device_map=STT_DEVICE,
            local_files_only=True
        )
        stt_processor = WhisperProcessor.from_pretrained(
            MODEL_ID['STT'],
            local_files_only=True
        )
        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=stt_model,
            tokenizer=stt_processor.tokenizer,
            feature_extractor=stt_processor.feature_extractor,
            dtype=torch.bfloat16,
        )
        print(f"  ✓ STT 모델 로드 완료 ({time.time() - start_time:.2f}초)")
        
        # RAG 모델 로드
        print("\n[2/3] RAG 모델 로드 중...")
        start_time = time.time()
        self.rag = ThreeStepRAG(gpu_id=RAG_DEVICE.split(':')[-1])
        self.rag.load()
        print(f"  ✓ RAG 모델 로드 완료 ({time.time() - start_time:.2f}초)")
        
        # TTS 모델 로드
        print("\n[3/3] TTS 모델 로드 중...")
        start_time = time.time()
        self.tts_model = Qwen3TTSModel.from_pretrained(
            MODEL_ID['TTS'],
            device_map=TTS_DEVICE,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(f"  ✓ TTS 모델 로드 완료 ({time.time() - start_time:.2f}초)")
        
        self._loaded = True
        print("\n" + "=" * 70)
        print("✅ 모든 모델 로드 완료! 시스템 준비 완료")
        print("=" * 70 + "\n")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """음성을 텍스트로 변환"""
        if not self._loaded:
            raise RuntimeError("먼저 load_models()를 호출하세요.")
        
        result = self.stt_pipe(
            {"array": audio_data, "sampling_rate": sample_rate},
            generate_kwargs={"language": "korean", "task": "transcribe"}
        )
        return result["text"].strip()
    
    def get_rag_response(self, text: str) -> Tuple[str, Optional[str]]:
        """RAG를 통해 응답 생성"""
        if not self._loaded:
            raise RuntimeError("먼저 load_models()를 호출하세요.")
        
        return self.rag.chat(text)
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> Tuple[List[np.ndarray], int]:
        """텍스트를 음성으로 변환"""
        if not self._loaded:
            raise RuntimeError("먼저 load_models()를 호출하세요.")
        
        sentences = re.split(r'(?<=[.?!])\s+', text)
        # 빈 문자열 제거 및 공백 정리
        sentences = [(s.strip()+" , ") for s in sentences if s.strip()]

        if not sentences:
            sentences = [text]

        batch_size = len(sentences)
        
        audio, sr = self.tts_model.generate_custom_voice(
            text= sentences,
            speaker=["Sohee"] * batch_size,
            language=["Korean"] * batch_size,
            instruct=[VOICE_INSTRUCT] * batch_size
        )
        
        return audio, sr
    
    # 추임새
    def _first_response(self, input_file='pre_sound.csv'):
        input_df = pd.read_csv(input_file)
        input_df = input_df[input_df['where'] == 'before'].reset_index(drop=True).copy()
        N = len(input_df)
        rand_index = random.randint(0, N-1)
        rand_row = input_df.iloc[rand_index]
        return rand_row['file'], rand_row['transcript']
    
    # 음성 파일 재생 -> 브라우저가 있어야 실행 가능
    async def _play_wav(self, event, file_path, text):
        """
        직접 재생 대신, 클라이언트(브라우저)에게 재생 명령을 보냅니다.
        """
        # 1. 클라이언트에게 "이 파일 재생해!"라고 메시지 전송
        # file_path는 'audio/filename.wav' 형태여야 웹에서 접근 가능합니다.
        await self.websocket.send_json({
            "type": "PLAY_AUDIO",
            "url": f"/{file_path}",
            "text": text 
        })

        print(f"추임새 생성: {text}")

        # 2. 클라이언트로부터 "재생 끝났어"라는 응답을 받을 때까지 대기
        # (또는 기존처럼 duration만큼 sleep 처리 가능)
        data, fs = sf.read(file_path)
        duration = len(data) / fs
        
        print(f"🔊 클라이언트에서 재생 중... ({duration:.2f}초)")
        await asyncio.sleep(duration) 
        
        event.set() # 재생 완료 신호
    
    # Rag 기반 LLM 답변 생성
    async def _rag_llm_process(self, event, transcribed_text):
        print("🤖 응답 생성 중...")
        start_time = time.time()
        response_text, ref_service, search_result = self.get_rag_response(transcribed_text)
        rag_time = time.time() - start_time

        await event.wait()
        return rag_time, response_text, ref_service, search_result

    # 병렬화 시작
    async def play_rag_parrallel(self, file_path, filter_text, transcribed_text):
        event = asyncio.Event()
        _, results = await asyncio.gather(
            # 음성 병렬화
            self._play_wav(event, file_path, filter_text),
            # rag 추론 병렬화
            self._rag_llm_process(event, transcribed_text)
        )
        return results  # (rag_time, response_text, ref_service, search_result)

    async def process_voice(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        output_audio_path: Optional[str] = None
    ) -> Dict:
        """전체 파이프라인 실행: 음성 → 텍스트 → RAG → 음성"""
        if not self._loaded:
            raise RuntimeError("먼저 load_models()를 호출하세요.")
        
        # 1. STT
        print("🎤 음성 인식 중...")
        start_time = time.time()
        transcribed_text = self.transcribe(audio_data, sample_rate)
        stt_time = time.time() - start_time
        print(f"  ✓ 인식 완료: '{transcribed_text}' ({stt_time:.2f}초)")

        # 2-A. 추임새 생성 및 재생
        first_response_path, first_response_text = self._first_response()
        print(f"  ✓ 응답 생성 완료 (추임새)")
        print(f"  📝 응답: {first_response_text}")

        rag_time, response_text, ref_service, search_result = await self.play_rag_parrallel(
            file_path=first_response_path, 
            filter_text=first_response_text,
            transcribed_text=transcribed_text,
        )

        # 2-C. RAG & LLM
        rag_time = time.time() - start_time
        print(f"  ✓ 응답 생성 완료 ({rag_time:.2f}초)")
        print(f"  📝 응답: {response_text}")
        print(f"  ℹ️ 참고 서비스: {ref_service}")

        # 3. 고정 오디오 파일 사용 여부 확인
        fixed_audio_path = None
        
        # GOODBYE 상황
        if search_result.get("final_result") == "goodbye":
            fixed_audio_path = FIXED_AUDIO["GOODBYE"]
            print(f"  🔊 고정 오디오 사용 (GOODBYE): {fixed_audio_path}")
        
        # NOT_FOUND 상황
        elif search_result.get("response_type") == "fixed_not_found":
            fixed_audio_path = FIXED_AUDIO["NOT_FOUND"]
            print(f"  🔊 고정 오디오 사용 (NOT_FOUND): {fixed_audio_path}")
        
        # 고정 오디오 파일이 있으면 해당 파일 로드
        if fixed_audio_path and os.path.exists(fixed_audio_path):
            print("🔊 고정 오디오 파일 로드 중...")
            start_time = time.time()
            audio_data_fixed, sr = sf.read(fixed_audio_path)
            audio_chunks = [audio_data_fixed]
            tts_time = time.time() - start_time
            print(f"  ✓ 고정 오디오 로드 완료 ({tts_time:.2f}초)")
        else:
            # 4. TTS 합성
            print("🔊 음성 합성 중...")
            start_time = time.time()
            audio_chunks, sr = self.synthesize(response_text)
            tts_time = time.time() - start_time
            print(f"  ✓ 음성 합성 완료 ({tts_time:.2f}초)")
        
        total_time = stt_time + rag_time + tts_time
        print(f"\n⏱️ 총 처리 시간: {total_time:.2f}초")
        
        return {
            "transcribed_text": transcribed_text,
            "response_text": response_text,
            "referenced_service": ref_service,
            "audio_chunks": audio_chunks,
            "sr": sr,
            "search_result": search_result,  # 검색 결과도 포함
            "timings": {
                "stt": stt_time,
                "rag": rag_time,
                "tts": tts_time,
                "total": total_time
            }
        }
    
    def reset_conversation(self):
        """대화 히스토리 초기화"""
        if self.rag:
            self.rag.reset_history()
    
    def reset_vad(self):
        """VAD 상태 및 오디오 버퍼 초기화"""
        self.eos_detector.reset()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.partial_text = ""
        self.last_transcribe_len = 0
        print("🔄 VAD 초기화 완료")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """오디오 청크를 버퍼에 추가"""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
    
    def get_buffered_audio(self) -> np.ndarray:
        """버퍼에 저장된 전체 오디오 반환"""
        return self.audio_buffer.copy()
    
    def transcribe_buffer(self) -> str:
        """버퍼에 있는 오디오를 텍스트로 변환 (중간 결과)"""
        if len(self.audio_buffer) < SAMPLE_RATE * 0.5:  # 최소 0.5초
            return ""
        
        try:
            result = self.stt_pipe(
                {"array": self.audio_buffer, "sampling_rate": SAMPLE_RATE},
                generate_kwargs={"language": "korean", "task": "transcribe"}
            )
            return result["text"].strip()
        except Exception as e:
            print(f"STT 오류: {e}")
            return ""
    
    def process_audio_chunk_vad(self, audio_chunk: np.ndarray) -> Tuple[EndOfSpeechResult, str]:
        """
        오디오 청크 처리 및 VAD 수행
        
        Returns:
            (EndOfSpeechResult, partial_text): 발화 종료 결과 및 현재까지의 인식 텍스트
        """
        # 버퍼에 청크 추가
        self.add_audio_chunk(audio_chunk)
        
        current_len = len(self.audio_buffer)
        
        # STT는 1초 간격으로만 호출 (너무 자주 호출하면 느려짐)
        # 최소 0.5초 이상이고, 마지막 호출 이후 1초 이상 지났을 때만 호출
        if current_len >= SAMPLE_RATE * 0.5 and (current_len - self.last_transcribe_len) >= SAMPLE_RATE * 1.0:
            self.partial_text = self.transcribe_buffer()
            self.last_transcribe_len = current_len
            print(f"   📝 중간 인식: '{self.partial_text[:50]}...' (버퍼: {current_len/SAMPLE_RATE:.1f}초)")
        
        # VAD 처리
        eos_result = self.eos_detector.process(audio_chunk, self.partial_text)
        
        return eos_result, self.partial_text


# =================================================================
# FastAPI 웹 애플리케이션
# =================================================================
app = FastAPI(title="음성 RAG 시스템")

# CORS 설정 (필요 시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

vrg_pipeline = VoiceRAGPipeline()


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    vrg_pipeline.load_models()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """메인 페이지"""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎤 음성 RAG 민원 안내</title>
    <style>
        :root { --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%); --bg-color: #f0f2f5; --chat-bg: #ffffff; --user-msg-bg: #e3f2fd; --bot-msg-bg: #f8f9fa; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Pretendard', sans-serif; background: var(--bg-color); height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { width: 100%; max-width: 600px; height: 90vh; background: var(--chat-bg); border-radius: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); display: flex; flex-direction: column; overflow: hidden; }
        .header { padding: 20px; background: var(--primary-gradient); color: white; text-align: center; }
        .chat-area { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; scroll-behavior: smooth; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 16px; font-size: 0.95rem; line-height: 1.5; animation: fadeIn 0.3s ease; position: relative; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: 0; } }
        .message.user { align-self: flex-end; background: var(--user-msg-bg); color: #1565c0; border-bottom-right-radius: 4px; }
        .message.assistant { align-self: flex-start; background: var(--bot-msg-bg); color: #333; border-bottom-left-radius: 4px; border: 1px solid #eee; }
        
        /* 오디오 숨김 */
        audio { display: none; }

        .controls { padding: 20px; background: white; border-top: 1px solid #eee; display: flex; gap: 10px; justify-content: center; align-items: center; }
        .btn { border: none; border-radius: 50px; padding: 12px 24px; font-size: 1rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; }
        .btn-record { background: var(--primary-gradient); color: white; width: 140px; justify-content: center; }
        .btn-record.recording { background: #ff5252; animation: pulse 1.5s infinite; }
        .btn-reset { background: #f1f3f5; color: #495057; padding: 12px; border-radius: 50%; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ 정부24 민원 안내: 미나</h1>
            <div id="status" style="font-size: 0.85rem; margin-top: 8px;">연결 대기 중...</div>
        </div>
        <div id="chatBox" class="chat-area">
            <!-- 초기 메시지는 서버에서 GREETING과 함께 전송됨 -->
        </div>
        <div class="controls">
            <button id="resetBtn" class="btn btn-reset">↻</button>
            <button id="recordBtn" class="btn btn-record">🎙️ 대화하기</button>
        </div>
    </div>
    <audio id="audioPlayer"></audio>

    <script>
        const recordBtn = document.getElementById('recordBtn');
        const resetBtn = document.getElementById('resetBtn');
        const statusDiv = document.getElementById('status');
        const chatBox = document.getElementById('chatBox');
        const audioPlayer = document.getElementById('audioPlayer');
        
        let ws;
        let isRecording = false;
        let audioContext = null;
        let mediaStream = null;
        let scriptProcessor = null;
        let sourceNode = null;
        let partialTextDiv = null;

        function connectWebSocket() {
            ws = new WebSocket((window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/ws');
            ws.onopen = () => { statusDiv.textContent = "상담 준비 완료"; recordBtn.disabled = false; };
            ws.onclose = () => setTimeout(connectWebSocket, 3000);
            ws.onmessage = (e) => handleServerMessage(JSON.parse(e.data));
        }

        function handleServerMessage(data) {
            if (data.type === 'STATUS') {
                statusDiv.textContent = data.message;
            }
            else if (data.type === 'GREETING') {
                // 시작 인사 - 텍스트 표시 및 오디오 재생
                addMessage(data.text, 'assistant');
                if (data.audio_base64) {
                    const audioBlob = base64ToBlob(data.audio_base64, 'audio/wav');
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayer.src = audioUrl;
                    audioPlayer.play().catch(console.error);
                }
                statusDiv.textContent = "상담 준비 완료";
            }
            else if (data.type === 'PARTIAL_TEXT') {
                // 중간 인식 결과 표시
                updatePartialText(data.text);
            }
            else if (data.type === 'STOP_RECORDING') {
                // 서버에서 발화 종료 감지 → 자동으로 녹음 중지
                console.log("발화 종료 감지:", data.reason);
                if (isRecording) {
                    stopRecording(false);  // 자동 중지 (서버가 이미 처리 시작)
                    statusDiv.textContent = "처리 중...";
                }
            }
            else if (data.type === 'PLAY_AUDIO') {
                // 추임새 (단일 파일)
                removePartialText();
                if (data.text) addMessage(data.text, 'assistant');
                const audioUrl = `/static/${data.url.replace(/^\\//, '')}`;
                audioPlayer.src = audioUrl;
                audioPlayer.play().catch(console.error);
            }
            else if (data.type === 'FINAL_RESULT') {
                removePartialText();
                if (data.user_text) addMessage(data.user_text, 'user');
                
                // 오디오 리스트가 있으면 순차 재생
                if (data.audio_base64_list && data.audio_base64_list.length > 0) {
                    playAudioSequence(data.bot_text, data.audio_base64_list);
                } else {
                    // 오디오가 없으면 텍스트만 출력
                    addMessage(data.bot_text, 'assistant');
                }
                
                // GOODBYE인 경우 버튼 비활성화
                if (data.is_goodbye) {
                    statusDiv.textContent = "상담 종료 중...";
                    recordBtn.disabled = true;
                } else {
                    statusDiv.textContent = "답변 완료";
                }
            }
            else if (data.type === 'CHAT_END') {
                // 채팅 완전 종료
                statusDiv.textContent = data.message;
                recordBtn.disabled = true;
                recordBtn.textContent = "상담 종료";
                recordBtn.classList.remove('recording');
                recordBtn.style.background = '#ccc';
                recordBtn.style.cursor = 'not-allowed';
                
                // 종료 메시지 추가
                const endDiv = document.createElement('div');
                endDiv.style.cssText = 'text-align: center; padding: 20px; color: #666; font-size: 0.9rem; border-top: 1px solid #eee; margin-top: 20px;';
                endDiv.innerHTML = '🙏 상담이 종료되었습니다.<br>이용해 주셔서 감사합니다.<br><br><small style="color:#999;">새 상담을 원하시면 ↻ 버튼을 눌러주세요.</small>';
                chatBox.appendChild(endDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            else if (data.type === 'ERROR') {
                removePartialText();
                alert(data.message);
            }
        }

        function updatePartialText(text) {
            if (!text) return;
            
            if (!partialTextDiv) {
                partialTextDiv = document.createElement('div');
                partialTextDiv.className = 'partial-text';
                partialTextDiv.style.cssText = 'font-size: 0.85rem; color: #888; font-style: italic; padding: 8px 12px;';
                chatBox.appendChild(partialTextDiv);
            }
            partialTextDiv.textContent = '🎤 ' + text + '...';
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function removePartialText() {
            if (partialTextDiv) {
                partialTextDiv.remove();
                partialTextDiv = null;
            }
        }

        function playAudioSequence(text, b64List) {
            // 텍스트는 한 번만 표시
            addMessage(text, 'assistant');

            let currentIndex = 0;

            function playNext() {
                if (currentIndex >= b64List.length) return;

                const audioBlob = base64ToBlob(b64List[currentIndex], 'audio/wav');
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl); 
                
                audio.onended = function() {
                    currentIndex++;
                    playNext();
                };

                audio.play().catch(e => console.log("Audio play error:", e));
            }

            playNext();
        }

        function addMessage(text, type) {
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.innerHTML = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function base64ToBlob(base64, mime) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) byteNumbers[i] = byteCharacters.charCodeAt(i);
            return new Blob([new Uint8Array(byteNumbers)], {type: mime});
        }

        function floatTo16BitPCM(float32Array) {
            const int16Array = new Int16Array(float32Array.length);
            for (let i = 0; i < float32Array.length; i++) {
                const s = Math.max(-1, Math.min(1, float32Array[i]));
                int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            return int16Array;
        }

        async function startRecording() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { 
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                sourceNode = audioContext.createMediaStreamSource(mediaStream);
                
                // ScriptProcessorNode로 실시간 오디오 청크 전송
                // 버퍼 크기: 4096 샘플 = 16000Hz에서 약 0.256초
                scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                
                scriptProcessor.onaudioprocess = (e) => {
                    if (!isRecording || ws.readyState !== WebSocket.OPEN) return;
                    
                    const float32Data = e.inputBuffer.getChannelData(0);
                    const int16Data = floatTo16BitPCM(float32Data);
                    
                    // 오디오 청크를 Base64로 인코딩하여 JSON으로 전송
                    const base64Data = btoa(String.fromCharCode(...new Uint8Array(int16Data.buffer)));
                    ws.send(JSON.stringify({
                        type: 'audio_chunk',
                        data: base64Data
                    }));
                };
                
                sourceNode.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);
                
                isRecording = true;
                recordBtn.textContent = "멈춤 ■";
                recordBtn.classList.add('recording');
                statusDiv.textContent = "듣고 있어요...";
                
                // 녹음 시작 알림
                ws.send(JSON.stringify({ type: 'start_recording' }));
                
            } catch (err) {
                console.error("마이크 오류:", err);
                alert("마이크 권한이 필요합니다.");
            }
        }

        function stopRecording(sendEndSignal = true) {
            if (!isRecording) return;
            
            isRecording = false;
            
            // 오디오 리소스 정리
            if (scriptProcessor) {
                scriptProcessor.disconnect();
                scriptProcessor = null;
            }
            if (sourceNode) {
                sourceNode.disconnect();
                sourceNode = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            // UI 업데이트
            recordBtn.textContent = "🎙️ 대화하기";
            recordBtn.classList.remove('recording');
            
            // 수동 중지 시에만 서버에 알림 (자동 중지는 서버가 이미 처리)
            if (sendEndSignal && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'end_recording' }));
                statusDiv.textContent = "처리 중...";
            }
        }

        recordBtn.onclick = async () => {
            if (!isRecording) {
                await startRecording();
            } else {
                stopRecording(true);  // 수동 중지
            }
        };

        resetBtn.onclick = () => { 
            if (ws.readyState === WebSocket.OPEN) { 
                ws.send(JSON.stringify({ type: 'reset' })); 
                chatBox.innerHTML = ''; 
                removePartialText();
                
                // 버튼 상태 복원 (종료 후 다시 시작할 때)
                recordBtn.disabled = false;
                recordBtn.textContent = "🎙️ 대화하기";
                recordBtn.style.background = '';
                recordBtn.style.cursor = '';
            } 
        };
        
        connectWebSocket();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    vrg_pipeline.websocket = websocket
    vrg_pipeline.reset_vad()  # VAD 초기화
    
    # 연결 시 GREETING 전송
    try:
        greeting_text = ThreeStepRAG.GREETING
        greeting_audio_path = FIXED_AUDIO["GREETING"]
        
        # GREETING 오디오 파일 로드 및 Base64 인코딩
        if os.path.exists(greeting_audio_path):
            audio_data, sr = sf.read(greeting_audio_path)
            out_io = io.BytesIO()
            sf.write(out_io, audio_data, sr, format='WAV')
            out_io.seek(0)
            audio_b64 = base64.b64encode(out_io.read()).decode('utf-8')
            
            await websocket.send_json({
                "type": "GREETING",
                "text": greeting_text,
                "audio_base64": audio_b64
            })
            print(f"🎉 GREETING 전송: {greeting_text}")
        else:
            # 오디오 파일이 없으면 텍스트만 전송
            await websocket.send_json({
                "type": "GREETING",
                "text": greeting_text,
                "audio_base64": None
            })
            print(f"⚠️ GREETING 오디오 파일 없음, 텍스트만 전송: {greeting_text}")
    except Exception as e:
        print(f"GREETING 전송 오류: {e}")
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                text_data = message["text"]
                
                try:
                    json_data = json.loads(text_data)
                    msg_type = json_data.get("type", "")
                    
                    if msg_type == "reset":
                        vrg_pipeline.reset_conversation()
                        vrg_pipeline.reset_vad()
                        
                        # reset 후에도 GREETING 재전송
                        greeting_text = ThreeStepRAG.GREETING
                        greeting_audio_path = FIXED_AUDIO["GREETING"]
                        
                        if os.path.exists(greeting_audio_path):
                            audio_data, sr = sf.read(greeting_audio_path)
                            out_io = io.BytesIO()
                            sf.write(out_io, audio_data, sr, format='WAV')
                            out_io.seek(0)
                            audio_b64 = base64.b64encode(out_io.read()).decode('utf-8')
                            
                            await websocket.send_json({
                                "type": "GREETING",
                                "text": greeting_text,
                                "audio_base64": audio_b64
                            })
                        else:
                            await websocket.send_json({
                                "type": "GREETING",
                                "text": greeting_text,
                                "audio_base64": None
                            })
                        
                        await websocket.send_json({"type": "STATUS", "message": "대화 내용이 초기화되었습니다."})
                    
                    elif msg_type == "start_recording":
                        # 녹음 시작 - VAD 초기화
                        vrg_pipeline.reset_vad()
                        print("🎤 녹음 시작")
                    
                    elif msg_type == "end_recording":
                        # 수동 녹음 종료 - 버퍼에 있는 오디오 처리
                        print("🛑 수동 녹음 종료")
                        await process_buffered_audio(websocket)
                    
                    elif msg_type == "audio_chunk":
                        # 실시간 오디오 청크 처리
                        if vrg_pipeline.is_processing:
                            continue
                        
                        # Base64 디코딩
                        audio_b64 = json_data.get("data", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32768.0
                        
                        # VAD 처리
                        eos_result, partial_text = vrg_pipeline.process_audio_chunk_vad(audio_float)
                        
                        # 중간 텍스트 전송 (변화 있을 때)
                        if partial_text and len(partial_text) > 2:
                            await websocket.send_json({
                                "type": "PARTIAL_TEXT",
                                "text": partial_text
                            })
                        
                        # 발화 종료 감지
                        if eos_result.is_end:
                            print(f"✅ 발화 종료 감지: {eos_result.reason} (confidence={eos_result.confidence:.2f})")
                            
                            # 클라이언트에 녹음 중지 요청
                            await websocket.send_json({
                                "type": "STOP_RECORDING",
                                "reason": eos_result.reason
                            })
                            
                            # 버퍼에 있는 오디오 처리
                            await process_buffered_audio(websocket)
                    
                except json.JSONDecodeError:
                    # 레거시 텍스트 메시지 처리
                    if text_data == "RESET":
                        vrg_pipeline.reset_conversation()
                        vrg_pipeline.reset_vad()
                        await websocket.send_json({"type": "STATUS", "message": "대화 내용이 초기화되었습니다."})
                    
            elif "bytes" in message:
                # 레거시: 전체 오디오 blob (webm) 처리 (이전 방식 호환용)
                audio_bytes = message["bytes"]
                
                await websocket.send_json({"type": "STATUS", "message": "음성을 분석하고 있어요..."})
                
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_path = tmp_file.name
                    
                    audio_data, sample_rate = librosa.load(tmp_path, sr=16000, mono=True)
                    
                except Exception as e:
                    print(f"오디오 로드 실패: {e}")
                    await websocket.send_json({"type": "ERROR", "message": "오디오 형식을 읽을 수 없습니다."})
                    continue
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                # 파이프라인 실행
                result = await vrg_pipeline.process_voice(audio_data, sample_rate)
                audio_chunks = result['audio_chunks']
                chunk_sr = result['sr']
                b64_list = []
                
                for chunk in audio_chunks:
                    out_io = io.BytesIO()
                    sf.write(out_io, chunk, chunk_sr, format='WAV')
                    out_io.seek(0)
                    b64_str = base64.b64encode(out_io.read()).decode('utf-8')
                    b64_list.append(b64_str)
                
                await websocket.send_json({
                    "type": "FINAL_RESULT",
                    "user_text": result["transcribed_text"],
                    "bot_text": result["response_text"],
                    "ref_service": result["referenced_service"],
                    "audio_base64_list": b64_list
                })
                
    except WebSocketDisconnect:
        print("클라이언트 연결 종료")
    except Exception as e:
        print(f"웹소켓 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vrg_pipeline.websocket == websocket:
            vrg_pipeline.websocket = None


async def process_buffered_audio(websocket: WebSocket):
    """버퍼에 저장된 오디오를 처리"""
    global vrg_pipeline
    
    audio_data = vrg_pipeline.get_buffered_audio()
    
    # 최소 길이 체크
    if len(audio_data) < SAMPLE_RATE * MIN_SPEECH_LENGTH_SEC:
        print("⚠️ 오디오가 너무 짧음, 무시")
        vrg_pipeline.reset_vad()
        return
    
    vrg_pipeline.is_processing = True
    
    try:
        await websocket.send_json({"type": "STATUS", "message": "음성을 분석하고 있어요..."})
        
        # 파이프라인 실행
        result = await vrg_pipeline.process_voice(audio_data, SAMPLE_RATE)
        audio_chunks = result['audio_chunks']
        chunk_sr = result['sr']
        search_result = result.get('search_result', {})
        b64_list = []
        
        for chunk in audio_chunks:
            out_io = io.BytesIO()
            sf.write(out_io, chunk, chunk_sr, format='WAV')
            out_io.seek(0)
            b64_str = base64.b64encode(out_io.read()).decode('utf-8')
            b64_list.append(b64_str)
        
        # GOODBYE 여부 전달
        is_goodbye = search_result.get("final_result") == "goodbye"
        
        await websocket.send_json({
            "type": "FINAL_RESULT",
            "user_text": result["transcribed_text"],
            "bot_text": result["response_text"],
            "ref_service": result["referenced_service"],
            "audio_base64_list": b64_list,
            "is_goodbye": is_goodbye  # 종료 여부 플래그
        })
        
        # GOODBYE 상황이면 채팅 종료
        if is_goodbye:
            print("👋 통화 종료, 채팅 종료")
            # 오디오 재생 시간 대기 후 종료 신호
            await asyncio.sleep(3.0)
            await websocket.send_json({
                "type": "CHAT_END",
                "message": "상담이 종료되었습니다. 이용해 주셔서 감사합니다."
            })
        
    except Exception as e:
        print(f"오디오 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({"type": "ERROR", "message": "처리 중 오류가 발생했습니다."})
    finally:
        vrg_pipeline.is_processing = False
        vrg_pipeline.reset_vad()

# =================================================================
# 메인 실행
# =================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 음성 RAG 시스템 서버 시작")
    print("=" * 70)
    print("\n웹 브라우저에서 http://localhost:8000 을 열어주세요.\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8015)
