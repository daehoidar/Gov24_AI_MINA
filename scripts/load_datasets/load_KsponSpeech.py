import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- 설정 ---
# 현재 스크립트가 실행되는 위치 (final_dataset/ 폴더)
BASE_DIR = Path(__file__).parent.resolve()

# 처리할 파일 리스트와 출력 파일 이름
TARGET_FILES = {
    "eval_clean.csv": "kspon_eval_clean.csv",
    "eval_other.csv": "kspon_eval_other.csv"
}

SAMPLE_RATE = 16_000
BIT_DEPTH = 16

def clean_text(text):
    """(철자)/(발음) -> 철자 추출 및 노이즈 태그 제거"""
    # 1. (철자)/(발음) -> 철자만 남김
    text = re.sub(r'\((.*?)\)\/\((.*?)\)', r'\1', text)
    # 2. b/, n/ 등 노이즈 태그 제거
    text = re.sub(r'[a-z]\/', '', text)
    # 3. 특수문자 (#, *, ?, !) 제거
    text = re.sub(r'[#|*|?|!]', '', text)
    return text.strip()

def process_trn_index(trn_filename, output_name):
    trn_path = BASE_DIR / trn_filename
    if not trn_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {trn_path}")
        return

    data_list = []
    
    # 인코딩 오류 방지를 위한 처리 (utf-8 우선, 실패 시 cp949)
    try:
        with open(trn_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(trn_path, 'r', encoding='cp949') as f:
            lines = f.readlines()

    print(f"🚀 {trn_filename} 처리 시작 (총 {len(lines)}개)")

    for line in tqdm(lines, desc=f"Processing {trn_filename}"):
        if " :: " not in line:
            continue
            
        # 1. 경로와 텍스트 분리
        rel_path_str, raw_text = line.split(" :: ")
        
        # 2. 경로 조인 (상대 경로 -> 절대 경로)
        # csv 내부 경로: Ksponspeech_eval/eval_clean/파일명.pcm
        full_pcm_path = BASE_DIR / rel_path_str.strip()
        
        if not full_pcm_path.exists():
            # 대소문자 확인 (Ksponspeech_eval vs KsponSpeech_eval)
            continue

        # 3. 텍스트 정제
        text_content = clean_text(raw_text)
        if not text_content:
            continue

        # 4. PCM duration 계산
        file_size = os.path.getsize(full_pcm_path)
        duration = file_size / (SAMPLE_RATE * (BIT_DEPTH // 8))

        data_list.append({
            "split": "test",
            "audio_path": str(full_pcm_path),
            "text": text_content,
            "duration": duration,
            "condition": "clean" if "clean" in trn_filename else "other"
        })

    # CSV 저장
    df = pd.DataFrame(data_list)
    df.to_csv(BASE_DIR / output_name, index=False, encoding='utf-8-sig')
    print(f"✨ 완료! {len(df)}개 항목이 {output_name}에 저장되었습니다.\n")
    return df

if __name__ == "__main__":
    all_dfs = []
    for trn_file, out_name in TARGET_FILES.items():
        df = process_trn_index(trn_file, out_name)
