import os
import wave
import random
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

random.seed(42)

def process_single_audio(wav_path_obj):
    """개별 파일 처리 워커 함수"""
    try:
        abs_wav_path = wav_path_obj.resolve()
        audio_folder_path = str(abs_wav_path)
        
        # split 결정
        split = 'Train' if 'Training' in audio_folder_path else 'Valid'

        # text 읽기
        txt_path = abs_wav_path.with_suffix('.txt')
        if not txt_path.exists():
            return None
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()

        # duration 계산
        with wave.open(str(abs_wav_path), 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        return {
            "split": split,
            "audio_path": audio_folder_path,
            "text": text_content,
            "duration": duration
        }
    except Exception:
        return None

def create_sampled_dataset(root_dir, output_csv, total_target=500_000):
    root_path = Path(root_dir)
    target_dirs = ['D60', 'D61', 'D62']
    samples_per_dir = total_target // len(target_dirs)
    
    selected_files = []

    # 1. 디렉토리별 균등 샘플링
    for d_name in target_dirs:
        dir_path = root_path / d_name
        if not dir_path.exists():
            print(f"Warning: {d_name} 디렉토리를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        print(f"🔍 {d_name}에서 파일 목록을 검색 중...")
        # 해당 디렉토리의 모든 wav 파일 탐색
        all_files = list(dir_path.glob("**/*.wav"))
        
        # 무작위 샘플링
        if len(all_files) > samples_per_dir:
            sampled = random.sample(all_files, samples_per_dir)
            selected_files.extend(sampled)
            print(f"✅ {d_name}: {len(all_files)}개 중 {samples_per_dir}개 샘플링 완료")
        else:
            selected_files.extend(all_files)
            print(f"⚠️ {d_name}: 파일 수가 부족하여 {len(all_files)}개 전체를 포함합니다.")

    # 2. 선택된 파일들에 대해 병렬 처리 진행
    print(f"\n🚀 총 {len(selected_files)}개 파일에 대한 병렬 처리를 시작합니다.")
    data_list = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
    # chunksize를 주면 50만 개를 묶어서 던지므로 통신 비용이 줄어듭니다.
        results = list(tqdm(executor.map(process_single_audio, selected_files, chunksize=100), 
                            total=len(selected_files), 
                            desc="Processing"))

        data_list = [r for r in results if r is not None]

    # 3. 결과 저장
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✨ 완료! {len(df)}개의 데이터가 {output_csv}에 저장되었습니다.")

# --- 설정 ---
ROOT_DIRECTORY = "/home/data/bootcamp/team/AIHUB_상담_음성_데이터/상담 음성/Validation"
OUTPUT_FILE = "ai_hub_counsel_sampled_0.5m_valid.csv"

if __name__ == "__main__":
    create_sampled_dataset(ROOT_DIRECTORY, OUTPUT_FILE)