import wave
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path('/home/data/bootcamp/team/ksb_room/ksb_script')

# 처리할 파일 리스트와 출력 파일 이름
INPUT_CSV = 'our_test_dataset.csv'
REULST_CSV = 'our_dataset.csv'

def process_csv_index(csv_filename, output_name):
    csv_path = BASE_DIR / csv_filename
    if not csv_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return

    data_list = []

    csv_df = pd.read_csv(csv_path)
    lines = csv_df.iterrows()

    print(f"🚀 {csv_filename} 처리 시작 (총 {len(csv_df)}개)")

    for idx, line in tqdm(lines, desc=f"Processing {csv_filename}"):
        text_content = line['transcript'].strip()
        full_path = BASE_DIR / 'audio' / line['file'].strip()

        with wave.open(str(full_path), 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        data_list.append({
            "split": "test",
            "audio_path": str(full_path),
            "text": text_content,
            "duration": duration,
        })

    # CSV 저장
    df = pd.DataFrame(data_list)
    df.to_csv(output_name, index=False, encoding='utf-8-sig')
    print(f"✨ 완료! {len(df)}개 항목이 {output_name}에 저장되었습니다.\n")
    return df

if __name__ == "__main__":
    process_csv_index(INPUT_CSV, REULST_CSV)