import os
import pandas as pd
from datasets import Dataset, Audio
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# =================================================================
# 01. 데이터 기본 정보 확인
# =================================================================

# Zeroth
zeroth_train_df = pd.read_csv('../csv_datasets/zeroth_sampled_train.csv', encoding='utf-8-sig')
zeroth_test_df = pd.read_csv('../csv_datasets/zeroth_sampled_test.csv', encoding='utf-8-sig')

print(f'\n ---------- Zeroth Data Info ----------')
print(f'Total time(hr): {(zeroth_train_df["duration"].sum() + zeroth_test_df["duration"].sum()) / 3600.0 : .2f}')
print(f'Train time(hr): {zeroth_train_df["duration"].sum() / 3600.0 : .2f}')
print(f'Test  time(hr): {zeroth_test_df["duration"].sum() / 3600.0 : .2f}')
print(f'Train Samples: {zeroth_train_df.shape[0]}')
print(f'Test  Samples: {zeroth_test_df.shape[0]}')

# 상담 음성
counsel_train_df = pd.read_csv('../csv_datasets/ai_hub_counsel_sampled_0.5m_train.csv')
counsel_test_df = pd.read_csv('../csv_datasets/ai_hub_counsel_sampled_0.5m_valid.csv')

print(f'\n ---------- 상담 음성 Data Info ----------')
print(f'Total time(hr): {(counsel_train_df["duration"].sum() + counsel_test_df["duration"].sum()) / 3600.0 : .2f}')
print(f'Train time(hr): {counsel_train_df["duration"].sum() / 3600.0 : .2f}')
print(f'Test  time(hr): {counsel_test_df["duration"].sum() / 3600.0 : .2f}')
print(f'Train Samples: {counsel_train_df.shape[0]}')
print(f'Test  Samples: {counsel_test_df.shape[0]}')


# 복지 분야 콜센터 데이터
welfare_train_df = pd.read_csv('../csv_datasets/ai_hub_welfare_sampled_train.csv')
welfare_test_df = pd.read_csv('../csv_datasets/ai_hub_welfare_sampled_valid.csv')

print(f'\n ---------- 복지 분야 콜센터 음성 Data Info ----------')
print(f'Total time(hr): {(welfare_train_df["duration"].sum() + welfare_test_df["duration"].sum()) / 3600.0 : .2f}')
print(f'Train time(hr): {welfare_train_df["duration"].sum() / 3600.0 : .2f}')
print(f'Test  time(hr): {welfare_test_df["duration"].sum() / 3600.0 : .2f}')
print(f'Train Samples: {welfare_train_df.shape[0]}')
print(f'Test  Samples: {welfare_test_df.shape[0]}')

# 한국어 음성 평가 데이터 (출처: AIHUB)
kspon_clean_df = pd.read_csv('../csv_datasets/kspon_eval_clean.csv') 
kspon_other_df = pd.read_csv('../csv_datasets/kspon_eval_other.csv')
print(f'\n ---------- 한국어 음성 Data Info ----------')
print(f'Total time(hr): {(kspon_clean_df["duration"].sum() + kspon_other_df["duration"].sum()) / 3600.0 : .2f}')
print(f'Clean time(hr): {kspon_clean_df["duration"].sum() / 3600.0 : .2f}')
print(f'Other time(hr): {kspon_other_df["duration"].sum() / 3600.0 : .2f}')
print(f'Clean Samples: {kspon_clean_df["duration"].shape[0]}')
print(f'Other Samples: {kspon_other_df["duration"].shape[0]}')

def to_hf_ds(df):
    return Dataset.from_dict({
        "audio": [str(p) for p in df['audio']],
        "text": df['text'].astype(str).tolist()
    })

# # =================================================================
# # 02. Train 데이터 구성하기
# # =================================================================
# # 학습 80시간
# # 복지분야 50시간 + 상담음성 20시간 + zeroth 10시간

welfare_train_sampled_df = welfare_train_df.sample(frac=0.13, random_state=RANDOM_STATE)
counsel_train_sampled_df = counsel_train_df.sample(frac=0.025, random_state=RANDOM_STATE)
zeroth_train_sampled_df= zeroth_train_df.sample(frac=0.20, random_state=RANDOM_STATE)

# train_df_list = [welfare_train_sampled_df, counsel_train_sampled_df, zeroth_train_sampled_df]
# total_df = pd.concat(train_df_list, axis=0, ignore_index=True)
# train_df, valid_df = train_test_split(
#     total_df,
#     test_size=0.10,
#     random_state=RANDOM_STATE,
#     shuffle=True
# )

# train_df.to_csv('final_train.csv', encoding='utf-8-sig', index=False)

# print(f'\n ---------- 최종 Train Data Info ----------')
# print(f'Train time(hr): {train_df["duration"].sum() / 3600.0 : .2f}')
# print(f'Train Samples: {train_df.shape[0]}')

# train_df_renamed = train_df.rename(columns={"audio_path": "audio"})
# train_dataset = to_hf_ds(train_df_renamed)
# train_dataset = train_dataset.shuffle(seed=RANDOM_STATE)
# train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

# # =================================================================
# # 03. Valid 데이터 구성하기
# # =================================================================
# # 검증 10시간
# # 복지 6.5시간 + 상담 2.5시간 + zeroth 1시간

welfare_valid_test_sampled_df = welfare_test_df.sample(frac=0.03, random_state=RANDOM_STATE)
welfare_valid_sampled_df, welfare_test_sampled_df = train_test_split(
    welfare_valid_test_sampled_df,
    test_size=0.28,
    random_state=RANDOM_STATE
)
counsel_valid_test_sampled_df = counsel_test_df.sample(frac=0.16, random_state=RANDOM_STATE)
counsel_valid_sampled_df, counsel_test_samped_df = train_test_split(
    counsel_valid_test_sampled_df,
    test_size=0.5,
    random_state=RANDOM_STATE
)

# valid_df_list = [welfare_valid_sampled_df, counsel_valid_sampled_df, zeroth_test_df]
# valid_df = pd.concat(valid_df_list, axis=0, ignore_index=True)
# valid_df.to_csv('final_valid.csv', encoding='utf-8-sig', index=False)

# print(f'\n ---------- 최종 Valid Data Info ----------')
# print(f'Valid time(hr): {valid_df["duration"].sum() / 3600.0 : .2f}')
# print(f'Valid Samples: {valid_df.shape[0]}')

# valid_df_renamed = valid_df.rename(columns={"audio_path": "audio"})
# valid_dataset = to_hf_ds(valid_df_renamed)
# valid_dataset = valid_dataset.shuffle(seed=RANDOM_STATE)
# valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16_000))

# =================================================================
# 04. Test 데이터 구성하기
# =================================================================
# 평가 10시간
# 복지 + 상담       -> 5시간
# 한국어 음성 데이터 -> 5시간
# 우리 음성 데이터

our_dataset_df = pd.read_csv('../csv_datasets/our_dataset.csv') # 우리 음성 데이터

test_df_list = [
    welfare_test_sampled_df,
    counsel_test_samped_df,
    kspon_clean_df,
    kspon_other_df,
    our_dataset_df
]

test_df = pd.concat(
    test_df_list,
    axis=0,
    ignore_index=True
)
test_df.to_csv('../csv_datasets/final_test.csv', encoding='utf-8-sig', index=False)

print(f'\n ---------- 최종 Test Data Info ----------')
print(f'test time(hr): {test_df["duration"].sum() / 3600.0 : .2f}')
print(f'test Samples: {test_df.shape[0]}')

test_df_renamed = test_df.rename(columns={"audio_path": "audio"})
test_dataset = to_hf_ds(test_df_renamed)
test_dataset = test_dataset.shuffle(seed=RANDOM_STATE)
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

# =================================================================
# 03. 데이터셋 저장
# =================================================================

SAVE_BASE_PATH = './'
# train_dataset.save_to_disk(os.path.join(SAVE_BASE_PATH, 'train'))
# valid_dataset.save_to_disk(os.path.join(SAVE_BASE_PATH, 'valid'))
test_dataset.save_to_disk(os.path.join(SAVE_BASE_PATH, 'test'))

print("✅ 모든 데이터셋이 각각의 디렉토리에 Arrow 형식으로 저장되었습니다.")