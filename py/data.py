import pickle
import pandas as pd
from typing import List

# --- 데이터 로드 (최초 1회) ---
try:
    with open("watch_hrv_total.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Time 파싱 (오류 발생 시 NaT - Not a Time)
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    data["time"] = data["Time"].dt.strftime("%H:%M")
    
    # NaT가 된 행은 제거하여 오류 방지
    data = data.dropna(subset=["Time"])

except FileNotFoundError:
    print("오류: 'watch_hrv_total.pkl' 파일을 찾을 수 없습니다. 빈 DataFrame을 사용합니다.")
    data = pd.DataFrame(columns=["Time", "time", "prt", "CollectionType", "Shift", "Stress", "PPG_MeanHR", "PPG_MeanNN", "PPG_SDNN", "PPG_LF", "PPG_LFn", "PPG_HFn"])
except Exception as e:
    print(f"데이터 로드 중 알 수 없는 오류 발생: {e}")
    data = pd.DataFrame(columns=["Time", "time", "prt", "CollectionType", "Shift", "Stress", "PPG_MeanHR", "PPG_MeanNN", "PPG_SDNN", "PPG_LF", "PPG_LFn", "PPG_HFn"])


def get_biosignal_records(
    prt: str,
    day: str,
    collection_type: str = "Automatic",
    target_hours: int = 12,  # ✅ simple_police_bio.py와의 호환성을 위해 유지
) -> List[dict]:
    """
    prt/day(당일 08:00 ~ 익일 07:59) 하루치 근무 데이터를 1시간 슬롯으로 정리.
    - 'OnShift' 상태인 데이터만 필터링
    - 6시간 이상 시간 차이가 나는 '외톨이 데이터' (예: 퇴근 기록)를 자동으로 제거
    - 관측이 없는 시간대는 N/A로 채워서 24칸 보장
    """
    df = data.copy()
    
    # 1) 조회 '일'을 '근무일' (당일 08:00 ~ 익일 07:59) 기준으로 변경
    try:
        day_dt = pd.to_datetime(day).normalize()
    except Exception as e:
        print(f"오류: 'day' 값('{day}')을 날짜로 변환할 수 없습니다. {e}")
        return [] # 빈 리스트 반환

    workday_start = day_dt + pd.Timedelta(hours=8)  # 당일 08:00:00
    workday_end = workday_start + pd.Timedelta(days=1)   # 익일 08:00:00

    # 2) 필터 (특정 prt / Automatic / OnShift / '근무일' 시간)
    #    .str.strip()으로 공백 제거
    df = df[
        (df["prt"].astype(str).str.strip() == str(prt).strip())
        & (df["CollectionType"].str.strip() == collection_type)
        & (df["Time"] >= workday_start)
        & (df["Time"] < workday_end)
    ].copy()

    if df.empty:
        # print(f"참고: {prt}의 {day} 근무일(08:00~) 동안 'OnShift' 데이터가 없습니다.")
        pass # 데이터가 없어도 빈 24칸을 반환하기 위해 계속 진행

    # 3) Stress → 0/1 (대/소문자 및 언어 통일)
    def _to_bin(x):
        s = str(x).strip().lower()
        return 1 if s in {"1", "yes", "y", "true", "양성"} else 0

    if 'Stress' in df.columns:
        df["Stress"] = df["Stress"].apply(_to_bin)

    # 4) 시간 정렬
    if not df.empty:
        df = df.sort_values("Time")

    # 4-5) '외톨이 데이터' 제거 (Clustering)
    if not df.empty and len(df) > 1: # 데이터가 2개 이상일 때만 비교
        # 데이터 포인트 간의 시간 차이 계산 (초 단위)
        df['time_diff_seconds'] = df['Time'].diff().dt.total_seconds()
        
        # 6시간(21600초)을 초과하는 가장 큰 시간 차이(gap) 찾기
        max_gap = df['time_diff_seconds'].max()
        
        if pd.notna(max_gap) and max_gap > 21600: # 6시간
            # print(f"참고: {prt}/{day}에서 {max_gap}초의 큰 시간차가 감지됨. 데이터 군집 분리 시도.")
            
            # 가장 큰 갭을 기준으로 데이터를 두 그룹으로 나눔
            gap_index = df['time_diff_seconds'].idxmax()
            df_group_1 = df.loc[:gap_index].iloc[:-1] # 갭 이전
            df_group_2 = df.loc[gap_index:]          # 갭 이후
            
            # 데이터가 더 많은 그룹(실제 근무)을 선택
            if len(df_group_1) > len(df_group_2):
                df = df_group_1
                # print(f"-> 더 큰 군집 1 (데이터 {len(df)}개)을 선택합니다.")
            else:
                df = df_group_2
                # print(f"-> 더 큰 군집 2 (데이터 {len(df)}개)을 선택합니다.")
        
        # 'hour' 컬럼 생성
        df["hour"] = df["Time"].dt.floor("h")
        
    elif not df.empty: # 데이터가 1개일 경우
        df["hour"] = df["Time"].dt.floor("h")


    # 5) '근무일' 시작 시각(08:00) 기준으로 24칸 슬롯 생성
    slots_start_hour = day_dt + pd.Timedelta(hours=8)
    slots = pd.date_range(start=slots_start_hour, periods=24, freq="1h")

    # 6) 생성된 24칸 슬롯에 데이터 매핑 (없는 시간대는 NaN)
    if not df.empty:
        # 중복된 시간이 있다면 평균 계산 (안전장치)
        # 20:00:02 와 20:30:05 데이터는 모두 20:00 슬롯으로 묶여서 평균 처리됨
        df_grouped = df.groupby("hour").mean(numeric_only=True)
        df = df_grouped.reindex(slots)
    else:
        # 데이터가 아예 없으면, 'slots'를 인덱스로 하는 빈 DataFrame 생성
        df = pd.DataFrame(index=slots)

    # 7) 표시용 HH:MM
    df = df.reset_index().rename(columns={"index": "hour"})
    df["time"] = df["hour"].dt.strftime("%H:%M")

    # 8) 필요한 컬럼만 선택(없으면 생성)
    needed = [
        "time",
        "Stress",
        "PPG_MeanHR",
        "PPG_MeanNN",
        "PPG_SDNN",
        "PPG_LF",
        "PPG_LFn",
        "PPG_HFn",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[needed].copy()

    # 9) NaN → "N/A", 숫자는 소수 2자리 반올림
    df = df.where(df.notna(), "N/A")
    for c in ["PPG_MeanHR", "PPG_MeanNN", "PPG_SDNN", "PPG_LF", "PPG_LFn", "PPG_HFn"]:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: round(x, 2) if isinstance(x, (int, float)) else x
            )

    return df.to_dict(orient="records")