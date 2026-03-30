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

    # NaT가 있는 행 제거
    data = data.dropna(subset=["Time"])

except FileNotFoundError:
    print("오류: 'watch_hrv_total.pkl' 파일을 찾을 수 없습니다. 빈 DataFrame을 사용합니다.")
    data = pd.DataFrame(
        columns=[
            "Time",
            "time",
            "prt",
            "CollectionType",
            "Shift",
            "Stress",
            "PPG_MeanHR",
            "PPG_MeanNN",
            "PPG_SDNN",
            "PPG_LF",
            "PPG_LFn",
            "PPG_HFn",
        ]
    )
except Exception as e:
    print(f"데이터 로드 중 예상하지 못한 오류 발생: {e}")
    data = pd.DataFrame(
        columns=[
            "Time",
            "time",
            "prt",
            "CollectionType",
            "Shift",
            "Stress",
            "PPG_MeanHR",
            "PPG_MeanNN",
            "PPG_SDNN",
            "PPG_LF",
            "PPG_LFn",
            "PPG_HFn",
        ]
    )


def get_biosignal_records(
    prt: str,
    day: str,
    collection_type: str = "Automatic",
    target_hours: int = 12,  # simple_police_bio.py와의 호환성을 위해 유지
) -> List[dict]:
    """
    prt/day(당일 08:00 ~ 익일 07:59) 근무일 데이터를 1시간 단위로 정리한다.
    - 관측이 없는 시간대는 N/A로 채워서 24칸 보장
    """
    df = data.copy()

    try:
        day_dt = pd.to_datetime(day).normalize()
    except Exception as e:
        print(f"오류: 'day' 값('{day}')을 날짜로 변환할 수 없습니다. {e}")
        return []

    workday_start = day_dt + pd.Timedelta(hours=8)
    workday_end = workday_start + pd.Timedelta(days=1)

    df = df[
        (df["prt"].astype(str).str.strip() == str(prt).strip())
        & (df["CollectionType"].astype(str).str.strip() == collection_type)
        & (df["Time"] >= workday_start)
        & (df["Time"] < workday_end)
    ].copy()

    def _to_bin(x):
        s = str(x).strip().lower()
        return 1 if s in {"1", "yes", "y", "true", "양성"} else 0

    if "Stress" in df.columns:
        df["Stress"] = df["Stress"].apply(_to_bin)

    if not df.empty:
        df = df.sort_values("Time")
        df["hour"] = df["Time"].dt.floor("h")

    slots_start_hour = day_dt + pd.Timedelta(hours=8)
    slots = pd.date_range(start=slots_start_hour, periods=24, freq="1h")

    if not df.empty:
        df_grouped = df.groupby("hour").mean(numeric_only=True)
        df = df_grouped.reindex(slots)
    else:
        df = pd.DataFrame(index=slots)

    df = df.reset_index().rename(columns={"index": "hour"})
    df["time"] = df["hour"].dt.strftime("%H:%M")

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

    df = df.where(df.notna(), "N/A")
    for c in ["PPG_MeanHR", "PPG_MeanNN", "PPG_SDNN", "PPG_LF", "PPG_LFn", "PPG_HFn"]:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: round(x, 2) if isinstance(x, (int, float)) else x
            )

    return df.to_dict(orient="records")
