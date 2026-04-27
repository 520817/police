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
    target_hours: int = 12,
    start_datetime=None,
    shift_type: str = "day",
) -> List[dict]:
    """
    start_datetime 기준 이전 12시간(duty는 24시간) 데이터를 읽어온다.
    stress_result가 "Stress O" 또는 "Stress X"인 슬롯만 반환한다.
    """
    df = data.copy()

    hours_back = 24 if shift_type == "duty" else 12

    if start_datetime is not None:
        end_dt = pd.to_datetime(start_datetime)
    else:
        try:
            end_dt = pd.to_datetime(day).normalize() + pd.Timedelta(hours=20)
        except Exception as e:
            print(f"오류: 'day' 값('{day}')을 날짜로 변환할 수 없습니다. {e}")
            return []

    start_dt = end_dt - pd.Timedelta(hours=hours_back)

    def _is_valid_stress(x):
        return str(x).strip() in {"Stress O", "Stress X"}

    def _to_bin(x):
        return 1 if str(x).strip() == "Stress O" else 0

    df = df[
        (df["prt"].astype(str).str.strip() == str(prt).strip())
        & (df["CollectionType"].astype(str).str.strip() == collection_type)
        & (df["Time"] >= start_dt)
        & (df["Time"] < end_dt)
        & df["Stress"].apply(_is_valid_stress)
    ].copy()

    if df.empty:
        return []

    df["Stress"] = df["Stress"].apply(_to_bin)
    df = df.sort_values("Time")
    df["hour"] = df["Time"].dt.floor("h")

    df_grouped = df.groupby("hour").mean(numeric_only=True)
    df_grouped["Stress"] = df_grouped["Stress"].round().astype(int)
    df_grouped = df_grouped.reset_index()
    df_grouped["time"] = df_grouped["hour"].dt.strftime("%H:%M")

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
        if col not in df_grouped.columns:
            df_grouped[col] = pd.NA
    df_grouped = df_grouped[needed].copy()

    for c in ["PPG_MeanHR", "PPG_MeanNN", "PPG_SDNN", "PPG_LF", "PPG_LFn", "PPG_HFn"]:
        if c in df_grouped.columns:
            df_grouped[c] = df_grouped[c].apply(
                lambda x: round(x, 2) if isinstance(x, (int, float)) else x
            )

    return df_grouped.to_dict(orient="records")
