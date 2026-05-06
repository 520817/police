import pandas as pd
from typing import List
import os
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List

MACHINE_TOKEN = os.getenv("MACHINE_TOKEN")
BASE_URL = "https://lst-police.plaidai.io/api/machine"
HEADERS = {"Authorization": f"Bearer {MACHINE_TOKEN}"}

def get_biosignal_records(
    prt: str,
    day: str,
    collection_type: str = "Automatic",  # 하위 호환성 유지
    target_hours: int = 12,               # 하위 호환성 유지
    start_datetime=None,
    shift_type: str = "day",
) -> List[dict]:
    """
    prt(전화번호)와 start_datetime(챗봇 시작 시각) 기준으로
    이전 12시간(duty는 24시간) 데이터를 읽어온다.
    stress_result가 "Stress O" 또는 "Stress X"인 슬롯만 반환한다.

    반환 형식:
    [
        {
            "time": "HH:MM",
            "Stress": 0 or 1,
            "PPG_MeanHR": float or "N/A",
            "PPG_MeanNN": float or "N/A",
            "PPG_SDNN":   float or "N/A",
            "PPG_LF":     float or "N/A",
            "PPG_LFn":    float or "N/A",
            "PPG_HFn":    float or "N/A",
        },
        ...
    ]
    """

    # ── 1. 시간 범위 계산 ──
    hours_back = 24 if shift_type == "duty" else 12

    if start_datetime is not None:
        end_dt = (
            datetime.fromisoformat(str(start_datetime))
            if isinstance(start_datetime, str)
            else start_datetime
        )
    else:
        # start_datetime 없으면 현재 KST 시각 사용
        end_dt = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)

    start_dt = end_dt - timedelta(hours=hours_back)

    print(f"[data] prt={prt}, shift_type={shift_type}, 범위={start_dt} ~ {end_dt}")

    # ── 2. API 호출 ──
    try:
        resp = requests.get(
            f"{BASE_URL}/collections",
            headers=HEADERS,
            params={
                "phone_number": prt,
                "limit": 500,
            },
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        print(f"[data] API 호출 실패: {e}")
        return []

    if not items:
        print(f"[data] collection 없음: prt={prt}")
        return []

    # ── 3. 유효한 슬롯 필터링 ──
    def _is_valid_stress(stress_result):
        return str(stress_result).strip() in {"Stress O", "Stress X"}

    def _to_bin(stress_result):
        return 1 if str(stress_result).strip() == "Stress O" else 0

    def _round(v):
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            return "N/A"

    # 시간대별로 자동(1) 우선, 없으면 맥파(3) 사용
    # slot_key(HH) → {"auto": record, "manual": record}
    slot_candidates: dict = {}

    for item in items:
        # start_time 파싱
        try:
            item_dt_str = item.get("start_time", "")
            item_dt = datetime.fromisoformat(item_dt_str.replace("Z", ""))
        except Exception:
            continue

        # 시간 범위 필터
        if not (start_dt <= item_dt < end_dt):
            continue

        # stress_result 유효성 검사 (NaN 제외)
        stress_result = item.get("stress_result")
        if not _is_valid_stress(stress_result):
            continue

        # feature_values 추출
        fv = item.get("feature_values") or {}

        record = {
            "time":       item_dt.strftime("%H:%M"),
            "Stress":     _to_bin(stress_result),
            "PPG_MeanHR": _round(fv.get("PPG_MeanHR")),
            "PPG_MeanNN": _round(fv.get("PPG_MeanNN")),
            "PPG_SDNN":   _round(fv.get("PPG_SDNN")),
            "PPG_LF":     _round(fv.get("PPG_LF")),
            "PPG_LFn":    _round(fv.get("PPG_LFn")),
            "PPG_HFn":    _round(fv.get("PPG_HFn")),
        }

        slot_key = item_dt.strftime("%H")
        collection_type = item.get("collection_type")
        bucket = slot_candidates.setdefault(slot_key, {})

        if collection_type == 1:
            bucket["auto"] = record
        elif collection_type == 3:
            if "manual" not in bucket:
                bucket["manual"] = record

    # 자동 우선, 없으면 맥파
    records = []
    for slot_key in sorted(slot_candidates.keys()):
        bucket = slot_candidates[slot_key]
        record = bucket.get("auto") or bucket.get("manual")
        if record:
            records.append(record)

    # ── 4. 시간순 정렬 ──
    records.sort(key=lambda x: x["time"])

    print(f"[data] 유효 슬롯 {len(records)}개 로드 완료")
    return records
