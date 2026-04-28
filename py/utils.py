from typing import Any, Iterable, List, Dict, Optional
import json
import os
import re
import uuid
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch
import urllib.request

matplotlib.use("Agg")  # 서버 환경에서 GUI 없이 렌더링
# 한글 폰트 설정 (Render Linux 환경)
font_path = "/tmp/NanumGothic.ttf"
if not os.path.exists(font_path):
    urllib.request.urlretrieve(
        "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
        font_path
    )
fm.fontManager.addfont(font_path)
matplotlib.rc("font", family="NanumGothic")  # 이 줄 추가
matplotlib.rcParams["axes.unicode_minus"] = False

def _to_list(seq: Iterable[Any]) -> List[Any]:
    if seq is None:
        return []
    if isinstance(seq, list):
        return seq
    return list(seq)


def _time_to_hhmm_list(seq: Iterable[Any]) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    s = pd.to_datetime(_to_list(seq), errors="coerce")
    for dt in s:
        if pd.isna(dt):
            out.append(None)
        else:
            out.append(pd.to_datetime(dt).strftime("%H:%M"))
    return out


def remove_ppg_prefix(obj):
    """
    dict 또는 list[dict]에서 키가 'PPG_'로 시작하면 접두사를 제거한다.
    원본 구조는 유지하고 키만 변경한다.
    """
    if isinstance(obj, list):
        return [remove_ppg_prefix(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = k
            if isinstance(k, str) and k.startswith("PPG_"):
                nk = k[4:]
            out[nk] = v
        return out
    return obj


def linebreak_by_sentence(text: str) -> str:
    if not text:
        return text
    paras = text.split("\n\n")
    out_paras = []
    for p in paras:
        s = re.sub(r"[ \t]+", " ", p.strip())
        s = re.sub(r'(?<=[\.\?\!。！？…])\s+', "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        out_paras.append(s)
    return "\n\n".join(out_paras)

def _parse_dt_series(day: str, time_series: pd.Series) -> pd.Series:
    """
    time 컬럼의 'HH:MM' 또는 'YYYY-MM-DD HH:MM' 문자열을 datetime으로 변환한다.
    '24:MM'은 다음 날 00:MM으로 보정한다.
    """

    def parse_one(s: str):
        s = str(s).strip()

        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            m = re.search(r"\s(24):(\d{2})$", s)
            if m:
                mm = m.group(2)
                base = pd.to_datetime(s[:10] + f" 00:{mm}", errors="coerce")
                return base + pd.Timedelta(days=1) if pd.notna(base) else pd.NaT
            return pd.to_datetime(s, errors="coerce")

        m = re.match(r"^(24):(\d{2})$", s)
        if m:
            mm = m.group(2)
            base = pd.to_datetime(f"{day} 00:{mm}", errors="coerce")
            return base + pd.Timedelta(days=1) if pd.notna(base) else pd.NaT

        dt = pd.to_datetime(f"{day} {s}", errors="coerce")
        if pd.notna(dt) and dt.hour < 8:
            dt = dt + pd.Timedelta(days=1)
        return dt

    return time_series.apply(parse_one)


def make_biosignal_overview_plot(
    valid_signals,
    session_id: str | None,
    prt: str,
    day: str,
    shift_type: str = "day",
    base_dir: str = "plots",
) -> str | None:
    if not valid_signals:
        return None

    save_dir = os.path.join(base_dir, prt, day)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(valid_signals)

    time_col = "time"
    stress_col = "Stress"

    if time_col not in df.columns:
        return None
    df = df[df[time_col].notna()].copy()
    if df.empty:
        return None

    if stress_col in df.columns:
        df[stress_col] = pd.to_numeric(df[stress_col], errors="coerce")
    else:
        df[stress_col] = np.nan

    df["_dt"] = _parse_dt_series(str(day), df[time_col].astype(str))
    df = df[df["_dt"].notna()].copy()
    if df.empty:
        return None

    df = df.sort_values("_dt")
    uniq_dt = pd.Series(df["_dt"].drop_duplicates().sort_values().to_list())
    if len(uniq_dt) == 0:
        return None

    if len(uniq_dt) == 1:
        freq_min = 60
    else:
        diffs = uniq_dt.diff().dropna()
        diff_mins = (diffs.dt.total_seconds() / 60.0).to_numpy()
        diff_mins = diff_mins[np.isfinite(diff_mins) & (diff_mins >= 1)]
        freq_min = 60 if len(diff_mins) == 0 else int(np.clip(int(round(np.min(diff_mins))), 5, 120))

    actual_count = len(uniq_dt)
    slot_periods = 24 if shift_type == "duty" else 12

    window = pd.Timedelta(minutes=freq_min * (min(slot_periods, actual_count) - 1))

    best_start, best_count = uniq_dt.iloc[0], -1
    for s in uniq_dt:
        cnt = int(((uniq_dt >= s) & (uniq_dt <= s + window)).sum())
        if cnt > best_count or (cnt == best_count and s > best_start):
            best_count, best_start = cnt, s

    if actual_count <= slot_periods:
        slots = pd.date_range(start=best_start, periods=slot_periods, freq=f"{freq_min}min")
    else:
        slots = pd.date_range(start=uniq_dt.iloc[0], periods=actual_count, freq=f"{freq_min}min")
    df = df.set_index("_dt").reindex(slots)

    times = slots.strftime("%H").tolist()
    x_idx = np.arange(len(slots))
    stress = df[stress_col].to_numpy(dtype=float) if stress_col in df.columns else np.full(len(slots), np.nan)

    valid_mask = np.isfinite(stress)
    if int(valid_mask.sum()) < 1:
        return None

    COLOR_STRESS = "#ef5350"
    COLOR_CALM   = "#42a5f5"
    COLOR_NODATA = "#e0e0e0"
    bar_colors = np.where(stress == 1, COLOR_STRESS, COLOR_CALM)

    shift_labels = {"day": "주간", "night": "야간", "off": "비번", "duty": "당직", "holiday": "휴무"}
    shift_label = shift_labels.get(shift_type, shift_type)
    hour_label = "24시간" if shift_type == "duty" else "12시간"

    tick_fontsize = 8 if shift_type == "duty" else 11
    fig_width = 10 if shift_type == "duty" else 8

    fig, ax = plt.subplots(1, 1, dpi=130, figsize=(fig_width, 1.8))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    for xi, has_data, color in zip(x_idx, np.isfinite(stress), bar_colors):
        if has_data:
            ax.bar(xi, 1, width=1.0, align="edge", color=color, alpha=0.65, zorder=1)
        else:
            ax.bar(xi, 1, width=1.0, align="edge", color=COLOR_NODATA, alpha=0.4, zorder=1)

    for xi in x_idx[1:]:
        ax.axvline(x=xi, color="#cccccc", linewidth=0.6, zorder=2)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(x_idx + 0.5)
    ax.set_xticklabels(times, rotation=0, ha="center", fontsize=tick_fontsize)
    ax.tick_params(axis="x", colors="#222222", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    legend_patches = [
        Patch(facecolor=COLOR_STRESS, alpha=0.65, label="스트레스"),
        Patch(facecolor=COLOR_CALM,   alpha=0.65, label="안정"),
        Patch(facecolor=COLOR_NODATA, alpha=0.4,  label="데이터 없음"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fontsize=10,
        framealpha=0.0,
        edgecolor="none",
    )

    fig.suptitle(
        f"오늘의 신체 상태  |  {shift_label} · {hour_label}",
        fontsize=13, fontweight="bold", color="#222222", y=1.28,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    fig.text(
        0.5, -0.12,
        "* 스트레스/안정 구분은 심박수 외 여러 HRV 지표를 종합해 판단합니다.",
        ha="center", fontsize=10, color="#666666",
    )

    sid = session_id or "nosess"
    img_name = f"biosignal_{sid}_{uuid.uuid4().hex[:8]}.png"
    img_path = os.path.join(save_dir, img_name)
    fig.savefig(img_path, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)

    return f"/plots/{prt}/{day}/{img_name}"

def get_validation_data(state: dict):
    analyses_raw = state.get("analyses", []) or []
    reset_points = state.get("validation_reset_points", []) or []

    start_idx = 0
    if reset_points:
        try:
            start_idx = max(0, int(reset_points[-1]))
        except Exception:
            start_idx = 0
    analyses_raw = analyses_raw[start_idx:]

    parsed = []
    for a in analyses_raw:
        try:
            obj = json.loads(a) if isinstance(a, str) else a
            if not isinstance(obj, dict):
                continue

            emotion = obj.get("emotion", {})
            if not isinstance(emotion, dict):
                emotion = {}

            main = str(emotion.get("main") or "").strip()
            sub = str(emotion.get("sub") or "").strip()
            if sub.lower() in ("null", "unknown", "none", ""):
                sub = ""

            parsed.append(
                {
                    "original_text": (obj.get("original_text") or "").strip(),
                    "prev_ai_text": (obj.get("prev_ai_text") or "").strip(),
                    "situation": (obj.get("situation") or "").strip(),
                    "main": main,
                    "sub": sub,
                }
            )
        except Exception as e:
            print("analysis parse error:", e)
            print("failed raw analysis:", a)

    if not parsed:
        return {
            "random_turns": [
                {
                    "original_text": "",
                    "prev_ai_text": "",
                    "situation": "",
                    "emotion": {"main": "", "sub": ""},
                },
                {
                    "original_text": "",
                    "prev_ai_text": "",
                    "situation": "",
                    "emotion": {"main": "", "sub": ""},
                },
            ],
            "top_emotions": [],
        }

    long_items = [x for x in parsed if len(x["original_text"]) >= 10]
    if len(long_items) < 3:
        return {"is_insufficient": True, "random_turns": [], "top_emotions": []}
    valid_items = long_items

    def _is_meaningful_situation(text: str) -> bool:
        s = (text or "").strip().lower()
        return bool(s and s not in {"unknown", "null", "none"})

    def _normalize_situation(text: str) -> str:
        s = re.sub(r"\s+", " ", (text or "").strip().lower())
        return re.sub(r"[^\w가-힣]+", "", s)

    preferred_pool = valid_items[max(0, len(valid_items) // 3):] or valid_items
    preferred_pool = [x for x in preferred_pool if _is_meaningful_situation(x.get("situation", ""))] or preferred_pool

    deduped_pool = []
    seen_situations = set()
    for item in preferred_pool:
        key = _normalize_situation(item.get("situation", ""))
        if key and key in seen_situations:
            continue
        if key:
            seen_situations.add(key)
        deduped_pool.append(item)

    candidate_pool = deduped_pool or preferred_pool or valid_items
    sample_size = min(2, len(candidate_pool))
    sampled_turns = random.sample(candidate_pool, sample_size) if sample_size else []

    while len(sampled_turns) < 2:
        sampled_turns.append(sampled_turns[0] if sampled_turns else {
            "original_text": "",
            "prev_ai_text": "",
            "situation": "",
            "main": "",
            "sub": "",
        })

    emotion_totals: Dict[str, float] = {}
    for item in parsed:
        main_item = item["main"]
        sub_item = item["sub"]

        if main_item and main_item != "unknown":
            emotion_totals[main_item] = emotion_totals.get(main_item, 0.0) + 1.0
        if sub_item:
            emotion_totals[sub_item] = emotion_totals.get(sub_item, 0.0) + 0.5

    top_3_emotions = [
        label for label, _ in sorted(
            emotion_totals.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
    ]

    return {
        "random_turns": [
            {
                "original_text": item["original_text"],
                "prev_ai_text": item.get("prev_ai_text", ""),
                "situation": item["situation"],
                "emotion": {
                    "main": item["main"],
                    "sub": item["sub"],
                },
            }
            for item in sampled_turns[:2]
        ],
        "top_emotions": top_3_emotions,
    }
