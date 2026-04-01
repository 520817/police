from typing import Any, Iterable, List, Dict, Optional
import json
import os
import re
import uuid
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
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
matplotlib.rc("font", family="NanumGothic")
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


def make_strategy_guide(resilience_score: float, stage: str = "engaging") -> str:
    """
    회복탄력성 점수와 대화 단계에 따라 응답 전략 가이드를 만든다.
    """
    try:
        s = float(resilience_score)
    except Exception:
        s = 4.5
    s = max(0.0, min(10.0, s))
    band = "low" if s < 3.0 else "mid"

    stage_guide = {
        "low": {
            "engaging": "지금은 속도를 낮추고 공감 중심으로 반응하세요. 한 번에 질문은 하나만 하고, 사용자가 스스로 말할 여지를 충분히 둡니다.",
            "evoking": "감정을 단정하지 말고 열린 질문으로 확인하세요. 원인 탐색보다 현재 느낌을 안전하게 말하게 하는 데 집중합니다.",
            "conclusion": "정리는 짧고 부드럽게 끝내세요. 실천 제안은 아주 작고 부담 없는 수준으로 1개만 제시합니다.",
        },
        "mid": {
            "engaging": "공감으로 시작하되 핵심 맥락을 빠르게 확인하세요. 현재 상황을 한 문장으로 반영한 뒤 짧은 질문을 덧붙입니다.",
            "evoking": "감정-상황 연결을 구체화하도록 돕습니다. 필요하면 한 단계 더 깊은 질문으로 생각의 흐름을 정리합니다.",
            "conclusion": "대화에서 얻은 점을 간단히 요약하고 다음 행동 한 가지를 합의합니다. 과도한 조언은 피합니다.",
        },
    }

    guide = stage_guide[band].get(stage, stage_guide[band]["engaging"])
    return f"[resilience_band:{band} score={s:.2f}] [stage:{stage}]\n{guide}"


def make_biosignal_overview_plot(
    valid_signals,
    session_id: str | None,
    prt: str,
    day: str,
    base_dir: str = "plots",
) -> str | None:
    if not valid_signals:
        return None

    save_dir = os.path.join(base_dir, prt, day)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(valid_signals)

    time_col = "time"
    hr_col = "MeanHR" if "MeanHR" in df.columns else "PPG_MeanHR"
    lfn_col = "LFn" if "LFn" in df.columns else "PPG_LFn"
    hfn_col = "HFn" if "HFn" in df.columns else "PPG_HFn"

    if time_col not in df.columns:
        return None
    df = df[df[time_col].notna()].copy()
    if df.empty:
        return None

    for c in [lfn_col, hfn_col, hr_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

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

    max_slots = 12
    window = pd.Timedelta(minutes=freq_min * (max_slots - 1))

    best_start, best_count = uniq_dt.iloc[0], -1
    for s in uniq_dt:
        cnt = int(((uniq_dt >= s) & (uniq_dt <= s + window)).sum())
        if cnt > best_count or (cnt == best_count and s > best_start):
            best_count, best_start = cnt, s

    slots = pd.date_range(start=best_start, periods=max_slots, freq=f"{freq_min}min")
    df = df.set_index("_dt").reindex(slots)

    times = slots.strftime("%H").tolist()
    x_idx = np.arange(len(slots))
    lfn = df[lfn_col].to_numpy(dtype=float) if lfn_col in df.columns else np.full(len(slots), np.nan)
    hfn = df[hfn_col].to_numpy(dtype=float) if hfn_col in df.columns else np.full(len(slots), np.nan)
    hr = df[hr_col].to_numpy(dtype=float) if hr_col in df.columns else np.full(len(slots), np.nan)

    valid_mask = np.isfinite(lfn) | np.isfinite(hfn) | np.isfinite(hr)
    if int(valid_mask.sum()) < 6:
        return None

    valid_lfn = np.where(np.isfinite(lfn), lfn, 0)
    valid_hfn = np.where(np.isfinite(hfn), hfn, 0)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=120)
    fig.patch.set_facecolor("#fafafa")
    ax1.set_facecolor("#fafafa")
    ax1.grid(False)

    ax1.bar(
        x_idx,
        valid_lfn,
        width=0.8,
        color="#ef5350",
        alpha=0.35,
        label="긴장",
        zorder=1,
    )
    ax1.bar(
        x_idx,
        valid_hfn,
        width=0.8,
        bottom=valid_lfn,
        color="#42a5f5",
        alpha=0.35,
        label="이완",
        zorder=1,
    )
    ax1.set_ylim(bottom=0, top=1.0)
    ax1.set_xlabel("시간", fontsize=13, color="#222222")
    ax1.set_ylabel("긴장/이완 비율", fontsize=13, color="#222222")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#aaaaaa")
    ax1.spines["bottom"].set_color("#aaaaaa")
    ax1.tick_params(axis="y", labelsize=11, colors="#222222")
    ax1.tick_params(axis="x", labelsize=11, colors="#222222")

    ax2 = None
    if np.isfinite(hr).any():
        ax2 = ax1.twinx()
        ax2.grid(False)

        valid_hr_mask = np.isfinite(hr)
        ax2.plot(
            x_idx[valid_hr_mask],
            hr[valid_hr_mask],
            color="#e53935",
            linewidth=1.5,
            linestyle="--",
            alpha=0.5,
            zorder=3,
        )
        ax2.scatter(
            x_idx[valid_hr_mask],
            hr[valid_hr_mask],
            color="#e53935",
            s=70,
            zorder=4,
            label="심박수(bpm)",
        )

        for xi, yi in zip(x_idx[valid_hr_mask], hr[valid_hr_mask]):
            ax2.annotate(
                f"{int(round(yi))}",
                xy=(xi, yi),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=10,
                color="#c62828",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax2.set_ylabel("심박수(bpm)", fontsize=13, color="#e53935")
        ax2.tick_params(axis="y", colors="#e53935", labelsize=13)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["right"].set_color("#aaaaaa")

        hr_valid = hr[valid_hr_mask]
        hr_min = np.nanmin(hr_valid)
        hr_max = np.nanmax(hr_valid)
        ax2.set_ylim(bottom=max(0, hr_min - 15), top=hr_max + 20)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            fontsize=11,
            framealpha=0.0,
            edgecolor="none",
        )
    else:
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            fontsize=12,
            framealpha=0.0,
            edgecolor="none",
        )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(times, rotation=0, ha="center", fontsize=13)

    plt.title("시간대별 생체신호", fontsize=14, fontweight="bold", color="#222222", pad=14)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

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
            "random_turn": {
                "original_text": "",
                "prev_ai_text": "",
                "situation": "",
                "emotion": {"main": "", "sub": ""},
            },
            "consistency": {
                "situation_match": 0.0,
                "emotion_match": 0.0,
                "overall_match": 0.0,
            },
            "top_emotions": [],
            "last": {"original_text": "", "situation": "", "emotion": ""},
        }

    valid_items = [x for x in parsed if x["original_text"]] or parsed
    random_turn = random.choice(valid_items)
    last_analysis = valid_items[-1]

    def _tokenize_kr_en(text: str) -> set[str]:
        if not text:
            return set()
        return {t for t in re.findall(r"[0-9A-Za-z가-힣]+", text.lower()) if len(t) >= 2}

    original_tokens = _tokenize_kr_en(random_turn.get("original_text", ""))
    situation_tokens = _tokenize_kr_en(random_turn.get("situation", ""))
    union = original_tokens | situation_tokens
    inter = original_tokens & situation_tokens
    situation_match = (len(inter) / len(union)) if union else 0.0

    emotion_match = 0.0
    main = (random_turn.get("main") or "").strip().lower()
    sub = (random_turn.get("sub") or "").strip().lower()
    user_text_l = (random_turn.get("original_text") or "").lower()

    if main and main in user_text_l:
        emotion_match += 0.7
    if sub and sub in user_text_l:
        emotion_match += 0.3
    emotion_match = min(1.0, emotion_match)

    overall_match = round((situation_match + emotion_match) / 2.0, 2)

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
        "random_turn": {
            "original_text": random_turn["original_text"],
            "prev_ai_text": random_turn.get("prev_ai_text", ""),
            "situation": random_turn["situation"],
            "emotion": {
                "main": random_turn["main"],
                "sub": random_turn["sub"],
            },
        },
        "consistency": {
            "situation_match": round(situation_match, 2),
            "emotion_match": round(emotion_match, 2),
            "overall_match": overall_match,
        },
        "top_emotions": top_3_emotions,
        "last": {
            "original_text": last_analysis["original_text"],
            "situation": last_analysis["situation"],
            "emotion": last_analysis["main"],
        },
    }
