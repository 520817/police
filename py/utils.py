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


def normalize_stage_transition(
    current_stage: str,
    proposed_stage: str,
    analyzed_turn_count: int = 0,
) -> str:
    current = str(current_stage or "engaging").strip().lower()
    proposed = str(proposed_stage or current or "engaging").strip().lower()
    valid = {"engaging", "evoking", "conclusion"}
    if current not in valid:
        current = "engaging"
    if proposed not in valid:
        proposed = current
    try:
        analyzed_turn_count = int(analyzed_turn_count)
    except Exception:
        analyzed_turn_count = 0

    stage_order = ["engaging", "evoking", "conclusion"]
    current_idx = stage_order.index(current)
    proposed_idx = stage_order.index(proposed)

    # 두 단계 이상 건너뛰기 금지 (engaging → conclusion 직행 포함)
    if proposed_idx - current_idx > 1:
        return stage_order[current_idx + 1]

    # conclusion에서 후퇴 금지
    if current == "conclusion" and proposed_idx < current_idx:
        return "conclusion"

    return proposed


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
    base_dir: str = "plots",
) -> str | None:
    if not valid_signals:
        return None

    save_dir = os.path.join(base_dir, prt, day)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(valid_signals)

    time_col = "time"
    hr_col = "MeanHR" if "MeanHR" in df.columns else "PPG_MeanHR"
    stress_col = "Stress"

    if time_col not in df.columns:
        return None
    df = df[df[time_col].notna()].copy()
    if df.empty:
        return None

    for c in [stress_col, hr_col]:
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
    stress = df[stress_col].to_numpy(dtype=float) if stress_col in df.columns else np.full(len(slots), np.nan)
    hr = df[hr_col].to_numpy(dtype=float) if hr_col in df.columns else np.full(len(slots), np.nan)

    valid_mask = np.isfinite(stress) | np.isfinite(hr)
    if int(valid_mask.sum()) < 6:
        return None

    COLOR_STRESS = "#ef5350"   # 빨강 (스트레스)
    COLOR_CALM   = "#42a5f5"   # 파랑 (안정)
    COLOR_NODATA = "#e0e0e0"   # 회색 (데이터 없음)
    bar_colors = np.where(stress == 1, COLOR_STRESS, COLOR_CALM)

    COLOR_DOT = "#546E7A"   # 심박수 점 색상 (청회색, 빨강/파랑과 안겹침)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, dpi=130,
        figsize=(8, 4.2),
        gridspec_kw={"height_ratios": [0.5, 2.2]},
    )
    fig.patch.set_facecolor("#fafafa")
    ax_top.set_facecolor("#fafafa")
    ax_bot.set_facecolor("#fafafa")

    # ── 상단 패널: 스트레스/안정 bar (텍스트 없음) ──
    for xi, has_data, color in zip(x_idx, np.isfinite(stress), bar_colors):
        if has_data:
            ax_top.bar(xi, 1, width=1.0, align="edge", color=color, alpha=0.55, zorder=1)
        else:
            ax_top.bar(xi, 1, width=1.0, align="edge", color=COLOR_NODATA, alpha=0.4, zorder=1)

    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_linewidth(0.8)
    ax_top.spines["bottom"].set_color("#cccccc")

    # ── 상단 패널: 슬롯 구분 세로선 ──
    for xi in x_idx[1:]:
        ax_top.axvline(x=xi, color="#cccccc", linewidth=0.6, zorder=2)

    # ── 하단 패널: BPM 선 그래프 ──
    valid_hr_mask = np.isfinite(hr)
    if valid_hr_mask.any():
        x_center = x_idx + 0.5   # bar 중앙 정렬

        # 전체 연결
        ax_bot.plot(
            x_center[valid_hr_mask],
            hr[valid_hr_mask],
            color="#bbbbbb",
            linewidth=1.2,
            zorder=2,
        )

        ax_bot.scatter(
            x_center[valid_hr_mask],
            hr[valid_hr_mask],
            c=COLOR_DOT,
            s=55,
            zorder=3,
        )
        for xi, yi in zip(x_center[valid_hr_mask], hr[valid_hr_mask]):
            ax_bot.annotate(
                f"{int(round(yi))}",
                xy=(xi, yi),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=10,
                color=COLOR_DOT,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        hr_valid = hr[valid_hr_mask]
        y_bot = min(30, int(np.nanmin(hr_valid)) - 5)
        y_top = max(120, int(np.nanmax(hr_valid)) + 10)
        y_top = min(y_top, 180)
        ax_bot.set_ylim(bottom=y_bot, top=y_top)
        ax_bot.set_yticks(range(y_bot, y_top + 1, 20))
        ax_bot.tick_params(axis="y", labelsize=10, colors="#555555")

    # 하단 패널 세로 grid (시간대 구분)
    ax_bot.set_xticks(x_idx + 0.5)
    ax_bot.set_xticklabels(times, rotation=0, ha="center", fontsize=11)
    ax_bot.tick_params(axis="x", labelsize=11, colors="#222222")
    for xi in x_idx[1:]:
        ax_bot.axvline(x=xi, color="#e0e0e0", linewidth=0.6, zorder=1)

    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.spines["left"].set_color("#cccccc")
    ax_bot.spines["bottom"].set_color("#cccccc")

    # ── 범례 (상단 패널 위) ──
    legend_patches = [
        Patch(facecolor=COLOR_STRESS, alpha=0.55, label="스트레스"),
        Patch(facecolor=COLOR_CALM,   alpha=0.55, label="안정"),
        Patch(facecolor=COLOR_NODATA, alpha=0.4,  label="데이터 없음"),
    ]
    from matplotlib.lines import Line2D
    hr_handle = Line2D([0], [0], marker="o", color="#bbbbbb", markerfacecolor=COLOR_DOT,
                       markersize=6, label="심박수(bpm)", linewidth=1.2)
    ax_top.legend(
        handles=legend_patches + [hr_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=10,
        framealpha=0.0,
        edgecolor="none",
    )

    # ── 공통 ──
    fig.suptitle("오늘의 신체 상태", fontsize=15, fontweight="bold", color="#222222", y=1.06)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, bottom=0.14)

    fig.text(
        0.5, 0.02,
        "* 스트레스/안정 구분은 심박수 외 여러 HRV 지표를 종합해 판단합니다.",
        ha="center", fontsize=11, color="#666666",
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
