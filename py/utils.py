from typing import Any, Iterable, List, Dict, Optional
import json
import os
import re
import random

import numpy as np
import pandas as pd

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


def make_biosignal_html(
    valid_signals,
    shift_type: str = "day",
) -> str | None:
    if not valid_signals:
        return None

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

    df["_dt"] = _parse_dt_series(
        df[time_col].iloc[0][:10] if str(df[time_col].iloc[0]).count("-") >= 2
        else str(pd.Timestamp.today().date()),
        df[time_col].astype(str),
    )
    df = df[df["_dt"].notna()].copy()
    if df.empty:
        return None

    df = df.sort_values("_dt")
    slot_map = {}
    for _, row in df.iterrows():
        h = row["_dt"].hour
        key = f"{h:02d}"
        if key not in slot_map:
            slot_map[key] = row.get(stress_col, np.nan)

    if not slot_map:
        return None

    is_duty = shift_type == "duty"
    slot_count = 24 if is_duty else 12

    base_h = int(sorted(slot_map.keys())[0])
    slots = []
    for i in range(slot_count):
        h = (base_h + i) % 24
        key = f"{h:02d}"
        val = slot_map.get(key, np.nan)
        if pd.isna(val):
            state = "n"
        elif int(val) == 1:
            state = "s"
        else:
            state = "c"
        slots.append({"h": key, "s": state})

    shift_labels = {
        "day": "주간", "night": "야간", "off": "비번",
        "duty": "당직", "holiday": "휴무",
    }
    shift_label = shift_labels.get(shift_type, shift_type)
    hour_label = "24시간" if is_duty else "12시간"
    badge_color = "duty" if is_duty else "day"

    def make_row_html(row_slots):
        time_cells = "".join(f'<div class="bio-tlbl">{d["h"]}</div>' for d in row_slots)
        tile_cells = "".join(f'<div class="bio-tile bio-tile--{d["s"]}"></div>' for d in row_slots)
        return (
            f'<div class="bio-timerow">{time_cells}</div>'
            f'<div class="bio-tilerow">{tile_cells}</div>'
        )

    if is_duty:
        rows_html = (
            '<div class="bio-rowlabel">오전</div>'
            + make_row_html(slots[:12])
            + '<div class="bio-rowlabel" style="margin-top:8px;">오후</div>'
            + make_row_html(slots[12:])
        )
    else:
        rows_html = make_row_html(slots)

    html = (
        '<div class="bio-card">'
        '<div class="bio-head">'
        '<span class="bio-title">오늘의 신체 상태</span>'
        f'<span class="bio-badge bio-badge--{badge_color}">{shift_label} · {hour_label}</span>'
        '</div>'
        '<div class="bio-legend">'
        '<span class="bio-leg"><span class="bio-dot bio-dot--s"></span>스트레스</span>'
        '<span class="bio-leg"><span class="bio-dot bio-dot--c"></span>안정</span>'
        '<span class="bio-leg"><span class="bio-dot bio-dot--n"></span>데이터 없음</span>'
        '</div>'
        f'<div class="bio-body">{rows_html}</div>'
        '</div>'
        '<style>'
        '.bio-card{background:#fff;border:0.5px solid #e0e0e0;border-radius:12px;padding:14px 16px 12px;max-width:100%;box-sizing:border-box;}'
        '.bio-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}'
        '.bio-title{font-size:13px;font-weight:500;color:#111;}'
        '.bio-badge{font-size:11px;padding:2px 8px;border-radius:10px;}'
        '.bio-badge--day{background:#E6F1FB;color:#185FA5;}'
        '.bio-badge--duty{background:#FAEEDA;color:#854F0B;}'
        '.bio-legend{display:flex;gap:12px;margin-bottom:10px;}'
        '.bio-leg{display:flex;align-items:center;gap:4px;font-size:11px;color:#666;}'
        '.bio-dot{display:inline-block;width:10px;height:10px;border-radius:2px;flex-shrink:0;}'
        '.bio-dot--s{background:#F09595;}'
        '.bio-dot--c{background:#85B7EB;}'
        '.bio-dot--n{background:#ebebeb;border:0.5px solid #ccc;}'
        '.bio-rowlabel{font-size:10px;color:#999;margin-bottom:2px;}'
        '.bio-timerow{display:flex;align-items:center;margin-bottom:3px;}'
        '.bio-tlbl{flex:1;display:flex;align-items:center;justify-content:center;text-align:center;font-size:11px;line-height:1;color:#666;padding-bottom:0;height:14px;}'
        '.bio-tilerow{display:flex;gap:2px;}'
        '.bio-tile{flex:1;height:32px;border-radius:3px;}'
        '.bio-tile--s{background:#F09595;}'
        '.bio-tile--c{background:#85B7EB;}'
        '.bio-tile--n{background:#ebebeb;border:0.5px solid #ddd;}'
        '</style>'
    )
    return html


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
                {"original_text": "", "prev_ai_text": "", "situation": "", "emotion": {"main": "", "sub": ""}},
                {"original_text": "", "prev_ai_text": "", "situation": "", "emotion": {"main": "", "sub": ""}},
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
            "original_text": "", "prev_ai_text": "", "situation": "", "main": "", "sub": "",
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
            emotion_totals.items(), key=lambda x: x[1], reverse=True,
        )[:3]
    ]

    return {
        "random_turns": [
            {
                "original_text": item["original_text"],
                "prev_ai_text": item.get("prev_ai_text", ""),
                "situation": item["situation"],
                "emotion": {"main": item["main"], "sub": item["sub"]},
            }
            for item in sampled_turns[:2]
        ],
        "top_emotions": top_3_emotions,
    }
