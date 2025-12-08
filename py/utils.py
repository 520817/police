from typing import Any, Iterable, List, Dict, Optional
import pandas as pd
import os
import uuid
import matplotlib
from matplotlib import font_manager, rcParams
matplotlib.use("Agg")  # 서버 사이드 렌더링 (디스플레이 없는 환경)
import matplotlib.pyplot as plt
rcParams["font.family"] = "Malgun Gothic"

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
    dict 또는 list[dict]에서 키가 'PPG_'로 시작하면 접두어 제거
    """
    if isinstance(obj, list):
        return [remove_ppg_prefix(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = k
            if isinstance(k, str) and k.startswith("PPG_"):
                nk = k[4:]  # 'PPG_' 제거 → PPG_MeanHR -> MeanHR
            out[nk] = v
        return out
    return obj

    
def make_biosignal_overview_plot(valid_signals, session_id: str | None, save_dir: str = "plots") -> str | None:
    """
    valid_signals: data.py에서 N/A 제거한 뒤 넘겨주는 list[dict]
    session_id: 없으면 'nosess'
    save_dir: FastAPI에서 /plots로 서빙할 디렉터리
    """
    import os
    import uuid
    import pandas as pd
    import matplotlib.pyplot as plt

    if not valid_signals:
        return None

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.DataFrame(valid_signals)

    time_col = "time"
    hr_col = "MeanHR" if "MeanHR" in df.columns else "PPG_MeanHR"
    lfn_col = "LFn" if "LFn" in df.columns else "PPG_LFn"
    hfn_col = "HFn" if "HFn" in df.columns else "PPG_HFn"

    for c in [lfn_col, hfn_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0
    if hr_col in df.columns:
        df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")

    times = df[time_col].tolist()
    lfn_vals = df[lfn_col].tolist()
    hfn_vals = df[hfn_col].tolist()
    hfn_stacked = (df[lfn_col] + df[hfn_col]).tolist()
    hr_vals = df[hr_col].tolist() if hr_col in df.columns else None

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=110)

    ax1.set_xlabel("시각")
    ax1.set_ylabel("긴장 / 회복 비율 (%)")

    # 그대로 두되 alpha=0.6 유지
    ax1.fill_between(
        times,
        lfn_vals,
        hfn_stacked,
        label="회복 신호 비율",
        color="#8ad2ff",
        alpha=0.3,
        step="mid",
    )
    ax1.fill_between(
        times,
        lfn_vals,
        0,
        label="긴장 신호 비율",
        color="#ff7373",
        alpha=0.3,
        step="mid",
    )

    ax1.set_ylim(bottom=0)

    if hr_vals is not None:
        ax2 = ax1.twinx()
    
        # 문자열 times 대신 숫자 인덱스로 플롯 (0,1,2,...)
        x_idx = list(range(len(times)))
        #line = ax2.plot(x_idx, hr_vals, marker="o", label="심장박동수(bpm)", color="#ff5252")
        ax2.plot(x_idx, hr_vals, color="#ff5252", linewidth=2, label="심장박동수(bpm)")
        
        for xi, yi in zip(x_idx, hr_vals):
            if pd.isna(yi):
                continue
            ax2.text(
                xi, yi, "♥", color="#ff5252",
                fontsize=12, ha="center", va="center"
            )
        
        ax2.set_ylabel("심장박동수 (bpm)")
    
        # 점 옆에 숫자 표시 (x축으로 살짝 오른쪽 이동)
        for i, y in enumerate(hr_vals):
            if pd.isna(y):
                continue
            ax2.text(
                i + 0.3,          # ▶ 점 오른쪽으로 0.3만큼 이동
                y,                # 같은 높이
                f"{int(y)}", 
                ha="left",        # 글자를 점 오른쪽에 맞추기
                va="center",
                fontsize=9,
            )
    
        # x축 라벨을 실제 시간 문자열로 교체
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(times, rotation=45, ha="right")
    
        # 범례 합치기
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    
    else:
        ax1.legend(loc="upper left")

    # 테두리 회색
    for ax in [ax1] + ([ax2] if hr_vals is not None else []):
        for spine in ax.spines.values():
            spine.set_color("gray")
            spine.set_alpha(0.2)
            spine.set_linewidth(1)

    plt.title("생체 반응 흐름")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    sess = session_id or "nosess"
    img_name = f"biosignal_{sess}_{uuid.uuid4().hex[:8]}.png"
    img_path = os.path.join(save_dir, img_name)
    fig.savefig(img_path, bbox_inches="tight")
    plt.close(fig)

    public_url = f"/plots/{img_name}"

    return public_url
