// src/vas_sam.js
import React, { useMemo, useState, useEffect } from "react";
import "./vas_sam.css";

/* ---------------------------
 * CheckDot
 * --------------------------- */
function CheckDot({ checked }) {
  return (
    <div className="checkdot" aria-hidden="true">
      {checked ? <div className="checkdot__inner" /> : null}
    </div>
  );
}

/* ---------------------------
 * SamCircle
 * --------------------------- */
function SamCircle({
  title,
  value,
  onChange,
  size = 380,
  faceSizeOuter = 68,
  faceSizeCenter = 72,
}) {
  const faces = useMemo(() => [1, 2, 3, 4, 5, 6, 7, 8, 9], []);
  const centerValue = 5;
  const outerFaces = faces.filter((v) => v !== centerValue);

  const center = size / 2;
  const radius = size * 0.35;

  const cardClass = (active) => `sam__card ${active ? "sam__card--active" : ""}`;

  return (
    <div className="sam">
      <div className="sam__title">{title}</div>

      <div className="sam__intro">
        지금 이 순간 느껴지는 감정에 가장 가까운 표정을 선택해 주세요.
      </div>

      {/* ✅ 세로축 + 스테이지 묶음 */}
      <div className="sam__wrap">
        {/* ✅ Arousal (세로 한 줄/끝 라벨) */}
        <div className="sam__axisY" style={{ height: size }} aria-hidden="true">
          <div className="sam__axisYTop">에너지↑</div>
          <div className="sam__axisYMid">Arousal</div>
          <div className="sam__axisYBot">에너지↓</div>
        </div>

        {/* ✅ SAM stage */}
        <div className="sam__stage" style={{ width: size, height: size }}>
          {/* 중앙(5) */}
          <button
            type="button"
            onClick={() => onChange(centerValue)}
            className={cardClass(value === centerValue)}
            style={{
              position: "absolute",
              left: center,
              top: center,
              transform: "translate(-50%, -50%)",
            }}
            aria-pressed={value === centerValue}
          >
            <img
              className="sam__img"
              src={`/sam${centerValue}.png`}
              alt={`sam ${centerValue}`}
              width={faceSizeCenter}
              height={faceSizeCenter}
            />
            <CheckDot checked={value === centerValue} />
          </button>

          {/* 바깥 8개 */}
          {outerFaces.map((v, i) => {
            const angle = (2 * Math.PI * i) / outerFaces.length - Math.PI / 2;
            const left = center + radius * Math.cos(angle);
            const top = center + radius * Math.sin(angle);

            return (
              <button
                key={v}
                type="button"
                onClick={() => onChange(v)}
                className={cardClass(value === v)}
                style={{
                  position: "absolute",
                  left,
                  top,
                  transform: "translate(-50%, -50%)",
                }}
                aria-pressed={value === v}
              >
                <img
                  className="sam__img"
                  src={`/sam${v}.png`}
                  alt={`sam ${v}`}
                  width={faceSizeOuter}
                  height={faceSizeOuter}
                />
                <CheckDot checked={value === v} />
              </button>
            );
          })}
        </div>
      </div>

      {/* ✅ Valence (가로 한 줄) */}
      <div className="sam__axisX" aria-hidden="true">
        <span className="sam__axisXLeft">부정/불쾌</span>
        <span className="sam__axisXLine">Valence</span>
        <span className="sam__axisXRight">긍정/쾌</span>
      </div>
    </div>
  );
}

/* ---------------------------
 * PrecheckModalInner (hooks here)
 * --------------------------- */
function PrecheckModalInner({
  onClose,
  onDone,
  initialVas = 0,
  initialSam = null,
  phase = "pre", // [ADD] "pre" | "post"
}) {
  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, []);

  const [vas, setVas] = useState(initialVas);
  const [vasTouched, setVasTouched] = useState(false);

  const [sam, setSam] = useState(initialSam ?? null);

  const canSubmit = vasTouched && sam !== null;

  const isMobile = typeof window !== "undefined" && window.innerWidth <= 480;

  // 모달 패딩 감안해서 안전하게 계산
  const samSize = useMemo(() => {
    if (typeof window === "undefined") return 320;
    const vw = window.innerWidth;
    const max = 380;
    const min = 240;
    const s = Math.floor(vw * (isMobile ? 0.78 : 0.62));
    return Math.max(min, Math.min(max, s));
  }, [isMobile]);

  const faceOuter = Math.round(samSize * 0.18);
  const faceCenter = Math.round(samSize * 0.19);

  // [ADD] phase별 문구
  const title = phase === "post" ? "대화 후 상태 체크" : "대화 전 상태 체크";
  const desc =
    phase === "post"
      ? "대화를 마치기 전, 현재 상태를 선택한 뒤 제출 버튼을 클릭하세요."
      : "대화를 시작하기 전, 현재 상태를 선택한 뒤 제출 버튼을 클릭하세요.";
  const submitLabel = phase === "post" ? "제출하고 대화 종료" : "제출하고 대화 시작";

  const handleSubmit = () => {
    if (!canSubmit) return;

    const payload = {
      type: phase, // [CHANGED] "pre" | "post"
      vas,
      sam: sam,
      created_at: new Date().toISOString(),
    };

    // 저장 키도 phase로 분리하고 싶으면 아래처럼:
    // localStorage.setItem(`${phase}check`, JSON.stringify(payload));
    localStorage.setItem("precheck", JSON.stringify(payload)); // 기존 유지(원하면 바꿔도 됨)

    onDone?.(payload);
    onClose?.();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="precheckOverlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div className="precheckModal">
        {/* 헤더 */}
        <div className="precheckHeader">
          <div>
            <h1>{title}</h1>
            <p>{desc}</p>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="닫기"
            className="precheckCloseBtn"
          >
            ✕
          </button>
        </div>

        {/* VAS */}
        <div className="vasBlock">
          <div className="vasLabel">현재 스트레스 정도 (0=전혀 없음, 100=매우 심함)</div>

          <div className="vasInlineRow">
            <input
              className="vasSlider"
              type="range"
              min="0"
              max="100"
              step="1"
              value={vas}
              onChange={(e) => {
                setVasTouched(true);
                setVas(Number(e.target.value));
              }}
            />
            <div className="vasValue">{vasTouched ? vas : "-"}</div>
          </div>

          {!vasTouched ? (
            <div className="vasHelp">슬라이더를 한 번 움직여서 값을 선택해 주세요.</div>
          ) : null}
        </div>

        {/* SAM */}
        <div className="samGrid">
          <div className="samCol">
            <SamCircle
              title="SAM"
              value={sam}
              onChange={setSam}
              size={samSize}
              faceSizeOuter={faceOuter}
              faceSizeCenter={faceCenter}
            />
          </div>
        </div>

        {/* 제출 */}
        <div className="precheckFooter">
          <button
            type="button"
            disabled={!canSubmit}
            onClick={handleSubmit}
            className={[
              "precheckSubmitBtn",
              canSubmit ? "precheckSubmitBtn--enabled" : "precheckSubmitBtn--disabled",
            ].join(" ")}
          >
            {submitLabel}
          </button>

          {!canSubmit ? (
            <div className="precheckWarn">스트레스 정도와 정서 상태를 모두 선택해야 합니다.</div>
          ) : (
            <div className="precheckOk">선택 완료!</div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------------------
 * Wrapper: open check here
 * --------------------------- */
export default function PrecheckModal(props) {
  if (!props.open) return null;
  return <PrecheckModalInner {...props} />;
}
