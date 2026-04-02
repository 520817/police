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
  const cardClass = (active) => `sam__card ${active ? "sam__card--active" : ""}`;

  const points = useMemo(() => {
    return [
      { key: 1, img: 9, x: 0.32, y: 0.21 },
      { key: 2, img: 1, x: 0.68, y: 0.21 },

      { key: 3, img: 8, x: 0.18, y: 0.50 },
      { key: 5, img: 5, x: 0.50, y: 0.50 },
      { key: 7, img: 2, x: 0.82, y: 0.50 },

      { key: 8, img: 7, x: 0.32, y: 0.79 },
      { key: 9, img: 3, x: 0.68, y: 0.79 },
    ];
  }, []);

  return (
    <div className="sam">
      <div className="sam__title">{title}</div>

      <div className="sam__intro">
        지금 이 순간 느껴지는 감정에 가장 가까운 표정을 선택해 주세요.
      </div>

      <div className="sam__wrap">
        <div className="sam__axisY" style={{ height: size }} aria-hidden="true">
          <div className="sam__axisYTop">에너지↑</div>
          <div className="sam__axisYBot">에너지↓</div>
        </div>

        <div className="sam__stage" style={{ width: size, height: size }}>
          {points.map(({ key, img, x, y }) => {
            const isCenter = key === 5;
            const faceSize = isCenter ? faceSizeCenter : faceSizeOuter;

            return (
              <button
                key={key}
                type="button"
                onClick={() => onChange(key)}
                className={cardClass(value === key)}
                style={{
                  position: "absolute",
                  left: size * x,
                  top: size * y,
                  transform: "translate(-50%, -50%)",
                }}
                aria-pressed={value === key}
              >
                <img
                  className="sam__img"
                  src={`/sam${img}.png`}
                  alt={`sam ${img}`}
                  width={faceSize}
                  height={faceSize}
                />
                <CheckDot checked={value === key} />
              </button>
            );
          })}
        </div>
      </div>

      <div className="sam__axisX" aria-hidden="true">
        <span className="sam__axisXLeft">부정적</span>
        <span className="sam__axisXRight">긍정적</span>
      </div>
    </div>
  );
}

// Russell(1980) circumplex model 기반으로 각 이모지에 valence/arousal 점수 할당.
// valence: 1(매우 부정) ~ 9(매우 긍정)
// arousal: 1(매우 낮음) ~ 9(매우 높음)
// SAM 원본 척도에 따라 홀수값만 사용.
const SAM_COORDS = {
  1: { valence: -3, arousal:  1 }, // 😤 분노 — 매우 부정, 고각성
  2: { valence:  3, arousal:  1 }, // 😄 기쁨 — 매우 긍정, 고각성
  3: { valence: -2, arousal:  0 }, // 😣 짜증 — 부정, 중각성
  5: { valence:  0, arousal:  0 }, // 😐 중립 — 중간, 중각성
  7: { valence:  2, arousal:  0 }, // 😊 만족 — 긍정, 중각성
  8: { valence: -1, arousal: -1 }, // 😟 우울 — 약부정, 저각성
  9: { valence:  1, arousal: -1 }, // 😌 평온 — 약긍정, 저각성
};

/* ---------------------------
 * Validation Step
 * --------------------------- */
function RatingRow({ number, question, originalText, situation, emotion, value, onChange }) {
  return (
    <div className="ratingRow">
      <div className="ratingRow__label">
        {number}. {question}
      </div>

      {(originalText || situation || emotion) && (
        <div className="analysisChips">
          {originalText && (
            <div className="analysisChip analysisChip--original">
              <span className="analysisChip__label">말씀하신 내용</span>
              <span className="analysisChip__text">"{originalText}"</span>
            </div>
          )}

          {situation && (
            <div className="analysisChip analysisChip--situation">
              <span className="analysisChip__label">상황</span>
              <span className="analysisChip__text">{situation}</span>
            </div>
          )}

          {emotion && (
            <div className="analysisChip analysisChip--emotion">
              <span className="analysisChip__label">감정</span>
              <span className="analysisChip__text">{emotion}</span>
            </div>
          )}
        </div>
      )}

      <div className="ratingInlineScale">
        <span className="ratingInlineScale__label">전혀 맞지 않음</span>

        <div className="ratingRow__buttons" role="radiogroup">
          {[1, 2, 3, 4, 5].map((n) => (
            <button
              key={n}
              type="button"
              className={`ratingBtn ${value === n ? "ratingBtn--active" : ""}`}
              onClick={() => onChange(n)}
              aria-pressed={value === n}
            >
              {n}
            </button>
          ))}
        </div>

        <span className="ratingInlineScale__label">매우 맞음</span>
      </div>
    </div>
  );
}

function ValidationStep({ validationData, scores, setScores }) {
  const peakOriginal = validationData?.random_turn?.original_text || "오늘 랜덤 발화(직전 AI까지)";
  const peakSituation = validationData?.random_turn?.situation || "상황 분석 결과";
  const peakEmotion = validationData?.random_turn?.emotion?.main || "감정 분석 결과";

  const lastSituation = validationData?.last?.situation || "마지막 상황 분석 결과";
  const lastEmotion = validationData?.last?.emotion || "마지막 감정 분석 결과";

  const mainEmotionRaw = validationData?.top_emotions?.[0] || "";

  const mainEmotion =
    mainEmotionRaw && mainEmotionRaw !== "unknown"
      ? mainEmotionRaw
      : "특정한 한 가지 감정으로 정리하기 어려운 상태";

  return (
    <div className="validationBlock">
      <RatingRow
        number={1}
        question="오늘 대화 중 한 순간을 AI가 이렇게 이해했어요. 실제와 맞았나요?"
        originalText={peakOriginal}
        situation={peakSituation}
        emotion={peakEmotion}
        value={scores.q1}
        onChange={(v) => setScores((prev) => ({ ...prev, q1: v }))}
      />

      <RatingRow
        number={2}
        question="오늘 대화에서 AI가 가장 많이 감지한 감정이에요. 실제로 공감되시나요?"
        emotion={mainEmotion}
        value={scores.q2}
        onChange={(v) => setScores((prev) => ({ ...prev, q2: v }))}
      />

      <RatingRow
        number={3}
        question="대화 마지막 순간 AI가 이렇게 이해했어요. 실제와 맞았나요?"
        situation={lastSituation}
        emotion={lastEmotion}
        value={scores.q3}
        onChange={(v) => setScores((prev) => ({ ...prev, q3: v }))}
      />
    </div>
  );
}

/* ---------------------------
 * PrecheckModalInner
 * --------------------------- */
function PrecheckModalInner({
  onClose,
  onDone,
  initialVas = 0,
  initialSam = null,
  phase = "pre",
  validationData = null,
}) {
  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, []);

  const [step, setStep] = useState(1);
  const [vas, setVas] = useState(initialVas);
  const [vasTouched, setVasTouched] = useState(false);
  const [sam, setSam] = useState(initialSam ?? null);

  const [scores, setScores] = useState({
    q1: null,
    q2: null,
    q3: null,
  });

  const canSubmitSam = vasTouched && sam !== null;
  const canSubmitValidation = scores.q1 && scores.q2 && scores.q3;

  const isDataInsufficient = useMemo(() => {
    if (!validationData) return true;
    const hasPeak = !!validationData.random_turn?.original_text;
    const hasEmotion =
      !!validationData.top_emotions?.[0] &&
      validationData.top_emotions[0] !== "unknown";
    return !hasPeak || !hasEmotion;
  }, [validationData]);

  const isMobile = typeof window !== "undefined" && window.innerWidth <= 480;
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

  const title =
    phase === "post"
      ? step === 1
        ? "대화 후 상태 체크"
        : "대화 내용 확인"
      : "대화 전 상태 체크";

  const desc = useMemo(() => {
    if (phase === "pre")
      return "대화를 시작하기 전, 현재 상태를 선택한 뒤 제출 버튼을 클릭하세요.";
    if (step === 1) {
      return isDataInsufficient
        ? "대화 내용이 짧아 분석 없이 종료합니다. 현재 상태를 선택해 주세요."
        : "대화를 마치기 전, 현재 상태를 선택한 뒤 다음 버튼을 눌러 주세요.";
    }
    return "오늘 나눈 대화에 대한 AI의 분석 결과가 적절했는지 평가해 주세요.";
  }, [phase, step, isDataInsufficient]);

  const handleNext = () => {
    if (!canSubmitSam) return;
    if (phase === "post" && isDataInsufficient) {
      handleFinalSubmit();
      return;
    }
    setStep(2);
  };

  const handleFinalSubmit = () => {
    const coords = SAM_COORDS[sam];
    const payload = {
      type: phase,
      vas,
      valence: coords?.valence ?? null, // 1(매우 부정) ~ 9(매우 긍정)
      arousal: coords?.arousal ?? null, // 1(매우 낮음) ~ 9(매우 높음)
      created_at: new Date().toISOString(),
    };

    if (phase === "post") {
      const peakSituation = validationData?.random_turn?.situation || "";
      const peakEmotion = validationData?.random_turn?.emotion?.main || "";
      const mainEmotion = validationData?.top_emotions?.[0] || "";
      const lastSituation = validationData?.last?.situation || "";
      const lastEmotion = validationData?.last?.emotion || "";

      payload.validation = {
        q1: scores.q1,
        q2: scores.q2,
        q3: scores.q3,
        q1_text: peakSituation + " / " + peakEmotion,  // AI가 분석한 상황+감정
        q2_text: mainEmotion,                            // AI가 감지한 주요 감정
        q3_text: lastSituation + " / " + lastEmotion,   // 마지막 상황+감정
      };
      payload.validation_detail = validationData ?? null;
      payload.is_insufficient = isDataInsufficient;
    }

    localStorage.setItem(`${phase}check`, JSON.stringify(payload));
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
        <div className="precheckHeader">
          <div>
            <h1>{title}</h1>
            <p
              style={{
                color:
                  isDataInsufficient && phase === "post" ? "#d32f2f" : "#666",
              }}
            >
              {desc}
            </p>
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

        {(phase === "pre" || (phase === "post" && step === 1)) && (
          <>
            <div className="vasBlock">
              <div className="vasLabel">
                현재 스트레스 정도 (0=전혀 없음, 100=매우 심함)
              </div>

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

              {!vasTouched && (
                <div className="vasHelp">
                  슬라이더를 움직여 값을 선택해 주세요.
                </div>
              )}
            </div>

            <div className="samGrid">
              <div className="samCol">
                <SamCircle
                  title="현재 정서 상태"
                  value={sam}
                  onChange={setSam}
                  size={samSize}
                  faceSizeOuter={faceOuter}
                  faceSizeCenter={faceCenter}
                />
              </div>
            </div>
          </>
        )}

        {phase === "post" && step === 2 && !isDataInsufficient && (
          <ValidationStep
            validationData={validationData}
            scores={scores}
            setScores={setScores}
          />
        )}

        <div className="precheckFooter">
          {phase === "pre" && (
            <button
              type="button"
              disabled={!canSubmitSam}
              onClick={handleFinalSubmit}
              className={`precheckSubmitBtn ${
                canSubmitSam
                  ? "precheckSubmitBtn--enabled"
                  : "precheckSubmitBtn--disabled"
              }`}
            >
              제출하고 대화 시작
            </button>
          )}

          {phase === "post" && step === 1 && (
            <button
              type="button"
              disabled={!canSubmitSam}
              onClick={handleNext}
              className={`precheckSubmitBtn ${
                canSubmitSam
                  ? "precheckSubmitBtn--enabled"
                  : "precheckSubmitBtn--disabled"
              }`}
            >
              {isDataInsufficient ? "제출하고 대화 종료" : "다음 (분석 결과 확인)"}
            </button>
          )}

          {phase === "post" && step === 2 && (
            <>
              <button
                type="button"
                onClick={() => setStep(1)}
                className="precheckBackBtn"
              >
                이전
              </button>

              <button
                type="button"
                disabled={!canSubmitValidation}
                onClick={handleFinalSubmit}
                className={`precheckSubmitBtn ${
                  canSubmitValidation
                    ? "precheckSubmitBtn--enabled"
                    : "precheckSubmitBtn--disabled"
                }`}
              >
                제출하고 대화 종료
              </button>
            </>
          )}

          {step === 1 && !canSubmitSam && (
            <div className="precheckWarn">
              스트레스 정도와 정서 상태를 모두 선택해야 합니다.
            </div>
          )}
          {step === 2 && !canSubmitValidation && (
            <div className="precheckWarn">세 문항 모두 점수를 선택해 주세요.</div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------------------
 * Wrapper
 * --------------------------- */
export default function PrecheckModal(props) {
  if (!props.open) return null;
  return <PrecheckModalInner {...props} />;
}
