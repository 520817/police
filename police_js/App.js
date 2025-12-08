// src/App.js
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const apiOrigin = "http://localhost:8000"; // 백엔드 origin

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function Typewriter({ text, speed = 30, onDone, onStep }) {
  const [out, setOut] = useState("");
  const iRef = useRef(0);

  useEffect(() => {
    setOut("");
    iRef.current = 0;
    const timer = setInterval(() => {
      iRef.current += 1;
      const next = text.slice(0, iRef.current);
      setOut(next);
      onStep?.();
      if (iRef.current >= text.length) {
        clearInterval(timer);
        onDone?.();
      }
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed, onDone, onStep]);

  return <span>{out}</span>;
}

// ✅ 이미지 말풍선 (생체신호 플롯)
function ImageMessageBubble({ src }) {
  if (!src) return null;
  return (
    <div className="ai-message">
      <div className="bio-plot-wrapper">
        <b>AI</b>:
        <div className="bio-plot-box">
          <img
            src={src}
            alt="생체신호 그래프"
            className="bio-plot-img"
            style={{
              maxWidth: "100%",
              borderRadius: "8px",
              border: "1px solid #ccc",
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [currentTypingId, setCurrentTypingId] = useState(null);

  // 🔥 sessionId 제거 → 대화 시작 여부만 관리
  const [started, setStarted] = useState(false);

  const [dept, setDept] = useState("");
  const [rank, setRank] = useState("");
  const [shiftType, setShiftType] = useState("day");
  const [starting, setStarting] = useState(false);

  // ✅ 생체신호 동의 상태
  // "unknown" | "accepted" | "declined" | "ended"
  const [consentState, setConsentState] = useState("unknown");

  // ✅ 전화번호 기반 user_id 관리
  const [userId, setUserId] = useState(null);
  const [showPhoneModal, setShowPhoneModal] = useState(false);
  const [phoneInput, setPhoneInput] = useState("");
  const [phoneError, setPhoneError] = useState("");

  const API_URL = `${apiOrigin}/chat`;

  // ✅ 처음 들어왔을 때 localStorage에서 user_id / dept / rank 복원
  useEffect(() => {
    const storedUserId = localStorage.getItem("user_id");
    if (storedUserId) {
      setUserId(storedUserId);
    } else {
      setShowPhoneModal(true); // 저장된 ID 없으면 모달 띄우기
    }

    const storedDept = localStorage.getItem("dept");
    if (storedDept) {
      setDept(storedDept);
    }

    const storedRank = localStorage.getItem("rank");
    if (storedRank) {
      setRank(storedRank);
    }
  }, []);

  // ✅ 전화번호 제출
  const handlePhoneSubmit = () => {
    const trimmed = phoneInput.trim();
    const regex = /^01[0-9]{9}$/; // 010 포함 11자리

    if (!regex.test(trimmed)) {
      setPhoneError("올바른 11자리 번호를 입력해 주세요. (예: 01012345678)");
      return;
    }

    setPhoneError("");
    setShowPhoneModal(false);

    // 여기서는 전화번호를 그대로 user_id로 사용
    localStorage.setItem("user_id", trimmed);
    setUserId(trimmed);
  };

  // =========================
  // 1. 대화 시작
  // =========================
  const handleStart = async () => {
    if (!userId) {
      alert("전화번호를 먼저 입력해 주세요.");
      return;
    }
    if (!dept.trim() || !rank.trim()) {
      alert("부서와 계급을 입력해 주세요.");
      return;
    }
    if (starting) return;

    // 새 대화 시작이니까 동의 상태 리셋
    setConsentState("unknown");
    setStarting(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "",
          dept,
          rank,
          shift_type: shiftType,
          user_id: userId,
          // 🔥 session_id 제거
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text()}`);
      const data = await res.json();

      if (data.consent_state) setConsentState(data.consent_state);

      // 대화 시작 플래그 on
      setStarted(true);

      // 새 세션용 동의 요청 메시지를 기존 기록 아래에 붙이기
      const consentAiMsg = {
        id: makeId(),
        role: "ai",
        type: "consent_prompt",
        isTyping: true,
        text:
          "오늘 수집된 생체신호를 참고해서 함께 살펴볼까요?\n\n분석에 동의하시면 ‘동의’를, 원치 않으시면 ‘거절’을 눌러 주세요.",
      };

      setMessages((prev) => [...prev, consentAiMsg]);
      setCurrentTypingId(consentAiMsg.id);
    } catch (e) {
      console.error(e);
      const errId = makeId();
      setMessages((prev) => [
        ...prev,
        {
          id: errId,
          role: "ai",
          text: "서버 통신 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
          isTyping: true,
        },
      ]);
      setCurrentTypingId(errId);
    } finally {
      setStarting(false);
    }
  };

  // =========================
  // 2. 대화 종료
  // =========================
  const handleEndConversation = () => {
    // 1) 마지막에 고정용 AI 종료 메시지 추가 (기록은 남김)
    const endId = makeId();
    setMessages((prev) => [
      ...prev,
      {
        id: endId,
        role: "ai",
        text:
          "오늘 대화는 여기에서 마무리하겠습니다.\n\n조금이라도 도움이 되셨다면 좋겠습니다.\n " +
          "나중에 또 필요하실 때 언제든지 편하게 다시 찾아와 주세요!",
        isTyping: false,
      },
    ]);

    setCurrentTypingId(endId);
    // 2) 상태 정리 (이전 기록은 그대로 둠)
    setStarted(false);
    setConsentState("ended");
    setStarting(false);
  };

  // =========================
  // 3. 동의/거절 버튼
  // =========================
  const handleConsent = async (consent) => {
    if (!started) {
      alert("먼저 대화를 시작해 주세요.");
      return;
    }
    setConsentState(consent);

    const typingId = makeId();
    const placeholderText =
      consent === "accepted" ? "생체신호 분석 중..." : "진행 중...";
    // 우선 placeholder 하나 박아두기
    setMessages((prev) => [
      ...prev,
      { id: typingId, role: "ai", text: placeholderText, isTyping: true },
    ]);
    setCurrentTypingId(typingId);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "",
          biosignal_consent: consent,
          user_id: userId,
          // 🔥 session_id 제거
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text()}`);
      const data = await res.json();

      if (data.consent_state) setConsentState(data.consent_state);

      const arr = Array.isArray(data.replies)
        ? data.replies
        : data.reply
        ? [data.reply]
        : [];
      const plotPath = data.plot_path || null;

      // ✅ 여기서 "이미지 → 텍스트" 순서를 한 번에 보장
      setMessages((prev) => {
        let next = [...prev];

        // 1) plot 먼저
        if (plotPath) {
          next.push({
            id: makeId(),
            role: "ai",
            type: "plot",
            src: apiOrigin + plotPath,
            isTyping: false,
          });
        }

        // 2) placeholder 교체
        next = next.map((m) =>
          m.id === typingId
            ? { ...m, text: arr[0] || "(응답 없음)", isTyping: false }
            : m
        );

        // 3) 나머지 텍스트 붙이기
        if (arr.length > 1) {
          const rest = arr.slice(1).map((t) => ({
            id: makeId(),
            role: "ai",
            text: t,
            isTyping: true,
          }));
          next = [...next, ...rest];
        }

        return next;
      });
    } catch (e) {
      console.error(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === typingId
            ? {
                ...m,
                text: "서버 통신 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                isTyping: false,
              }
            : m
        )
      );
    }
  };

  // =========================
  // 4. 일반 메시지 전송
  // =========================
  const handleSendMessage = async (message) => {
    if (!message.trim()) return;
    if (!userId) {
      alert("전화번호를 먼저 입력해 주세요.");
      return;
    }
    if (!started) {
      alert("먼저 대화를 시작해 주세요.");
      return;
    }
    if (consentState === "unknown") {
      alert("먼저 생체신호 분석 동의 또는 거절을 선택해 주세요.");
      return;
    }

    const userMsg = {
      id: makeId(),
      role: "user",
      text: message,
      isTyping: false,
    };
    const typingId = makeId();
    const aiTypingMsg = {
      id: typingId,
      role: "ai",
      text: "답변 생성 중...",
      isTyping: true,
    };
    setMessages((prev) => [...prev, userMsg, aiTypingMsg]);
    setCurrentTypingId(typingId);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: message,
          user_id: userId,
          // 🔥 session_id 제거
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text()}`);
      const data = await res.json();

      const arr = Array.isArray(data.replies)
        ? data.replies
        : data.reply
        ? [data.reply]
        : [];
      const plotPath = data.plot_path || null;

      // 일반 턴에서도 plot 있으면 이미지 먼저
      setMessages((prev) => {
        let next = [...prev];

        if (plotPath) {
          next.push({
            id: makeId(),
            role: "ai",
            type: "plot",
            src: apiOrigin + plotPath,
            isTyping: false,
          });
        }

        // placeholder 교체
        next = next.map((m) =>
          m.id === typingId
            ? { ...m, text: arr[0] || "(응답 없음)", isTyping: false }
            : m
        );

        // 추가 응답
        if (arr.length > 1) {
          const rest = arr.slice(1).map((t) => ({
            id: makeId(),
            role: "ai",
            text: t,
            isTyping: true,
          }));
          next = [...next, ...rest];
        }

        return next;
      });
    } catch (e) {
      console.error(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === typingId
            ? {
                ...m,
                text: "서버 통신 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                isTyping: false,
              }
            : m
        )
      );
    }
  };

  // =========================
  // 타이핑 끝났을 때
  // =========================
  const handleEndTyping = (id) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, isTyping: false } : m))
    );
    setCurrentTypingId(null);
  };

  // 다음 타이핑 잡아주기
  useEffect(() => {
    if (currentTypingId !== null) return;
    const next = messages.find((m) => m.role === "ai" && m.isTyping);
    if (next) setCurrentTypingId(next.id);
  }, [messages, currentTypingId]);

  return (
    <div className="app">
      {/* ✅ 전화번호 입력 모달 */}
      {showPhoneModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 999,
          }}
        >
          <div
            style={{
              background: "white",
              padding: 24,
              borderRadius: 12,
              width: 320,
              boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
            }}
          >
            <h2 style={{ marginBottom: 8 }}>전화번호 확인</h2>
            <p style={{ fontSize: 14, color: "#555", marginBottom: 8 }}>
              서비스를 사용하기 위해 전화번호 11자리를 입력해 주세요.
              <br />
              (예: 01012345678)
            </p>
            <input
              type="tel"
              value={phoneInput}
              onChange={(e) => setPhoneInput(e.target.value)}
              maxLength={11}
              placeholder="01012345678"
              style={{
                width: "100%",
                padding: 8,
                borderRadius: 6,
                border: "1px solid #ccc",
              }}
            />
            {phoneError && (
              <div
                style={{
                  color: "red",
                  fontSize: 12,
                  marginTop: 4,
                }}
              >
                {phoneError}
              </div>
            )}
            <button
              onClick={handlePhoneSubmit}
              style={{
                marginTop: 12,
                width: "100%",
                padding: 8,
                borderRadius: 6,
                border: "none",
                cursor: "pointer",
              }}
            >
              확인
            </button>
          </div>
        </div>
      )}

      <div className="chat-box">
        <div className="chat-header">
          <h1>경찰관 전용 AI 챗봇</h1>
          <div className="logo-group">
            <img src="./images/police.png" alt="경찰청 로고" className="chat-logo" />
            <img src="./images/kist.png" alt="키스트 로고" className="chat-logo" />
          </div>
        </div>

        {/* 프로필 입력 라인 */}
        <div
          style={{
            display: "flex",
            gap: 8,
            marginBottom: 8,
            flexWrap: "wrap",
          }}
        >
          <input
            placeholder="부서 (예: 형사과, 교통과)"
            value={dept}
            onChange={(e) => {
              const v = e.target.value;
              setDept(v);
              localStorage.setItem("dept", v); // 자동 저장
            }}
            className="message-input"
            disabled={started}
          />
          <input
            placeholder="계급 (예: 순경, 경위)"
            value={rank}
            onChange={(e) => {
              const v = e.target.value;
              setRank(v);
              localStorage.setItem("rank", v); // 자동 저장
            }}
            className="message-input rank-input"
            disabled={started}
          />
          <select
            value={shiftType}
            onChange={(e) => setShiftType(e.target.value)}
            className="message-input shift-input"
            disabled={started}
            style={{ minWidth: 140 }}
            aria-label="근무타입"
            title="근무타입"
          >
            <option value="day">주간</option>
            <option value="night">야간</option>
          </select>

          {!started ? (
            <button
              className="send-button"
              onClick={handleStart}
              disabled={starting || !dept || !rank || !userId}
            >
              대화 시작
            </button>
          ) : (
            <button
              className="send-button end-button"
              onClick={handleEndConversation}
            >
              대화 종료
            </button>
          )}
        </div>

        <MessageList
          messages={messages}
          currentTypingId={currentTypingId}
          onEndTyping={handleEndTyping}
          onInlineAccept={() => handleConsent("accepted")}
          onInlineDecline={() => handleConsent("declined")}
          consentState={consentState}
        />

        <MessageForm
          onSendMessage={handleSendMessage}
          disabled={!started || consentState === "unknown" || !userId}
        />
      </div>
    </div>
  );
}

function MessageList({
  messages,
  currentTypingId,
  onEndTyping,
  onInlineAccept,
  onInlineDecline,
  consentState,
}) {
  const bottomRef = useRef(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentTypingId]);

  // 가장 마지막 consent_prompt 위치 찾기
  const consentIdx = [...messages]
    .map((m, i) => (m.type === "consent_prompt" ? i : -1))
    .filter((i) => i >= 0)
    .pop();

  return (
    <div className="messages-list">
      {messages.map((m, i) => (
        <React.Fragment key={m.id}>
          {m.type === "plot" ? (
            <ImageMessageBubble src={m.src} />
          ) : (
            <Message
              id={m.id}
              role={m.role}
              text={m.text}
              isTyping={m.isTyping}
              currentTypingId={currentTypingId}
              onEndTyping={onEndTyping}
              onTypingStep={() =>
                bottomRef.current?.scrollIntoView({ behavior: "smooth" })
              }
            />
          )}

          {/* 동의 버튼 표시 (해당 세션 시작 시점에만) */}
          {i === consentIdx && consentState === "unknown" && (
            <InlineConsent
              onAccept={onInlineAccept}
              onDecline={onInlineDecline}
              disabled={Boolean(currentTypingId)}
            />
          )}
        </React.Fragment>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

function InlineConsent({ onAccept, onDecline, disabled }) {
  return (
    <div className="inline-consent">
      <button className="btn-accept" onClick={onAccept} disabled={disabled}>
        동의
      </button>
      <button className="btn-decline" onClick={onDecline} disabled={disabled}>
        거절
      </button>
    </div>
  );
}

function Message({
  id,
  role,
  text,
  isTyping,
  currentTypingId,
  onEndTyping,
  onTypingStep,
}) {
  const isCurrentTyping = isTyping && currentTypingId === id;
  const label = role === "user" ? "User" : "AI";
  return (
    <div className={role === "user" ? "user-message" : "ai-message"}>
      <p>
        <b>{label}</b>:{" "}
        {isCurrentTyping ? (
          <Typewriter
            text={text}
            speed={30}
            onDone={() => onEndTyping(id)}
            onStep={onTypingStep}
          />
        ) : (
          text
        )}
      </p>
    </div>
  );
}

function MessageForm({ onSendMessage, disabled }) {
  const [value, setValue] = useState("");
  const isDisabled = useMemo(
    () => disabled || !value.trim(),
    [disabled, value]
  );

  const onSubmit = (e) => {
    e.preventDefault();
    if (isDisabled) return;
    onSendMessage(value);
    setValue("");
  };

  return (
    <form className="message-form" onSubmit={onSubmit}>
      <input
        className="message-input"
        type="text"
        placeholder={
          disabled
            ? "상단의 프로필을 입력하고 대화 시작을 눌러주세요."
            : "메시지를 입력하세요."
        }
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={disabled}
      />
      <button className="send-button" type="submit" disabled={isDisabled}>
        전송
      </button>
    </form>
  );
}
