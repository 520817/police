// src/App.js
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import PrecheckModal from "./vas_sam";

// const apiOrigin = "http://localhost:8000"; // 백엔드 origin
const apiOrigin = "https://police-pwfu.onrender.com";

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

// 이미지 말풍선 (생체신호 플롯)
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
  const [showPrecheck, setShowPrecheck] = useState(false); // precheck 모달 표시 여부
  const [precheckData, setPrecheckData] = useState(null); // pre/post 결과 저장

  const [messages, setMessages] = useState([]);
  const [currentTypingId, setCurrentTypingId] = useState(null);

  // 대화 시작 여부
  const [started, setStarted] = useState(false);

  const [dept, setDept] = useState("");
  const [rank, setRank] = useState("");
  const [shiftType, setShiftType] = useState("day");
  const [starting, setStarting] = useState(false);

  // 생체신호 동의 상태
  // "unknown" | "accepted" | "declined" | "ended"
  const [consentState, setConsentState] = useState("unknown");

  // 전화번호 기반 user_id 관리
  const [userId, setUserId] = useState(null);

  // pre/post 체크 단계
  const [precheckPhase, setPrecheckPhase] = useState("pre"); // "pre" | "post"
  // 종료 확정 대기 (post 제출하면 진짜 종료)
  const [pendingEnd, setPendingEnd] = useState(false);

  // precheck 결과 있으면 시작 허용
  const canChat = Boolean(precheckData);

  const [showPhoneModal, setShowPhoneModal] = useState(false);
  const [phoneInput, setPhoneInput] = useState("");
  const [phoneError, setPhoneError] = useState("");

  const API_URL = `${apiOrigin}/chat`;

  const [sessionId, setSessionId] = useState(null);

  // 처음 들어왔을 때 localStorage에서 user_id / dept / rank 복원
  useEffect(() => {
    const storedUserId = localStorage.getItem("user_id");
    if (storedUserId) {
      setUserId(storedUserId);
      // 첫 진입은 pre 체크부터
      setPrecheckPhase("pre");
      setPendingEnd(false);
      setPrecheckData(null);
      setShowPrecheck(true);
    } else {
      setShowPhoneModal(true);
    }

    const storedDept = localStorage.getItem("dept");
    if (storedDept) setDept(storedDept);

    const storedRank = localStorage.getItem("rank");
    if (storedRank) setRank(storedRank);
  }, []);

  // 전화번호 제출
  const handlePhoneSubmit = () => {
    const trimmed = phoneInput.trim();
    const regex = /^01[0-9]{9}$/;

    if (!regex.test(trimmed)) {
      setPhoneError("올바른 11자리 번호를 입력해 주세요. (예: 01012345678)");
      return;
    }

    setPhoneError("");
    setShowPhoneModal(false);

    localStorage.setItem("user_id", trimmed);
    setUserId(trimmed);

    setPrecheckPhase("pre");
    setPendingEnd(false);
    setPrecheckData(null);
    setShowPrecheck(true);
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
    if (!canChat) {
      alert("대화 전 상태 체크를 먼저 완료해 주세요.");
      return;
    }
    if (starting) return;

    setPrecheckPhase("pre");
    setPendingEnd(false);

    // ✅ FIX: session_id가 서버에서 안 오더라도,
    // 프론트에서는 userId를 sessionId로 미리 세팅해 두면 UI가 안정적임
    setConsentState("unknown");
    setSessionId(userId); // ✅ FIX (중요)
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
          session_id: "", // 서버가 안 써도 됨
        }),
      });

      const raw = await res.text();
      console.log("[START] status:", res.status);
      console.log("[START] raw response:", raw);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status} ${raw}`);
      }

      let data = null;
      try {
        data = JSON.parse(raw);
      } catch (e) {
        throw new Error("서버 응답이 JSON이 아닙니다. raw를 콘솔에서 확인하세요.");
      }

      console.log("[START] parsed data:", data);

      // 서버가 session_id를 안 주는 스펙이므로 fallback 유지
      const sid =
        data.session_id ??
        data.sessionId ??
        data.session ??
        data.sid ??
        null;

      if (sid) {
        setSessionId(sid);
      } else {
        console.warn(
          "[START] session_id 없음(서버 스펙). user_id를 session 대체로 사용",
          Object.keys(data)
        );
        setSessionId(userId); // ✅ FIX
      }

      setStarted(true);

      const consentAiMsg = {
        id: makeId(),
        role: "ai",
        type: "consent_prompt",
        isTyping: false,
        text:
          "오늘 수집된 생체신호를 참고해서 함께 살펴볼까요?\n\n분석에 동의하시면 ‘동의’를, 원치 않으시면 ‘거절’을 눌러 주세요.",
      };

      setMessages((prev) => [...prev, consentAiMsg]);
    } catch (e) {
      console.error(e);

      const errId = makeId();
      setMessages((prev) => [
        ...prev,
        {
          id: errId,
          role: "ai",
          text: `서버 통신 오류(START): ${String(e?.message || e)}`,
          isTyping: false,
        },
      ]);
      setCurrentTypingId(errId);

      // ✅ FIX: 시작 실패 시 sessionId 정리
      setSessionId(null);
    } finally {
      setStarting(false);
    }
  };

  // =========================
  // 2. 대화 종료
  // =========================
  const handleEndConversation = () => {
    if (!started) return;

    setPrecheckPhase("post");
    setPendingEnd(true);

    setPrecheckData(null);
    setShowPrecheck(true);
  };

  const finalizeEndConversation = () => {
    const endId = makeId();
    setMessages((prev) => [
      ...prev,
      {
        id: endId,
        role: "ai",
        text:
          "오늘 대화는 여기에서 마무리하겠습니다.\n\n조금이라도 도움이 되셨다면 좋겠습니다.\n" +
          "나중에 또 필요하실 때 언제든지 편하게 다시 찾아와 주세요!",
        isTyping: false,
      },
    ]);
    setCurrentTypingId(endId);

    setStarted(false);
    setConsentState("ended");
    setStarting(false);
    setSessionId(null);
    setPendingEnd(false);
  };

  // =========================
  // 3. 동의/거절
  // =========================
  const handleConsent = async (consent) => {
    if (pendingEnd) {
      alert("종료 절차 진행 중입니다. 상태 체크 제출을 완료해 주세요.");
      return;
    }
    if (!started) {
      alert("먼저 대화를 시작해 주세요.");
      return;
    }

    // ✅ FIX: sessionId가 없으면 userId로 fallback
    const effectiveSessionId = sessionId || userId; // ✅ FIX
    if (!effectiveSessionId) {
      alert("세션이 아직 준비되지 않았습니다. 다시 '대화 시작'을 눌러 주세요.");
      return;
    }

    setConsentState(consent);

    const typingId = makeId();
    const placeholderText =
      consent === "accepted" ? "생체신호 분석 중..." : "진행 중...";

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
          dept,
          rank,
          shift_type: shiftType,
          biosignal_consent: consent,
          user_id: userId,
          session_id: effectiveSessionId, // ✅ FIX
        }),
      });

      const raw = await res.text();
      console.log("[CONSENT] status:", res.status);
      console.log("[CONSENT] raw:", raw);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status} ${raw}`);
      }

      let data = null;
      try {
        data = JSON.parse(raw);
      } catch (e) {
        throw new Error("CONSENT 응답이 JSON이 아닙니다.");
      }

      // ✅ FIX: session_id가 없으면 유지
      const sid =
        data.session_id ??
        data.sessionId ??
        data.session ??
        data.sid ??
        null;
      if (sid) setSessionId(sid);

      const arr = Array.isArray(data.replies)
        ? data.replies
        : data.reply
        ? [data.reply]
        : [];
      const plotPath = data.plot_path || null;

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

        next = next.map((m) =>
          m.id === typingId
            ? { ...m, text: arr[0] || "(응답 없음)", isTyping: false }
            : m
        );

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
                text: `서버 통신 오류(CONSENT): ${String(e?.message || e)}`,
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
    if (pendingEnd) {
      alert("종료 절차 진행 중입니다. 상태 체크 제출을 완료해 주세요.");
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

    // ✅ FIX: sessionId 없으면 userId
    const effectiveSessionId = sessionId || userId; // ✅ FIX
    if (!effectiveSessionId) {
      alert("세션이 아직 준비되지 않았습니다. 다시 '대화 시작'을 눌러 주세요.");
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
          dept,
          rank,
          shift_type: shiftType,
          user_id: userId,
          session_id: effectiveSessionId, // ✅ FIX
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        console.error("API ERROR", res.status, t);
        throw new Error(`HTTP ${res.status} ${t}`);
      }

      const data = await res.json();

      // ✅ FIX: session_id 있으면 갱신
      const sid =
        data.session_id ??
        data.sessionId ??
        data.session ??
        data.sid ??
        null;
      if (sid) setSessionId(sid);

      const arr = Array.isArray(data.replies)
        ? data.replies
        : data.reply
        ? [data.reply]
        : [];
      const plotPath = data.plot_path || null;

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

        next = next.map((m) =>
          m.id === typingId
            ? { ...m, text: arr[0] || "(응답 없음)", isTyping: false }
            : m
        );

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
                text: `서버 통신 오류(CHAT): ${String(e?.message || e)}`,
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
      <PrecheckModal
        open={showPrecheck && !showPhoneModal}
        phase={precheckPhase}
        onClose={() => {
          if (!precheckData) return;
          setShowPrecheck(false);
        }}
        onDone={(payload) => {
          setPrecheckData(payload);
          setShowPrecheck(false);

          if (precheckPhase === "post" && pendingEnd) {
            finalizeEndConversation();
            return;
          }

          console.log("precheck:", payload);
        }}
      />

      {/* 전화번호 입력 모달 */}
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
              <div style={{ color: "red", fontSize: 12, marginTop: 4 }}>
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

      <div
        className={`chat-box ${showPrecheck ? "chat-box--hidden" : ""}`}
        aria-hidden={showPrecheck}
      >
        <div className="chat-header">
          <h1>경찰관 전용 AI 챗봇</h1>
          <div className="logo-group">
            <img src="/images/police.PNG" alt="경찰청 로고" className="chat-logo" />
            <img src="/images/kist.PNG" alt="키스트 로고" className="chat-logo" />
          </div>
        </div>

        {/* 프로필 입력 라인 */}
        <div className="profile-row">
          <input
            placeholder="부서 (예: 형사과, 교통과)"
            value={dept}
            onChange={(e) => {
              const v = e.target.value;
              setDept(v);
              localStorage.setItem("dept", v);
            }}
            className="message-input dept-input"
            disabled={started || pendingEnd}
          />
          <input
            placeholder="계급 (예: 순경, 경위)"
            value={rank}
            onChange={(e) => {
              const v = e.target.value;
              setRank(v);
              localStorage.setItem("rank", v);
            }}
            className="message-input rank-input"
            disabled={started || pendingEnd}
          />
          <select
            value={shiftType}
            onChange={(e) => setShiftType(e.target.value)}
            className="message-input shift-input"
            disabled={started || pendingEnd}
            aria-label="근무타입"
            title="근무타입"
          >
            <option value="day">주간</option>
            <option value="night">야간</option>
          </select>

          {!started ? (
            <button
              className="send-button start-button"
              onClick={handleStart}
              disabled={starting || !dept || !rank || !userId || !canChat}
            >
              대화 시작
            </button>
          ) : (
            <button
              className="send-button end-button start-button"
              onClick={handleEndConversation}
              disabled={pendingEnd}
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
          pendingEnd={pendingEnd}
          sessionId={sessionId}
        />

        <MessageForm
          onSendMessage={handleSendMessage}
          disabled={!started || consentState === "unknown" || !userId || pendingEnd}
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
  pendingEnd,
  sessionId,
}) {
  const bottomRef = useRef(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentTypingId]);

  const consentIdx = messages.findLastIndex((m) => m.type === "consent_prompt");

  console.log(
    "consentIdx",
    consentIdx,
    "consentState",
    consentState,
    "pendingEnd",
    pendingEnd,
    "lastMsg",
    messages[messages.length - 1]
  );

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

          {i === consentIdx &&
            consentIdx !== -1 &&
            consentState === "unknown" &&
            !pendingEnd && (
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
