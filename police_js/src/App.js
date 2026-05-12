// src/App.js
import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import PrecheckModal from "./vas_sam";

// const apiOrigin = "http://localhost:8000"; // 백엔드 origin
const apiOrigin = "https://police-pwfu.onrender.com"; // render backend 링크

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
function ImageMessageBubble({ html }) {
  if (!html) return null;
  return (
    <div className="ai-message">
      <div className="ai-label">K폴담</div>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

export default function App() {
  const getKstDay = () =>
    new Date().toLocaleDateString("sv-SE", { timeZone: "Asia/Seoul" }); // YYYY-MM-DD

  const [showPrecheck, setShowPrecheck] = useState(false);
  const [precheckData, setPrecheckData] = useState(null);
  const [precheckPhase, setPrecheckPhase] = useState("pre"); // "pre" | "post"
  const [pendingEnd, setPendingEnd] = useState(false);
  const [validationData, setValidationData] = useState(null);

  const [messages, setMessages] = useState([]);
  const [currentTypingId, setCurrentTypingId] = useState(null);
  const [isAiBusy, setIsAiBusy] = useState(false); // 입력창 잠금용 상태

  const [started, setStarted] = useState(false);

  const [dept, setDept] = useState("");
  const [userRank, setUserRank] = useState("");
  const [shiftType, setShiftType] = useState("");
  const [starting, setStarting] = useState(false);
  const [submitAttempted, setSubmitAttempted] = useState(false);

  // "unknown" | "accepted" | "declined" | "ended"
  const [consentState, setConsentState] = useState("unknown");

  const [userId, setUserId] = useState(null);

  const [showPhoneModal, setShowPhoneModal] = useState(false);
  const [phoneInput, setPhoneInput] = useState("");
  const [phoneError, setPhoneError] = useState("");

  const API_URL = `${apiOrigin}/chat`;

  const [sessionId, setSessionId] = useState(null);

  const persistSessionId = (sid) => {
    if (!sid) return;
    localStorage.setItem("session_id", sid);
    localStorage.setItem("session_day", getKstDay());
    setSessionId(sid);
  };

  const buildSessionMeta = () => ({
    prt: userId,
    dept: dept.trim(),
    user_rank: userRank.trim(),
    shift_type: shiftType,
  });

  // 처음 들어왔을 때 localStorage에서 prt / dept / user_rank 복원
  useEffect(() => {
    const storedUserId = localStorage.getItem("prt");
    if (storedUserId) {
      setUserId(storedUserId);
      setShowPrecheck(true);
    } else {
      setShowPhoneModal(true);
    }

    const storedDept = localStorage.getItem("dept");
    if (storedDept) {
      setDept(storedDept);
    }

    const storedUserRank = localStorage.getItem("user_rank");
    if (storedUserRank) {
      setUserRank(storedUserRank);
    }

    const storedSessionId = localStorage.getItem("session_id");
    const storedSessionDay = localStorage.getItem("session_day");
    if (storedSessionId && storedSessionDay === getKstDay()) {
      setSessionId(storedSessionId);
    } else {
      localStorage.removeItem("session_id");
      localStorage.removeItem("session_day");
    }
  }, []);

  // 대화 중 페이지 이탈 경고
  useEffect(() => {
    const handler = (e) => {
      if (!started || consentState === "ended") return;
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [started, consentState]);

  // 전화번호 제출
  const handlePhoneSubmit = () => {
    const trimmed = phoneInput.trim();
    const regex = /^01[0-9]{9}$/; // 010 포함 11자리

    if (!regex.test(trimmed)) {
      setPhoneError("올바른 11자리 번호를 입력해 주세요. (예: 01012345678)");
      return;
    }

    setPhoneError("");
    setShowPhoneModal(false);

    localStorage.setItem("prt", trimmed);
    localStorage.removeItem("session_id");
    localStorage.removeItem("session_day");
    setSessionId(null);
    setUserId(trimmed);

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
    if (!dept.trim() || !userRank.trim() || !shiftType) {
      setSubmitAttempted(true);
      return;
    }
    setSubmitAttempted(false);
    if (starting) return;

    setPrecheckPhase("pre");
    setPendingEnd(false);
    setConsentState("unknown");
    setStarting(true);

    // 새 시작 시 화면 초기화
    setMessages([{
      id: makeId(),
      role: "ai",
      type: "loading",
      isTyping: false,
      text: "대화를 준비하고 있습니다. 잠시만 기다려 주세요.",
    }]);
    setCurrentTypingId(null);
    setIsAiBusy(false);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "",
          ...buildSessionMeta(),
          session_id: sessionId || "",
          modal_submit: true,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text()}`);
      const data = await res.json();

      if (data.session_id) persistSessionId(data.session_id);

      setConsentState("unknown");
      setStarted(true);

      const consentAiMsg = {
        id: makeId(),
        role: "ai",
        type: "consent_prompt",
        isTyping: false, // 바로 보이게
        text:
          "오늘 수집된 생체신호를 참고해서 함께 살펴볼까요?\n\n분석에 동의하시면 ‘동의’를, 원치 않으시면 ‘거절’을 눌러 주세요.",
      };

      setMessages([consentAiMsg]);
      setCurrentTypingId(null);
    } catch (e) {
      console.error(e);
      const errId = makeId();
      setMessages([
        {
          id: errId,
          role: "ai",
          text: "서버 통신 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
          isTyping: false,
        },
      ]);
      setCurrentTypingId(null);
    } finally {
      setStarting(false);
    }
  };

  // =========================
  // 2. 대화 종료
  // =========================
  const handleEndConversation = async () => {
    if (!started) return;
    if (!sessionId) {
      alert("세션 정보가 없습니다.");
      return;
    }

    setPrecheckPhase("post");
    setPendingEnd(true);
    setPrecheckData(null);

    try {
      const res = await fetch(`${apiOrigin}/validation/${sessionId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      if (data.status === "success") {
        setValidationData(data.validation_data ?? null);
      } else {
        console.error("validation fetch failed:", data.message);
        setValidationData(null);
      }
    } catch (e) {
      console.error("validation fetch error:", e);
      setValidationData(null);
    }

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
          "오늘 대화는 여기에서 마무리하겠습니다.\n\n조금이라도 도움이 되셨다면 좋겠습니다.\n나중에 또 필요하실 때 언제든지 편하게 다시 찾아와 주세요!",
        isTyping: false,
      },
    ]);

    setCurrentTypingId(null);
    setIsAiBusy(false);
    setStarted(false);
    setConsentState("ended");
    setStarting(false);

    setPendingEnd(false);
    setPrecheckPhase("pre");
    setValidationData(null);
  };

  // =========================
  // 3. 동의/거절 버튼
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

    if (isAiBusy) return;

    setConsentState(consent);
    setStarted(true);
    setIsAiBusy(true);

    const typingId = makeId();
    const placeholderText =
      consent === "accepted" ? "생체신호 분석 중..." : "진행 중...";

    // 분석 중 문구는 타이핑 없이 바로 출력
    setMessages((prev) => [
      ...prev,
      { id: typingId, role: "ai", text: placeholderText, isTyping: false },
    ]);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "",
          ...buildSessionMeta(),
          biosignal_consent: consent,
          session_id: sessionId,
          modal_submit: true,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      console.log("백엔드 응답 데이터:", data);
      console.log("변경 전 consentState:", consent);
      console.log("변경 후 consentState(from DB):", data.consent_state);

      if (data.session_id) persistSessionId(data.session_id);
      if (data.consent_state) setConsentState(data.consent_state);

      const arr = Array.isArray(data.replies)
        ? data.replies
        : data.reply
        ? [data.reply]
        : [];
      const bioHtml = data.bio_html || null;

      setMessages((prev) => {
        let next = [...prev];

        // placeholder를 실제 첫 답변으로 교체 (타이핑 없이 바로 표시)
        next = next.map((m) =>
          m.id === typingId
            ? { ...m, text: arr[0] || "(응답 없음)", isTyping: false }
            : m
        );

        // plot은 추가
        if (bioHtml) {
          next.push({
            id: makeId(),
            role: "ai",
            type: "plot",
            html: bioHtml,
            isTyping: false,
          });
        }

        // 나머지 텍스트
        if (arr.length > 1) {
          const rest = arr.slice(1).map((t) => ({
            id: makeId(),
            role: "ai",
            text: t,
            isTyping: false,
          }));
          next = [...next, ...rest];
        }

        return next;
      });

      
      setCurrentTypingId(null);
      setIsAiBusy(false);
    } catch (e) {
      console.error(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === typingId
            ? {
                ...m,
                text: "오류가 발생했습니다. 다시 시도해 주세요.",
                isTyping: false,
              }
            : m
        )
      );
      setCurrentTypingId(null);
      setIsAiBusy(false);
    }
  };

  // =========================
  // 4. 일반 메시지 전송
  // =========================
  const handleSendMessage = async (message) => {
    if (!message.trim() || isAiBusy) return;

    const userMsg = {
      id: makeId(),
      role: "user",
      text: message,
      isTyping: false,
    };
    const typingId = makeId();

    setIsAiBusy(true);

    const aiPendingMsg = {
      id: typingId,
      role: "ai",
      text: "답변을 작성하고 있습니다...",
      isTyping: false, // 로딩 문구는 바로 보이게
    };

    setMessages((prev) => [...prev, userMsg, aiPendingMsg]);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: message,
          ...buildSessionMeta(),
          session_id: sessionId,
          modal_submit: false,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();

      if (data.session_id) persistSessionId(data.session_id);

      const arr = Array.isArray(data.replies)
        ? data.replies
        : data.reply
        ? [data.reply]
        : [];
      const bioHtml = data.bio_html || null;

      setMessages((prev) => {
        let next = [...prev];

        // placeholder를 실제 첫 답변으로 바꾸고 타이핑 시작
        next = next.map((m) =>
          m.id === typingId
            ? { ...m, text: arr[0] || "(응답 없음)", isTyping: true }
            : m
        );

        if (bioHtml) {
          next.push({
            id: makeId(),
            role: "ai",
            type: "plot",
            html: bioHtml,
            isTyping: false,
          });
        }

        if (arr.length > 1) {
          const rest = arr.slice(1).map((t) => ({
            id: makeId(),
            role: "ai",
            text: t,
            isTyping: false,
          }));
          next = [...next, ...rest];
        }

        return next;
      });

      setCurrentTypingId(typingId);
      
    } catch (e) {
      console.error(e);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === typingId
            ? {
                ...m,
                text: "오류가 발생했습니다. 다시 시도해 주세요.",
                isTyping: false,
              }
            : m
        )
      );
      setCurrentTypingId(null);
      setIsAiBusy(false);
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
    setIsAiBusy(false);
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
        validationData={validationData}
        onClose={() => {
          if (!precheckData) return;
          setShowPrecheck(false);
        }}
        onDone={async (payload) => {
          setPrecheckData(payload);
          setShowPrecheck(false);

          const day = getKstDay();
          const actualSessionId = sessionId || (userId ? `${userId}_${day}` : null);

          try {
            console.log(`${precheckPhase} 전송 시작 - ID:`, actualSessionId);

            const response = await fetch(`${apiOrigin}/survey`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                session_id: actualSessionId,
                ...buildSessionMeta(),
                score_type: precheckPhase,
                valence: payload.valence,
                arousal: payload.arousal,
                vas: payload.vas,
                validation_q1: payload.validation?.q1 ?? null,
                validation_q2: payload.validation?.q2 ?? null,
                validation_q3: payload.validation?.q3 ?? null,
                validation_q1_text: payload.validation?.q1_text ?? null,
                validation_q2_text: payload.validation?.q2_text ?? null,
                validation_q3_text: payload.validation?.q3_text ?? null,
                is_insufficient: payload.is_insufficient ?? null,
              }),
            });

            if (!response.ok) {
              throw new Error("설문 저장 실패");
            }

            const result = await response.json();
            console.log("DB 저장 완료:", result);
          } catch (error) {
            console.error("DB 저장 에러:", error);
          }

          if (precheckPhase === "post" && pendingEnd) {
            finalizeEndConversation();
          }
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
                boxSizing: "border-box",
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
                boxSizing: "border-box",
              }}
            >
              확인
            </button>
          </div>
        </div>
      )}

      <div className="chat-box">
        <div className="chat-header">
          <h1>
            <span className="header-sub">경찰의 마음을 귀담아듣는,</span>
            <span className="header-main">K폴담</span>
          </h1>
          <div className="logo-group">
            <img src="/images/police.PNG" alt="경찰청 로고" className="chat-logo" />
            <img src="/images/kist.PNG" alt="키스트 로고" className="chat-logo" />
          </div>
        </div>

        {/* 프로필 입력 라인: 대화 시작 전만 표시 */}
        {!started && (
          <div className="profile-row">
            <input
              placeholder="부서 (예: 형사과, 교통과)"
              value={dept}
              onChange={(e) => {
                const v = e.target.value;
                setDept(v);
                localStorage.setItem("dept", v);
              }}
              className={`message-input dept-input${submitAttempted && !dept.trim() ? " input--error" : ""}`}
              disabled={pendingEnd}
            />
            <input
              placeholder="계급 (예: 순경, 경위)"
              value={userRank}
              onChange={(e) => {
                const v = e.target.value;
                setUserRank(v);
                localStorage.setItem("user_rank", v);
              }}
              className={`message-input rank-input${submitAttempted && !userRank.trim() ? " input--error" : ""}`}
              disabled={pendingEnd}
            />
            <select
              value={shiftType}
              onChange={(e) => {
                setShiftType(e.target.value);
              }}
              className={`message-input shift-input${submitAttempted && !shiftType ? " input--error" : ""}`}
              disabled={pendingEnd}
              aria-label="근무타입"
              title="근무타입"
            >
              <option value="" disabled hidden></option>
              <option value="day">주간</option>
              <option value="night">야간</option>
              <option value="off">비번</option>
              <option value="duty">당직</option>
              <option value="holiday">휴무</option>
            </select>
            <button
              className="send-button start-button"
              onClick={handleStart}
              disabled={starting}
            >
              대화 시작
            </button>
          </div>
        )}
        {!started && submitAttempted && (!dept.trim() || !userRank.trim() || !shiftType) && (
          <div className="profile-error">
            {[!dept.trim() && "부서", !userRank.trim() && "계급", !shiftType && "근무 유형"]
              .filter(Boolean)
              .join(", ")}을 입력해 주세요.
          </div>
        )}

        {/* 대화 중: 종료 버튼만 오른쪽 정렬 */}
        {started && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
            <button
              className="send-button end-button"
              onClick={handleEndConversation}
              disabled={pendingEnd || isAiBusy}
            >
              대화 종료
            </button>
          </div>
        )}

        <MessageList
          messages={messages}
          currentTypingId={currentTypingId}
          onEndTyping={handleEndTyping}
          onInlineAccept={() => handleConsent("accepted")}
          onInlineDecline={() => handleConsent("declined")}
          consentState={consentState}
          isAiBusy={isAiBusy}
        />

        {console.log("🛠️ 채팅창 활성화 로직 체크:", {
          "1.대화시작여부(started)": started,
          "2.동의상태(consentState)": consentState,
          "3.유저ID존재": !!userId,
          "4.종료절차중": pendingEnd,
          "5.AI타이핑중(id)": currentTypingId,
          "6.AI처리중(isAiBusy)": isAiBusy,
          "결과(disabled)":
            !started ||
            consentState === "unknown" ||
            !userId ||
            pendingEnd ||
            isAiBusy,
        })}

        <MessageForm
          onSendMessage={handleSendMessage}
          disabled={
            !started ||
            consentState === "unknown" ||
            !userId ||
            pendingEnd ||
            isAiBusy
          }
          isStarted={started}
          consentState={consentState}
          isAiBusy={isAiBusy}
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
  isAiBusy,
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentTypingId]);

  const consentIdx = [...messages]
    .map((m, i) => (m.type === "consent_prompt" ? i : -1))
    .filter((i) => i >= 0)
    .pop();

  return (
    <div className="messages-list">
      {messages.map((m, i) => (
        <React.Fragment key={m.id}>
          {m.type === "plot" ? (
            <ImageMessageBubble html={m.html} />
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

          {i === consentIdx && consentState === "unknown" && (
            <InlineConsent
              onAccept={onInlineAccept}
              onDecline={onInlineDecline}
              disabled={isAiBusy}
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
  const displayText = (text || "").replace(/\\n/g, "\n");

  return (
    <div className={role === "user" ? "user-message" : "ai-message"}>
      {role === "ai" && <div className="ai-label">K폴담</div>}
      <p style={{ whiteSpace: "pre-wrap" }}>
        {isCurrentTyping ? (
          <Typewriter
            text={displayText}
            speed={30}
            onDone={() => onEndTyping(id)}
            onStep={onTypingStep}
          />
        ) : (
          displayText
        )}
      </p>
    </div>
  );
}

function MessageForm({
  onSendMessage,
  disabled,
  isStarted,
  consentState,
  isAiBusy,
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  const isDisabled = useMemo(() => disabled || !value.trim(), [disabled, value]);

  const getPlaceholder = () => {
    if (!isStarted) return "상단의 프로필을 입력하고 대화 시작을 눌러주세요.";
    if (consentState === "unknown") return "생체신호 분석 동의 여부를 선택해주세요.";
    if (isAiBusy) return "답변을 작성 중입니다...";
    return "메시지를 입력하세요.";
  };

  const handleChange = (e) => {
    setValue(e.target.value);
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = el.scrollHeight + "px";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isDisabled) return;
      onSendMessage(value);
      setValue("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    if (isDisabled) return;
    onSendMessage(value);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  return (
    <form className="message-form" onSubmit={onSubmit}>
      <textarea
        ref={textareaRef}
        className="message-input message-textarea"
        placeholder={getPlaceholder()}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
      />
      <button className="send-button" type="submit" disabled={isDisabled}>
        전송
      </button>
    </form>
  );
}

