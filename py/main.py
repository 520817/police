import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Literal
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from zoneinfo import ZoneInfo

from db import *
from utils import get_validation_data
from simple_police_bio import predict, SESSION_STATES, get_graph

# --------------------------------------------------------------
# TEST 모드: CSV 기반 테스트를 위해 prt/day를 고정
USE_FIXED_TEST_CONTEXT = False
TEST_PRT = "prt2098"
TEST_DAY = "2025-09-20"
# --------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://police-front.onrender.com",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 플롯 이미지 서빙
app.mount("/plots", StaticFiles(directory="plots"), name="plots")


def today_kst():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")

def resolve_runtime_context(input_prt: Optional[str]):
    if USE_FIXED_TEST_CONTEXT:
        return TEST_PRT, TEST_DAY
    # 운영: 프론트에서 전달받은 사용자 PK(prt) + 오늘 날짜
    return (input_prt or "").strip(), today_kst()
    
# 프론트에서 오는 요청 스키마
class ChatInput(BaseModel):
    text: str
    dept: Optional[str] = None
    user_rank: Optional[str] = None
    shift_type: Optional[Literal["day", "night", "off", "duty", "holiday"]] = None
    biosignal_consent: Optional[Literal["accepted", "declined", "unknown"]] = None
    modal_submit: Optional[bool] = False

    # 여기서 prt(전화번호) 받기
    prt: Optional[str] = None
    session_id: Optional[str] = None


class SurveyInput(BaseModel):
    session_id: str
    score_type: Literal["pre", "post"]
    valence: int
    arousal: int
    vas: int

    validation_q1_text: Optional[str] = None 
    validation_q2_text: Optional[str] = None  
    validation_q3_text: Optional[str] = None
    validation_q1: Optional[int] = None
    validation_q2: Optional[int] = None
    validation_q3: Optional[int] = None
    is_insufficient: Optional[bool] = None

@app.post("/survey")
def save_survey(input: SurveyInput):
   
    conn = get_db_connection()
    if not conn:
        return {"status": "error", "message": "DB connection failed"}
    
    try:
        save_survey_scores(
            session_id=input.session_id,
            score_type=input.score_type,
            valence=input.valence,
            arousal=input.arousal,
            vas=input.vas,
            validation_q1_text=input.validation_q1_text,  
            validation_q2_text=input.validation_q2_text,  
            validation_q3_text=input.validation_q3_text,
            validation_q1=input.validation_q1,
            validation_q2=input.validation_q2,
            validation_q3=input.validation_q3,
            is_insufficient=input.is_insufficient,
        )

        return {"status": "success", "message": f"{input.score_type} survey saved."}
    except Exception as e:
        print(f"❌ 서버 에러 발생: {e}") # 터미널에서 에러 확인용
        return {"status": "error", "message": str(e)}

    
@app.post("/chat")
def chat(input: ChatInput):
    effective_prt, effective_day = resolve_runtime_context(input.prt)
    if not effective_prt:
        return {"replies": ["전화번호(prt)가 필요합니다."], "session_id": None}

    # 프론트가 기존 session_id를 보내면 재사용
    target_session_id = (input.session_id or "").strip() or None
    
    resp = predict(
        user_text=input.text,
        dept=input.dept or "",
        user_rank=input.user_rank or "",
        shift_type=(input.shift_type or "day"),
        prt=effective_prt,      # 운영 시 input.prt(=사용자 PK) 사용
        day=effective_day,      # 운영 시 서버 기준 오늘 날짜 사용
        biosignal_consent=input.biosignal_consent,
        modal_submit=bool(input.modal_submit),
        session_id=target_session_id,
    )

    resolved_session_id = resp.get("session_id") or target_session_id
    conn = get_db_connection()
    if conn and resolved_session_id:
        try:
            cur = conn.cursor()
            sql = """
            INSERT INTO sessions (session_id, prt, dept, user_rank, shift_type)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                dept = EXCLUDED.dept,
                user_rank = EXCLUDED.user_rank,
                shift_type = EXCLUDED.shift_type
            """
            cur.execute(
                sql,
                (
                    resolved_session_id,
                    effective_prt,
                    input.dept or "",
                    input.user_rank or "",
                    input.shift_type or "day",
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[DB Error] Session creation failed: {e}")

    replies = resp.get("replies", [])
    if isinstance(replies, str):
        replies = [replies]

    #  프론트에서 상태 동기화와 디버깅에 도움되는 필드도 함께 리턴
    return {
        "replies": replies,
        "created": resp.get("created", False),
        "biosignal_first_emit": resp.get("biosignal_first_emit", False),
        "consent_state": resp.get("consent_state", "unknown"),
        "logs": resp.get("logs", []),
        "plot_path": resp.get("plot_path"),
        "session_id": resolved_session_id, # 실제 사용된 session_id 반환
    }

@app.get("/validation/{session_id}")
def get_validation(session_id: str):
    print("validation requested:", session_id)
    state = SESSION_STATES.get(session_id)
    print("state exists:", state is not None)
    if state:
        print("analyses len:", len(state.get("analyses", [])))
        
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    
    # DB에서 해당 thread의 최신 스냅샷 가져오기
    state_snapshot = graph.get_state(config)
    state = state_snapshot.values # 이게 우리가 쓰던 AppState 딕셔너리입니다.

    if not state or len(state.get("analyses", [])) < 1:
        return {
            "status": "insufficient_data",
            "message": "분석을 위한 대화 내용이 부족합니다.",
            "validation_data": None
        }

    try:
        validation_data = get_validation_data(state)

        if validation_data.get("is_insufficient"):
            return {
                "status": "insufficient_data",
                "message": "분석 가능한 대화가 부족합니다.",
                "validation_data": None
            }

        random_turns = validation_data.get("random_turns") or []
        peak_original = ""
        for item in random_turns:
            peak_original = (item.get("original_text") or "").strip()
            if peak_original:
                break
        top_emotions = validation_data.get("top_emotions") or []

        # 분석 결과 내부가 사실상 비어있는 경우
        if not peak_original and not top_emotions:
            return {
                "status": "insufficient_data",
                "message": "분석 결과가 충분하지 않습니다.",
                "validation_data": None
            }

        return {
            "status": "success",
            "validation_data": validation_data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "validation_data": None
        }

@app.get("/ping")
def ping():
    return {"ok": True}

