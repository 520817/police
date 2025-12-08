import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Literal
from fastapi.staticfiles import StaticFiles

from .simple_police_bio import predict

# --------------------------------------------------------------    
# ✅ 테스트 중이라 prt/day를 고정시킴
prt = "prt2099"                 # 내가 보고 싶은 경찰관
day = "2025-09-08"              # 분석하고 싶은 날짜
# 이후에는 today = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")
# --------------------------------------------------------------   

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

app = FastAPI()

# 플롯 이미지 서빙
app.mount("/plots", StaticFiles(directory="plots"), name="plots")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트에서 오는 요청 스키마
class ChatInput(BaseModel):
    text: str
    dept: Optional[str] = None
    rank: Optional[str] = None
    shift_type: Optional[str] = None
    biosignal_consent: Optional[Literal["accepted", "declined", "unknown"]] = None

    # 여기서 user_id(전화번호) 받기
    user_id: Optional[str] = None


@app.post("/chat")
def chat(input: ChatInput):
    # 디버깅용 로그 (전화번호 기반 user_id 잘 오는지 확인)
    print(f"👤 user_id from frontend: {input.user_id}")
    print(f"💬 text: {input.text}")

    # predict에 biosignal_consent 전달 (지금까지와 동일)
    # 아직 simple_police_bio.predict에 user_id 인자가 없다면,
    # 여기서는 받기만 하고 넘기지 않고, 나중에 predict 시그니처를 확장하면 됨.
    resp = predict(
        user_text=input.text,
        dept=input.dept or "",
        rank=input.rank or "",
        shift_type=(input.shift_type or "day"),
        prt=prt,      # 현재는 백엔드에서 고정
        day=day,      # 현재는 백엔드에서 고정
        biosignal_consent=input.biosignal_consent,
        # 나중에 simple_police_bio.predict에 user_id 추가하면:
        # user_id=input.user_id,
    )

    print("📤 predict returned:", resp)  # 디버깅 로그

    replies = resp.get("replies", [])
    created = resp.get("created", False)
    biosignal_first_emit = resp.get("biosignal_first_emit", False)
    consent_state = resp.get("consent_state", "unknown")
    logs = resp.get("logs", [])
    plot_path = resp.get("plot_path")

    if isinstance(replies, str):
        replies = [replies]

    #  프론트에서 상태 동기화와 디버깅에 도움되는 필드도 함께 리턴
    return {
        "replies": replies,
        "created": created,
        "biosignal_first_emit": biosignal_first_emit,
        "consent_state": consent_state,
        "logs": logs,
        "plot_path": plot_path,
    }


@app.get("/ping")
def ping():

    return {"ok": True}

