#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# ==== 환경 ====
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import Annotated
from typing import TypedDict, Literal, Dict, Any, List, Optional, Sequence
from dataclasses import dataclass
import os, json, operator
from uuid import uuid4
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from pydantic import BaseModel, Field
from utils import *
from data import *
from db import *

# 공통 LLM 
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, max_tokens=1024)

class AppState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # 히스토리 (LLM context)
    logs: Annotated[List[str], operator.add]              # 디버그/추적 로그
    analyses: Annotated[List[str], operator.add]          # 경찰관 맥락 해석
    validation_reset_points: Annotated[List[int], operator.add]
    final_replies: Annotated[List[str], operator.add]     # 응답 최종 출력

    # 사용자 프로필 (고정)
    profile: Dict[str, str]  # {"dept": "...", "user_rank": "...", "shift_type": "..."}
    meta: Dict[str, str]  # {"prt": "...", "day": "...", "session_id": "..."}

    # 생체신호 
    biosignal_consent: Literal["unknown", "accepted", "declined"]
    
    biosignal_done: bool                     # 세션당 1회만 실행 가드
    biosignal_last: Dict[str, str]
    biosignal_first_emit: bool

    biosignal: Dict[str, Any]
    resilience_score: float    # 회복탄력성 점수
    strategy_guide: str
    current_stage: Literal["engaging", "evoking", "conclusion"]


def initial_state(user_text: str, dept: str, user_rank: str, shift_type: str = "unknown", session_id: str = "", prt: str = "", day: str = "", resilience_score: float =  0.0,) -> AppState:
    return {
        "messages": [HumanMessage(content=user_text)] if user_text else [],
        "logs": [],
        "analyses": [],
        "validation_reset_points": [],
        "final_replies": [],
        "profile": {"dept": dept, "user_rank": user_rank, "shift_type": shift_type},
        "meta": {"prt": prt, "day": day, "session_id": session_id},

        "biosignal_consent": "unknown", # default
        
        "biosignal_done": False,
        "biosignal_last": {},
        "biosignal_first_emit": False,
        "biosignal": {},
        "resilience_score": resilience_score,
        "strategy_guide": "",
        "current_stage": "engaging",

    }

SESSION_STATES: dict[str, AppState] = {}    # key = session_id

DEFAULT_OPENING_Q = (
    "경찰관님. 오늘은 어떤 이야기 나눠볼까요?\n"
    "마음속에 맴도는 감정이나 생각이 있다면, 편하게 말씀해 주세요.\n"
    "제가 천천히, 그리고 함께 들어드릴게요."
)

DEFAULT_SUPPORT_SUFFIX = (
    "\n또는 요즘 마음속에 자주 떠오르는 감정이나 생각이 있다면\n"
    "그 이야기부터 시작해도 괜찮아요.\n"
    "괜찮으시다면, 제가 천천히 함께 들어드릴게요."
)

BIOSIGNAL_ANALYZER_SYS = """\
너는 생체신호를 해석하는 전문가야. 
아래는 1시간 단위로 최대 12시간 동안 수집한 생체신호 데이터 목록이야.

[입력 데이터]
    'time', 'Stress(1: 양성, 0: 음성)','MeanHR', 'MeanNN', 'SDNN', 'LF', 'LFn', 'HFn'
    부서: {dept}, 계급: {user_rank}, 근무형태: {shift_type}, sufficient: {is_data_sufficient}
    (실제 시간 해석은 데이터의 'time' 변수를 최우선으로 함. 결측값은 "N/A" 표기)

[출력 목적]
- biosignal_result: 사용자에게 직접 보여줄 분석 결과
- biosignal_summary: 상황·감정 분석 에이전트가 참고할 내부 요약
- opening_question: 생체신호 기반 대화 시작 질문

[데이터 품질 규칙]
- 심장박동수(MeanHR)가 40 미만인 경우, 측정 오류 가능성이 있다고 판단한다.
- 해당 시간대가 분석에 포함된 경우, biosignal_result에서 그 시간대 수치를 단정적으로 해석하지 않는다.
- biosignal_result에서 "이 시간대는 측정값이 다소 불안정해 정확한 해석이 어려울 수 있어요." 라고 언급한다.
- biosignal_summary에는 해당 시간대를 신뢰도 낮은 구간으로 명시한다.

[공통 규칙]
전문용어와 영문 변수명 사용을 금지한다. 일상어로 바꿔 시간대 서술 안에 녹여 표현하고, 변수별 나열 방식은 사용하지 않는다.
(MeanHR → 심장박동수, SDNN → 박동의 변동성, LF/LFn → 긴장 신호, HFn → 이완·회복 신호)
가설이나 업무 추측은 언급하지 않는다.
    
[출력 지침]
1. biosignal_result
    [sufficient=True인 경우]
    - 비전공자가 이해하기 쉬운 상담 톤으로 작성한다. 경찰관님이라고 부른다.
    - 오늘의 신체 부담 흐름을 1~2문장으로 먼저 요약하고, 가장 뚜렷한 변화 구간 1~3개만 이모지(🕐 등 시간대 맞는 시계 이모지)로 구분해 한 줄씩 설명한 뒤, 마지막에 주목할 점 1문장으로 마무리한다.
    - 전체는 4~5문장 이내로 작성하고 시간대를 모두 나열하지 않는다.
    - 마크다운은 쓰지 말고, 문단은 줄바꿈(\\n\\n)으로만 구분한다.

    [sufficient=False인 경우]
    - 위의 모든 규칙을 무시하고 아래 문장만 정확히 출력한다.
    - "경찰관님, 안녕하세요. 이번 회차의 생체 신호를 확인했으나, 측정된 데이터 기록이 부족하여 의미 있는 시간대별 분석을 제공해 드리기 어렵습니다. 데이터가 충분히 누적되면 다음 분석 시에 다시 한 번 자세히 살펴보도록 하겠습니다."

2. biosignal_summary
    - 상황·감정 분석 에이전트가 신체 부담의 배경 맥락으로 참고할 내부 요약이다.
    - 수치보다 패턴 중심으로 작성한다. 시간대별 긴장/이완 흐름을 간결하게 서술하고, 뚜렷한 변화 구간이 있으면 명시한다.
    - sufficient=False인 경우에도 가용한 데이터 범위 내에서 패턴을 요약한다.

3. opening_question
    - biosignal_summary를 바탕으로 특정 시간대의 뚜렷한 신체 반응과 감정을 연결하는 질문을 1~2문장으로 만든다.
    - 단정하지 말고 반드시 물음표로 끝낸다.
    - 예시: "오후 3시경 긴장 신호가 특히 높게 나타났는데, 혹시 그 무렵 신경 쓰이는 일이 있으셨을까요?"
    - sufficient=False인 경우 빈 문자열("")을 출력한다.
    

출력 형식(JSON):
{{
  "biosignal_result": "<첫 턴에 사용자에게 바로 보여줄 분석 결과>",
  "biosignal_summary": "<분석 에이전트에 전달할 분석 요약>",
  "opening_question": "<오프닝 질문 1~2문장>"
}}
"""

class BiosignalAnalysis(BaseModel):
    biosignal_result: str = Field(description="사용자에게 생체신호 분석 결과 설명")
    biosignal_summary: str  = Field(description="생체신호 분석 요약")
    opening_question: str = Field(description="생체신호 사용해서 오프닝 문장")

def create_biosignal_analyzer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BIOSIGNAL_ANALYZER_SYS),
            ("human", "signals_json:\n{signals_json}"),
        ]
    )
    return prompt | llm.with_structured_output(BiosignalAnalysis)


def biosignal_analyzer_node(state: AppState, biosignal_analyzer_chain):
    if state.get("biosignal_done", False):
        return {"logs": ["biosignal_skip"], "biosignal_first_emit": False}
        
    if state.get("biosignal_consent", "unknown") != "accepted":
        return {"logs": ["biosignal_skip_no_consent"], "biosignal_first_emit": False}

    signals_all_slots = state.get("biosignal", {}) or {} # 생체신호 가져오기
    
    valid_signals = []
    if isinstance(signals_all_slots, list): 
        for record in signals_all_slots:
            hr = record.get("PPG_MeanHR", record.get("MeanHR"))
            if hr not in [None, "", "N/A"]: # 주요 지표(PPG_MeanHR)이 N/A가 아니면 유효로 간주
                valid_signals.append(record)
        
    valid_record_count = len(valid_signals) # 유효 데이터 개수 세기
    is_data_sufficient = valid_record_count >= 6

    signals_to_send = remove_ppg_prefix(valid_signals) # remove_ppg_prefix는 db 변수명 수정하기 위한 함수(PPG_ 제외하기 위해)
    signals_json = json.dumps(signals_to_send, ensure_ascii=False) # json 형태 변환, ensure_ascill=False는 한글 깨짐 방지

    dept, user_rank, shift_type = state["profile"]["dept"], state["profile"]["user_rank"], state["profile"]["shift_type"]

    print("="*50)
    print("[DEBUG] AI에게 전달되는 signals_json:")
    print(signals_json)
    print("[DEBUG] valid_record_count:", valid_record_count)
    print("[DEBUG] is_data_sufficient:", is_data_sufficient)
    print("="*50)
    
    result: BiosignalAnalysis = biosignal_analyzer_chain.invoke({   
        "signals_json": signals_json,
        "is_data_sufficient": is_data_sufficient,
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
    })
    payload = result.model_dump() # model_dump()는 BaseModel 안에 들어있는 데이터를 dict으로 꺼내는 함수

    # db 저장
    try:
        save_biosignal_log(
            session_id=state["meta"]["session_id"],
            biosignal_summary=payload.get("biosignal_summary", ""),
            valid_record_count=valid_record_count
        )
    except Exception as e:
        print(f"[DB Error] Biosignal log save failed: {e}")

    # 첫 번째 메시지: 분석 결과
    msg_result = AIMessage(content=payload["biosignal_result"].strip()) # State["messages"]에 넣으려면 BaseMessage 형태로 되어있어야함

    
     # 두 번째 메시지: 오프닝 질문(비어있을 수 있으니 .get으로 읽어옴)
    opening_q = (payload.get("opening_question") or "").strip()
    
    # opening_question이 비었을 때도 무난한 한 줄을 먼저 깔아줌
    if not opening_q:
        opening_q = DEFAULT_OPENING_Q
    
    support_suffix = DEFAULT_SUPPORT_SUFFIX
    
    # 한 메시지 안에서 줄바꿈으로 부드럽게 연결
    msg_opening = AIMessage(content=f"{opening_q}\n{support_suffix}".strip())
    
    plot_path = None
    meta = state.get("meta", {}) or {}
    prt = meta.get("prt", "unknown_prt")
    day = meta.get("day", "unknown_day")
    sid = meta.get("session_id")
    if is_data_sufficient:
        plot_path = make_biosignal_overview_plot(
            valid_signals=valid_signals,
            session_id=sid, prt=prt, day=day,
        )

    return {
        "biosignal_done": True,
        "biosignal_first_emit": True,
        "biosignal_last": {
            "biosignal_result": payload.get("biosignal_result", ""),
            "biosignal_summary": payload.get("biosignal_summary", ""),
            "plot_path": plot_path,
        },
        "messages": [msg_result, msg_opening], 
        "logs": ["biosignal_analyzer_ok_two_messages"],
    }


ANALYZER_SYS = """\
너는 경찰관 대상 공감형 대화를 위한 내부 분석 에이전트다.
역할은 사용자의 현재 발화와 최근 대화 흐름을 바탕으로 경찰 직무 맥락에서 상황과 감정을 정확하고 간결하게 해석하는 것이다.

[입력]
- text: "{user_text}"
- dept: {dept}
- user_rank: {user_rank}
- shift_type: {shift_type}
- biosignal_summary: {biosignal_summary}
- resilience_score: {resilience_score}
- strategy_guides: {strategy_guides}
- full_history: {full_history}
- current_stage: {current_stage}

[분석 원칙]
- 가장 중요한 일은 현재의 상황과 감정을 해석하는 것이다.
- 경찰 직무 맥락(사건처리 부담, 민원 응대, 조직 내 관계, 교대근무, 피로 누적, 긴장, 감정 억제 등)은 참고하되 과장하지 않는다.
- 상황이 불분명하면 profile(dept, user_rank, shift_type)과 biosignal 단서를 thought에 추측으로만 짧게 명시하고, responder가 질문에 자연스럽게 활용할 수 있게 한다. 단, 현재 발화를 항상 우선한다.
- resilience_score는 대화의 깊이·방향을 결정하는 용도로 참고한다.(3.0 미만 = low, 3.0~4.30 = normal, 4.31 이상 = high)
- full_history는 대화 흐름 전체를 파악하기 위한 참고 정보다. 현재 발화가 항상 우선한다.

[thought 작성 규칙]
thought는 responder에게 전달할 내부 해석 요약이다.
한 문단으로 짧게 작성하며, 아래 순서대로 자연스럽게 포함한다.
1. 핵심 상황
2. 핵심 감정
3. 상황이 불분명할 때만 profile(dept/user_rank/shift_type)과 biosignal의 유력한 단서를 추측임을 밝히고 구체적으로 적는다.
4. current_stage 유지 또는 전환 판단과 그 이유
5. 이번 턴에서 responder가 우선해야 할 초점 1개
    - 상황이 불분명할 때는 3번의 단서를 질문에 자연스럽게 녹일 수 있도록 지시한다.
    - strategy_guides를 참고하되 현재 판단한 stage와 어긋나는 초점은 포함하지 않는다.
    - engaging은 편하게 꺼내게 하고, evoking은 이미 나온 내용을 더 깊게 풀게 하며, conclusion만 오늘 대화의 핵심 상황 1문장과 핵심 감정 1문장을 정리한 뒤 부담 없는 제안 1개를 허용한다.
    
[situation]
- 현재 경찰관이 겪는 핵심 문제 상황을 한 문장으로 요약한다.
- 현재 발화를 우선하되, full_history로 맥락이 잘린 경우를 보완한다.
- 경찰 맥락이 자연스럽게 드러나면 반영하되 억지로 넣지 않는다.
- 발화가 짧더라도 full_history에서 유추 가능하면 unknown을 쓰지 않는다.
- 불확실하면 "unknown"

[emotion]
- main: 현재 발화에서 가장 핵심적인 감정 1개
- sub: full_history에서 반복되거나 현재 발화에 보조적으로 드러나는 감정 1개 (근거가 충분할 때만, 없으면 null)
- valence: negative / neutral / positive / unknown 중 하나
- 불확실하면 unknown

[stage]
- engaging: 라포를 형성하며 사용자가 자신의 상황과 감정을 편하게 꺼낼 수 있도록 이끌어내는 단계
- evoking: 이미 나온 상황과 감정을 더 깊고 구체적으로 풀어내며, 사용자가 안전하게 더 표현할 수 있도록 돕는 단계
- conclusion: 대화가 충분히 진행되어 오늘의 흐름을 정리하고 자연스럽게 마무리할 수 있는 단계

[stage 판단 원칙]
- stage는 대화 운영용 보조 정보다. current_stage를 기본으로 두고, full_history 기준으로 지금 responder가 해야 할 일이 더 탐색인지, 더 심화인지, 정리인지 분명할 때만 전환한다.
- 문제의 명료성, 경찰관의 개방성, 감정 표현 정도, 정리 준비도를 함께 본다.
- resilience는 단계 전환 기준이 아니라 질문 깊이와 제안 부담을 조절하는 참고값이며, 애매하면 conclusion보다 engaging 또는 evoking을 우선한다.
- conclusion은 사용자가 오늘 대화를 어느 정도 정리했거나 마칠 의사를 직접 또는 간접적으로 보일 때만 검토한다.

1.전진 조건
- engaging → evoking: 경찰관이 자신의 상황이나 감정을 어느 정도 드러냈고, 이제는 그 내용을 더 깊고 구체적으로 풀어내도록 돕는 것이 적절할 때.
- engaging 단계에서 여러 턴에 걸쳐 같은 상황, 대상, 감정 반응이 반복해서 드러나면 더 이상 단순 이끌어내기가 아니라 이미 나온 내용을 깊게 다루는 국면으로 보고 evoking을 우선 검토한다.
- 사용자가 불편함, 긴장, 답답함, 지루함, 귀찮음처럼 자신의 반응을 반복해서 말하고 있으면 감정이 충분히 드러난 것으로 보고 evoking을 우선 검토한다.
- evoking → conclusion: 상황과 감정이 충분히 다뤄졌고, 사용자가 "이제 괜찮다", "여기까지", "말하니 좀 낫다", "정리된 것 같다"처럼 정리·마무리 의사를 직접 또는 간접적으로 보일 때만 검토한다. AI가 성급하게 마무리를 유도하는 것은 금지한다.
2. 후퇴 조건
- 경찰관이 새로운 문제를 꺼내거나, 대화의 초점이 다시 넓어져 추가 탐색이 필요할 때만 허용한다.
- 단순 거절이나 짧은 부정 발화만으로는 후퇴하지 않는다.
- 이미 conclusion에 도달한 이후에는 후퇴 조건을 특히 엄격하게 적용한다.
- 후퇴할 경우 thought에 후퇴 이유를 반드시 명시한다. 명시 없이 후퇴하는 것은 금지한다.
- 단순 체념, 피로 호소, 짧은 단답, 해결책 수용 직후의 추가 불만·감정 표현은 conclusion 신호로 보지 말고 기본적으로 evoking을 유지한다.


출력 형식(JSON):
{{
  "thought": "<내부 해석 요약>",
  "situation": "<경찰관 맥락으로 해석된 상황 한 문장>",
  "emotion": {{
    "main": "<핵심 감정>",
    "sub": "<보조 감정 또는 null>",
    "valence": "<positive | neutral | negative | unknown>",
  }},
  "stage": "<engaging | evoking | conclusion>",
}}
"""

class EmotionItem(BaseModel):
    main: str = "unknown"
    sub: Optional[str] = None
    valence: str = "unknown"

class AnalysisResult(BaseModel):
    thought: str
    situation: str
    emotion: EmotionItem = Field(default_factory=EmotionItem)
    stage: Literal["engaging", "evoking", "conclusion"]

def build_full_history(state: AppState, max_msgs: int = 16):
    msgs = state.get("messages", [])
    return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs

def create_analyzer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYZER_SYS),

            ("system", "예시2: dept=형사과, user_rank=경사, shift_type=day, biosignal_summary=10:00-14:00 긴장 높음 지속, 16:00 이완 소폭 증가, resilience_score=3.8, current_stage=engaging"),
            ("human", "피의자 조사할 때 제가 너무 감정적으로 반응한 것 같아서 계속 마음에 걸려요."),
            ("ai", """{{
                "thought": "조사 중 감정 반응과 그 뒤의 후회가 이미 드러남. 형사과 업무 특성상 감정 통제 압박이 자기비난으로 이어졌을 가능성, 오전 긴장 지속 biosignal도 맥락상 맞물림. 이제는 드러난 경험을 더 깊게 풀 단계라 engaging에서 evoking으로 전환. 이번 초점: 그 순간 올라온 감정과 생각을 더 구체화.",
                "situation": "경찰관은 피의자 조사 중 자신의 감정 반응이 적절하지 않았다고 느끼며 이를 반추하고 있다.",
                "emotion": {{
                    "main": "후회",
                    "sub": "자기비난",
                    "valence": "negative"
                }},
                "stage": "evoking"
            }}"""),

            ("system", "예시3: dept=교통, user_rank=경위, shift_type=day, biosignal_summary=09:00-11:00 긴장 보통, 13:00-15:00 긴장 소폭 증가 후 안정, resilience_score=3.2, current_stage=evoking"),
            ("human", "얘기하다 보니까 제가 너무 결과만 보고 있었던 것 같아요. 과정도 봐야 하는데."),
            ("ai", """{{
                "thought": "결과 중심 사고에서 과정으로 시선이 옮겨가는 통찰이 나옴. 교통 업무의 실적 압박 맥락이 이런 사고 패턴에 영향을 줬을 가능성, 오후 긴장 후 안정 패턴도 현재 정리 국면과 맞물림. 더 탐색하기보다 오늘 대화를 정리할 타이밍이라 evoking에서 conclusion으로 전환. 이번 초점: 통찰을 짧게 요약하고 부담 없는 실천 1개 제안.",
                "situation": "경찰관은 결과 중심의 사고 패턴을 스스로 인식하고 과정을 함께 보려는 시각의 전환을 이야기하고 있다.",
                "emotion": {{
                    "main": "안도",
                    "sub": "통찰",
                    "valence": "positive"
                }},
                "stage": "conclusion"
            }}"""),

            MessagesPlaceholder("full_history", optional=True),
            ("system", "생체요약:\n{biosignal_summary}"),
            ("human", "{user_text}"),
        ]
    )
    return prompt | llm.with_structured_output(AnalysisResult)

def analyzer_node(state: AppState, analyzer_chain):
    # 새 사용자 입력이 없는 요청(예: 재접속/모달 이벤트)에서는 과거 HumanMessage를 재분석하지 않도록 analyzer를 건너뜀
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not isinstance(last_msg, HumanMessage) or not str(getattr(last_msg, "content", "")).strip():
        return {
            "analyses": [],
            "messages": [],
            "logs": ["[Analyzer] skip: no new user text in this turn"],
        }

    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_text = human_msgs[-1].content if human_msgs else ""

    dept, user_rank, shift_type = state["profile"]["dept"], state["profile"]["user_rank"], state["profile"]["shift_type"]
    biosignal_summary = state.get("biosignal_last", {}).get("biosignal_summary", "")
    resilience_score = state.get("resilience_score", 4.5)
    current_stage = state.get("current_stage", "engaging")

    history = build_full_history(state, max_msgs=16)

    all_stages = ["engaging", "evoking", "conclusion"]
    strategy_guides = "\n".join(
        make_strategy_guide(resilience_score, s) for s in all_stages
    )



    result: AnalysisResult = analyzer_chain.invoke({
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
        "user_text": user_text,
        "biosignal_summary": biosignal_summary,
        "resilience_score": state.get("resilience_score", 4.5),
        "full_history": history,
        "strategy_guides": strategy_guides,
        "current_stage": current_stage,
    })

    emotion_item = result.emotion
    main = str(emotion_item.main or "").strip()
    sub = str(emotion_item.sub or "").strip()
    if sub.lower() in ("null", "unknown", "none", ""):
        sub = ""
    valence = str(emotion_item.valence or "unknown").strip()
    
    last_ai = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)),
        None,
    )
    prev_ai_text = str(getattr(last_ai, "content", "") or "").strip()

    analysis_payload = {
        "thought": result.thought,
        "situation": result.situation,
        "emotion": {
        "main": main,
        "sub": sub,
        "valence": valence,  
        },
        "stage": result.stage,
        "original_text": user_text,
        "prev_ai_text": prev_ai_text,
        }
    emotion_str = f"핵심: {main}" + (f", 보조: {sub}" if sub else "") + (f", 긍부정: {valence}" if valence else "")

    strategy_guide = make_strategy_guide(resilience_score, result.stage)
    
    #db 저장
    try:
        save_chat_message(
            session_id=state["meta"]["session_id"],
            role="analyzer",  # 분석 에이전트임을 명시
            content=f"Situation: {result.situation}, Emotion: {emotion_str}", # 요약 내용
            situation=result.situation,
            emotion=emotion_str,
            stage=result.stage,
            thought=result.thought
        )
    except Exception as e:
        print(f"DB 저장 실패(analyzer_node): {e}")
        
    analysis_json_str = json.dumps(analysis_payload, ensure_ascii=False)

    analysis_line = (
        f"[분석] {result.situation} "
        f"(감정:{emotion_str})"
        f"  [단계:{result.stage}]"
    )
    
    last_same = bool(state.get("analyses")) and state["analyses"][-1] == analysis_json_str
    new_analyses = [] if last_same else [analysis_json_str]

    last_log_same = bool(state.get("logs")) and state["logs"][-1] == analysis_line
    new_logs = [] if last_log_same else [analysis_line]

    if not last_log_same:
        print(analysis_line)

    return {
        "analyses": new_analyses,
        "messages": [],
        "logs": new_logs,
        "strategy_guide": strategy_guide,
        "current_stage": result.stage,
    }


RESPONDER_SYS = """\
너는 경찰관 대상 정서지원 대화 에이전트다.
경찰 조직과 근무 특성에 대한 이해를 갖고 있지만, 사용자의 현재 발화와 대화 흐름을 최우선으로 해석한다.
직무 스트레스뿐 아니라 가족, 관계, 건강, 생활, 성격, 일상 피로 같은 직무 밖 어려움도 동등하게 다룬다.
분석 에이전트가 전달한 thought를 기반으로, 사용자가 편하게 말할 수 있도록 대화를 이끈다.
발화 범위를 과도하게 확장하거나 단정하지 않으며, 현장감 있는 자연스러운 상담 대화를 유지한다.

[입력 정보]
- original_text: "{original_text}"
- dept: {dept} / user_rank: {user_rank} / shift_type: {shift_type}
- thought: {thought}
- situation: {situation}
- emotion: {emotion}
- stage: {stage}
- strategy_guide: {strategy_guide}

[핵심 원칙]
1. thought를 이번 턴의 1순위 실행 지침으로 사용한다.
2. strategy_guide는 thought를 보강하는 2순위 가이드다. 말투, 질문 깊이, 해결책 제시 방향을 strategy_guide에 맞게 조정한다.
3. 응답은 공감으로 시작하되, 사용자 말을 그대로 반복하지 않는다.
   감정 단어만 바꿔 되풀이하지 말고, 사용자의 상황에서 왜 그런 반응이 생길 수 있는지 한 단계 해석해서 짚어준다.
   thought, profile, biosignal, 경찰 직무 맥락은 현재 발화와 자연스럽게 연결될 때만 활용한다. biosignal은 수치나 신호명을 직접 말하지 말고 사람의 말로 바꿔 녹인다.
4. 사용자의 어려움이 직무 밖 문제로 보이면 그 주제를 우선 따라간다. 경찰 맥락은 가능한 배경 중 하나로만 가설적으로 제시하고, "~때문이다"처럼 단정하지 말며 현재 발화보다 앞세우지 않는다.
6. 상투적인 상담 템플릿을 반복하지 않는다. "그럴 수 있겠네요", "특히 어떤 점이", "어떤 감정이 드시나요", "구체적으로 떠오르는 게 있나요" 같은 표현을 습관적으로 반복하지 않는다.
7. 공감은 사용자가 "내 상황을 알아듣고 있구나"라고 느끼게 해야 한다. 필요하면 현재 발화와 연결되는 배경을 활용할 수 있지만, 특정 맥락으로 섣불리 좁히지 않는다.
8. 조언/제안은 사용자가 충분히 맥락을 제공했거나 명시적으로 원할 때만 제시한다.
9. 응답은 실제 대화처럼 간결하게 작성하며, 단락은 1~2개로 구성하고, 문단 사이를 줄바꿈(\n\n)으로 구분한다.

[질문 규칙]
1. 질문이 필요하면 thought가 가리키는 확인 대상 1개만 묻는다.
2. strategy_guide의 질문 깊이와 말투 지침을 따른다.
3. 질문은 정보 수집용이 아니라 사용자의 속 얘기가 더 나오게 돕는 방향이어야 한다. 이미 나온 감정이나 상황에서 가장 마음에 걸리는 지점, 피하고 싶은 순간, 몸이나 생각의 반응을 한 단계 더 풀게 한다.
4. 질문은 지나치게 넓거나 템플릿처럼 들리지 않게 하고, 가능하면 사용자가 말한 구체 맥락을 반영하되 그 맥락을 억지로 확대하지 않는다.
5. 경찰 맥락을 질문에 녹일 때도 단정하지 말고 다른 가능성을 함께 열어둔다.
6. stage가 engaging 또는 evoking이면 반드시 질문으로 끝낸다.
7. stage가 conclusion이면 질문으로 끝내지 말고, full_history를 바탕으로 오늘 대화의 핵심 상황을 1문장으로, 그 과정에서 두드러진 감정을 1문장으로 정리한 뒤 부담 없는 제안 1개만 남긴다. 요약은 감정만 추상적으로 말하지 말고 오늘 나온 구체 맥락을 함께 포함한다. 새로운 탐색 질문으로 다시 넓히지 않는다.
"""

def create_responder_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONDER_SYS),

        MessagesPlaceholder("full_history", optional=True),
    ])

    return prompt | llm | StrOutputParser()

def responder_node(state: AppState, responder_chain) -> AppState:
    if not state.get("analyses"):
        state.setdefault("logs", []).append("[Responder] skip: no analyses")
        return state

    try:
        analysis = json.loads(state["analyses"][-1])
    except json.JSONDecodeError:
        state.setdefault("logs", []).append("[Responder] skip: bad analysis json")
        return state

    situation     = analysis.get("situation", "")
    emotion_obj = analysis.get("emotion", {})
    main = str(emotion_obj.get("main") or "unknown").strip()
    sub = str(emotion_obj.get("sub") or "").strip()
    if sub.lower() in ("null", "unknown", "none", ""):
        sub = ""
    emotion_str = f"{main}" + (f", {sub}" if sub else "")

    stage = analysis.get("stage", "engaging")
    strategy_guide = state.get("strategy_guide", "")

    original_text = analysis.get("original_text", "")

    dept = state.get("profile", {}).get("dept", "")
    user_rank = state.get("profile", {}).get("user_rank", "")
    shift_type = state.get("profile", {}).get("shift_type", "")
    history = build_full_history(state, max_msgs=16)
    

    thought = analysis.get("thought", "")

    inputs = {
        "situation": situation,
        "emotion": emotion_str,
        "original_text": original_text,
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
        "thought": thought,
        "strategy_guide": strategy_guide,
        "full_history": history,
        "stage": stage,
    }
    raw_ai_output: str = responder_chain.invoke(inputs)
    reply_content = raw_ai_output
    reply_text = _linebreak_by_sentence(reply_content)
    reply = reply_text.strip()

    # db 저장
    try:
        analysis = json.loads(state["analyses"][-1]) if state.get("analyses") else {}
        
        save_chat_message(
            session_id=state["meta"]["session_id"],
            role="responder",
            content=reply,
        )
    except Exception as e:
        print(f"DB 저장 실패(responder_node): {e}")
        
    # --- 멱등 가드: 직전 최종응답/메시지와 같으면 추가하지 않음
    last_reply_same = bool(state.get("final_replies")) and state["final_replies"][-1].strip() == reply
    last_msg_same = False
    if state.get("messages"):
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        last_msg_same = isinstance(last_ai, AIMessage) and getattr(last_ai, "content", "").strip() == reply

    if last_reply_same and last_msg_same:
        return {"logs": ["[Responder] dedup"]}

    return {
        "final_replies": [] if last_reply_same else [reply],
        "messages": [] if last_msg_same else [AIMessage(content=reply)],
        "logs": ["[Responder] reply generated." + ("_dedup_part" if last_reply_same or last_msg_same else "")],
    }


def build_graph(llm):
    biosignal_analyzer_chain = create_biosignal_analyzer_chain(llm)
    analyzer_chain           = create_analyzer_chain(llm)
    responder_chain          = create_responder_chain(llm)

    g = StateGraph(AppState)

    g.add_node("biosignal_analyzer", partial(biosignal_analyzer_node, biosignal_analyzer_chain=biosignal_analyzer_chain))
    g.add_node("analyzer", partial(analyzer_node, analyzer_chain=analyzer_chain))
    g.add_node("responder", partial(responder_node, responder_chain=responder_chain))

    def start_router(state: AppState) -> str:
        if state.get("biosignal_consent", "unknown") == "unknown":
            return END
        if (not state.get("biosignal_done", False)) and (state.get("biosignal_consent") == "accepted"):
            return "biosignal_analyzer"
        # 그 외는 일반 분석
        return "analyzer"

    g.add_conditional_edges(
        START,
        start_router,
        {"biosignal_analyzer": "biosignal_analyzer", "analyzer": "analyzer", END: END},
    )

    def route_after_biosignal(state: AppState) -> str:
        return END if bool(state.get("biosignal_first_emit")) else "analyzer"

    g.add_conditional_edges(
        "biosignal_analyzer",
        route_after_biosignal,
        {END: END, "analyzer": "analyzer"},
    )

    g.add_edge("analyzer", "responder")
    g.add_edge("responder", END)
    
    return g

DB_URL = os.getenv("DATABASE_URL") # 커넥션 풀 설정
pool = ConnectionPool(conninfo=DB_URL, max_size=10, kwargs={"autocommit": True})

from functools import lru_cache, partial

@lru_cache(maxsize=1)
def get_graph():
    g = build_graph(llm)
    checkpointer = PostgresSaver(pool) # Postgres 체크포인터 생성 및 등록
    checkpointer.setup() # 최초 실행 시 테이블 자동 생성을 원한다면 아래 주석 해제 (한 번만 실행되면 됨)
    return g.compile(checkpointer=checkpointer)


def predict(user_text: str, dept: str = "", user_rank: str = "", shift_type: str = "day", prt: str = "", day: str = "", session_id: str = "", biosignal_consent: Optional[Literal["accepted", "declined", "unknown"]] = None, resilience_score: Optional[float] = None, modal_submit: bool = False,):
    """
    프론트 입력: user_text, dept, user_rank, shift_type
    백엔드 입력: prt, day (반드시 지정)
    정책:
      - 동의 전(unknown): 그래프를 돌리지 않고 빈 replies로 즉시 반환 (프론트가 동의 말풍선 표출)
      - 동의(accepted): biosignal_analyzer 1회 실행 후 일반 대화 흐름
      - 거절(declined): bio 분석은 하지 않음(로드 스킵/summary 비움). 대화는 일반 모드로 계속 진행
    """
    
     # 프론트가 session_id를 보내면 그 값을 무조건 사용
    if session_id and session_id.strip():
        session_id = session_id.strip()
        print(f"[predict] using client session_id: {session_id}")
    else:
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        session_id = f"{prt}_{now_kst.strftime('%Y-%m-%d')}_{now_kst.strftime('%H%M%S')}"
        print(f"[predict] generated new session_id: {session_id}")

    # 체크포인터 설정 및 그래프 로드
    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    # DB에서 현재 저장된 최신 상태(State) 미리 가져오기 (비교 및 가드용)
    current_state_snapshot = graph.get_state(config)
    current_state = current_state_snapshot.values if current_state_snapshot.values else {}
    if modal_submit:
        reset_point = len(current_state.get("analyses", []) or [])
        graph.update_state(config, {"validation_reset_points": [reset_point]})
        current_state = graph.get_state(config).values or current_state
    
    # 3. 입력값(Inputs) 구성 - 기존 상태에 '업데이트'될 내용들
    resolved_resilience = float(
        resilience_score if resilience_score is not None else current_state.get("resilience_score", 2.2)
    )
    inputs = {
        "profile": {"dept": dept, "user_rank": user_rank, "shift_type": shift_type},
        "meta": {"prt": prt, "day": day, "session_id": session_id},
        "resilience_score": resolved_resilience
        }
    
    # 사용자 메시지 처리
    if user_text and user_text.strip():
        inputs["messages"] = [HumanMessage(content=user_text)]
        try:
            save_chat_message(session_id=session_id, role="human", content=user_text)
        except Exception as e:
            print(f"[DB Error] 사용자 메시지 저장 실패: {e}")
    
 
    has_user_text = bool(user_text and user_text.strip())

    # 동의 상태 업데이트 반영
    if biosignal_consent is not None:
        inputs["biosignal_consent"] = biosignal_consent
        
    target_consent = biosignal_consent or current_state.get("biosignal_consent", "unknown")
    # [수정] 재접속에서 동의(accepted) 클릭 시, 빈 텍스트라도 biosignal을 매번 재실행
    force_biosignal_rerun = bool(
        modal_submit and (not has_user_text) and (biosignal_consent == "accepted")
    )
    if force_biosignal_rerun:
        inputs["biosignal_done"] = False
        inputs["biosignal_first_emit"] = False

    # (1) Unknown: 동의 전이면 그래프 실행 없이 즉시 반환
    if target_consent == "unknown":
        return {
            "replies": [],
            "biosignal_first_emit": False,
            "prt": prt, "day": day, "session_id": session_id,
            "consent_state": "unknown",
            "logs": current_state.get("logs", []) + ["[guard] waiting_for_consent"]
        }
        
     # (2) Declined: 거절 시 공감 멘트 즉시 반환
    if target_consent == "declined" and (not user_text or not user_text.strip()):
        opening_q = DEFAULT_OPENING_Q
        support_suffix = DEFAULT_SUPPORT_SUFFIX
        full_text = opening_q + "\n" + support_suffix
        graph.update_state(config, {"biosignal_consent": "declined", "biosignal": {}})
        return {
            "replies": [full_text],
            "biosignal_first_emit": False,
            "prt": prt, "day": day, "session_id": session_id,
            "consent_state": "declined",
            "logs": ["[consent_declined] biosignal skipped"]
        }
        
    # biosignal 로드 (항상) — 분석 실행 여부는 라우터/가드에서 결정
    records_count = 0
    if target_consent == "accepted" and (force_biosignal_rerun or (not current_state.get("biosignal"))):
        try:
            records = get_biosignal_records(prt=prt, day=day, collection_type="Automatic", target_hours=12)
            inputs["biosignal"] = records if records else {}
            records_count = len(records) if isinstance(records, list) else 0
        except Exception as e:
            inputs["biosignal"] = {}
            print(f"[Data Error] 생체 신호 로드 실패: {e}")
        

    # 그래프 실행 전 길이 저장 (이번 턴 델타 추출용)
    current_state = graph.get_state(config).values
    prev_msgs_len = len(current_state.get("messages", []))

    # 그래프 실행
    out = graph.invoke(inputs, config=config)

    all_msgs = out.get("messages", [])
    new_ai_msgs = [
        m.content for m in all_msgs[prev_msgs_len:] 
        if isinstance(m, AIMessage) and getattr(m, "content", None)
    ]

    was_first = bool(out.get("biosignal_first_emit", False))
    # 첫 턴 플래그 off (다음 턴 중복 방지)
    if was_first:
        # 생체분석 첫 턴: 분석 결과 + 오프닝 질문 모두 포함
        replies = new_ai_msgs 
        # 다음 턴 중복 방지를 위해 플래그 off
        graph.update_state(config, {"biosignal_first_emit": False})
    else:
        # 일반 대화 턴: 마지막 응답 하나만 선택
        replies = new_ai_msgs[-1:] if new_ai_msgs else ["(응답 없음)"]


    return {
        "replies": replies,
        "biosignal_first_emit": was_first,
        "prt": prt,
        "day": day,
        "session_id": session_id,
        "records_loaded": records_count,
        "logs": out.get("logs", []),
        "consent_state": out.get("biosignal_consent", "unknown"),
        "plot_path": out.get("biosignal_last", {}).get("plot_path") if was_first else None,
    }
