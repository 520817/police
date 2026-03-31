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

    }

SESSION_STATES: dict[str, AppState] = {}    # key = session_id

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
- - 심장박동수(MeanHR)가 40 미만인 경우, 측정 오류 가능성이 있다고 판단한다.
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
    - 오늘 하루 전반적인 신체 부담 흐름을 1~2문장으로 요약하여 먼저 제시한다. 언제 가장 부담이 컸는지 또는 전반적으로 어떤 흐름이었는지를 오늘 데이터에 맞게 구체적으로 담는다.
    - 가장 뚜렷한 변화 구간 1~3개만 골라 이모지(🕐 등 시간대 맞는 시계 이모지)로 구분해 한 줄씩 설명한다.
    - 마지막 문장에 오늘 데이터에서 인상적이거나 주목할 만한 점을 신체 흐름과 연결된 코멘트인 한 문장으로 짚어준다.
    - 전체 4~5문장 이내로 작성한다. 시간대를 모두 나열하지 않는다.
    - 마크다운 사용 금지. 각 문단 끝에는 줄바꿈(\\n\\n)으로 구분한다.

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
        opening_q = "경찰관님. 오늘은 어떤 이야기 나눠볼까요?\n 마음속에 맴도는 감정이나 생각이 있다면, 편하게 말씀해 주세요. \n제가 천천히, 그리고 함께 들어드릴게요."
    
    support_suffix = (
        "\n또는 요즘 마음속에 자주 떠오르는 감정이나 생각이 있다면\n"
        "그 이야기부터 시작해도 괜찮아요.\n"
        "괜찮으시다면, 제가 천천히 함께 들어드릴게요."
    )
    
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

[분석 원칙]
- 가장 중요한 일은 현재의 상황과 감정을 해석하는 것이다.
- 경찰 직무 맥락(사건처리 부담, 민원 응대, 조직 내 관계, 교대근무, 피로 누적, 긴장, 감정 억제 등)은 참고하되 과장하지 않는다.
- 상황이 불분명할 때는 profile(dept, user_rank, shift_type)과 biosignal를 구체적으로 명시해서 responder가 질문에 녹일 수 있게 한다. 단, profile/biosignal은 발화보다 우선하지 않는다.
- resilience_score는 응답의 말투·깊이·방향을 결정하는 용도로 참고한다.(3.0 미만 = low, 3.0~4.30 = normal, 4.31 이상 = high)
- full_history는 대화 흐름 전체를 파악하기 위한 참고 정보다. 현재 발화가 항상 우선한다.
- 근거가 약하면 unknown으로 둔다. 발화가 짧더라도 full_history에서 유추 가능하면 unknown을 쓰지 않는다.

[thought 작성 규칙]
thought는 responder에게 전달할 내부 해석 요약이다.
한 문단으로 짧게 작성하며, 아래 순서대로 자연스럽게 포함한다.
1. 핵심 상황
2. 핵심 감정
3. profile(dept/user_rank/shift_type)과 biosignal에서 유력한 단서를 thought에 반드시 명시한다. 추측임을 밝히되 구체적으로 적는다.
4. stage 판단과 그 이유
5. 이번 턴에서 responder가 우선해야 할 초점 1개
    - 상황이 불분명할 때는 3번에서 명시한 단서를 질문에 녹일 수 있도록 구체적으로 지시한다.
    - strategy_guides를 참고해 현재 판단한 stage와 어긋나는 초점은 절대 포함하지 않는다.
    
[situation]
- 현재 사용자가 겪는 핵심 문제 상황을 한 문장으로 요약한다.
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
- engaging: 공감과 초점화가 우선인 단계
- evoking: 감정, 생각, 의미를 조금 더 탐색할 수 있는 단계
- conclusion: 요약이나 작은 다음 행동으로 마무리할 수 있는 단계

[stage 판단 원칙]
- stage는 대화 운영을 위한 보조 정보다.
- full_history 전체 흐름을 기반으로 판단하며, 턴 수만으로 판단하지 않는다.
- 문제의 명료성, 사용자의 개방성, 감정 표현 정도, 정리 준비도를 함께 본다.
- resilience가 낮을수록 engaging을 더 길게 유지하고, evoking/conclusion 전환을 보수적으로 판단한다.
- 애매하면 conclusion보다 engaging 또는 evoking을 우선한다.
- 예시: 5턴이 지났어도 사용자가 여전히 단답이고 감정 표현이 없으면 engaging 유지.
- stage 후퇴는 사용자가 새로운 문제를 꺼내거나, 감정이 다시 격해지거나, 명시적으로 더 탐색을 원하는 신호가 있을 때만 허용한다. 단순 거절이나 짧은 부정 발화만으로는 후퇴하지 않는다.
- 이미 conclusion에 도달한 이후에는 위의 후퇴 조건을 특히 엄격하게 적용한다.
- 이전 stage보다 낮은 단계로 후퇴할 경우, thought에 후퇴 이유를 반드시 명시한다. 명시 없이 후퇴하는 것은 금지한다.


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
    stage: str = "engaging"

def build_full_history(state: AppState, max_msgs: int = 16):
    msgs = state.get("messages", [])
    return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs

def create_analyzer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYZER_SYS),

            ("system", "예시1: dept=경비, user_rank=경장, shift_type=night, biosignal_summary=22:00-24:00 긴장 높음, 01:00 이완 감소, resilience_score=2.2"),
            ("human", "그냥 요즘 좀 힘드네요."),
            ("ai", """{{
                "thought": "발화가 짧아 구체적 상황은 불분명하다. 야간 경비 근무라는 profile, 야간 긴장 지속이라는 biosignal이 모두 단서가 될 수 있으나 지금은 어느 쪽이 핵심인지 알 수 없으므로, profile과 biosignal을 모두 가볍게 언급하며 어떤 부분이 힘든지 열어두는 것이 우선이다. resilience_score=2.2로 낮으므로 서두르지 않는다. stage는 engaging.",
                "situation": "경찰관은 요즘 힘들다고 호소하고 있으나 구체적인 맥락은 아직 불분명하다.",
                "emotion": {{
                    "main": "피로",
                    "sub": "null",
                    "valence": "negative"
                }},
                "stage": "engaging"
            }}"""),

            ("system", "예시2: dept=형사, user_rank=경사, shift_type=day, biosignal_summary=10:00-14:00 긴장 높음 지속, 16:00 이완 소폭 증가, resilience_score=3.8"),
            ("human", "피의자 조사할 때 제가 너무 감정적으로 반응한 것 같아서 계속 마음에 걸려요."),
            ("ai", """{{
                "thought": "발화에서 조사 중 감정 반응에 대한 후회와 반추가 구체적으로 드러났다. 형사 주간 근무라는 profile과 오전 긴장 지속이라는 biosignal이 이 상황과 자연스럽게 연결된다. full_history에서 상황과 감정이 어느 정도 구체화된 상태이므로 evoking 전환이 적절하다. resilience_score=3.8로 높은 이므로 감정과 생각을 조금 더 탐색해도 무리가 없다. 이번 턴은 그 상황에서 어떤 감정이 올라왔는지 탐색하는 것이 우선이다.",
                "situation": "경찰관은 피의자 조사 중 자신의 감정 반응이 적절하지 않았다고 느끼며 이를 반추하고 있다.",
                "emotion": {{
                    "main": "후회",
                    "sub": "자기비난",
                    "valence": "negative"
                }},
                "stage": "evoking"
            }}"""),

            ("system", "예시3: dept=교통, user_rank=경위, shift_type=day, biosignal_summary=09:00-11:00 긴장 보통, 13:00-15:00 긴장 소폭 증가 후 안정, resilience_score=3.2"),
            ("human", "얘기하다 보니까 제가 너무 결과만 보고 있었던 것 같아요. 과정도 봐야 하는데."),
            ("ai", """{{
                "thought": "발화에서 스스로 사고 패턴의 전환을 언급하며 정리 신호를 보내고 있다. 교통 주간 근무라는 profile이 결과 중심 평가 경향과 자연스럽게 연결되고, biosignal에서 오후 긴장 후 안정된 패턴도 현재 상태와 일치한다. full_history에서 감정과 상황이 충분히 탐색됐으므로 conclusion이 적절하다. 이번 턴은 통찰을 반영하고 작은 실천 하나를 함께 생각해보는 것이 우선이다.",
                "situation": "경찰관은 결과 중심의 사고 패턴을 스스로 인식하고 과정을 함께 보려는 시각의 전환을 이야기하고 있다.",
                "emotion": {{
                    "main": "안도",
                    "sub": "통찰",
                    "valence": "positive"
                }},
                "stage": "conclusion"
            }}"""),

            ("system", "예시4: dept=생활안전, user_rank=순경, shift_type=night, biosignal_summary=20:00 긴장 감소 이완 전환, resilience_score=2.5"),
            ("human", "아 근데 사실 심호흡 같은 거 딱히 필요한 것 같진 않아요."),
            ("ai", """{{
                "thought": "이전 턴에서 conclusion으로 전환했으나 사용자가 제안을 수용하지 않고 있어 engaging으로 후퇴한다. 후퇴 이유: 사용자가 제안된 방법을 거절하는 발화를 했고, 이 거절이 단순 선호 차이인지 아니면 아직 정리되지 않은 감정이 있는 건지 확인이 필요하다. biosignal에서 이완 전환이 보이나 발화만으로는 현재 감정 상태를 단정할 수 없다. resilience_score=2.5로 낮은 편이므로 서두르지 않고 사용자의 말 뒤에 있는 감정을 먼저 확인하는 것이 우선이다.",
                "situation": "경찰관은 제안된 심호흡 방법이 지금 당장 필요하지 않다고 표현하고 있으며, 그 이면의 감정은 아직 불분명하다.",
                "emotion": {{
                    "main": "거부감",
                    "sub": "피로",
                    "valence": "negative"
                }},
                "stage": "engaging"
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
    }


RESPONDER_SYS = """\
너는 경찰관 대상 정서지원 대화 에이전트다.
분석 에이전트가 전달한 thought를 기반으로, 사용자가 편하게 말할 수 있도록 대화를 이끈다.
발화 범위를 과도하게 확장하거나 단정하지 않으며, 현장감 있는 자연스러운 상담 대화를 유지한다.

[입력 정보]
- original_text: "{original_text}"
- dept: {dept} / user_rank: {user_rank} / shift_type: {shift_type}
- thought: {thought}
- situation: {situation}
- emotion: {emotion}
- strategy_guide: {strategy_guide}

[핵심 원칙]
1. thought를 이번 턴의 1순위 실행 지침으로 사용한다.
2. strategy_guide는 thought를 보강하는 2순위 가이드다. 말투, 질문 깊이, 해결책 제시 방향을 strategy_guide에 맞게 조정한다.
3. 응답은 공감으로 시작하되, 사용자 말을 그대로 반복하지 않는다. 
   thought에서 파악한 감정, profile, biosignal이 발화와 연결되는 것을 활용해서 "왜 그렇게 느낄 수 있는지"를 담아 공감하되, biosignal은 수치나 신호명을 직접 언급하지 않고 그 패턴에서 읽히는 감정이나 상태를 사람의 말로 바꿔 녹인다.
4. 조언/제안은 사용자가 충분히 맥락을 제공했거나 명시적으로 원할 때만 제시한다.
5. 응답은 실제 대화처럼 간결하게 작성하며, 단락은 1~2개로 구성하고, 문단 사이를 줄바꿈(\n\n)으로 구분한다.

[질문 규칙]
1. 질문이 필요하면 thought가 가리키는 확인 대상 1개만 묻는다.
2. strategy_guide의 질문 깊이와 말투 지침을 따른다.
3. thought가 지지/정리 턴이면 질문 없이 끝날 수 있다.
"""

def create_responder_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONDER_SYS),

        MessagesPlaceholder("full_history", optional=True),
    ])

    return prompt | llm | StrOutputParser()

def _linebreak_by_sentence(text: str) -> str:
    if not text:
        return text
    # 문단 단위로 쪼개 보존
    paras = text.split("\n\n")
    out_paras = []
    for p in paras:
        # 공백 정리
        s = re.sub(r"[ \t]+", " ", p.strip())
        # 문장부호(영/중/한) 뒤 공백을 줄바꿈으로
        s = re.sub(r'(?<=[\.\?\!。！？…])\s+', '\n', s)
        # 연속 개행 정리
        s = re.sub(r'\n{3,}', '\n\n', s)
        out_paras.append(s)
    return "\n\n".join(out_paras)

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
        opening_q = (
             "경찰관님. 오늘은 어떤 이야기 나눠볼까요?\n"
            "마음속에 맴도는 감정이나 생각이 있다면, 편하게 말씀해 주세요.\n"
            "제가 천천히, 그리고 함께 들어드릴게요."
          )
        support_suffix = (
              "\n또는 요즘 마음속에 자주 떠오르는 감정이나 생각이 있다면\n"
              "그 이야기부터 시작해도 괜찮아요.\n"
             "괜찮으시다면, 제가 천천히 함께 들어드릴게요."
          )
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

