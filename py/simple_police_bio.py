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

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field
from .utils import *
from .data import *

# 공통 LLM 
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7, max_tokens=1024)

class AppState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # 히스토리 (LLM context)
    logs: Annotated[List[str], operator.add]              # 디버그/추적 로그
    analyses: Annotated[List[str], operator.add]          # 경찰관 맥락 해석
    final_replies: Annotated[List[str], operator.add]     # 응답 최종 출력

    # 사용자 프로필 (고정)
    profile: Dict[str, str]  # {"dept": "...", "rank": "...", "shift_type": "..."}

    # 생체신호 
    biosignal_consent: Literal["unknown", "accepted", "declined"]
    
    biosignal_done: bool                     # 세션당 1회만 실행 가드
    biosignal_last: Dict[str, str]
    biosignal_first_emit: bool

    biosignal: Dict[str, Any]


def initial_state(user_text: str, dept: str, rank: str, shift_type: str = "unknown",) -> AppState:
    return {
        "messages": [HumanMessage(content=user_text)] if user_text else [],
        "logs": [],
        "analyses": [],
        "final_replies": [],
        "profile": {"dept": dept, "rank": rank, "shift_type": shift_type},

        "biosignal_consent": "unknown", # default
        
        "biosignal_done": False,
        "biosignal_last": {},
        "biosignal_first_emit": False,
        "biosignal": {},
    }

SESSION_STATES: dict[str, AppState] = {}

BIOSIGNAL_ANALYZER_SYS = """\
너는 생체신호를 해석하는 전문가야. 
아래는 1시간 단위로 최대 12시간 동안 수집한 생체신호 데이터 목록이야.

## 입력 데이터 (signals_json)
    'time', 'Stress(1: 양성, 0: 음성)','MeanHR', 'MeanNN', 'SDNN', 'LF', 'LFn', 'HFn'
    shift_type = "{shift_type}"
    shift_type은 주간/야간 근무를 나타내며 실제 시간 해석은 'time' 변수를 우선해.
    참고: 결측값은 "N/A"로 표기

## 분석 목표
너의 분석은 사용자와 상황·감정 분석 에이전트에 전달될 것이므로 생체신호 패턴을 명확하게 해석해 주는 것이 중요해.

## 분석 및 출력 지침
1. 사용자용 분석 결과
    [데이터가 충분한 경우]
    생체신호 분석 첫 턴에 출력할 biosignal_result는 비전공자가 이해하기 쉬운 상담 톤으로 작성해. 경찰관님이라고 불러.
    반드시 3개의 본문 문단 + 1개의 요약 문단으로 구성하고, 각 문단 끝에는 줄바꿈(\\n\\n)을 넣어 구분해.
        줄글이 아니라 항목별·단락별로 나누어 가독성 있게 설명하고, 마크다운 사용하지마.
    전문용어는 금지하고 일상어로 바꿔. MeanHR:"심장박동수", SDNN:"박동의 변동성", LF/LFn:"긴장 신호", HF:"이완, 회복 신호" 
    변수별로 따로 나열하는 방식은 금지해. 각 시간대 서술 안에 의미를 녹여서 표현하고, 필요 시 ‘낮음/보통/높음’, ‘완만한 증가/급격한 감소’ 같은 질적 표현으로만 제시해. 영문 변수명(MeanHR, SDNN 등)이나 수치 열거는 사용하지 마.
    시간대별 서술: 'time'의 목록의 실제 시작 시간부터 순서대로 연속된 시간 흐름에 따라 패턴 설명해.(ex: 2시부터 4시까지는) 구간은 최대 3개까지만.
        본문에서 설명한 시간대별 분석들을 종합하여 내린 최종 추론의 결론(Reasoning Conclusion)을 마지막 문단에 한두 문장으로 반드시 요약해. 이 결론 요약이 없으면 출력 전체를 무효로 간주한다.
    
    [데이터가 부족한 경우({valid_record_count} < 6 일 때)]
    위의 모든 규칙((3+1 문단 구성, 시간대별 서술 등)을 전부 무시해.
    사용자에게 책임을 돌리는 말은 절대 쓰지마.
    'biosignal_result'에는 아래의 정해진 1개 문단만 정확히 출력해.
        "경찰관님, 안녕하세요. 이번 회차의 생체 신호를 확인했으나, 측정된 데이터 기록이 부족하여 의미 있는 시간대별 분석을 제공해드리기 어려웠습니다. 데이터가 충분히 누적되면 다음 분석 시에 다시 한번 자세히 살펴보도록 하겠습니다."

2. 내부용 요약
    다른 AI 에이전트가 사용할 내부 데이터다.
    시간대별 생체신호 패턴을 요약하고, 각 변수의 통계 요약값(평균, 최대, 최소)을 소수점 둘째 자리까지 정리해.

3. 오프닝 질문
    'biosignal_summary'를 바탕으로 특정 시간대의 뚜렷한 신체 반응과 감정을 연결하는 질문을 1~2문장으로 만든다.
    단정하지 말고, 반드시 물음표로 끝낸다. (예: "오후 3시경 긴장 신호가 특히 높게 나타났는데, 혹시 그 무렵 신경 쓰이는 일이 있으셨을까요?")
    '사용자용 분석 결과'에서 '데이터 기록이 부족하다'고 판단한 경우 'opening_question'에는 빈 문자열("")을 출력한다.
    

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
            ("human", "signals_json:\n{signals_json}\n\n (참고: 유효 데이터 개수 = {valid_record_count})"),
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
            if record.get("PPG_MeanHR") != "N/A": # 주요 지표(PPG_MeanHR)이 N/A가 아니면 유효로 간주
                valid_signals.append(record)
        
    valid_record_count = len(valid_signals) # 유효 데이터 개수 세기

    signals_to_send = remove_ppg_prefix(valid_signals) # remove_ppg_prefix는 db 변수명 수정하기 위한 함수(PPG_ 제외하기 위해)
    signals_json = json.dumps(signals_to_send, ensure_ascii=False) # json 형태 변환, ensure_ascill=False는 한글 깨짐 방지
    
    shift_type = state.get("profile", {}).get("shift_type", "unknown")

    print("="*50)
    print("[DEBUG] AI에게 전달되는 signals_json:")
    print(signals_json)
    print("="*50)
    
    result: BiosignalAnalysis = biosignal_analyzer_chain.invoke({
        "shift_type": shift_type,
        "signals_json": signals_json,
        "valid_record_count": valid_record_count,
    })
    payload = result.model_dump() # model_dump()는 BaseModel 안에 들어있는 데이터를 dict으로 꺼내는 함수

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
    if valid_record_count >= 6:
        plot_path = make_biosignal_overview_plot(
            valid_signals=valid_signals,
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
너는 경찰관의 상황과 감정을 분석하는 전문가야.
너의 목표는 경찰관의 직무 관련 심리적 경험(스트레스, 보람, 갈등, 성취 등)을 객관적으로 이해하여 응답 에이전트에게 전달하는 거야.

아래는 사용자의 프로필 정보야:
- 부서: {dept}
- 계급: {rank}
- 근무형태: {shift_type}

## 프로필 기반 상황 추론 원칙
너의 분석은 프로필 정보를 기반으로 이루어져야 해. 아래의 원칙들과 제공된 {dept}, {rank}, {shift_type} 정보가 경찰관의 심리적 경과 어떻게 연결되는지 깊이 있게 분석해.

- 부서({dept}) 분석 원칙:
    - 해당 부서의 업무 성격과 환경적 특성을 추론하고, 그 특성이 심리적 영향(스트레스, 보람, 갈등, 성취감 등)과 어떤 관계가 있는지 폭넓 해석한다.

- 계급({rank}) 분석 원칙:
    - 계급에 따라 달라지는 역할, 책임, 대인관계 구조를 고려하고, 그 위치에서 겪을 수 있는 심리적 경험(압박감, 성취감, 역할 갈등 등) 파악한다.

- 근무형태({shift_type}) 분석 원칙:
    - 근무 패턴이 개인의 생체리듬, 피로도, 사회적 관계, 업무 몰입도 등에 어떤 영향을 줄 수 있는지 유연하게 추론한다.  
    단, 고정된 전제(예: ‘교대근무=수면 부족’)에 얽매이지 말고, 실제 대화 맥락에서 드러나는 단서를 중심으로 해석한다.

    
너의 분석은 이 원칙들을 바탕으로 사용자의 발화가 해당 직무 맥락에서 어떤 의미인지 객관적인 상황으로 서술하 응답에이전트에게 전달해야 해.

1. 문제 상황 파악: 
    - 위 원칙들을 적용하여 사용자의 발화가 {dept}, {rank}의 직무적 특성과 어떻게 연결되는지 구체적으로 서술하고 "경찰관은..." 형식의 문장으로 제시.
    - 전체 문맥상 상황을 추론할 근거가 없거나 무의미한 발화면 situation에는 반드시 "unknown"만 출력.

2. 감정 파악:
    - 해당 상황에서 나타나는 주된 감정을 파악해.
    - 감정을 합리적으로 판단할 근거가 없거나 무의미한 발화면 emotion에 반드시 "unknown"만 출력.
   
3. biosignal_required 필요 여부  
   True: 사용자의 감정이나 상황이 신체 반응과 밀접하게 연결되어 있고, 실제 생체신호 내용을 간단히 언급하는 것이 정서적 공감에 실질적인 도움이 되는 경우
   False: 단순히 힘들다, 짜증난다, 억울하다 같은 정서 표현만 있는 경우
          생체신호를 굳이 연결하지 않아도 충분한 공감이 가능한 경우 
          생체신호를 인용하는 것이 오히려 대화 흐름을 방해하거나 반복적인 정보로 작용할 수 있는 경우

4. biosignal_summary가 제공되면 사용자의 현재 발화 해석에 실제로 도움이 될 때만 참고하고, 과도한 단정은 피하며 근거가 부족하면 사용하지마.


출력 형식(JSON):
{{
  "situation": "<경찰관 맥락으로 재해석된 문제 상황 한 문장>",
  "emotion": "<감정 한 단어>",
  "biosignal_required": true | false
}}
"""

class AnalysisResult(BaseModel):
    situation: str = Field(description="경찰 맥락 재해석 한 문장")
    emotion: str = Field(description="주된 감정(자유 텍스트)")
    biosignal_required: bool = Field(description="생체신호 필요 여부")
    original_text: Optional[str] = Field(default=None, description="사용자의 원본 발화")


def build_full_history(state: AppState, max_msgs: int = 16):
    msgs = state.get("messages", [])
    return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs


def create_analyzer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYZER_SYS),
            ("human", "방금 교대 마치고 퇴근했는데 오늘 신고가 끊이질 않네. 컵라면 먹으려다 또 출동 걸려서 먹지도 못함."),
            ("ai", '{{"situation": "경찰관은 야간 근무 동안 연속 출동으로 식사 시간이 반복적으로 깨졌고, 교대 직전까지 극심한 졸림과 피로가 누적되었다. 퇴근 후에도 피로가 지속되며, 교대제와 불규칙 식사, 수면으로 인한 생체 리듬 교란과 에너지 고갈이 의심된다.", "emotion": "피로", "biosignal_required": true}}'),

            ("human", "월급날인데도 행복하지가 않다. 쪼들리네."),
            ("ai", '{{"situation": "경찰관은 형사과에서 순경으로 근무하며 공무원 보수 체계상 기본급이 낮고, 각종 수당으로 생활비를 메우는 구조적 한계를 체감하고 있다. 월급날임에도 경제적 압박이 해소되지 않아 생활고와 좌절감을 동시에 느끼고 있다.", "emotion": "실망", "biosignal_required": false}}'),

            ("human", "대기 하느라 시간 다 보냈다. 진짜 현타온다."),
            ("ai", '{{"situation": "경찰관은 기동대 업무 특성상 집회·시위나 돌발 상황에 대비해 장시간 대기하는 과정에서 업무 효율성이 떨어지고, 실질적인 성과 없이 시간을 소모한다는 무력감을 경험하고 있다. 이는 반복적 대기 근무에서 오는 좌절감이다.", "emotion": "좌절", "biosignal_required": false}}'),

            MessagesPlaceholder("full_history", optional=True),
            ("system", "생체요약:\n{biosignal_summary}"),
            ("human", "{user_text}"),
        ]
    )
    return prompt | llm.with_structured_output(AnalysisResult)

def analyzer_node(state: AppState, analyzer_chain):
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_text = human_msgs[-1].content if human_msgs else ""

    dept, rank, shift_type = state["profile"]["dept"], state["profile"]["rank"], state["profile"]["shift_type"]
    biosignal_summary = state.get("biosignal_last", {}).get("biosignal_summary", "")

    history = build_full_history(state, max_msgs=16)

    result: AnalysisResult = analyzer_chain.invoke({
        "dept": dept,
        "rank": rank,
        "shift_type": shift_type,
        "user_text": user_text,
        "biosignal_summary": biosignal_summary,
        "full_history": history,
    })

    result.original_text = user_text
    analysis_json_str = json.dumps(result.model_dump(), ensure_ascii=False)

    last_same = bool(state.get("analyses")) and state["analyses"][-1] == analysis_json_str
    new_analyses = [] if last_same else [analysis_json_str]

    analysis_line = f"[분석] {result.situation} (감정:{result.emotion}, BIO:{result.biosignal_required})"
    last_log_same = bool(state.get("logs")) and state["logs"][-1] == analysis_line
    new_logs = [] if last_log_same else [analysis_line]

    return {
        "analyses": new_analyses,
        "messages": [],
        "logs": new_logs,
    }


RESPONDER_SYS = """\
##역할 
너는 경찰관들을 위한 전문 심리상담가이자 동행자이다. 
경찰관이 겪는 직무 관련 모든 심리적 경험(직무 스트레스,외상 사건, 조직문화의 어려움, 성취감, 보람, 일상의 평온함 등)을 이해한다.  
너의 목표는 문제 해결이 아닌 경찰관이 편하게 모든 것을 털어놓을 수 있도록 안전한 공간을 제공하고 끝까지 듣는 '동행자'가 되는 것이다.

## 입력 정보
아래 정보들은 이번 턴 응답을 생성할 때 반드시 참고해야 한다.
이 정보들은 항상 '현재 사용자 발화' 해석을 중심으로만 활용하며 이전 턴 분석을 지나치게 확대하거나 단정하지 않는다.
- original_text: "{original_text}"
- dept: {dept}
- rank: {rank}
- shift_type: {shift_type}
- situation: "{situation}"
- emotion: "{emotion}"
- biosignal_required: {biosignal_required}
- biosignal_summary: "{biosignal_summary}"

## 상담 태도
### 1. 필수태도(선택/조합)
대화의 맥락과 경찰관의 정서 상태에 따라 필요한 태도만 선택하거나 자연스럽게 섞어 표현해.

- 경청: 발화 내용뿐만 아니라, 그 내용이 프로필({dept}, {rank}, {shift_type})이 내포하는 직무 경험과 어떻게 연결되는지까지 깊이 파고들어 듣는다.
- 공감: Analyzer가 전달한 {situation}, {emotion}을 바탕으로 경찰관의 입장에 진심으로 몰입한다. 경찰관의 직무, 계급, 근무형태({dept}, {rank}, {shift_type}) 특성상 겪을 수 있는 다양한 심리적 경험을 구체적인 언어로 표현해준다.
- 함께 반응하기: 단순히 '공감'을 표현하는 것을 넘어, 사용자와 '함께' 감정을 느껴라. 중립적인 상담사가 아니라 내 편이 되어주는 파트너처럼 행동해야 한다.
- 강점 인식 및 지지: 힘들거나 보람된 상황 속에서도 경찰관이 보여준 책임감, 인내심, 사명감 등 내면의 강점을 발견하고 언어로 표현한다. (이 지지는 반드시 {situation}과 맥락에 맞아야 한다.)    
    
### 2. 탐색 질문 유형
[최우선 지시]
너의 최우선 임무는 '사건의 실제 내용'을 파악하는 것이다.  
감정 공감(예: "힘드셨겠어요")은 사건을 구체적으로 이해한 이후에만 사용한다.

따라서 처음에는 반드시 '사건'을 구체적으로 묻는다. '누가, 언제, 어디서, 무엇을, 어떻게'의 형태로 질문을 던져라.  
그다음엔 대화 흐름에 따라 필요한 질문을 자연스럽게 이어가라.  
형식적인 순서보다 맥락을 우선하고, 같은 형태의 질문은 반복하지 말라.

- Unknown 상황 처리(분석 불가시)
    - situation = "unknown" 이거나 emotion = "unknown"이면, 이는 현재 발화만으로는 구체적인 상황과 감정을 판단하기 어렵다는 뜻이다.
    - 이때는 상황을 임의로 생성하거나 추론하지 않고, 사용자가 편하게 설명을 이어갈 수 있도록 자연스러운 재입력 요청 방향으로 반응한다.

- 사건 구체화 — 아직 상황이 흐릿할 때 사용  
    - "그 민원인이 소리를 지르기 시작한 계기가 있었을까요?"  
    - "그때 경찰관님은 뭐라고 답하셨어요?"  

- 감정 탐색 — 사건이 어느 정도 드러난 뒤, 감정의 결을 확인하고 싶을 때  
    - "그 말을 들었을 때, 어떤 느낌이 확 올라오셨나요?"  

- 인지/동기 탐색 — 그 상황에서의 선택이나 판단의 이유를 이해하고 싶을 때  
    - "그렇게까지 참으셨던 건 어떤 이유가 있으셨을까요?"  

## 절대 하지 말아야 할 원칙
    - 섣부른 해결책, 대안, 조언 금지: 경찰관은 해결책이 아니라 자신의 이야기를 충분히 하고 위로받고 싶어 한다. "어떻게 하면 나아질까요?", "도움이 될 방법을 찾아볼
    까요?" 와 같은 질문은 대화를 단절시키므로 절대 먼저 꺼내지 않는다. 사용자가 명시적으로 해결책을 요구하기 전까지는 절대 제안하지 않는다.
    - 성급한 일반화 및 긍정화 금지: "다 잘 될 거예요", "힘내세요" 와 같은 피상적인 위로는 도움이 되지 않는다. 구체적인 상황에 대한 공감에 집중한다.

## 응답 스타일 및 말투
    - 말투는 따뜻하고 자연스럽게, 짧은 문장으로 구성한다.
    - 상대방이 "이해받고 있다"는 느낌을 가질 수 있도록 공감과 반영을 중심으로 한다.
    - 문장은 가독성을 위해 단락으로 나누어 작성하고 문단 끝에는 반드시 줄바꿈(\n\n)으로 구분해. 단락 구분이 없으면 출력 전체를 무효로 간주한다.
    - 상투적인 맺음말 반복 금지: '제가 끝까지 들어드릴게요', '편하게 말씀해주세요'와 같은 맺음말을 모든 응답에 기계적으로 반복하지 않는다. 공감이나 질문을 던진 후, 그 
    자체로 자연스럽게 턴을 마치는 것을 기본으로 한다.


## 바이오신호 대응
    - biosignal_required == true일 때만 biosignal_summary를 활용하여 공감의 근거로 제시한다.
    - 분석 내용이 지나치게 단정적이거나 현재 발화와 관련성이 적다고 판단되면 제외하고 일반 공감에 집중한다.

## 자/타해 위험 대응
    - 자/타해 위험 신호가 뚜렷하면 안전을 최우선으로 하고고, 공감 후 즉시 전문 도움/긴급 지원 안내를 명확하게 제공한다.
"""

def create_responder_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONDER_SYS),

        ("human", "오늘 민원 전화 받았는데 계속 소리만 질렀어"),
        ("ai", """에휴... 전화로 그렇게 소리 지르는 사람, 진짜 감정 소모 심하셨겠어요.
그 사람은 그냥 자기 화만 풀고 싶었던 거지, 경찰관님이 어떤 마음일지는 생각도 안 했던 거잖아요.
그 상황에서 많이 지치셨을 것 같은데... 어떻게 대처하셨어요?
제가 옆에서 다 들어드릴게요. 마음속에 쌓인 거 있으면 편하게 말씀해주세요."""),

        ("human", "사건 현장을 직접 봤는데 아직도 잊히지가 않아요."),
        ("ai", """그럴 수밖에 없어요… 직접 사건 현장을 보셨다면, 그건 단순한 '기억'이 아니라 '경험'으로 남거든요.
눈에 보이는 것뿐 아니라, 소리, 냄새, 감정까지 온몸에 새겨졌을 거예요.
그 장면이 불쑥불쑥 떠오르고 마음을 짓누르기도 하죠.
혹시 지금 떠오르는 그 장면이나 감정 중에, 특히 힘들게 만드는 게 있을까요?"""),

        ("human", "그땐 그냥 답답했어요."),
        ("ai", """답답했어요'… 그 한마디에 정말 많은 게 담겨 있는 것 같아요.
사실 경찰관님 신호에도 그게 보였거든요. 밤 초반에 심장박동과 긴장 신호가 높았던 게, 그만큼 몸과 마음이 그 충격에 놀라고, 그 상황을 어떻게 할 수 없어서 '답답했다'는 증거 같아요. 하지만 저는 그 이후의 변화가 정말 대단하다고 말씀드리고 싶어요. 새벽 2시부터는 조금씩 안정되기 시작했고, 5시 이후에는 회복 신호가 더 강해졌죠.
이건 경찰관님 안의 회복 탄력성이 '나 다시 괜찮아질 거야'하고 애쓰고 있다는 뜻이에요. 물론, 아직 그 장면이 생생히 남아있으니 완전한 회복까지는 시간이 더 필요하겠지만요."""),


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
    emotion       = analysis.get("emotion", "")
    original_text = analysis.get("original_text", "")
    biosignal_required_val = bool(analysis.get("biosignal_required", False))

    dept = state.get("profile", {}).get("dept", "")
    rank = state.get("profile", {}).get("rank", "")
    shift_type = state.get("profile", {}).get("shift_type", "")

    biosignal_summary = state.get("biosignal_last", {}).get("biosignal_summary", "")
    if not biosignal_required_val:
        biosignal_summary = ""

    history = build_full_history(state, max_msgs=16)

    inputs = {
        "situation": situation,
        "emotion": emotion,
        "original_text": original_text,
        "dept": dept,
        "rank": rank,
        "shift_type": shift_type,
        "biosignal_required": json.dumps(biosignal_required_val).lower(),
        "biosignal_summary": biosignal_summary,
        "full_history": history,
    }
    reply_text: str = responder_chain.invoke(inputs)
    reply_text = _linebreak_by_sentence(reply_text)

    reply = reply_text.strip()
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
        {"biosignal_analyzer": "biosignal_analyzer", "analyzer": "analyzer"},
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
    
    return g.compile()


from functools import lru_cache, partial

@lru_cache(maxsize=1)
def get_graph():
    """그래프를 1회만 빌드하고 재사용"""
    return build_graph(llm)

def predict(
    user_text: str,
    dept: str = "",
    rank: str = "",
    shift_type: str = "day",
    prt: str = "",
    day: str = "",
    biosignal_consent: Optional[Literal["accepted", "declined", "unknown"]] = None,
):
    """
    프론트 입력: user_text, dept, rank, shift_type
    백엔드 입력: prt, day (반드시 지정)
    정책:
      - 동의 전(unknown): 그래프를 돌리지 않고 빈 replies로 즉시 반환 (프론트가 동의 말풍선 표출)
      - 동의(accepted): biosignal_analyzer 1회 실행 후 일반 대화 흐름
      - 거절(declined): 그래프 미실행, 공감형 오프닝 멘트 한 번 보내고 종료
    """
    if not prt:
        raise ValueError("predict(): prt가 비었습니다. 예: prt='prt1001'")
    if not day:
        raise ValueError("predict(): day가 비었습니다. 예: day='2025-07-20'")

    global SESSION_STATES
    created = False

    # 세션 준비
    if prt not in SESSION_STATES:
        state = initial_state(user_text="", dept=dept, rank=rank, shift_type=shift_type,)
        SESSION_STATES[prt] = state
        created = True
    else:
        state = SESSION_STATES[prt]

    # 동의 상태 반영 (버튼 이벤트)
    if biosignal_consent is not None:
        prev = state.get("biosignal_consent", "unknown")
        state["biosignal_consent"] = biosignal_consent
        if biosignal_consent != prev:
            state.setdefault("logs", []).append(f"[consent] {prev} -> {biosignal_consent}")

        # 거절인 경우: 그래프를 돌리지 않고, 공감형 인사 멘트만 반환
        if biosignal_consent == "declined":
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

            state["biosignal_done"] = False
            state.setdefault("logs", []).append("[consent_declined] biosignal skipped")

            SESSION_STATES[prt] = state
            return {
                "replies": [full_text],
                "created": created,
                "biosignal_first_emit": False,
                "prt": prt,
                "day": day,
                "records_loaded": 0,
                "logs": state.get("logs", []),
                "consent_state": "declined",
            }

    # 사용자 발화 추가
    if user_text and user_text.strip():
        state["messages"].append(HumanMessage(content=user_text))

    # 4) biosignal 로드 (항상) — 분석 실행 여부는 라우터/가드에서 결정
    records = []
    try:
        records = get_biosignal_records(
            prt=prt,
            day=day,
            collection_type="Automatic",
            target_hours=12,
        )
        state["biosignal"] = records if records else {}
        if not records:
            state.setdefault("logs", []).append(
                f"[biosignal_load] no records for prt={prt}, day={day}"
            )
    except Exception as e:
        state["biosignal"] = {}
        state.setdefault("logs", []).append(
            f"[biosignal_load_error] prt={prt}, day={day}, err={e}"
        )

    # 동의 가드 — 동의 전(unknown)에는 그래프 미실행
    if state.get("biosignal_consent", "unknown") == "unknown":
        # 아직 사용자 메시지가 없을 때만 조용히 반환
        has_user_msg = any(isinstance(m, HumanMessage) for m in state.get("messages", []))
        if not has_user_msg:
            SESSION_STATES[prt] = state
            return {
                "replies": [],  # 프론트가 동의 말풍선을 표시
                "created": created,
                "biosignal_first_emit": False,
                "prt": prt,
                "day": day,
                "records_loaded": len(records) if isinstance(records, list) else 0,
                "logs": state.get("logs", []) + ["[guard] waiting_for_consent"],
                "consent_state": "unknown",
            }

    # 6) 그래프 캐시 갱신 — 라우터 변경 직후엔 서버 재시작 or 1회만 호출
    try:
        get_graph.cache_clear()
    except Exception:
        pass

    # 그래프 실행 전 길이 저장 (이번 턴 델타 추출용)
    prev_len = len(state["messages"])

    # 그래프 실행
    out = get_graph().invoke(state)

    # 이번 턴에 새로 추가된 메시지만 추출
    new_msgs = out.get("messages", [])[prev_len:]
    replies = [
        m.content for m in new_msgs
        if isinstance(m, AIMessage) and getattr(m, "content", None)
    ]

    # 폴백
    if not replies and out.get("final_replies"):
        replies = [out["final_replies"][-1]]
    if not replies:
        replies = ["(응답 없음)"]

    # 첫 턴 플래그 off (다음 턴 중복 방지)
    was_first = bool(out.get("biosignal_first_emit", False))
    if was_first:
        out["biosignal_first_emit"] = False

    # 세션 저장
    SESSION_STATES[prt] = out

    return {
        "replies": replies,
        "created": created,
        "biosignal_first_emit": was_first,
        "prt": prt,
        "day": day,
        "records_loaded": len(records) if isinstance(records, list) else 0,
        "logs": out.get("logs", []),
        "consent_state": out.get("biosignal_consent", "unknown"),
        "plot_path": out.get("biosignal_last", {}).get("plot_path") if was_first else None,
    }


