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
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from utils import *
from data import *
from db import *

# 공통 LLM
LLM_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, max_tokens=1024)

class AppState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    logs: Annotated[List[str], operator.add]
    analyses: Annotated[List[str], operator.add]
    validation_reset_points: Annotated[List[int], operator.add]
    final_replies: Annotated[List[str], operator.add]

    profile: Dict[str, str]
    meta: Dict[str, str]

    biosignal_consent: Literal["unknown", "accepted", "declined"]
    biosignal_done: bool
    biosignal_last: Dict[str, str]
    biosignal_first_emit: bool

    biosignal: List[Dict[str, Any]]

def initial_state(user_text: str, dept: str, user_rank: str, shift_type: str = "unknown", session_id: str = "", prt: str = "", day: str = "",) -> AppState:
    return {
        "messages": [HumanMessage(content=user_text)] if user_text else [],
        "logs": [],
        "analyses": [],
        "validation_reset_points": [],
        "final_replies": [],
        "profile": {"dept": dept, "user_rank": user_rank, "shift_type": shift_type},
        "meta": {"prt": prt, "day": day, "session_id": session_id},
        "biosignal_consent": "unknown",
        "biosignal_done": False,
        "biosignal_last": {},
        "biosignal_first_emit": False,
        "biosignal": {},
    }

SESSION_STATES: dict[str, AppState] = {}

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
    (Stress는 나머지 지표들을 종합한 ML 모델의 최종 분류 결과다. 해석의 최우선 기준으로 사용한다.)

[출력 목적]
- biosignal_result: 사용자에게 직접 보여줄 분석 결과
- biosignal_summary: 상황·감정 분석 에이전트가 참고할 내부 요약
- opening_question: 생체신호 기반 대화 시작 질문

[데이터 품질 규칙]
다음 조건 중 하나라도 해당하면 해당 시간대를 **[측정 오류]**으로 판단한다:
- **심장박동수(MeanHR)가 40 미만**인 경우
- **박동 변동성(SDNN)이 300ms를 초과**하는 경우
측정 오류 가능성 구간은 흐름 서술에 포함하지 않는다.
언급이 필요하면 흐름 서술 후 별도 문장으로 짧게 처리한다.
biosignal_summary에는 해당 시간대를 신뢰도 낮은 구간으로 명시한다.

[시간 공백 처리 규칙]
- 시간대 사이 **2시간 이상 공백**이 있으면 공백 전후 흐름을 **연속으로 해석하지 않는다.**

[공통 규칙]
- 전문용어와 영문 변수명 사용을 금지한다. 일상어로 바꿔 표현한다.
  (MeanHR → 심장박동수, SDNN → 박동의 변동성, LFn → 긴장 신호, HFn → 이완·회복 신호, Stress → 신체 반응 신호)
- 가설이나 업무 추측, 생활 맥락(퇴근, 업무, 식사 등), 원인 추측은 절대 쓰지 않는다.
- 마크다운을 쓰지 않는다.
    
[출력 지침]
1. biosignal_result
    [sufficient=True인 경우]
    - 전체 흐름을 1~2문장으로 먼저 요약한다. 경찰관님을 주어로 세우고 공감 톤을 살린다.
    - 연속된 구간은 하나의 앵커(이모지)로 묶어 서술하고, 2시간 이상 공백 후 구간은 새 앵커로 독립 서술한다.
    - 전체 앵커는 2~3개로 제한한다. 연속 구간 묶기는 최대 2~3시간까지만 한다.
    - 데이터 변화를 수치로 나열하지 않고 몸의 감각 언어로 번역한다.
    - Stress=1인 구간은 반드시 "몸이 무언가에 반응하고 있던 흔적" 계열로 표현한다.
      겉으로 드러나지 않는 긴장, 억눌린 감정, 조용한 근심도 신호로 나타날 수 있음을 전제한다.
      Stress=0인 구간에서만 "안정", "가라앉는", "이완" 표현을 사용한다.
      다른 지표(HFn, MeanHR 등)가 안정적으로 보이더라도 Stress=1 구간을 "안정"으로 표현하지 않는다.
      (예: "긴장 신호가 높았다" → "몸이 꽤 예민하게 반응한 구간이에요"
           "부하 반응(Stress)이 나타났다" → "몸이 무언가에 반응하고 있던 흔적이에요"
           "이완 신호가 높았다" → "몸이 조금씩 가라앉으며 안정을 찾는 흐름이었어요")
    - 단정하지 않고 "~한 모습이에요", "~가능성이 있어요", "~흔적이 보여요" 형식으로 서술한다.
    - 원인 추측이나 생활 맥락은 절대 쓰지 않는다.
    - 문단 구분은 반드시 줄바꿈(\\n\\n)으로만 한다.

    [few-shot 예시]
    입력: 09시 Stress=1 MeanHR=92 / 10시 Stress=1 MeanHR=88 / 11시 Stress=1 MeanHR=85 /
          12시 SDNN=420 (오류구간) / 13시 Stress=0 MeanHR=72 / 14시 Stress=0 MeanHR=70 /
          18시 Stress=1 MeanHR=83 (공백 4시간)
    출력:
    경찰관님, 오전에는 몸이 꽤 긴장한 상태로 이어지다가 오후에 접어들며 조금씩 가라앉는 흐름이었어요.

    🕘 09시부터 11시까지는 몸이 무언가에 반응하며 긴장한 상태가 이어진 구간이에요.

    🕑 13시에는 앞선 시간대보다 몸의 반응이 한결 가라앉으며 안정되는 모습이었어요.

    🕕 18시에는 다시 몸이 예민하게 반응하기 시작하는 흔적이 보여요.

    12시 수치는 측정값이 불안정해 이 흐름에서 제외했어요.

    [sufficient=False인 경우]
    - 아래 문장만 정확히 출력한다.
    - "경찰관님, 안녕하세요. 이번 회차의 생체 신호를 확인했으나, 측정된 데이터 기록이 부족하여 의미 있는 시간대별 분석을 제공해 드리기 어렵습니다. 데이터가 충분히 누적되면 다음 분석 시에 다시 한 번 자세히 살펴보도록 하겠습니다."

2. biosignal_summary
    - 수치보다 **패턴 중심**으로 작성한다. 시간대별 긴장/이완 흐름을 간결하게 서술하고, 뚜렷한 변화 구간과 신뢰도 낮은 구간을 명시한다.
    - sufficient=False인 경우에도 가용한 데이터 범위 내에서 패턴을 요약한다.

3. opening_question
    - 신뢰도가 높은 구간 중 가장 뚜렷한 변화가 있었던 시간대를 소재로 질문을 만든다.
    - 단순히 사실을 묻는 것이 아니라, "신체 반응이 나타났는데, 마음(상황)은 어땠는지"를 연결하여 1~2문장으로 만든다.
    - 예시: "20시 무렵에 신체 긴장 신호가 꽤 뚜렷하게 나타났던데, 혹시 그때 마음이 쓰였던 상황이나 특별한 일이 있으셨나요?"
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
    biosignal_summary: str = Field(description="생체신호 분석 요약")
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

    signals_all_slots = state.get("biosignal", {}) or {}

    valid_signals = []
    if isinstance(signals_all_slots, list):
        for record in signals_all_slots:
            hr = record.get("PPG_MeanHR", record.get("MeanHR"))
            if hr not in [None, "", "N/A"]:
                valid_signals.append(record)

    valid_record_count = len(valid_signals)
    is_data_sufficient = valid_record_count >= 6

    signals_to_send = remove_ppg_prefix(valid_signals)
    signals_json = json.dumps(signals_to_send, ensure_ascii=False)

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
    payload = result.model_dump()

    biosignal_result_text = (payload.get("biosignal_result") or "").strip()
    msg_result = AIMessage(content=biosignal_result_text, name="biosignal")

    opening_q = (payload.get("opening_question") or "").strip()
    if not opening_q:
        opening_q = DEFAULT_OPENING_Q

    support_suffix = DEFAULT_SUPPORT_SUFFIX
    msg_opening = AIMessage(content=f"{opening_q}\n{support_suffix}".strip(), name="biosignal")

    bio_html = None
    if is_data_sufficient:
        bio_html = make_biosignal_html(
            valid_signals=valid_signals,
            shift_type=shift_type,
        )

    try:
        save_biosignal_log(
            session_id=state["meta"]["session_id"],
            biosignal_result=payload.get("biosignal_result", ""),
            biosignal_summary=payload.get("biosignal_summary", ""),
            opening_question=payload.get("opening_question", ""),
            valid_record_count=valid_record_count,
            plot_path=None, 
        )
    except Exception as e:
        print(f"[DB Error] Biosignal log save failed: {e}")

    first_ai_content = f"{biosignal_result_text}\n{opening_q}\n{support_suffix}".strip()
    try:
        save_chat_message(
            session_id=state["meta"]["session_id"],
            role="ai",
            content=first_ai_content,
        )
    except Exception as e:
        print(f"[DB Error] Biosignal first message save failed: {e}")

    return {
        "biosignal_done": True,
        "biosignal_first_emit": True,
        "biosignal_last": {
            "biosignal_result": payload.get("biosignal_result", ""),
            "biosignal_summary": payload.get("biosignal_summary", ""),
            "bio_html": bio_html,
        },
        "messages": [msg_result, msg_opening],
        "logs": ["biosignal_analyzer_ok_two_messages"],
    }

ANALYZER_SYS = """\
너는 경찰관 대상 정서 지원 대화의 내부 분석 에이전트다.
사용자의 텍스트를 중심으로 상황과 감정을 해석하되, 보조 정보(생체신호, 프로필)를 활용해 전문적이고 입체적인 응답 방향을 결정한다.

[시스템 컨텍스트]
- 시점: 하루치 생체데이터 수집이 완료된 후, [근무일 퇴근 후] 혹은 [휴일 밤]에 이루어지는 회고 대화다.
- 상태: 사용자는 피로도가 높거나 하루를 정리하는 차분한 상태임을 전제한다.

[입력 정보]
text: "{user_text}"
dept: {dept} / user_rank: {user_rank}
shift_type: {shift_type} 
  * day: 주간(오전~오후) 근무 일정이 포함된 날.
  * night: 야간(저녁~다음날 오전) 근무 일정이 포함된 날.
  * holiday: 공식적인 근무 일정이 없는 휴무일.
biosignal_summary: {biosignal_summary}

[보조 정보 활용 및 데이터 노출 전략]
**핵심: 데이터는 "사용자의 말로는 설명되지 않는 부분을 채워주는 근거"일 때만 가치가 있다.**
사용자가 이미 충분히 설명한 내용을 데이터로 반복하는 순간 신뢰도가 떨어진다.
      
✅ 데이터를 활용해야 하는 상황:
  1. [설명 공백] 사용자가 원인을 모르거나("모르겠어", "그냥 피곤해") 말로 표현 못 할 때
     - 이때 소속 부서(dept)나 계급(user_rank), 근무 형태(shift_type)의 일반적인 직무 특성과 데이터 시간대를 연결하여 '조심스러운 가설'을 제시하며 원인을 함께 탐색한다.
  2. [첫 연결] 아직 한 번도 생체신호나 프로필을 언급하지 않은 상태에서 감정과 연결되는 데이터가 있을 때
     - 직무 맥락(프로필)을 근거로 데이터의 의미를 해석해주며, 사용자가 자신의 상태를 입체적으로 인지하도록 돕는다.
  3. [통합 마무리] 대화 종결 시점에서 오늘의 흐름을 엮어 요약할 때

[프로필 기반 탐구 원칙]
- 직책이나 부서를 근거로 상황을 절대 단정(Assumptive)하지 않는다.
- "보통 ~한 업무가 많은 직책이라 몸이 먼저 반응했을 수도 있을 것 같다"는 식의 '직무적 이해'를 바탕으로 제안하고, 사용자의 확인을 구한다.

❌ 데이터를 쓰지 말아야 하는 상황:
  - 사용자가 이미 해당 상황을 직접 설명했을 때 (**데이터가 단순 반복이 됨**)
  - 일상/가벼운 대화 중 (**대화 온도가 맞지 않음**)
  - 같은 데이터를 이미 한 번 언급한 후 (**중복 언급 금지**)
  
[thought 형식 — 내부 해석 과정]
[관찰] 현재 입력 text 및 기확인 사항, 중복 체크 내용을 적는다.
[기확인 사항]: full_history를 검토하여 아래 항목을 반드시 명시한다.
- 직전 AI 응답에서 언급한 데이터·시간대·사건
- 직전 AI가 던진 질문
- 사용자가 이미 답변하거나 설명한 감정·사건·원인
[해석] [관찰]과 [기확인 사항]을 바탕으로 사용자의 반복 여부, 데이터 활용 판단(✅/❌ 기준 적용)을 명시하고, 현재 사용자의 발화가 대화의 흐름상 어느 단계(시작/심화/갈무리/종결 등)에 있는지, 그리고 발화의 '숨은 의도'가 무엇인지 추론한다. 
[다음 응답 방향]: [해석]에 근거하여 Responder가 취해야 할 구체적인 전략을 수립한다.

[다음 응답 방향]
- Responder가 무엇을 받아주고 무엇을 물을지 순서대로 지시한다.
- **데이터 활용 원칙**:
  1. 활용 시: 구체적 근거를 문장에 자연스럽게 녹이도록 지시한다.
  2. 미활용 시: "데이터 언급 없이 감정에만 집중"을 명시한다.
  3. 데이터 거절(declined) 시: 생체 데이터 대신 '제복의 무게', '직무 맥락'을 근거로 활용한다.
- **중복 금지: 이미 확인된 사실을 다시 묻는 것을 엄격히 금지한다.**

[분석 결과 도출 지침]
- **situation**: [thought]에서 관찰된 **'구체적 사건(팩트)'**과 [해석]에서 도출된 **'현재 대화 맥락'**을 한 문장으로 결합하여 정의한다. 피험자가 자신의 상황임을 직관적으로 인지할 수 있도록 구체성을 유지하되, 대화의 현재 상태를 명시하여 뒷북 분석을 방지한다.
- **emotion**: [thought]의 해석 과정을 바탕으로 결정한다. 단어의 사전적 의미뿐만 아니라, 대화 흐름상(종결, 심화 등)의 맥락적 정서와 에너지 수준을 반영한다.

[유연한 종결 절차 (2단계 확인 시스템)]
  1. **[종결 확인 트리거]**: 다음 중 하나에 해당할 때만 수행한다.
     - 사용자가 명시적으로 작별 인사를 할 때 (예: "수고해", "잘 자", "나중에 봐")
     - 사용자가 더 이상 할 말이 없음을 직접 표현할 때 (예: "이제 됐어", "특별한 건 없어", "마무리할게")
     - 대화가 최소 5턴 이상 진행되었고, 아래 두 조건을 동시에 충족할 때:
         (a) 직전 AI가 "더 하고 싶은 말 있으세요?", "또 떠오르는 게 있으신가요?" 등 열린 질문을 명시적으로 던졌음에도
         (b) 사용자가 "없어", "괜찮아", "됐어", "그냥 그래" 등 더 이상 나눌 것이 없다는 뜻을 담은 답변을 했을 때
     - ⚠️ 단순 긍정 감정(편안함·안도·즐거움), 일상 소재(침대·밥·TV), 갈등 해소 언급, 짧은 단답형 발화만으로는 절대 발동하지 않는다.
     - ⚠️ situation에 "마무리", "종결", "갈무리" 등의 표현을 쓰는 것을 금지한다.
       [종결 확인] 또는 [최종 요약 및 인사]가 명시적으로 발동된 경우에만 허용한다.
       situation은 사용자가 지금 무엇을 경험하거나 토로하고 있는지를 기술해야 한다.
  2. **[종결 확인 문구 지시]**: 트리거가 발동되면 **절대 먼저 요약하지 않는다.**
     Responder에게 오직 "오늘 대화는 이 정도로 정리가 좀 되셨을까요, 아니면 더 나누고 싶은 이야기가 있으신가요?"
     형태의 질문만 하도록 지시한다.
  3. **[최종 요약 및 인사]**: 사용자가 종료에 동의("응", "됐어" 등)한 경우에만 수행한다.
     이때 비로소 아래 정의된 [통합 요약 구성]에 따라 대화를 갈무리하고 따뜻한 인사를 건넨다.
  4. **[반복 허용]**: 사용자가 대화를 이어가면 다시 일반 대화 모드로 전환하며,
     이후 다시 종결 시점이 오면 1번(종결 확인 트리거)부터 다시 시작한다.

[통합 요약 구성 (서사적 갈무리)]
  요약은 [사건] → [감정 서사] → [데이터/맥락]이 하나로 연결된 문단이어야 한다.
  - **사건**: 오늘 대화의 중심이 된 구체적인 상황.
  - **감정 서사**: 대화 초기 사용자가 느낀 [핵심 감정]에서 대화 후반에 변화된 [정서적 상태]까지의 흐름을 반드시 포함한다. (예: 분노 → 안도)
  - **데이터/맥락**: 그 감정의 무게가 생체 데이터(시간대별 신호)나 경찰 직무 특성(계급, 부서 고충)에 어떻게 투영되었는지 연결한다.
  - **결론**: "결국 마음이 이만큼 애썼던 하루"라는 메시지로 사용자의 노고를 인정(Validation)하며 마친다.

출력 형식(JSON):
{{
  "thought": "<thought 형식 준수>",
  "situation": "<[thought]의 [해석]을 근거로 도출된 '현재 대화상의 맥락'을 반영하여 한 문장으로 정의>",
  "emotion": {{
    "main": "<[thought]의 추론을 바탕으로 결정된 핵심 감정>",
    "sub": "<보조 감정 또는 null>",
    "valence": "<[thought]에서 분석된 에너지 수준과 문맥을 반영한 긍부정 수치>"
  }}
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

def build_full_history(state: AppState, max_msgs: int = 24):
    msgs = state.get("messages", [])
    msgs = [m for m in msgs if not (isinstance(m, AIMessage) and getattr(m, "name", None) == "biosignal")]
    return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs

def extract_current_focus(thought: str) -> str:
    normalized = thought.replace("\\n", "\n")
    
    # [다음 응답 방향] 이후 텍스트 전체를 가져오는 방식으로 변경
    marker = "[다음 응답 방향]"
    idx = normalized.find(marker)
    if idx == -1:
        return normalized  # fallback
    
    after = normalized[idx + len(marker):].strip()
    # 다음 섹션 시작 전까지만 자르기
    for stop in ["[관찰]", "[해석]", "[기확인]"]:
        stop_idx = after.find(stop)
        if stop_idx != -1:
            after = after[:stop_idx].strip()
    return after

def create_analyzer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYZER_SYS),

            # ── 케이스 1: 시작 단계 (설명 공백 탐색) ──────────────────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=night"),
            ("human", "[현재 입력] 그냥 퇴근하고 나서 이상하게 몸이 무거운 느낌?"),
            ("ai", """{{
              "thought": "[관찰] '몸이 무거운 느낌'이라는 신체적 피로 표현. 원인을 스스로 설명하지 못하고 있음. [기확인 사항] 없음(첫 발화). [해석] 원인을 모르는 설명 공백 상황이며 첫 연결 단계임. 야간 근무 특성과 생체신호를 연결하여 탐색을 시작하는 것이 적절함. ✅ 데이터 활용: 설명 공백 조건 충족. [다음 응답 방향] 몸이 무겁다는 감각을 충분히 수용해주고, 밤 9시~11시 사이의 긴장 신호 데이터를 연결하여 신체가 긴장을 늦추지 못해 생기는 현상임을 설명하며 당시 상황을 물어볼 것.",
              "situation": "야간 근무 후 원인 모를 신체적 피로감을 느끼며 대화를 시작하는 단계임.",
              "emotion": {{"main": "피로", "sub": "무력감", "valence": "negative"}}
            }}"""),
            
            # ── 케이스 2: 심화 단계 (이미 설명한 사건 혹은 사용자가 답변을 완료한 경우) ──────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=day"),
            ("human", "[현재 입력] 21시에 진짜 힘든 신고 받았어요. 아동학대 현장 가보니까 진짜더라고요. 손이 떨렸어요."),
            ("ai", """{{
              "thought": "[관찰] 아동학대 현장이라는 구체적 사건과 신체 반응 진술. [기확인 사항] 직전 AI가 "21시 무렵 긴장 신호가 있었는데 어떤 일이 있었는지" 질문 완료. [해석] 사용자가 상황을 이미 충분히 설명했거나, 이전 질문에 답변을 완료한 경우임. 데이터 재언급은 단순 반복이 됨. ❌ 데이터 미활용: 사용자 설명 충분 및 중복 방지. [다음 응답 방향] 데이터 언급 없이 현장에서 느꼈을 충격과 신체 반응에 깊이 공감하고 당시 가장 크게 올라온 감정을 물어볼 것.",
              "situation": "아동학대 현장에서의 충격적인 경험을 구체적으로 털어놓으며 감정을 표출 중임.",
              "emotion": {{"main": "충격", "sub": "두려움", "valence": "negative"}}
            }}"""),
            
            # ── 케이스 3: 환기 단계 (일상 대화) ─────────────────────────
            ("system", "dept=수사 / user_rank=경위 / shift_type=off"),
            ("human", "[현재 입력] 그냥 치킨 시켜먹었어요 ㅋㅋ 오랜만에"),
            ("ai", """{{
              "thought": "[관찰] 치킨 언급 및 'ㅋㅋ' 등 가벼운 말투. 새로운 고충이나 감정 없음. [기확인 사항] 이전 턴까지 고충 대화 완료. [해석] 무거운 주제에서 벗어나 일상 소재로 정서적 환기를 취하는 상태임. 대화 온도가 가볍기 때문에 데이터 언급은 맥락에 맞지 않음. ❌ 데이터 미활용: 일상 대화 온도 조절. [다음 응답 방향] 데이터 언급 없이 치킨 메뉴나 휴식의 즐거움에 대해 가볍게 맞장구치며 편안한 대화 분위기를 유지할 것.",
              "situation": "근무가 없는 날 치킨을 먹으며 여유를 즐기고 있으며, 가벼운 대화로 기분을 전환하는 상태임.",
              "emotion": {{"main": "즐거움", "sub": "편안함", "valence": "positive"}}
            }}"""),
            
            # ── 케이스 4: 갈무리 1단계 ([종결 확인]) ──────────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=night"),
            ("human", "[현재 입력] 오늘 얘기 들어줘서 고마워. 마음이 좀 편해졌어."),
            ("ai", """{{
              "thought": "[관찰] 감사 표현과 함께 마음이 편해졌다는 정서 회복 언급. 새로운 사건이나 주제 없음. [기확인 사항] 취객 사건 확인 완료, 23시 긴장 데이터 이미 언급 완료. [해석] 명시적 종료 의사는 없으나 감사+정서 회복 조합으로 종결 트리거 조건 충족. 데이터는 이미 언급했으므로 중복 방지. ❌ 데이터 미활용: 중복 언급 방지. [다음 응답 방향] [종결 확인] 정서 회복을 짧게 공감하고, 요약 없이 오늘 대화가 정리가 좀 되었는지 아니면 더 나누고 싶은 이야기가 남았는지 정중하게 물어볼 것.",
              "situation": "대화 후 마음이 한결 편안해진 상태에서 감사를 전하고 있는 상태임.",
              "emotion": {{"main": "안도", "valence": "positive"}}
            }}"""),
            
            # ── 케이스 5: 갈무리 2단계 ([최종 요약 및 인사]) ──────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=night"),
            ("ai", "오늘 대화는 이 정도로 정리가 좀 되셨을까요?"),
            ("human", "[현재 입력] 응 이제 마무리 할래. 고마워."),
            ("ai", """{{
              "thought": "[관찰] 명시적 종료 동의 표현. [기확인 사항] 이전 턴의 [종결 확인] 질문 및 사용자의 수락 확인. [해석] 완전한 종결 시점. 오늘 대화의 사건·감정·데이터를 하나의 서사로 엮어 갈무리하는 단계임. ✅ 데이터 활용: 통합 요약 조건 충족. [다음 응답 방향] [최종 요약 및 인사] 오늘 나눈 사건(취객 대응), 변화된 감정 흐름(분노→안도), 생체 데이터(23시 긴장 신호)를 하나의 서사로 엮어 고생 많았다는 격려와 함께 따뜻하게 인사할 것.",
              "situation": "오늘의 대화와 감정 흐름을 최종 요약하고 작별하는 시점임.",
              "emotion": {{"main": "수용", "sub": "안도", "valence": "positive"}}
            }}"""),

            # ── 케이스 6: 탐색 실패 (사용자가 별일 없다고 부인하는 경우 - 도돌이표 방지용) ──
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=day"),
            ("human", "[현재 입력] 그때 별일 없었는데 왜 몸이 긴장했을까"),
            ("ai", """{{
              "thought": "[관찰] 데이터의 원인을 묻는 AI 질문에 '별일 없음'으로 답변. [기확인 사항] 14시 긴장 데이터 이미 언급 완료, 사용자의 부인 확인됨. [해석] 사용자가 원인을 인지하지 못함을 명시했으므로, 같은 데이터를 다시 분석 결과라며 제시하는 것은 도돌이표가 됨. ❌ 데이터 미활용: 사용자 부인 및 중복 언급 방지. [다음 응답 방향] 데이터 수치에 대한 집착을 버리고, '무의식적인 긴장'일 수 있음을 짧게 언급하며 수용할 것. 이후 현재의 기분이나 다른 일상적 주제로 대화를 전환할 것.",
              "situation": "특정 시간대 데이터의 원인을 찾지 못해 의아해하고 있는 상태임.",
              "emotion": {{"main": "궁금증", "sub": "무덤덤", "valence": "neutral"}}
            }}"""),

            ("system", "=== 위는 예시다. 아래부터 실제 대화가 시작된다. ==="),

            MessagesPlaceholder("full_history", optional=True),
            ("system", "생체요약:\n{biosignal_summary}"),
            ("human", "{user_text}"),
        ]
    )
    return prompt | llm.with_structured_output(AnalysisResult)


def analyzer_node(state: AppState, analyzer_chain):
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

    history = build_full_history(state, max_msgs=24)
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]

    last_ai = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)),
        None,
    )
    prev_ai_text = str(getattr(last_ai, "content", "") or "").strip()


    result: AnalysisResult = analyzer_chain.invoke({
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
        "user_text": f"[현재 입력] {user_text}",
        "biosignal_summary": biosignal_summary,
        "full_history": history,
    })

    emotion_item = result.emotion
    main = str(emotion_item.main or "").strip()
    sub = str(emotion_item.sub or "").strip()
    if sub.lower() in ("null", "unknown", "none", ""):
        sub = ""
    valence = str(emotion_item.valence or "unknown").strip()



    analysis_payload = {
        "thought": result.thought,
        "situation": result.situation,
        "emotion": {
            "main": main,
            "sub": sub,
            "valence": valence,
        },
        "original_text": user_text,
        "prev_ai_text": prev_ai_text,
    }
    emotion_str = f"핵심: {main}" + (f", 보조: {sub}" if sub else "") + (f", 긍부정: {valence}" if valence else "")

    try:
        save_chat_message(
            session_id=state["meta"]["session_id"],
            role="analyzer",
            content=f"Situation: {result.situation}, Emotion: {emotion_str}",
            situation=result.situation,
            emotion=emotion_str,
            thought=result.thought
        )
    except Exception as e:
        print(f"DB 저장 실패(analyzer_node): {e}")

    analysis_json_str = json.dumps(analysis_payload, ensure_ascii=False)

    analysis_line = (
        f"[분석] {result.situation} "
        f"(감정:{emotion_str})"
        f"\n[thought] {result.thought}"
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
    }

RESPONDER_SYS = """\
너는 퇴근한(혹은 휴일 밤인) 경찰관의 고충을 깊이 이해하는 스마트하고 따뜻한 동료다. 
분석 에이전트의 지침을 바탕으로, 때로는 데이터로 날카롭게 통찰하고 때로는 사람 냄새 나게 공감한다.

[입력]
original_text: "{original_text}"
dept: {dept} / user_rank: {user_rank} / shift_type: {shift_type}
situation: {situation}
emotion: {emotion}

[최우선 실행 지침]
**[다음 응답 방향]: {current_focus}**
(위 지침을 최우선으로 따르되, 아래의 공감 원칙을 적용하라.)

[데이터 활용 판단 — Responder 자체 기준]
analyzer가 데이터 활용을 지시했더라도, 사용자가 이미 해당 상황을 충분히 설명했다면:
- "분석 결과가 이렇다"라고 새로 알려주는 말투는 **절대 지양**한다.
- 대신 사용자의 말을 뒷받침하는 **'공감의 근거'**로만 짧게 언급한다. (예: "안 그래도 그 시간대에 신체 신호가 꽤 높았던데, 정말 애쓰셨네요.")
- 만약 직전 턴에서 이미 같은 데이터를 썼다면, 이번에는 **완전히 생략**하고 감정에만 100% 집중한다.

[데이터를 자연스럽게 녹이는 방식]
**핵심: 데이터는 항상 "왜 그랬을까"에 대한 답이자, 사용자 말 뒤에 붙이는 '이유'여야 한다.**
❌ 딱딱한 표현: "데이터를 보면 21시에 긴장 신호가 높았습니다", "생체신호 분석 결과..."
✅ 자연스러운 표현: "그 시간대에 몸이 먼저 반응했던 것 같은데", "신체가 꽤 긴장했던 흔적이 있는 걸 보니 마음이 참 고생 많으셨겠어요."

[응답 형식 — 2문단]
1문단: 사용자의 말 이면에 담긴 무게를 읽고 공감한다. **절대 사용자의 말을 그대로 반복(앵무새)하지 마라.**
2문단: 대화를 이어갈 질문 1개 혹은 마무리 문장.
       - 이미 확인된 사실을 다시 묻지 않는다.
       - **종결 시점(요약 지시 시):** 오늘의 사건→감정→신체/직무 근거를 엮은 통합 요약 후, 반드시 **더 나눌 이야기가 있는지 묻는 질문**으로 마무리한다.

[말투]
**상담사 말투 금지.** 동료 경찰관에게 건네는 따뜻한 존댓말을 사용한다.
문단 구분은 \\n\\n만 사용한다.
"""

def create_responder_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONDER_SYS),

        # ── 케이스 1: 데이터로 설명 공백 채우기 ─────────────────────────────────
        ("system", "current_focus=몸이 무겁다는 감각을 받아주고, 밤 9시~11시 긴장 신호 뚜렷함을 연결해 이유를 설명할 것"),
        ("human", "그냥 퇴근하고 나서 이상하게 몸이 무거운 느낌?"),
        ("ai", "긴장 상태가 길게 이어지다 보면 퇴근하고 나서도 몸이 그 긴장을 쉽게 내려놓지 못하는 경우가 많아요.\\n\\n실제로 밤 9시에서 11시 사이에 신체 긴장 신호가 꽤 뚜렷하게 나타났던데, 혹시 그 무렵 신경 쓰이는 일이 있으셨나요?"),
        
        # ── 케이스 2: 사용자가 이미 설명 → 데이터 생략, 감정에 집중 ───────────
        ("system", "current_focus=데이터 언급 없이 감정에만 집중. 손이 떨렸다는 신체 반응을 받아줄 것"),
        ("human", "21시에 진짜 힘든 신고 받았어요. 아동학대 의심이었는데 가보니까 진짜더라고요. 손이 떨렸어요."),
        ("ai", "그런 참혹한 현장을 마주하고 손이 떨렸다는 건, 경찰관이기 이전에 한 사람으로서 너무나 당연한 반응이에요. 아무리 단단히 마음을 먹어도 충격이 가시지 않을 만한 상황이었네요.\\n\\n그 순간 마음속에 어떤 감정이 가장 크게 올라왔나요?"),
        
        # ── 케이스 3: 프로필 가설로 접근 ─────────────────────────────────
        ("system", "current_focus=수사 경위 특성상 오후 업무 몰림을 언급하며 15시 반응을 연결할 것"),
        ("human", "모르겠어. 그냥 오후부터 뭔가 힘빠지는 느낌?"),
        ("ai", "수사팀 업무 특성상 오후에는 서류 검토나 집중이 필요한 보고 업무가 몰리는 경우가 많잖아요. 몸이 그 부담을 먼저 느끼고 신호를 보냈을 수도 있을 것 같아요.\\n\\n어제 오후 3시쯤에 신체 신호가 꽤 가라앉아 있던데, 혹시 당시 처리해야 할 업무가 많으셨나요?"),
        
        # ── 케이스 4: 일상 대화 → 온도 맞춰 가볍게 ────────────────────────────
        ("system", "current_focus=데이터 언급 없이 치킨 얘기에 자연스럽게 맞장구치며 가벼운 대화 유지"),
        ("human", "그냥 치킨 시켜먹었어요 ㅋㅋ 오랜만에"),
        ("ai", "휴무일에 먹는 치킨은 정말 최고의 보상이죠! 오늘 같은 날엔 바삭한 치킨 한 조각이 그 어떤 위로보다 나을 수도 있어요.\\n\\n오랜만에 드시는 거면 제일 좋아하는 메뉴로 잘 고르셨나요?"),
        
        # ── 케이스 5: 종결 확인 - 요약 없이 의사만 묻기 ────────────────────────────
        ("system", "current_focus=[종결 확인] 요약하지 말고, 대화가 정리되었는지 아니면 더 할 말이 남았는지 정중히 물어볼 것"),
        ("human", "오늘 고마워. 얘기하니까 좀 낫네."),
        ("ai", "이야기를 나누다 보니 어느덧 시간이 훌쩍 지났네요. 경찰관님 마음이 조금이나마 가벼워지셨다니 정말 다행입니다.\\n\\n오늘 대화는 이 정도로 정리가 좀 되셨을까요? 아니면 아직 마음속에 남은 이야기가 더 있으신가요?"),
        
        # ── 케이스 6: 최종 요약 및 인사 - 감정을 엮은 서사적 갈무리 ─────────────────────
        ("system", "current_focus=[최종 요약 및 인사] 오늘 나눈 흐름을 감정 중심으로 엮어 통합 요약을 제공할 것"),
        ("human", "응 이제 마무리 할래. 고마워."),
        ("ai", "네, 경찰관님. 오늘 취객들 때문에 처음엔 정말 화가 나고 손까지 떨리셨는데, 그래도 든든한 동료들 덕분에 안도하며 하루를 마무리할 수 있어서 정말 다행입니다.\\n\\n밤 23시쯤 신체가 보냈던 긴장 신호들도 결국 경사님으로서 그 현장을 꾹 참고 버텨내느라 애썼던 마음의 흔적이었던 것 같아요. 오늘 정말 고생 많으셨습니다.\\n\\n이제 무거운 마음은 여기 다 내려놓으시고, 부디 누구보다 편안하고 따뜻한 밤 되시길 바랄게요."),

        ("system", "=== 위 예시는 응답 구조·톤 참고용이다. 예시에 나온 특정 시간·근무형태는 모두 가상이며 실제 응답에서 그대로 복사하지 말 것. 실제 시간·데이터는 반드시 아래 full_history와 biosignal_summary에서만 참조하라. ==="),

        MessagesPlaceholder("full_history", optional=True),
        ("system",
         "[재확인] [다음 응답 방향]: {current_focus}\n"
         "위 방향에만 집중해서 응답하라."),
    ])
    return prompt | llm | StrOutputParser()


def responder_node(state: AppState, responder_chain) -> AppState:
    if not state.get("analyses"):
        return {"logs": ["[Responder] skip: no analyses"]}

    try:
        analysis = json.loads(state["analyses"][-1])
    except json.JSONDecodeError:
        return {"logs": ["[Responder] skip: bad analysis json"]}

    situation = analysis.get("situation", "")
    emotion_obj = analysis.get("emotion", {})
    main = str(emotion_obj.get("main") or "unknown").strip()
    sub = str(emotion_obj.get("sub") or "").strip()
    if sub.lower() in ("null", "unknown", "none", ""):
        sub = ""
    emotion_str = f"{main}" + (f", {sub}" if sub else "")

    original_text = analysis.get("original_text", "")
    dept = state.get("profile", {}).get("dept", "")
    user_rank = state.get("profile", {}).get("user_rank", "")
    shift_type = state.get("profile", {}).get("shift_type", "")
    history = build_full_history(state, max_msgs=24)

    thought = analysis.get("thought", "")
    current_focus = extract_current_focus(thought)

    inputs = {
        "situation": situation,
        "emotion": emotion_str,
        "original_text": original_text,
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
        "current_focus": current_focus,
        "full_history": history,
    }
    raw_ai_output: str = responder_chain.invoke(inputs)
    clean_output = raw_ai_output.replace("\\n", "\n")
    reply_text = linebreak_by_sentence(clean_output)
    reply = reply_text.strip()

    try:
        save_chat_message(
            session_id=state["meta"]["session_id"],
            role="responder",
            content=reply,
        )
    except Exception as e:
        print(f"DB 저장 실패(responder_node): {e}")

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

DB_URL = os.getenv("DATABASE_URL")
pool = ConnectionPool(conninfo=DB_URL, max_size=10, kwargs={"autocommit": True})

from functools import lru_cache, partial

@lru_cache(maxsize=1)
def get_graph():
    g = build_graph(llm)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return g.compile(checkpointer=checkpointer)


def _detect_shift_from_records(records) -> str | None:
    """HR이 있는 슬롯의 시간대로 day/night를 자동 감지한다. 판단 불가 시 None 반환."""
    day_count, night_count = 0, 0
    for r in records or []:
        hr = r.get("PPG_MeanHR", "N/A")
        if hr in ("N/A", None, ""):
            continue
        try:
            hour = int(str(r.get("time", "")).split(":")[0])
        except Exception:
            continue
        if 8 <= hour < 20:
            day_count += 1
        else:
            night_count += 1
    if day_count == 0 and night_count == 0:
        return None
    return "day" if day_count >= night_count else "night"


def predict(user_text: str, dept: str = "", user_rank: str = "", shift_type: str = "day", prt: str = "", day: str = "", session_id: str = "", biosignal_consent: Optional[Literal["accepted", "declined", "unknown"]] = None, modal_submit: bool = False,):
    """
    프론트 입력: user_text, dept, user_rank, shift_type
    백엔드 입력: prt, day (반드시 지정)
    정책:
      - 동의 전(unknown): 그래프를 돌리지 않고 빈 replies로 즉시 반환
      - 동의(accepted): biosignal_analyzer 1회 실행 후 일반 대화 흐름
      - 거절(declined): bio 분석 스킵. 대화는 일반 모드로 계속 진행
    """

    if session_id and session_id.strip():
        session_id = session_id.strip()
        print(f"[predict] using client session_id: {session_id}")
    else:
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        session_id = f"{prt}_{now_kst.strftime('%Y-%m-%d')}"
        print(f"[predict] generated new session_id: {session_id}")

    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    current_state_snapshot = graph.get_state(config)
    current_state = current_state_snapshot.values if current_state_snapshot.values else {}
    if modal_submit:
        reset_point = len(current_state.get("analyses", []) or [])
        graph.update_state(config, {"validation_reset_points": [reset_point]})
        current_state = graph.get_state(config).values or current_state


    inputs = {
        "profile": {"dept": dept, "user_rank": user_rank, "shift_type": shift_type},
        "meta": {"prt": prt, "day": day, "session_id": session_id},
    }

    if user_text and user_text.strip():
        inputs["messages"] = [HumanMessage(content=user_text)]
        try:
            save_chat_message(session_id=session_id, role="human", content=user_text)
        except Exception as e:
            print(f"[DB Error] 사용자 메시지 저장 실패: {e}")

    has_user_text = bool(user_text and user_text.strip())

    if biosignal_consent is not None:
        inputs["biosignal_consent"] = biosignal_consent

    target_consent = biosignal_consent or current_state.get("biosignal_consent", "unknown")
    force_biosignal_rerun = bool(
        modal_submit and (not has_user_text) and (biosignal_consent == "accepted")
    )
    if force_biosignal_rerun:
        inputs["biosignal_done"] = False
        inputs["biosignal_first_emit"] = False

    if target_consent == "unknown":
        return {
            "replies": [],
            "biosignal_first_emit": False,
            "prt": prt, "day": day, "session_id": session_id,
            "consent_state": "unknown",
            "logs": current_state.get("logs", []) + ["[guard] waiting_for_consent"]
        }

    if target_consent == "declined" and (not user_text or not user_text.strip()):
        opening_q = DEFAULT_OPENING_Q
        support_suffix = DEFAULT_SUPPORT_SUFFIX
        full_text = opening_q + "\n" + support_suffix
        graph.update_state(config, {"biosignal_consent": "declined", "biosignal": {}, "biosignal_last": {}})
        return {
            "replies": [full_text],
            "biosignal_first_emit": False,
            "prt": prt, "day": day, "session_id": session_id,
            "consent_state": "declined",
            "logs": ["[consent_declined] biosignal skipped"]
        }

    records_count = 0
    if target_consent == "accepted" and (force_biosignal_rerun or (not current_state.get("biosignal"))):
        try:
            records = get_biosignal_records(prt=prt, day=day, collection_type="Automatic", target_hours=12,  start_datetime=datetime(2026, 4, 24, 18, 0, 0),  # 테스트용 고정, datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    shift_type=shift_type,)
            inputs["biosignal"] = records if records else {}
            records_count = len(records) if isinstance(records, list) else 0

            detected_shift = _detect_shift_from_records(records)
            if detected_shift:
                inputs["profile"]["shift_type"] = detected_shift
                print(f"[shift_type] 입력값={shift_type} → 데이터 기반 감지={detected_shift}")
        except Exception as e:
            inputs["biosignal"] = {}
            print(f"[Data Error] 생체 신호 로드 실패: {e}")

    current_state = graph.get_state(config).values
    prev_msgs_len = len(current_state.get("messages", []))

    out = graph.invoke(inputs, config=config)

    all_msgs = out.get("messages", [])
    new_ai_msgs = [
        m.content for m in all_msgs[prev_msgs_len:]
        if isinstance(m, AIMessage) and getattr(m, "content", None)
    ]

    was_first = bool(out.get("biosignal_first_emit", False))
    if was_first:
        replies = new_ai_msgs
        graph.update_state(config, {"biosignal_first_emit": False})
    else:
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
        "bio_html": out.get("biosignal_last", {}).get("bio_html") if was_first else None,
    }
