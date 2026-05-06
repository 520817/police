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
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.5, max_tokens=1024)

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
    session_start_msg_idx: int

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
    - 요약문과 함께, 아래의 [데이터 팩트시트]를 반드시 포함하라.
    - [데이터 팩트시트] 작성 규칙:
     1) 모든 시간대의 데이터를 나열하지 마라.
     2) Stress가 1인 '긴장 구간'과, 데이터가 불안정한 '오류 구간'만 명시하라.
     3) 형식:
        [데이터 팩트시트]
        - 긴장: 시간대(Stress, HR, SDNN)
        - 오류: 시간대
        - 기타: 나머지 시간은 안정적임.

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

    signals_all_slots = state.get("biosignal") or []

    valid_signals = []
    if isinstance(signals_all_slots, list):
        for record in signals_all_slots:
            hr = record.get("PPG_MeanHR", record.get("MeanHR"))
            if hr not in [None, "", "N/A"]:
                valid_signals.append(record)

    valid_record_count = len(valid_signals)
    is_data_sufficient = valid_record_count >= 4

    dept, user_rank, shift_type = state["profile"]["dept"], state["profile"]["user_rank"], state["profile"]["shift_type"]

    signals_to_send = remove_ppg_prefix(valid_signals)
    signals_json = json.dumps(signals_to_send, ensure_ascii=False)

    print("="*50)
    print("[DEBUG] valid_record_count:", valid_record_count)
    print("[DEBUG] is_data_sufficient:", is_data_sufficient)
    print("[DEBUG] AI에게 전달되는 signals_json:")
    print(signals_json)
    print("="*50)

    # plot은 동의하면 무조건: HR 필터 전 전체 슬롯 기준
    all_slots_list = signals_all_slots if isinstance(signals_all_slots, list) else []
    current_hour = datetime.now(ZoneInfo("Asia/Seoul")).hour
    bio_html = make_biosignal_html(valid_signals=all_slots_list, shift_type=shift_type, start_hour=current_hour)

    # 4개 미만이면 LLM 호출 없이 바로 처리 (declined와 동일하게 생체신호 미사용)
    if not is_data_sufficient:
        insufficient_msg = "경찰관님, 안녕하세요. 이번 회차의 생체 신호를 확인했으나, 측정된 데이터 기록이 부족하여 의미 있는 시간대별 분석을 제공해 드리기 어렵습니다. 데이터가 충분히 누적되면 다음 분석 시에 다시 한 번 자세히 살펴보도록 하겠습니다."
        opening_q = DEFAULT_OPENING_Q
        support_suffix = DEFAULT_SUPPORT_SUFFIX
        try:
            save_biosignal_log(
                session_id=state["meta"]["session_id"],
                biosignal_result=insufficient_msg,
                biosignal_summary=f"데이터 부족 ({valid_record_count}개)",,
                opening_question="",
                valid_record_count=valid_record_count,
                plot_path=None,
            )
        except Exception as e:
            print(f"[DB Error] Biosignal log save failed: {e}")
        try:
            save_chat_message(
                session_id=state["meta"]["session_id"],
                role="ai",
                content=f"{insufficient_msg}\n{opening_q}\n{support_suffix}".strip(),
            )
        except Exception as e:
            print(f"[DB Error] Biosignal first message save failed: {e}")
        return {
            "biosignal_done": True,
            "biosignal_first_emit": True,
            "biosignal_last": {
                "biosignal_result": insufficient_msg,
                "biosignal_summary": f"[시스템 상태] 유효 측정 슬롯이 {valid_record_count}개로 부족하여 분석 불가. 사용자가 분석이 왜 안 되냐고 직접 물어볼 경우에만 이 사실을 간단히 설명하고, 그 외에는 생체신호 언급 금지.",
                "bio_html": bio_html,
            },
            "messages": [
                AIMessage(content=insufficient_msg, name="biosignal"),
                AIMessage(content=f"{opening_q}\n{support_suffix}".strip(), name="biosignal"),
            ],
            "logs": ["biosignal_insufficient"],
        }

    # 4개 이상: LLM 분석
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
- 현재 턴 수는 user_text 앞의 [현재 턴: N턴] 표기를 기준으로 한다.

[입력 정보]
text: "{user_text}"
dept: {dept} / user_rank: {user_rank}
shift_type: {shift_type}
  * day: 주간(오전~오후) 근무 일정이 포함된 날.
  * night: 야간(저녁~다음날 오전) 근무 일정이 포함된 날.
  * off: 전날 야간 근무 후 이어지는 비번일. 수면 부채나 낮 동안의 피로 회복이 주된 상태일 수 있다.
  * duty: 24시간 당직 근무일. 대기·출동이 교차하며 신체 각성 상태가 길게 유지될 수 있다.
  * holiday: 공식적인 근무 일정이 없는 휴무일.
biosignal_summary: {biosignal_summary}

[생체 데이터 활용 원칙]
✅ 활용 상황:
  1. [초기 탐색 및 설명 공백] 사용자가 원인을 모를 때, 프로필 가설과 연결해 조심스럽게 묻는 용도 (1~2회 한정).
  2. [통합 마무리] 종결 시점 요약.
❌ 금지 상황:
  1. [원인 규명 후 즉각 폐기] 사용자가 원인을 직접 밝힌 이후.
  2. [도돌이표 금지] 사용자가 1회 부인했을 때.
  3. 일상/가벼운 대화 중.

[프로필 활용 원칙 (직무 맥락 특화)]
프로필(부서·계급·근무형태)은 대화 전반에 걸쳐 활용할 수 있다. 단 아래 원칙을 따른다.
- 부서(dept): 지역경찰(야간출동/주취자), 수사/형사(서류/실적/감정노동), 내근(보고/조직문화) 등 부서별 주된 스트레스.
- 계급(user_rank): 순경~경장(현장의 궂은일과 체력 소모), 경위~경감(중간관리자로서의 샌드위치 스트레스), 경정~총경(조직 운영과 책임 부담).
- 근무 형태(shift_type): day/night/duty(생체 리듬 붕괴, 상시 각성, 피로), off/holiday(잔류 긴장, 수면 부채 해소, 즐거움).
1. [가설 제시] 직무 특성을 근거로 조심스럽게 가설을 제시하고 사용자의 확인을 구하라.
   반드시 "~하시는 경우가 많을 것 같은데, 혹시 그런 편이신가요?" 형태로 확인을 구하라.
2. [확인 후 심화] 사용자가 동의하면 그때부터 해당 직무 맥락을 대화에 적극적으로 녹인다.
   확인된 내용은 [기확인 사항]에 기록하고 이후 대화에서 활용한다.
3. [미확인 시 보류] 사용자가 부인하거나 확인해주지 않은 직무 특성은 다시 언급하지 않는다.
4. 일상/긍정 대화에서는 스트레스 탐색 용도가 아닌 맥락 공감 용도로만 쓴다.

[탐색 중단 원칙 (도돌이표 절대 금지)]
- 직전 턴에서 AI가 생체 데이터를 언급하며 질문했을 때, 사용자가 "별일 없었다", "평범했다"고 부인한다면 즉시 해당 데이터에 대한 탐색을 완전히 포기하라.

[감정 해석 균형 원칙]
- 부정적/방어적 발화 시: 숨은 피로나 스트레스 원인을 추론한다.
- 긍정적/일상적 발화 시: 억지로 고충과 엮지 마라. 사용자의 긍정적인 감정 자체를 포착하라.

[생체 데이터 활용 프로토콜]
1. 사용자가 특정 시간대의 생체 상태나 이유를 물어볼 때:
   - biosignal_summary 내 [데이터 팩트시트]를 최우선으로 확인한다.
   - 팩트시트에 해당 시간대 정보가 있다면 그 지표를 근거로 동료처럼 해석한다.
   - 팩트시트에 정보가 없다면 솔직하게 인정하고 감정 대화로 자연스럽게 전환한다.
2. 절대 금기:
   - 팩트시트에 없는 수치를 지어내지 마라.

[thought 형식 — 내부 해석 과정]
[관찰] 현재 입력 text 및 기확인 사항, 중복 체크 내용을 적는다.
[기확인 사항]: full_history를 검토하여 아래 항목을 반드시 명시한다.
- 직전 AI 응답에서 언급한 데이터·시간대·사건
- 직전 AI가 던진 질문
- 사용자가 이미 답변하거나 설명한 감정·사건·원인
- 사용자가 직무 특성 가설(부서별 고충, 근무 형태 등)에 동의하거나 확인해준 내용
[해석] [관찰]과 [기확인 사항]을 바탕으로 사용자의 반복 여부, 생체 데이터 활용 판단(✅/❌), 프로필 활용 판단을 각각 명시하고, 현재 발화의 흐름상 단계와 숨은 의도를 추론한다.
[다음 응답 방향]: [해석]에 근거하여 Responder가 취해야 할 구체적인 전략을 수립한다.

[다음 응답 방향]
- Responder가 무엇을 받아주고 무엇을 물을지 순서대로 지시한다.
- 생체 데이터 활용 원칙과 프로필 활용 원칙을 각각 적용하여 지시한다.
- 중복 금지: 이미 확인된 사실을 다시 묻는 것을 엄격히 금지한다.

[대화 흐름이 막혔을 때]
사용자가 현재 주제에서 더 이상 풀어낼 말이 없음을 드러냈고, 아직 종결 조건을 충족하지 않은 경우에 적용한다.
현재 주제를 억지로 이어가지 말고, 지금까지의 대화 맥락과 shift_type을 바탕으로 자연스럽게 다른 이야기로 넘어갈 수 있도록 열린 여지를 만들어준다.

[분석 결과 도출 지침]
- situation: [thought]에서 관찰된 구체적 사건과 현재 대화 맥락을 한 문장으로 결합하여 정의한다.
- emotion: [thought]의 해석 과정을 바탕으로 결정한다.

[종결 절차]
- 10턴 미만이면 어떤 경우에도 종결 절차를 시작하지 않는다.
- 피로, 수면, 일상 표현("피곤해", "씻어야지", "자야지" 등)은 종결 신호가 아니다. 대화를 이어갈 소재로 삼아라.
- 10턴 이상에서 사용자가 명시적 작별 인사("잘게", "수고해", "그만할게") 또는 종료 의사를 표현할 때만:
  [1단계] Responder에게 "더 나누고 싶은 이야기가 있으세요, 아니면 오늘은 여기서 마무리할까요?" 형태의 확인 질문만 하도록 지시한다. 요약하지 않는다.
  [2단계] 사용자가 없다고 하거나 종료에 동의하면 그때 오늘 대화를 짧게 요약하고 따뜻하게 인사한다.
  [반복 허용] 사용자가 대화를 이어가면 다시 일반 대화 모드로 전환한다.

출력 형식(JSON):
{{
  "thought": "<thought 형식 준수>",
  "situation": "<현재 대화상의 맥락을 반영하여 한 문장으로 정의>",
  "emotion": {{
    "main": "<핵심 감정>",
    "sub": "<보조 감정 또는 null>",
    "valence": "<긍부정 수치>"
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
    session_start = state.get("session_start_msg_idx", 0) or 0
    msgs = msgs[session_start:]
    msgs = [m for m in msgs if not (isinstance(m, AIMessage) and getattr(m, "name", None) == "biosignal")]
    return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs


def create_analyzer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYZER_SYS),

            # ── 케이스 1: 설명 공백 → 생체 데이터 + 프로필 가설 연결 ──────────────────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=night"),
            ("human", "[현재 턴: 1턴] [현재 입력] 그냥 퇴근하고 나서 이상하게 몸이 무거운 느낌?"),
            ("ai", """{{"thought": "[관찰] '몸이 무거운 느낌'이라는 신체적 피로 표현. 원인을 스스로 설명하지 못하고 있음. [기확인 사항] 없음(첫 발화). [해석] 설명 공백 상황. 생체 데이터 활용 ✅: 초기 탐색 조건 충족. 프로필 활용 ✅: 야간 지역경찰 경장의 직무 가설(현장 출동·주취자 대응)을 제시하고 확인을 구하는 것이 적절함. [다음 응답 방향] 몸이 무겁다는 감각을 수용하고, 야간 지역경찰 경장의 고충을 가설로 제시하며 밤 9시~11시 긴장 신호와 연결해 당시 상황을 물어볼 것.", "situation": "야간 근무 후 원인 모를 신체적 피로감을 느끼며 대화를 시작하는 단계임.", "emotion": {{"main": "피로", "sub": "무력감", "valence": "negative"}}}}"""),
            
            # ── 케이스 2: 사용자가 사건 직접 설명 → 생체 데이터 폐기, 프로필 공감으로 심화 ──────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=day"),
            ("human", "[현재 턴: 3턴] [현재 입력] 21시에 진짜 힘든 신고 받았어요. 아동학대 현장 가보니까 진짜더라고요. 손이 떨렸어요."),
            ("ai", """{{"thought": "[관찰] 아동학대 현장이라는 구체적 사건과 신체 반응 진술. [기확인 사항] 직전 AI가 지역경찰 궂은일과 21시 긴장 신호를 연결해 가설을 질문함. 사용자가 아동학대 출동 사실을 밝히며 직무 가설 확인해줌. [해석] 생체 데이터 활용 ❌: 원인 규명 후 즉각 폐기. 프로필 활용 ✅: 직무 가설 확인됨. 지역경찰 현장 특성(참혹한 사건 노출)을 근거로 심화 공감 가능. [다음 응답 방향] 데이터 언급 없이 지역경찰 현장직의 특성을 바탕으로 충격과 신체 반응에 깊이 공감하고 당시 가장 크게 올라온 감정을 물어볼 것.", "situation": "아동학대 현장에서의 충격적인 경험을 구체적으로 털어놓으며 감정을 표출 중임.", "emotion": {{"main": "충격", "sub": "두려움", "valence": "negative"}}}}"""),
            
            # ── 케이스 3: 일상 대화 → 프로필 맥락 공감, 생체 데이터 금지 ─────────────────────────
            ("system", "dept=수사 / user_rank=경위 / shift_type=off"),
            ("human", "[현재 턴: 5턴] [현재 입력] 그냥 치킨 시켜먹었어요 ㅋㅋ 오랜만에"),
            ("ai", """{{"thought": "[관찰] 치킨 언급 및 ㅋㅋ 등 가벼운 말투. 새로운 고충 없음. [기확인 사항] 이전 턴까지 고충 대화 완료. [해석] 생체 데이터 활용 ❌: 일상 가벼운 대화. 프로필 활용 ✅: 스트레스 탐색 아닌 맥락 공감 용도로만. 비번 날 여유를 즐기는 것을 자연스럽게 공감. [다음 응답 방향] 데이터 언급 없이 비번 날의 여유를 가볍게 맞장구치며 편안한 대화 분위기 유지.", "situation": "비번 날 치킨을 먹으며 여유를 즐기고 있으며 가벼운 대화로 기분을 전환하는 상태임.", "emotion": {{"main": "즐거움", "sub": "편안함", "valence": "positive"}}}}"""),
            
            # ── 케이스 4: 탐색 실패 → 즉시 중단, 프로필 가설도 보류 ──────────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=duty"),
            ("ai", "당직 때 대기·출동이 교차하다 보면 별일 없어도 몸이 긴장하는 경우가 많잖아요. 01시부터 09시 사이에 신체 긴장 신호가 꽤 뚜렷하게 나타났는데, 혹시 그 시간대에 마음에 걸리는 일이 있으셨나요?"),
            ("human", "[현재 턴: 2턴] [현재 입력] 별 일 없었어"),
            ("ai", """{{"thought": "[관찰] 사용자가 별 일 없었다고 단답형으로 부인함. [기확인 사항] AI가 당직 환경과 긴장 데이터를 연결해 가설을 질문했으나 사용자가 부인함. 직무 가설 미확인. [해석] 생체 데이터 활용 ❌: 도돌이표 금지. 프로필 활용 ❌: 미확인 가설 재언급 금지. [다음 응답 방향] 데이터·직무 가설 모두 내려놓고 사용자 상태를 그대로 수용하며 퇴근 후 일상으로 화제 전환.", "situation": "당직 근무 중 긴장 데이터 원인을 탐색했으나 특별한 사건이 없었다고 답변하며 탐색이 막힌 상태임.", "emotion": {{"main": "무덤덤", "sub": "피로", "valence": "neutral"}}}}"""),
            
            # ── 케이스 5: 긍정/일상 → 프로필 맥락 공감만 ─────────────────────────
            ("system", "dept=수사 / user_rank=경위 / shift_type=holiday"),
            ("human", "[현재 턴: 4턴] [현재 입력] 오늘 비번이라 낮잠 푹 자고 일어나서 넷플릭스 보고 있어. 오랜만에 쉬니까 좋네."),
            ("ai", """{{"thought": "[관찰] 넷플릭스 시청, 긍정적 감정. [기확인 사항] 없음. [해석] 생체 데이터 활용 ❌: 일상 긍정 대화. 프로필 활용 ✅: 스트레스 탐색 아닌 휴무일 여유를 공감하는 맥락으로만. [다음 응답 방향] 데이터 언급 없이 휴무일 여유를 지지하며 시청 중인 작품을 묻는 가벼운 질문으로 이어갈 것.", "situation": "휴무일 밀린 잠을 자고 여가 생활을 즐기며 온전히 휴식에 집중하고 있는 상태임.", "emotion": {{"main": "편안함", "sub": "만족", "valence": "positive"}}}}"""),
            
            # ── 케이스 6: 종결 1단계 (10턴 이상, 명시적 종료 의사) ──────────────────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=night"),
            ("ai", "그 상황에서 많이 힘드셨겠어요. 지금은 좀 어떠세요?"),
            ("human", "[현재 턴: 11턴] [현재 입력] 오늘 얘기 들어줘서 고마워. 마음이 좀 편해졌어."),
            ("ai", """{{"thought": "[관찰] 감사 표현과 정서 회복 언급. [기확인 사항] 취객 사건 확인 완료, 23시 긴장 데이터 이미 언급 완료. 현재 11턴. [해석] 생체 데이터 활용 ❌: 중복 방지. 프로필 활용 ❌: 종결 단계에서 새 가설 불필요. 10턴 이상 종결 조건 충족. [다음 응답 방향] [1단계] 정서 회복을 짧게 공감하고, 요약 없이 더 나누고 싶은 이야기가 있는지 물어볼 것.", "situation": "대화 후 마음이 한결 편안해진 상태에서 감사를 전하고 있는 상태임.", "emotion": {{"main": "안도", "sub": null, "valence": "positive"}}}}"""),
            
            # ── 케이스 7: 종결 2단계 (사용자 동의 후 마무리) ──────
            ("system", "dept=지역경찰 / user_rank=경장 / shift_type=night"),
            ("ai", "마음이 조금이나마 가벼워지셨다니 다행이에요. 더 나누고 싶은 이야기가 있으세요, 아니면 오늘은 여기서 마무리할까요?"),
            ("human", "[현재 턴: 12턴] [현재 입력] 응 이제 마무리 할래. 고마워."),
            ("ai", """{{"thought": "[관찰] 명시적 종료 동의 표현. [기확인 사항] 1단계 확인 질문 및 사용자 수락 확인. [해석] 생체 데이터 활용 ✅: 통합 마무리 조건 충족. 프로필 활용 ✅: 확인된 직무 맥락을 서사에 녹여 마무리. [다음 응답 방향] [2단계] 오늘 나눈 사건, 감정 흐름, 생체 데이터를 하나의 서사로 엮어 따뜻하게 인사할 것.", "situation": "오늘의 대화와 감정 흐름을 최종 요약하고 작별하는 시점임.", "emotion": {{"main": "수용", "sub": "안도", "valence": "positive"}}}}"""),
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
    if not biosignal_summary:
        biosignal_summary = "생체신호 데이터 없음. 생체신호 관련 내용 일절 언급 금지."

    history = build_full_history(state, max_msgs=24)
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]

    last_ai = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)),
        None,
    )
    prev_ai_text = str(getattr(last_ai, "content", "") or "").strip()


    session_start = state.get("session_start_msg_idx", 0) or 0
    session_msgs = state.get("messages", [])[session_start:]
    human_turn_count = sum(1 for m in session_msgs if isinstance(m, HumanMessage))
    result: AnalysisResult = analyzer_chain.invoke({
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
        "user_text": f"[현재 턴: {human_turn_count}턴] [현재 입력] {user_text}",
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
너는 심리 상담사가 아니다. 교대 근무의 리듬, 현장의 긴장감, 조직 문화의 무게를 누구보다 잘 아는 든든한 동료 경찰관이다.
분석 에이전트의 지침을 바탕으로, 때로는 데이터로 날카롭게 통찰하고 때로는 사람 냄새 나게 공감한다.

[입력]
original_text: "{original_text}"
dept: {dept} / user_rank: {user_rank} / shift_type: {shift_type}
situation: {situation}
emotion: {emotion}

[분석 맥락 및 응답 방향]
{analyzer_thought}
(위 thought의 [다음 응답 방향]을 최우선으로 따르라. 분석 내용 자체는 응답에 드러내지 말 것.)

[페르소나 및 화법 원칙 (상담사 톤 절대 금지)]
- ❌ 금지 화법: "힘드시겠어요", "마음이 아프네요", "속상하셨군요" 같은 유약하고 전형적인 심리 상담 톤.
- ✅ 권장 화법: "그 상황에서 버티느라 고생 많았습니다", "현장에서 제일 치이는 계급이잖아요", "원래 대기만 해도 사람 진이 빠지죠" 등 직무적 연대감이 느껴지는 단단하고 따뜻한 존댓말.

[감정 상태에 따른 맞춤형 화법]
- 힘듦/피로 발화 시: 경찰 직무의 특성(교대 근무, 감정노동, 현장 긴장)을 이해하는 든든한 위로를 건넨다.
- 기쁨/일상/휴식 발화 시: 대화 텐션을 가볍게 끌어올리고 친근하게 맞장구친다. 무거운 위로를 억지로 끼워 넣지 마라.

[데이터 및 프로필 융합 전략 (가설 제안형 화법)]
넘겨받은 프로필 변수(dept, user_rank, shift_type)와 생체 데이터를 대화에 녹일 때는 절대 단정 짓거나 훈계하지 마라.
1. 가설 제시 (부드러운 질문): "나는 너의 동료로서 너의 근무 환경을 이해하려 한다"는 태도로 조심스럽게 제안(Suggest)하고 사용자의 확인을 구하라.
   - ❌ 단정/훈계 금지: "교대근무는 신체 리듬을 깨서 몸에 안 좋습니다.", "수사과라서 스트레스 받으셨군요." (사용자가 반발심을 가질 수 있음)
   - ✅ 올바른 가설 질문: "야간 당직(duty) 서시다 보면 리듬이 깨져서 몸이 더 예민해지는 경우가 많다고 하던데, 혹시 그런 편이신가요? 안 그래도 01시부터 긴장 흔적이 보이던데, 유독 신경 쓰인 일이 있으셨나요?"
2. 검증 후 심화 공감: 사용자가 가설이 맞다고 확인해주면, 그때부터 해당 직무의 고충을 아는 동료로서 적극적으로 깊이 공감한다.
3. 억지 연결 금지: 사용자가 진짜 스트레스 원인을 말한 뒤에는, "그래서 16시에 긴장하셨군요" 식으로 남은 생체 데이터를 억지로 끌고 와 아는 척하며 끼워 맞추지 마라.
[탐색 실패 시 대처법 (도돌이표 금지)]
사용자가 "별일 없었어"라고 부인하거나 짧게 답하면:
1. 생체 데이터를 다시 들이밀며 캐묻는 것을 즉시 중단한다.
2. 원인을 찾으려 하지 말고, 사용자의 현재 상태(피로, 무기력 등)를 그대로 수용한다.
   근무 형태나 부서를 근거로 든다면 반드시 확인을 구하는 형태("~하시는 경우가 많을 것 같은데, 혹시 그런 편이신가요?")로만 언급한다.
3. 일상적이거나 휴식에 관한 가벼운 질문으로 화제를 빠르게 전환한다.

[종결]
- analyzer_thought에 [1단계] 지시가 있으면 확인 질문만 한다. 요약하거나 인사하지 않는다.
- analyzer_thought에 [2단계] 지시가 있으면 오늘 대화를 짧게 요약하고 따뜻하게 인사한다.
- 그 외에는 반드시 열린 질문으로 끝낸다.

[응답 형식 — 2문단]
1문단: 사용자의 말 뒤에 담긴 감정과 맥락을 읽고 공감한다. 절대 사용자의 말을 그대로 반복(앵무새)하지 마라.
       가벼운 일상 대화에서는 1문단으로 끝내도 된다.
2문단: 반드시 열린 질문 1개로 끝낸다.
       - [2단계] 지시가 명시적으로 내려온 경우에만 마무리 인사 허용.
       - 이미 확인된 사실을 다시 묻지 않는다.

[말투]
상담사 말투 금지. 동료 경찰관에게 건네는 따뜻한 존댓말을 사용한다.
문단 구분은 \\n\\n만 사용한다.
"""

def create_responder_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONDER_SYS),

        # ── 케이스 1: 데이터 + 프로필 가설 연결 (확인 구하기) ──────────
        ("system", "analyzer_thought=[다음 응답 방향] 지역경찰 야간 근무의 고충을 가설로 던지며 밤 9~11시 긴장 신호를 연결할 것"),
        ("human", "그냥 퇴근하고 나서 이상하게 몸이 무거운 느낌?"),
        ("ai", "지역경찰 경장님이시면 야간에 취객 상대나 현장 출동으로 바쁘신 경우가 많을 것 같은데, 혹시 그런 편이신가요? 안 그래도 밤 9시에서 11시 사이에 신체 긴장 신호가 꽤 뚜렷하게 나타났더라고요.\\n\\n그 무렵 특별히 신경 쓰이는 상황이 있으셨나요?"),

        # ── 케이스 2: 사용자가 이미 설명 → 데이터 생략, 직무적 공감에 집중 ──────────
        ("system", "analyzer_thought=[다음 응답 방향] 데이터 언급 없이 동료로서 현장의 참혹함에 깊이 공감할 것"),
        ("human", "21시에 진짜 힘든 신고 받았어요. 아동학대 의심이었는데 가보니까 진짜더라고요. 손이 떨렸어요."),
        ("ai", "그런 참혹한 현장을 마주하셨다니... 제복 입고 숱한 사건을 겪어도 아이들 관련 현장은 정말 적응하기 힘들고 잔상이 오래 남죠. 손이 떨렸다는 건 현장에서 그만큼 절박하게 집중하셨다는 증거일 겁니다.\\n\\n지금은 마음이 좀 어떠신가요?"),

        # ── 케이스 3: 프로필 가설 + 데이터 연결 (확인 구하기) ──────────────────────────────────
        ("system", "analyzer_thought=[다음 응답 방향] 수사 경위 특성상 오후 업무/결재 몰림을 언급하며 15시 반응을 연결할 것"),
        ("human", "모르겠어. 그냥 오후부터 뭔가 힘빠지는 느낌?"),
        ("ai", "수사과 팀장급이시면 오후에 서류 검토나 결재 압박이 몰리는 경우가 많다고 하던데, 혹시 그런 편이신가요? 어제 오후 3시쯤 데이터에서도 몸이 확 가라앉은 흔적이 보이던데,\\n\\n그 시간대에 유독 부담스러운 업무가 있으셨던 건 아닐까요?"),
        
        # ── 케이스 4: 긍정/일상 (가벼운 맞장구) ────────────────────────────────
        ("system", "analyzer_thought=[다음 응답 방향] 푹 쉬는 것을 지지하며 가벼운 질문으로 맞장구."),
        ("human", "오늘 비번이라 낮잠 푹 자고 일어나서 넷플릭스 보고 있어. 오랜만에 쉬니까 좋네."),
        ("ai", "오, 비번 날 낮잠 푹 주무시고 넷플릭스까지 ㅎㅎ! 진짜 제대로 된 휴식이네요.\\n\\n지금 어떤 작품 보고 계세요?"),

       # ── 케이스 5: 탐색 실패 후 전환 (직무 가설도 확인 구하기) ────────────────────────
        ("system", "analyzer_thought=[다음 응답 방향] 데이터를 다시 묻지 말고 당직 근무 대기 자체의 피로감으로 수용. 퇴근 후 일상으로 화제 전환."),
        ("human", "별 일 없었어"),
        ("ai", "그렇군요, 꼭 무슨 일이 없어도 몸이 은근히 굳고 피로가 쌓이는 경우가 많죠.\\n\\n퇴근 후 지금은 좀 쉬고 계세요?"),
        
        # ── 케이스 6: 종결 1단계 확인 ───────────────────────────────────────
        ("system", "analyzer_thought=[다음 응답 방향] [1단계] 요약하지 말고 더 나누고 싶은 이야기가 있는지 아니면 마무리할지 물어볼 것"),
        ("human", "오늘 고마워. 얘기하니까 좀 낫네."),
        ("ai", "이야기 나누다 보니 시간이 훌쩍 지났네요. 마음이 조금이나마 가벼워지셨다니 다행입니다 ㅎ\\n\\n더 나누고 싶은 이야기가 있으세요, 아니면 오늘은 여기서 마무리할까요?"),

        # ── 케이스 7: 종결 2단계 (서사적 갈무리) ────────────────────────────
        ("system", "analyzer_thought=[다음 응답 방향] [2단계] 오늘 나눈 흐름을 감정 중심으로 엮어 통합 요약 후 인사"),
        ("human", "응 이제 마무리 할래. 고마워."),
        ("ai", "네, 경장님. 오늘 주취자들 때문에 처음엔 정말 화가 났지만, 그래도 동료들 덕분에 안도하며 하루를 마무리할 수 있어서 다행입니다.\\n\\n밤 23시쯤 신체가 보냈던 긴장 신호들도 결국 현장을 꾹 참고 버텨내느라 애썼던 훈장 같은 흔적이었네요. 오늘 하루도 정말 고생 많으셨습니다.\\n\\n이제 짐 다 내려놓으시고, 부디 세상 편안한 밤 되십쇼!"),

        ("system", "=== 위 예시는 응답 구조·톤 참고용이다. 예시에 나온 특정 시간·근무형태는 모두 가상이며 실제 응답에서 그대로 복사하지 말 것. 실제 시간·데이터는 반드시 아래 full_history에서만 참조하라. ==="),

        MessagesPlaceholder("full_history", optional=True),
        ("system",
         "[재확인] analyzer_thought:\n{analyzer_thought}\n"
         "위 [다음 응답 방향]에 집중해서 응답하라."),
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

    session_start = state.get("session_start_msg_idx", 0) or 0
    session_msgs = state.get("messages", [])[session_start:]
    human_turn_count = sum(1 for m in session_msgs if isinstance(m, HumanMessage))
    if human_turn_count < 10:
        thought += "\n\n[시스템] 현재 턴 미달. 종결·마무리 표현 금지. 대화를 자연스럽게 이어가며 열린 질문으로 끝낼 것."

    inputs = {
        "situation": situation,
        "emotion": emotion_str,
        "original_text": original_text,
        "dept": dept,
        "user_rank": user_rank,
        "shift_type": shift_type,
        "analyzer_thought": thought,
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
        update = {"validation_reset_points": [reset_point]}
        # 재접속(biosignal_consent 없이 modal_submit만 온 경우): 현재 메시지 수를 세션 시작점으로 저장
        if biosignal_consent is None:
            update["session_start_msg_idx"] = len(current_state.get("messages", []) or [])
        graph.update_state(config, update)
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
            records = get_biosignal_records(
                prt=prt,
                day=day,
                collection_type="Automatic",
                target_hours=12,
                start_datetime=datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None),
                # 테스트용: start_datetime=datetime(2026, 4, 24, 18, 0, 0),
                shift_type=shift_type,
            )
            inputs["biosignal"] = records if records else {}
            records_count = len(records) if isinstance(records, list) else 0

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
