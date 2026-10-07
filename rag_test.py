#!/usr/bin/env python3
"""
RAG 테스트: 빈손 테스트와 같은 모델·온도·질문으로, 검색한 근거를 붙여 답하게 한다.
답변은 결론 문단(해도 되는지와 핵심 유의점) → 지도할 때 유의할 점 → 회원에게서 지켜볼 점(신호 → 문제 → 이유 → 대처) → 근거 순서.
질문마다 먼저 범위(당뇨·비만·고혈압)를 판정해서, 범위 밖이면 검색·답변 없이 안내만 하고
일부만 범위 안이면 범위 밖 부분은 답하지 않도록 답변 프롬프트에 알려준다.
빈손 테스트 결과와 같은 채점 칸을 만들어서 두 결과를 나란히 비교할 수 있다.

실행: python rag_test.py                          # fixed, 10개(공공 5개 이상, 한 문서당 2개), 영어 번역 검색 포함
      python rag_test.py --strategy section_ctx --k 8 --per-doc 2
필요: python chunk.py → python index.py, .env 에 OPENAI_API_KEY
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from blank_test import MODEL, QUESTIONS_PATH, RESULTS_DIR, TEMPERATURE
from rag import Retriever, _client, cite, search_text, translate_query

SYSTEM_PROMPT = """너는 당뇨·비만·고혈압이 있는 회원을 지도하는 트레이너를 돕는 운동생리 보조 AI야.
트레이너는 회원의 질환과 유형, 연령, 성별, 운동 경력이 조합된 구체적인 상황을 물어. 운동 방법을 처음부터 가르치는 게 아니라,
그 조합의 회원이 그 상황에서 무엇을 주의해야 하는지, 하려는 것을 해도 되는지를 근거로 판단해 주는 게 네 역할이야.

답변 모양 (굵은 제목 세 개는 이 글자 그대로 반드시 넣어. 근거에 없는 칸은 '제공된 문서에서 찾지 못함')

(제목 없는 첫 문단)

**지도할 때 유의할 점**
- (항목) [번호]
- (항목) [번호]
- (항목) [번호]

**회원에게서 지켜볼 점**
- (신호) → (의심되는 문제) → (왜 생기는지) → (대처) [번호]
- (신호) → (의심되는 문제) → (왜 생기는지) → (대처) [번호]

**근거**
- [번호] 문서명 · 쪽(또는 섹션)

각 칸에 쓸 내용
- 첫 문단: 하려는 것을 해도 되는지를 트레이너에게 말하듯 자연스러운 문장으로. 예: "증량은 진행해도 되지만, 혈압이 조절되는지 먼저 확인하고 숨을 참지 않게 지도해야 합니다."
  '판단:', '상황 정리:' 같은 머리말이나 질문을 다시 요약하는 문장은 쓰지 마. 일반론보다 이 회원의 조건(질환 유형·연령·성별·경력)에 맞춰서.
  질문에 질환 유형, 약(인슐린·혈압약 등), 오늘 혈당·혈압 상태처럼 판단에 꼭 필요한 조건이 없으면
  첫 문장을 "판단하려면 먼저 ~을 확인해 주세요"로 시작해 무엇이 필요한지 되묻고, 조건에 따라 어떻게 달라지는지 짧게 나눠 말해.
- 지도할 때 유의할 점: 이 조합이라 운동을 어떻게 다르게 지도해야 하는지(강도·중량·호흡·휴식·운동 전 확인 등). 한 항목에 한 가지씩 3~6개.
- 회원에게서 지켜볼 점: 임상 대상이니 트레이너가 운동 전·중·후에 회원의 겉모습과 말·행동에서 주의 깊게 봐야 할 신호.
  이 상황에서 실제로 생길 수 있는 문제(근거에 있는 것)만 2~5개 골라 써. 일반적인 증상을 빠짐없이 나열하지 마.
  한 줄 예(내용은 따라 하지 말고 모양만): "- 식은땀·손 떨림 → 저혈당 의심 → 인슐린이 작용하는 중에 운동으로 혈당이 빨리 떨어짐 → 운동을 멈추고 당을 먹게 한 뒤 혈당 재확인 [3]"
- 근거: 인용한 번호마다 한 줄.

근거 사용 규칙
- 아래 [근거]에 있는 내용만 사용해. 근거에 없는 수치나 기준은 만들지 마.
- 맨 앞 문단과 목록 항목마다 뒷받침하는 근거 번호를 [1]처럼 붙여. 근거 번호를 못 붙이는 내용은 쓰지 마.
- [학술](논문·학회지)을 먼저 근거로 쓰고, [공공](정부 안내서) 자료는 한국 기준 수치를 보충할 때 써.
- 문서마다 기준이 다르면 섞지 말고 출처별로 나눠서 보여줘. 같은 문서의 연도판이 여러 개면 최신판을 우선해.
  같은 기준의 수치가 근거마다 다르면(예: 운동을 피할 공복혈당이 한 근거는 250 mg/dL, 다른 근거는 300 mg/dL) 하나만 고르지 말고
  "[1]은 ~, [2]는 ~"처럼 둘 다 밝히고, 더 보수적인 쪽으로 지도하라고 써.
- 근거가 다른 대상(예: 질문은 비만인데 근거는 당뇨 환자 기준)에 관한 것이면 그 점을 밝혀.
- 영어 근거는 한국어로 옮겨서 설명해.
- [OCR] 표시 근거는 그림 속 글자를 읽은 것이라 인식 오류가 있을 수 있어. 숫자가 단위·문맥과 맞지 않으면(예: '3002 이상') 그 숫자는 쓰지 마.

범위 규칙
- 이 도구가 다루는 것은 당뇨(1형·2형·임신당뇨와 그 합병증), 비만, 고혈압(원발성·이차성·임신 고혈압)이 있는 사람의 운동뿐이야.
- 근골격계 질환(관절염·디스크·부상 등), 암, 심장·폐·신장 질환처럼 범위 밖 질환의 운동 방법은 근거에 있더라도 조언하지 마.
- 근거에 범위 밖 질환의 운동 내용이 들어 있어도 그 내용으로 조언하지 마.

경계 규칙
- 진단이나 약 조절은 하지 말고 의료기관을 안내해. 약이 운동 반응에 주는 영향은 근거에 있으면 설명하되, 조절은 의사와 상의하라고 해.
- 대처에 인슐린·약의 용량 조절이나 추가 투약을 쓰지 마. 트레이너가 할 수 있는 것(운동 중단, 당 섭취, 휴식, 측정, 의료기관 안내)만 써."""


def build_context(chunks):
    return "\n\n".join(f"[{c['rank']}] {cite(c)}\n{c['text']}" for c in chunks)


# 진단 요청: 수치·증상으로 무슨 병인지 판정해 달라는 질문 → 답변 프롬프트에 규칙을 더한다
DIAG_TERMS = r"진단해|진단을 (내려|해)|무슨 병|어떤 병|병인지|병명"
DIAG_RULE = """
[진단 요청 처리] 이 질문은 수치나 증상으로 무슨 병인지 판정해 달라는 진단 요청이야.
- 첫 문단 첫 문장에서 진단은 할 수 없다고 먼저 말해. '~이 있는 것으로 보인다', '~에 해당한다'처럼 회원의 질환을 판정하는 표현은 쓰지 마.
- 근거에 있는 공신력 있는 기준 범위는 회원 얘기가 아닌 일반 정보로만 알려 줘. 예: "일반적으로 수축기 140 mmHg 이상 또는 이완기 90 mmHg 이상을 고혈압 기준으로 봅니다"
  '이 회원은 ~에 해당합니다', '~ 상태입니다', '~이 있는 것으로 보입니다'처럼 회원의 수치를 기준에 대어 판정하는 문장은 쓰지 마.
- 수치가 근거의 기준을 크게 벗어났거나 증상이 있으면 오늘 운동은 보류하고, 어느 경우든 의료기관에서 확인받도록 권해. 운동 강도·종류 처방은 하지 마.
- 지도할 때 유의할 점에는 운동 보류와 의료기관 안내, 다시 운동할 때 확인할 것만 써.
- 회원에게서 지켜볼 점에는 흉통·심한 두통·어지럼 같은 응급 신호와 대처만 써."""


# 진단 요청 답변에 이런 판정 문장이 남으면 한 번 다시 쓰게 한다
DIAG_BANNED = (r"(고혈압|당뇨병?|비만|저혈당|고혈당|혈당 ?장애)[^.,]{0,12}(에 해당|해당합니다|해당됩니다|입니다|으로 보(이|일)|로 보(이|일)|분류|간주|판단됩니다|가능성|의심됩니다)"
               r"|있는 것으로 보|상태로 보(이|일)")


# 약·인슐린을 조절하라는 문장이 의료진 언급 없이 나오면 한 번 다시 쓰게 한다 (경계 위반)
MED_ADJUST = r"(인슐린|혈압약|약물?)[을를은는이가]?\s?(\S+\s){0,2}(조절|추가|감량|증량|줄이|늘리|끊|중단|더 맞)"


MED_CHECK_PROMPT = """트레이너용 답변의 문장들을 보고, 트레이너나 회원에게 약·인슐린을 줄이거나 늘리거나 추가하거나 끊도록
권하는 문장이 있는지 판정한다. 다음은 위반이 아니다: 약 조절은 의료진과 상의하라는 문장, 임의로 끊으면 안 된다는 문장,
'인슐린이 조절된 상태라면'처럼 상태를 말하는 문장, 약 사용 여부·용량을 확인하라는 문장.
JSON으로만 답한다: {"violation": true | false, "sentence": "위반 문장 (없으면 빈 문자열)"}"""


def med_violation(client, answer):
    """의료진에게 넘기지 않고 약·인슐린 조절을 권하는 문장 (없으면 None).
    정규식으로 후보 문장을 고르고, 후보가 있을 때만 판정기로 확인한다."""
    sents = [x.strip() for x in re.split(r"(?<=[.다요])\s+|\n", answer.split("**근거**")[0])]
    cands = [x for x in sents if re.search(MED_ADJUST, x)]
    if not cands:
        return None
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": MED_CHECK_PROMPT}, {"role": "user", "content": "\n".join(cands)}])
    try:
        out = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return None
    return (out.get("sentence") or "위반 문장") if out.get("violation") else None


def drop_sentences(answer, pattern):
    """다시 써도 남은 위반 문장을 답변 본문(근거 칸 앞)에서 뺀다"""
    body, sep, refs = answer.partition("**근거**")
    lines = []
    for line in body.split("\n"):
        parts = re.split(r"(?<=[.다요])\s+", line)
        kept = [x for x in parts if not re.search(pattern, x)]
        if kept or not line.strip():
            lines.append(" ".join(kept))
    return "\n".join(lines) + sep + refs


def ask(client, question, chunks, extra=""):
    user = f"[근거]\n{build_context(chunks)}\n\n[질문]\n{question}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "system", "content": SYSTEM_PROMPT + extra},
                  {"role": "user", "content": user}],
    )
    return re.sub(r"(?m)^(\s*)- - ", r"\1- ", resp.choices[0].message.content)


# ── 범위 판정 ────────────────────────────────────────
SCOPE_PROMPT = """너는 질문이 운동지도 보조 도구의 범위 안인지 판정한다.
범위: 당뇨, 비만, 고혈압이 있는 사람의 운동·신체활동에 관한 질문.
- 당뇨: 1형·2형·임신당뇨, 당뇨 합병증(신경병증, 자율신경병증, 저혈당 등)
- 비만: 과체중, 고도비만, 소아·청소년 비만, 근감소성 비만, 체중 감량
- 고혈압: 원발성·이차성·임신 고혈압, 운동 중 혈압 반응, 혈압 수치·혈압약에 관한 질문
  (진단·약 조절 요청도 주제가 고혈압·당뇨면 범위 안이다. 거절은 답변 단계에서 한다)
- 위 질환이 있는 사람의 연령·성별·임신 조건, 운동 중 증상(어지럼, 통증, 저혈당 증상 등)도 범위 안이다.
- 폐경·임신·노화·월경 주기 같은 생애 단계는 질환이 아니라 대상 조건이므로 범위 밖 질환으로 보지 않는다.
범위 밖: 위 질환이 아닌 다른 질환이 질문의 주제인 경우. 근골격계 질환(관절염, 디스크, 오십견, 회전근개 파열, 골절, 부상 재활), 암, 심장·폐·신장 질환 등. 그리고 위 질환이 전혀 언급되지 않은 일반 운동 질문.
판정:
- in: 전부 범위 안
- partial: 범위 질환과, 따로 진단된 범위 밖 질환이 함께 나옴 (예: 비만 + 무릎 관절염 → 범위 밖은 "무릎 관절염")
- out: 범위 질환이 없음 (예: 허리 디스크 회원 데드리프트, 건강한 20대 벌크업)
예시: "고도비만 회원이 운동하면 무릎이 아프대요" → in (비만 회원의 운동 중 통증, 진단된 다른 질환 없음)
예시: "비만 회원이 심박수가 높게 나와 심장에 문제 있는 거 아니냐고 걱정해요" → in (진단이 아니라 걱정이나 의심일 뿐이면 범위 밖 질환으로 보지 않는다)
예시: "혈압 180/110인데 무슨 병인지 진단해 주세요" → in (고혈압 주제, 진단 거절은 답변 단계)
JSON으로만 답한다: {"scope": "in" | "partial" | "out", "in_scope": [범위 안 주제], "out_of_scope": [범위 밖 질환],
 "in_scope_question": "partial일 때 범위 밖 질환을 빼고 다시 쓴 질문 (나머지 조건은 그대로)", "reason": "한 문장"}
예시: "비만이면서 무릎 관절염이 있는 회원의 운동은?" → partial, in_scope_question: "비만이 있는 회원의 운동은 어떻게 하나요?"
예시: "신장 질환 때문에 이차성 고혈압이 생긴 회원이 일반 고혈압처럼 운동하겠대요" → partial, out_of_scope: ["신장 질환"],
 in_scope_question: "이차성 고혈압이 있는 회원이 일반 고혈압 회원처럼 운동하겠대요. 어떻게 생각해?" (원인 질환이 범위 밖이어도 범위 질환은 답한다)
"""

# 질문에 이 단어가 있으면 범위 질환이 언급된 것으로 보고 '범위 밖'으로 막지 않는다 (판정기 오판 대비)
SCOPE_TERMS = r"당뇨|혈당|인슐린|비만|체중|과체중|고혈압|혈압|diabet|obes|hypertens"
# 질문에 이 질환이 진단명으로 나오면 판정기가 '범위 안'이라 해도 '일부만 범위 안'으로 바꾼다 (판정기 오판 대비)
# 심장은 넣지 않는다: "심장에 문제 있는 거 아니냐"처럼 걱정만 하는 질문이 많아서
OUT_DISEASES = r"신장 ?질환|신부전|콩팥|투석|관절염|디스크|골다공증|오십견|회전근개|골절"
# 생애 단계는 질환이 아니라 대상 조건: 판정기가 범위 밖으로 넣어도 뺀다
LIFE_STAGES = r"폐경|임신|노화|노인|월경|생리|사춘기|성장기"
# 질문에 이런 회원 조건이 하나도 없으면 답변 첫머리에서 되묻게 한다
COND_TERMS = r"\d+\s?대|\d+\s?세|노인|어르신|청소년|학생|아이|어린이|임신|남성|여성|남자|여자|부부|1형|2형|원발성|이차성|초보|숙련|경력"
ASKBACK_RULE = """
[조건 누락] 이 질문에는 회원의 나이·성별·질환 유형·약·운동 경력이 없어. 첫 문장은 반드시 "판단하려면 먼저 ~을(를) 확인해 주세요"로 시작해 판단에 필요한 조건을 묻고, 조건에 따라 판단이 어떻게 달라지는지 짧게 말해."""


def check_scope(client, question):
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SCOPE_PROMPT}, {"role": "user", "content": question}])
    try:
        scope = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        scope = {}
    if scope.get("scope") not in ("in", "partial", "out"):
        return {"scope": "in", "in_scope": [], "out_of_scope": [], "reason": "판정 실패 → 범위 안으로 처리"}
    # 범위 단어(혈압·비만 등)가 범위 밖 목록에 섞여 들어오면 빼고, 남은 게 없으면 범위 안
    scope["out_of_scope"] = [t for t in scope.get("out_of_scope") or []
                             if not re.search(SCOPE_TERMS, t, re.I) and not re.search(LIFE_STAGES, t)]
    if scope["scope"] == "out" and re.search(SCOPE_TERMS, question, re.I):
        scope["scope"] = "partial" if scope["out_of_scope"] else "in"
        scope["reason"] = "범위 단어가 있어 막지 않고 답변 단계로 넘김. " + scope.get("reason", "")
    if scope["scope"] == "partial" and not re.search(SCOPE_TERMS, question, re.I):
        scope["scope"] = "out"     # '일부만 범위 안'인데 질문에 당뇨·비만·고혈압이 없으면 범위 밖
        scope["reason"] = "질문에 범위 질환이 없음. " + scope.get("reason", "")
    if scope["scope"] == "partial" and not scope["out_of_scope"]:
        scope["scope"] = "in"
    found = re.findall(OUT_DISEASES, question)
    if scope["scope"] == "in" and found:
        scope["scope"] = "partial"
        scope["out_of_scope"] = sorted(set(found))
        scope["reason"] = f"범위 밖 질환({', '.join(scope['out_of_scope'])})이 함께 나옴. " + scope.get("reason", "")
    return scope


def out_of_scope_message(scope):
    topics = ", ".join(scope.get("out_of_scope") or []) or "질문한 내용"
    return (f"이 도구는 당뇨·비만·고혈압이 있는 사람의 운동 지도만 다룹니다. "
            f"범위 밖 주제({topics})라서 운동 방법이나 주의사항을 답하지 않습니다. "
            f"의료기관이나 해당 분야 전문가와 상의해 주세요.\n\n"
            f"회원에게 당뇨·비만·고혈압 중 해당하는 질환이 있다면 함께 알려 주시면 그 범위 안에서 답하겠습니다.")


def run(question, retriever, client, k=10, per_doc=2, public=5, translate=True, public_per_doc=1):
    """범위 판정 → 검색 → 답변.
    범위 밖: 검색·답변 없이 안내 문구만.
    일부만 범위 안: 범위 밖 질환을 뺀 질문으로 검색·답변하고, 범위 밖 안내는 코드가 정해진 문구로 붙인다
    (답변 모델이 범위 밖 질환을 아예 보지 않게 해서 조언이 섞이지 않도록)."""
    scope = check_scope(client, question)
    if scope["scope"] == "out":
        return {"scope": scope, "asked": None, "query_en": None, "chunks": [],
                "answer": out_of_scope_message(scope)}
    asked = question
    if scope["scope"] == "partial" and (scope.get("in_scope_question") or "").strip():
        asked = scope["in_scope_question"].strip()
    text = search_text(asked)          # 질문 끝 공통 요청("지도할 때 주의해야 할 점은?")은 빼고 검색
    query_en = translate_query(text) if translate else None
    chunks = retriever.search(text, k=k, per_doc=per_doc, public=public, query_en=query_en,
                              public_per_doc=public_per_doc)
    diagnosis = bool(re.search(DIAG_TERMS, question))
    extra = DIAG_RULE if diagnosis else ""
    if not diagnosis and not re.search(r"약|진단", question) and not re.search(COND_TERMS, question):
        extra += ASKBACK_RULE
    if scope["scope"] == "partial":   # 다시 쓴 질문에 범위 밖 질환이 남아 있어도 조언하지 않도록
        extra += f"\n[범위 밖 질환] {', '.join(scope['out_of_scope'])}에 대한 운동 조언이나 판정은 하지 마. 그 질환 때문에 생긴 범위 질환(예: 이차성 고혈압)은 답하되, 원인 질환 자체의 관리는 의료진에게 넘겨."
    answer = ask(client, asked, chunks, extra)
    bad = med_violation(client, answer)
    if bad:
        answer = ask(client, asked, chunks, extra + f"\n- 직전 답변에 약·인슐린 조절을 권하는 문장('{bad[:60]}')이 있었어. 약·인슐린 조절은 담당 의료진과 상의하라고만 쓰고 다시 써.")
        bad = med_violation(client, answer)
        if bad:                       # 다시 써도 남으면 그 문장을 뺀다
            answer = drop_sentences(answer, re.escape(bad.strip()[:40]))
    if diagnosis and re.search(DIAG_BANNED, answer.split("**근거**")[0]):
        hit = re.search(DIAG_BANNED, answer).group(0)
        answer = ask(client, asked, chunks, extra + f"\n- 직전 답변에 회원을 판정하는 표현('{hit}')이 있었어. 이번에는 회원의 수치를 기준에 대어 판정하지 말고 다시 써.")
        if re.search(DIAG_BANNED, answer.split("**근거**")[0]):
            answer = drop_sentences(answer, DIAG_BANNED)
    if scope["scope"] == "partial":
        topics = ", ".join(scope["out_of_scope"])
        answer += (f"\n\n※ 범위 밖 주제({topics})는 답하지 않았습니다. 이 도구는 당뇨·비만·고혈압만 다룹니다. "
                   f"해당 질환에 맞는 운동은 의료기관이나 해당 분야 전문가와 상의해 주세요.")
    return {"scope": scope, "asked": asked, "search": text, "query_en": query_en, "chunks": chunks, "answer": answer,
            "diagnosis_request": diagnosis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="fixed")
    ap.add_argument("--questions", default=QUESTIONS_PATH, help="질문 CSV (번호, 질문 칸 필요)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--per-doc", type=int, default=2, help="한 문서에서 가져올 최대 청크 수")
    ap.add_argument("--public", type=int, default=5, help="최소 [공공] 청크 수 (한국 기준 수치용)")
    ap.add_argument("--public-per-doc", type=int, default=1, help="공공 몫을 채울 때 한 문서 최대 수")
    ap.add_argument("--no-translate", action="store_true", help="영어 번역 검색 끄기")
    args = ap.parse_args()

    retriever = Retriever(args.strategy)
    client = _client()
    df = pd.read_csv(args.questions, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"질문 {len(df)}개, 모델 {MODEL}, 인덱스 {args.strategy}, 상위 {args.k}개\n")

    answers, sources, scopes, asked = [], [], [], []
    for _, row in df.iterrows():
        print(f"[{row['번호']}/{len(df)}] {row['질문'][:30]}...")
        try:
            res = run(row["질문"], retriever, client, k=args.k, per_doc=args.per_doc,
                      public=args.public, translate=not args.no_translate, public_per_doc=args.public_per_doc)
            answers.append(res["answer"])
            sources.append("\n".join(f"[{c['rank']}] {cite(c)} ({c['score']:.2f})" for c in res["chunks"]))
            scopes.append(f"{res['scope']['scope']}: {res['scope'].get('reason', '')}")
            asked.append(res.get("search") or "")
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속
            answers.append(f"ERROR: {e}")
            sources.append("")
            scopes.append("")
            asked.append("")
            print(f"  실패: {e}")
        time.sleep(0.5)

    df["범위판정"] = scopes
    df["검색질문"] = asked      # 실제로 검색한 문장 (질문 끝 공통 요청 제외, 일부만 범위 안이면 범위 밖 질환도 뺌)
    df["답변"] = answers
    df["검색결과"] = sources
    df["모델"] = MODEL
    df["인덱스"] = (f"{args.strategy}@{args.k}/doc{args.per_doc}/pub{args.public}/ppd{args.public_per_doc}"
                   + ("" if args.no_translate else "/en"))
    for col in ["충족수", "근거표시", "되묻기", "경계위반", "메모"]:
        df[col] = ""

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    qname = Path(args.questions).stem
    out = f"{RESULTS_DIR}/rag_{MODEL}_{args.strategy}_{qname}_{stamp}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    failed = sum(a.startswith("ERROR") for a in answers)
    print(f"\n저장: {out}")
    print(f"성공 {len(answers) - failed}개 / 실패 {failed}개")


if __name__ == "__main__":
    main()
