#!/usr/bin/env python3
"""
RAG 테스트: 빈손 테스트와 같은 모델·온도·질문으로, 검색한 근거를 붙여 답하게 한다.
답변은 상황 정리 → 판단 → 주의사항(이 조합이라 조심할 점, 운동 시 생길 수 있는 문제) → 근거 순서.
질문마다 먼저 범위(당뇨·비만·고혈압)를 판정해서, 범위 밖이면 검색·답변 없이 안내만 하고
일부만 범위 안이면 범위 밖 부분은 답하지 않도록 답변 프롬프트에 알려준다.
빈손 테스트 결과와 같은 채점 칸을 만들어서 두 결과를 나란히 비교할 수 있다.

실행: python rag_test.py                          # section_ctx, 10개(공공 5개 이상, 한 문서당 2개), 영어 번역 검색 포함
      python rag_test.py --strategy fixed_ctx --k 8 --per-doc 2
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
from rag import Retriever, _client, cite, translate_query

SYSTEM_PROMPT = """너는 당뇨·비만·고혈압이 있는 회원을 지도하는 트레이너를 돕는 운동생리 보조 AI야.
트레이너는 회원의 질환과 유형, 연령, 성별, 운동 경력이 조합된 구체적인 상황을 물어. 운동 방법을 처음부터 가르치는 게 아니라,
그 조합의 회원이 그 상황에서 무엇을 주의해야 하는지, 하려는 것을 해도 되는지를 근거로 판단해 주는 게 네 역할이야.

답변 형식 (해당 내용이 근거에 없으면 그 칸에 '제공된 문서에서 찾지 못함'이라고 적어)
1. 상황 정리: 질문에서 읽은 회원 조건(질환과 유형, 연령, 성별, 운동 경력, 약·식사·컨디션)과 하려는 것.
   판단에 꼭 필요한 조건이 빠졌으면 무엇이 필요한지 먼저 되물어.
2. 판단: 이 조합에서 하려는 것을 해도 되는지, 된다면 어떤 조건·범위에서인지. 일반론보다 이 회원의 조건에 맞춰서.
3. 주의사항: 이 조합(연령·성별·질환 유형·경력)이라 특히 조심할 점과 운동 전·중·후에 확인할 것,
   그리고 이 상황에서 생길 수 있는 문제(예: 저혈당, 혈압 급상승, 운동 후 어지럼, 열 손상)를
   무엇이 생기는지 → 왜 생기는지 → 어떤 증상인지 → 어떻게 대처하는지로
4. 근거: 번호별 문서명과 쪽(또는 섹션)

근거 사용 규칙
- 아래 [근거]에 있는 내용만 사용해. 근거에 없는 수치나 기준은 만들지 마.
- 문장마다 뒷받침하는 근거 번호를 [1]처럼 붙여.
- [학술](논문·학회지)을 먼저 근거로 쓰고, [공공](정부 안내서) 자료는 한국 기준 수치를 보충할 때 써.
- 문서마다 기준이 다르면 섞지 말고 출처별로 나눠서 보여줘. 같은 문서의 연도판이 여러 개면 최신판을 우선해.
- 근거가 다른 대상(예: 질문은 비만인데 근거는 당뇨 환자 기준)에 관한 것이면 그 점을 밝혀.
- 영어 근거는 한국어로 옮겨서 설명해.

범위 규칙
- 이 도구가 다루는 것은 당뇨(1형·2형·임신당뇨와 그 합병증), 비만, 고혈압(원발성·이차성·임신 고혈압)이 있는 사람의 운동뿐이야.
- 근골격계 질환(관절염·디스크·부상 등), 암, 심장·폐·신장 질환처럼 범위 밖 질환의 운동 방법은 근거에 있더라도 조언하지 마.
- 근거에 범위 밖 질환의 운동 내용이 들어 있어도 그 내용으로 조언하지 마.

경계 규칙
- 진단이나 약 조절은 하지 말고 의료기관을 안내해. 약이 운동 반응에 주는 영향은 근거에 있으면 설명하되, 조절은 의사와 상의하라고 해."""


def build_context(chunks):
    return "\n\n".join(f"[{c['rank']}] {cite(c)}\n{c['text']}" for c in chunks)


def ask(client, question, chunks):
    user = f"[근거]\n{build_context(chunks)}\n\n[질문]\n{question}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


# ── 범위 판정 ────────────────────────────────────────
SCOPE_PROMPT = """너는 질문이 운동지도 보조 도구의 범위 안인지 판정한다.
범위: 당뇨, 비만, 고혈압이 있는 사람의 운동·신체활동에 관한 질문.
- 당뇨: 1형·2형·임신당뇨, 당뇨 합병증(신경병증, 자율신경병증, 저혈당 등)
- 비만: 과체중, 고도비만, 소아·청소년 비만, 근감소성 비만, 체중 감량
- 고혈압: 원발성·이차성·임신 고혈압, 운동 중 혈압 반응, 혈압 수치·혈압약에 관한 질문
  (진단·약 조절 요청도 주제가 고혈압·당뇨면 범위 안이다. 거절은 답변 단계에서 한다)
- 위 질환이 있는 사람의 연령·성별·임신 조건, 운동 중 증상(어지럼, 통증, 저혈당 증상 등)도 범위 안이다.
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
"""

# 질문에 이 단어가 있으면 범위 질환이 언급된 것으로 보고 '범위 밖'으로 막지 않는다 (판정기 오판 대비)
SCOPE_TERMS = r"당뇨|혈당|인슐린|비만|체중|과체중|고혈압|혈압|diabet|obes|hypertens"


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
    scope["out_of_scope"] = [t for t in scope.get("out_of_scope") or [] if not re.search(SCOPE_TERMS, t, re.I)]
    if scope["scope"] == "out" and re.search(SCOPE_TERMS, question, re.I):
        scope["scope"] = "partial" if scope["out_of_scope"] else "in"
        scope["reason"] = "범위 단어가 있어 막지 않고 답변 단계로 넘김. " + scope.get("reason", "")
    if scope["scope"] == "partial" and not re.search(SCOPE_TERMS, question, re.I):
        scope["scope"] = "out"     # '일부만 범위 안'인데 질문에 당뇨·비만·고혈압이 없으면 범위 밖
        scope["reason"] = "질문에 범위 질환이 없음. " + scope.get("reason", "")
    if scope["scope"] == "partial" and not scope["out_of_scope"]:
        scope["scope"] = "in"
    return scope


def out_of_scope_message(scope):
    topics = ", ".join(scope.get("out_of_scope") or []) or "질문한 내용"
    return (f"이 도구는 당뇨·비만·고혈압이 있는 사람의 운동 지도만 다룹니다. "
            f"범위 밖 주제({topics})라서 운동 방법이나 주의사항을 답하지 않습니다. "
            f"의료기관이나 해당 분야 전문가와 상의해 주세요.\n\n"
            f"회원에게 당뇨·비만·고혈압 중 해당하는 질환이 있다면 함께 알려 주시면 그 범위 안에서 답하겠습니다.")


def run(question, retriever, client, k=10, per_doc=2, public=5, translate=True):
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
    query_en = translate_query(asked) if translate else None
    chunks = retriever.search(asked, k=k, per_doc=per_doc, public=public, query_en=query_en)
    answer = ask(client, asked, chunks)
    if scope["scope"] == "partial":
        topics = ", ".join(scope["out_of_scope"])
        answer += (f"\n\n※ 범위 밖 주제({topics})는 답하지 않았습니다. 이 도구는 당뇨·비만·고혈압만 다룹니다. "
                   f"해당 질환에 맞는 운동은 의료기관이나 해당 분야 전문가와 상의해 주세요.")
    return {"scope": scope, "asked": asked, "query_en": query_en, "chunks": chunks, "answer": answer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="section_ctx")
    ap.add_argument("--questions", default=QUESTIONS_PATH, help="질문 CSV (번호, 질문 칸 필요)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--per-doc", type=int, default=2, help="한 문서에서 가져올 최대 청크 수")
    ap.add_argument("--public", type=int, default=5, help="최소 [공공] 청크 수 (한국 기준 수치용)")
    ap.add_argument("--no-translate", action="store_true", help="영어 번역 검색 끄기")
    args = ap.parse_args()

    retriever = Retriever(args.strategy)
    client = _client()
    df = pd.read_csv(args.questions, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"질문 {len(df)}개, 모델 {MODEL}, 인덱스 {args.strategy}, 상위 {args.k}개\n")

    answers, sources, scopes = [], [], []
    for _, row in df.iterrows():
        print(f"[{row['번호']}/{len(df)}] {row['질문'][:30]}...")
        try:
            res = run(row["질문"], retriever, client, k=args.k, per_doc=args.per_doc,
                      public=args.public, translate=not args.no_translate)
            answers.append(res["answer"])
            sources.append("\n".join(f"[{c['rank']}] {cite(c)} ({c['score']:.2f})" for c in res["chunks"]))
            scopes.append(f"{res['scope']['scope']}: {res['scope'].get('reason', '')}")
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속
            answers.append(f"ERROR: {e}")
            sources.append("")
            scopes.append("")
            print(f"  실패: {e}")
        time.sleep(0.5)

    df["범위판정"] = scopes
    df["답변"] = answers
    df["검색결과"] = sources
    df["모델"] = MODEL
    df["인덱스"] = (f"{args.strategy}@{args.k}/doc{args.per_doc}/pub{args.public}"
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
