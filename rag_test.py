#!/usr/bin/env python3
"""
RAG 테스트: 빈손 테스트와 같은 모델·온도·질문으로, 검색한 근거를 붙여 답하게 한다.
답변은 대상 확인 → 권장 운동 → 주의사항 → 항상성이 깨질 때 생기는 문제 → 근거 순서.
빈손 테스트 결과와 같은 채점 칸을 만들어서 두 결과를 나란히 비교할 수 있다.

실행: python rag_test.py                          # section_ctx, 10개(공공 5개 이상, 한 문서당 2개), 영어 번역 검색 포함
      python rag_test.py --strategy fixed_ctx --k 8 --per-doc 2
필요: python chunk.py → python index.py, .env 에 OPENAI_API_KEY
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from blank_test import MODEL, QUESTIONS_PATH, RESULTS_DIR, TEMPERATURE
from rag import Retriever, _client, cite, translate_query

SYSTEM_PROMPT = """너는 당뇨·비만·고혈압이 있는 사람의 운동을 지도하는 트레이너를 돕는 운동생리 보조 AI야.
연령(성인·노인·청소년), 성별, 질환 유형(1형·2형 당뇨, 원발성·이차성 고혈압 등), 임신 같은 대상별로
운동 방법과 주의사항, 그리고 질환 때문에 운동 중 항상성이 깨질 때 생기는 문제를 근거로 설명해.

답변 형식 (해당 내용이 근거에 없으면 그 칸에 '제공된 문서에서 찾지 못함'이라고 적어)
1. 대상 확인: 질문에서 읽은 대상(연령·성별·질환 유형·약·식사·컨디션). 판단에 꼭 필요한 정보가 빠졌으면 먼저 되물어.
2. 권장 운동: 종류·강도·시간·빈도
3. 주의사항: 운동 전·중·후에 확인할 것, 피할 동작이나 상황
4. 항상성이 깨질 때 생기는 문제: 무엇이(예: 혈당·혈압·체온·자율신경) → 어떤 기전으로 → 어떤 증상·위험으로 나타나는지
5. 근거: 번호별 문서명과 쪽(또는 섹션)

근거 사용 규칙
- 아래 [근거]에 있는 내용만 사용해. 근거에 없는 수치나 기준은 만들지 마.
- 문장마다 뒷받침하는 근거 번호를 [1]처럼 붙여.
- [학술](논문·학회지)을 먼저 근거로 쓰고, [공공](정부 안내서) 자료는 한국 기준 수치를 보충할 때 써.
- 문서마다 기준이 다르면 섞지 말고 출처별로 나눠서 보여줘. 같은 문서의 연도판이 여러 개면 최신판을 우선해.
- 근거가 다른 대상(예: 질문은 비만인데 근거는 당뇨 환자 기준)에 관한 것이면 그 점을 밝혀.
- 영어 근거는 한국어로 옮겨서 설명해.

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

    answers, sources = [], []
    for _, row in df.iterrows():
        print(f"[{row['번호']}/{len(df)}] {row['질문'][:30]}...")
        en = None if args.no_translate else translate_query(row["질문"])
        chunks = retriever.search(row["질문"], k=args.k, per_doc=args.per_doc,
                                  public=args.public, query_en=en)
        sources.append("\n".join(f"[{c['rank']}] {cite(c)} ({c['score']:.2f})" for c in chunks))
        try:
            answers.append(ask(client, row["질문"], chunks))
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속
            answers.append(f"ERROR: {e}")
            print(f"  실패: {e}")
        time.sleep(0.5)

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
