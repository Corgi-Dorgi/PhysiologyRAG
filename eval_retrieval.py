#!/usr/bin/env python3
"""
검색 평가: 청킹 전략별로 정답 근거가 검색 상위 K개 안에 드는지 잰다.

정답 근거는 eval/questions.csv 의 '근거위치' 칸에 '문서ID:인쇄쪽' 또는 '문서ID'로 적는다.
  예) khepi-ob-001:105-106; pmc-htn-ex-007
  근거위치가 빈 문항(문서에 없는 질문)은 검색 평가에서 뺀다.

문항마다 세 가지를 따로 기록해서 실패 원인을 나눈다.
  추출됨   정답 쪽의 텍스트가 청크로 존재하는가 (아니면 파싱·OCR 문제 → 검색으로는 못 고침)
  문서적중 상위 K개에 정답 문서가 있는가
  쪽적중   상위 K개에 정답 문서의 정답 쪽이 있는가 (= README의 '검색 적중률')

실행: python eval_retrieval.py            # 모든 인덱스, K=5
      python eval_retrieval.py --k 10 --only section_ctx
      python eval_retrieval.py --per-doc 2       # 한 문서당 최대 2개 (상위 독점 방지)
      python eval_retrieval.py --k 10 --per-doc 2 --public 5 --translate   # 현재 RAG 기본 설정
결과: eval/results/retrieval_<날짜>.csv (문항×전략 상세), 화면에 전략별 요약
"""
import argparse
import re
from datetime import datetime

import pandas as pd

from pathlib import Path

from rag import INDEX, ROOT, EmbedCache, Retriever, search_text, translate_query

QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "eval" / "results"


def parse_gold(spec):
    """'a:105-106,110; b:49; c' → {'a': {105, 106, 110}, 'b': {49}, 'c': set()}
    쪽 없이 문서ID만 적으면 그 문서의 어느 부분이든 정답으로 친다 (논문·건강정보처럼 쪽이 없는 문서)."""
    gold = {}
    for part in re.split(r"[;\n]", spec or ""):
        part = part.strip()
        if not part:
            continue
        doc, pages = part.split(":", 1) if ":" in part else (part, "")
        s = gold.setdefault(doc.strip(), set())
        for p in pages.split(","):
            p = p.strip()
            if re.fullmatch(r"\d+-\d+", p):
                a, b = map(int, p.split("-"))
                s.update(range(a, b + 1))
            elif p.isdigit():
                s.add(int(p))
    return gold


def label_pages(label):
    """청크의 인쇄 쪽 표시('105', '104-105') → 쪽 집합. 논문 섹션 이름이면 빈 집합"""
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", str(label))
    if not m:
        return set()
    a = int(m.group(1))
    return set(range(a, int(m.group(2) or a) + 1))


def hits(chunk, gold):
    """청크에 실제로 텍스트가 들어간 쪽(pages)이 정답 쪽과 겹치는가.
    '48-50쪽' 청크라도 49쪽이 빈 쪽이면 49쪽 정답으로 치지 않는다."""
    pages = gold.get(chunk["doc_id"])
    if pages is None:
        return False
    if not pages:
        return True
    got = set().union(*(label_pages(p) for p in chunk.get("pages", [chunk["label"]])))
    return bool(got & pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--only", help="이 전략만 (쉼표로 여러 개)")
    ap.add_argument("--per-doc", type=int, default=None, help="한 문서에서 가져올 최대 청크 수")
    ap.add_argument("--academic", type=int, default=0, help="결과 중 최소 [학술] 청크 수")
    ap.add_argument("--public", type=int, default=0, help="결과 중 최소 [공공] 청크 수 (한국어 질문 기준)")
    ap.add_argument("--translate", action="store_true", help="영어 번역 질문도 함께 검색")
    ap.add_argument("--expand", action="store_true", help="번역 검색어에 상황이 암시하는 위험·기전 이름을 덧붙임")
    ap.add_argument("--public-per-doc", type=int, default=None, help="공공 몫을 채울 때 한 문서 최대 수")
    ap.add_argument("--raw", action="store_true", help="질문 끝의 공통 요청을 빼지 않고 그대로 검색")
    ap.add_argument("--questions", default=str(QUESTIONS), help="평가 질문 CSV")
    args = ap.parse_args()

    qs = pd.read_csv(args.questions, encoding="utf-8-sig", dtype=str).fillna("")
    qs = qs[qs["근거위치"].str.strip() != ""]
    names = sorted(p.name for p in INDEX.iterdir() if (p / "vectors.npy").exists())
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    cache = EmbedCache()
    setting = (f"+doc{args.per_doc}" if args.per_doc else "") + (f"+ac{args.academic}" if args.academic else "") \
        + (f"+pub{args.public}" if args.public else "") \
        + ("+en" if args.translate else "") + ("x" if args.expand else "") \
        + (f"+ppd{args.public_per_doc}" if args.public_per_doc else "") + ("+raw" if args.raw else "")

    rows = []
    for name in names:
        ret = Retriever(name, cache)
        for _, q in qs.iterrows():
            gold = parse_gold(q["근거위치"])
            reachable = any(hits(c, gold) for c in ret.chunks)
            text = q["질문"] if args.raw else search_text(q["질문"])
            en = translate_query(text, expand=args.expand) if args.translate else None
            top = ret.search(text, k=args.k, per_doc=args.per_doc, academic=args.academic,
                             public=args.public, query_en=en, public_per_doc=args.public_per_doc)
            first = next((c["rank"] for c in top if hits(c, gold)), None)
            rows.append({
                "전략": name + setting, "번호": q["번호"], "유형": q["유형"], "질문": q["질문"],
                "근거위치": q["근거위치"], "추출됨": reachable,
                "문서적중": any(c["doc_id"] in gold for c in top),
                "쪽적중": first is not None, "정답순위": first or "",
                "학술수": sum(c.get("doc_type") == "학술" for c in top),
                "상위결과": " / ".join(f"{c['doc_id']}:{c['label']}({c['score']:.2f})" for c in top),
            })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"retrieval_{Path(args.questions).stem}_{datetime.now():%Y%m%d_%H%M}_k{args.k}{setting.replace('+', '_')}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    summary = df.groupby("전략").agg(
        문항=("번호", "count"), 추출됨=("추출됨", "sum"),
        문서적중=("문서적중", "sum"), 쪽적중=("쪽적중", "sum"), 평균학술수=("학술수", "mean"))
    reach = df[df["추출됨"]].groupby("전략")["쪽적중"].agg(["sum", "count"])
    summary["평균학술수"] = summary["평균학술수"].round(1)
    summary["쪽적중률"] = (summary["쪽적중"] / summary["문항"]).map("{:.0%}".format)
    summary["추출된 문항 중 쪽적중률"] = (reach["sum"] / reach["count"]).map("{:.0%}".format)
    print(f"상위 {args.k}개 기준, 근거위치가 있는 문항 {len(qs)}개\n")
    print(summary.to_string())
    print(f"\n저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
