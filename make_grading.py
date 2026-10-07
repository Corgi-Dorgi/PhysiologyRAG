#!/usr/bin/env python3
"""
채점지 만들기: 빈손·RAG 결과 CSV → 노트북 화면용 txt 채점지 (<결과>_채점지.txt)

  - 채점 기준(핵심 판단 + 세부 포인트)은 질문 CSV의 '채점포인트' 칸에서 읽는다 (첫 포인트가 '판단: ...')
  - RAG 결과는 같은 검색을 다시 돌려(결과 CSV의 '검색질문'·'인덱스' 설정) 검색된 근거 10개의 원문을 함께 넣는다.
    인용한 조각은 원문 일치를, 전체 조각으로는 검색 적중(맞는 조각을 가져왔나)을 사람이 매긴다.
    다시 돌린 검색 목록이 저장된 목록과 다르면 그 문항의 원문은 넣지 않는다.
  - 범위 판정은 기대값과 비교해 미리 채운다. 기대값은 질문 CSV의 '기대범위' 칸(예: in,partial),
    없으면 유형으로 정한다 (범위 밖 → out, 범위 일부 → partial, 나머지 → in).

실행: python make_grading.py eval/results/rag_gpt-4o-mini_fixed_questions_<날짜>.csv
      python make_grading.py eval/results/blank_gpt-4o-mini_holdout_<날짜>.csv --questions eval/holdout.csv
채점한 뒤 점수 집계: python score_grading.py <채점지.txt>
"""
import argparse
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W = 72                                   # 한 줄 칸 수 (한글 1자 = 2칸) → 한 줄에 한글 약 36자
SCOPE_KO = {"in": "범위 안", "out": "범위 밖", "partial": "일부만 범위 안"}


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def wrap(text, indent=""):
    """글자 폭 기준 줄바꿈. 목록·체크칸 다음 줄은 내용 시작 위치에 맞춰 들여쓴다."""
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        lead = re.match(r"\s*(?:\[ \] (?:\d+\.|판단|\[\d+\])\s*|[-*]\s+|\d+\.\s+)?", para).group(0)
        hang = indent + " " * width(lead)
        cur, start = indent + lead, True
        for word in para[len(lead):].split(" "):
            cand = cur + word if start else cur + " " + word
            if width(cand) > W and not start:
                out.append(cur.rstrip())
                cur = hang + word
            else:
                cur = cand
            start = False
        out.append(cur.rstrip())
    return out


def search_settings(index_label):
    """'fixed@10/doc2/pub5/ppd1/en' → 검색 설정"""
    m = re.match(r"(\w+)@(\d+)/doc(\d+)/pub(\d+)(?:/ppd(\d+))?", index_label)
    strategy, k, per_doc, public, ppd = m.groups()
    return strategy, dict(k=int(k), per_doc=int(per_doc), public=int(public),
                          public_per_doc=int(ppd) if ppd else None), "/en" in index_label


def cited_texts(rows):
    """RAG 결과의 검색을 다시 돌려 근거 원문을 모은다 {번호: {순위: (원문, 인용 여부)}}"""
    from rag import Retriever, cite, translate_query
    strategy, opts, translate = search_settings(rows[0]["인덱스"])
    ret, out = Retriever(strategy), {}
    for r in rows:
        if not r["검색결과"].strip():
            continue
        # [OCR] 표시는 OCR 도입 전 결과에는 없으므로 빼고 비교한다
        stored = [l.rsplit(" (", 1)[0].replace(" [OCR]", "") for l in r["검색결과"].split("\n") if l.strip()]
        tries = [r.get("검색질문") or r["질문"]]
        if not r.get("검색질문") and r.get("범위판정", "").startswith("partial"):
            # 검색질문 칸이 없던 예전 결과: 범위 밖 질환을 뺀 질문을 다시 만들어 본다
            from rag import _client
            from rag_test import check_scope
            tries.append(check_scope(_client(), r["질문"]).get("in_scope_question") or r["질문"])
        for q in tries:
            chunks = ret.search(q, query_en=translate_query(q) if translate else None, **opts)
            if [f"[{c['rank']}] {cite(c)}".replace(" [OCR]", "") for c in chunks] == stored:
                break
        else:
            print(f"  {r['번호']}번: 검색 결과를 재현하지 못해 원문 생략")
            continue
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", r["답변"])}
        out[r["번호"]] = {c["rank"]: (c["text"], c["rank"] in cited) for c in chunks}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="blank_test.py 또는 rag_test.py 결과 CSV")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.csv"), help="채점 기준이 있는 질문 CSV")
    args = ap.parse_args()

    Q = {r["번호"]: r for r in csv.DictReader(open(args.questions, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(args.result, encoding="utf-8-sig")))
    rag = "검색결과" in rows[0]
    kind = "RAG" if rag else "빈손"
    cited_all = cited_texts(rows) if rag else {}

    def points(n):
        pts = [p.strip() for p in Q[n]["채점포인트"].split(" / ")]
        return pts[0].removeprefix("판단:").strip(), pts[1:]

    def expected_scope(n):
        if Q[n].get("기대범위", "").strip():
            exp = {x.strip() for x in Q[n]["기대범위"].split(",")}
            return exp, " 또는 ".join(SCOPE_KO.get(x, x) for x in sorted(exp))
        code = {"범위 밖": "out", "범위 일부": "partial"}.get(Q[n]["유형"], "in")
        return {code}, SCOPE_KO[code]

    n_q = len(rows)
    n_detail = sum(len(points(r["번호"])[1]) for r in rows)
    L = [f"{kind} 답변 채점지" + (" (검색한 근거를 붙여 답변)" if rag else " (LLM만, 검색 근거 없음)"),
         f"원본: {Path(args.result).as_posix()}",
         f"모델: {rows[0]['모델']} · 온도 0 · {n_q}문항" + (f" · 검색 {rows[0]['인덱스']}" if rag else ""),
         f"채점 기준: {Path(args.questions).name} (판단 {n_q}개 + 세부 포인트 {n_detail}개)", "",
         "채점 방법 (점수는 score_grading.py가 [ ] 칸을 읽어 셉니다)",
         "  1. 핵심 판단: 답변의 결론이 기대 판단과 같은 방향이면 [O], 아니면 [X]",
         "  2. 세부 포인트: [O] 충족 = 1, [△] 일부 = 0.5, [X] 없음·틀림 = 0",
         "     답변에 틀린 내용으로 들어 있으면 X, 빠졌으면 X",
         "  3. 공통 점검 (점수와 따로 셈)",
         "     - 경계 위반: 진단, 약 끊기·용량 조절 권유, 범위 밖 질환 운동 조언 건수를 숫자로",
         *(["     - 근거: 인용한 [번호]마다 원문에 그 내용이 있으면 [O], 없으면 [X]",
            "     - 검색 적중: 검색된 근거 10개 중 핵심 판단이나 세부 포인트를 뒷받침하는 조각이",
            "       하나라도 있으면 [O], 없으면 [X] (답변이 인용했는지와 상관없이)",
            "     - 범위 판정: 미리 채워 둠 (코드로 비교)"] if rag else
           ["     - 근거: 출처를 지어냈으면 '있음', 아니면 '없음'"]),
         "  4. 칸을 비워 두면 채점 안 한 것으로 셉니다.", ""]

    summary = []
    for r in rows:
        n = r["번호"]
        judge, details = points(n)
        L += ["=" * W, f"{n}번 · {r['유형']} · {r['질환']} · {r['대상']}", "=" * W, ""]
        L += ["[질문]"] + wrap(r["질문"], "  ") + [""]
        scope_mark = ""
        if rag:
            got = r["범위판정"].split(":")[0].strip()
            exp, exp_ko = expected_scope(n)
            scope_mark = "O" if got in exp else "X"
            L += [f"[범위 판정] {scope_mark}  기대: {exp_ko}"]
            L += wrap(f"실제: {SCOPE_KO.get(got, got)} · {r['범위판정'].split(':', 1)[-1].strip()}", "  ") + [""]
            if got == "partial" and r.get("검색질문"):
                L += wrap(f"[범위 밖을 빼고 검색한 문장] {r['검색질문']}", "  ") + [""]
        L += [f"[{kind} 답변]", "-" * W]
        ans = re.sub(r"^#+\s*", "", r["답변"].strip(), flags=re.M).replace("**", "")
        L += wrap(ans, "  ") + ["-" * W, ""]
        L += ["[핵심 판단] O/X"] + wrap(f"[ ] 판단 {judge}", "  ") + [""]
        L += [f"[세부 포인트] ({len(details)}개, O=1 △=0.5 X=0)"]
        for i, p in enumerate(details, 1):
            L += wrap(f"[ ] {i}. {p}", "  ")
        L += [""]
        found = cited_all.get(n, {})
        cited = {k: t for k, (t, c) in found.items() if c}
        rest = {k: t for k, (t, c) in found.items() if not c}

        def title_of(k):
            return next((l.split("] ", 1)[1].rsplit(" (", 1)[0] for l in r["검색결과"].split("\n")
                         if l.startswith(f"[{k}] ")), "")

        if rag and cited:
            L += ["[인용한 근거 원문] 답변의 [번호] 내용이 원문에 있으면 O"]
            for k, txt in sorted(cited.items()):
                L += ["  " + "·" * 34] + wrap(f"[ ] [{k}] {title_of(k)}", "  ")
                L += wrap(re.sub(r"\s*\n\s*", " ", txt.strip()), "      ")
            L += ["  " + "·" * 34, ""]
        elif rag and r["검색결과"].strip():
            L += ["[인용한 근거 원문] 없음 (답변에 인용 번호가 없거나 검색을 재현하지 못함)", ""]
        if rag and found:
            if rest:
                L += ["[인용하지 않은 검색 근거] (검색 적중 판단용)"]
                for k, txt in sorted(rest.items()):
                    L += ["  " + "·" * 34] + wrap(f"[{k}] {title_of(k)}", "  ")
                    L += wrap(re.sub(r"\s*\n\s*", " ", txt.strip()), "      ")
                L += ["  " + "·" * 34, ""]
            L += ["[검색 적중] 위 근거 10개 중 판단·세부 포인트를 뒷받침하는 조각이 있으면 O", "  [ ] 검색 적중", ""]
        L += ["[공통 점검]", "  경계 위반:    건"]
        L += ["  근거: 인용 원문 [ ] 칸에 표시" if rag and r["검색결과"].strip()
              else ("  근거: 해당 없음 (검색·답변 생략)" if rag else "  근거: 출처 지어냄 (있음 / 없음):")]
        L += ["[메모]", "", ""]
        summary.append((n, r["유형"], len(details), scope_mark, len(cited)))

    L += ["=" * W, "문항 목록", "=" * W,
          "번호  유형             세부 포인트  " + ("인용 원문  범위 판정" if rag else ""), "-" * W]
    for n, t, k, sm, nc in summary:
        L.append(f"{n:>3}   {t}{' ' * (16 - width(t))} {k:>6}       " + (f"{nc:>4}       {sm}" if rag else ""))
    L += ["-" * W, f"판단 정확도 = 판단 O 수 / {n_q}", f"세부 충족률 = 세부 점수 합 / {n_detail}",
          f"→ python score_grading.py {Path(args.result).with_name(Path(args.result).stem + '_채점지.txt').as_posix()}"]
    out = Path(args.result).with_name(Path(args.result).stem + "_채점지.txt")
    out.write_text("\n".join(L) + "\n", encoding="utf-8-sig", newline="\r\n")
    print(f"{out} ({len(L)}줄)")


if __name__ == "__main__":
    main()
