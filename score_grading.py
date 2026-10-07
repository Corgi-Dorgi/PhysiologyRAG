#!/usr/bin/env python3
"""
채점지 점수 집계: 사람이 채운 _채점지.txt 를 읽어 문항별·전체 점수를 낸다.

읽는 칸 (make_grading.py가 만든 칸 그대로)
  [O] 판단 ...            핵심 판단: O = 맞음, X = 틀림
  [O] 1. ... / [△] / [X]  세부 포인트: O = 1, △ = 0.5, X = 0
  [O] [3] 문서명           인용 원문 일치 (RAG만)
  [O] 검색 적중            검색된 근거 10개 중 맞는 조각이 있었나 (RAG만)
  경계 위반:  2 건         경계 위반 건수
  [ ] 로 비워 둔 칸은 채점 안 한 것으로 보고 분모에서 뺀다 (몇 개 비었는지 함께 보여줌)
  O 대신 ○·o, X 대신 ×·x, △ 대신 ▲·0.5 도 받는다.

실행: python score_grading.py eval/results/<결과>_채점지.txt [<다른 채점지> ...]
결과: 화면에 요약, <채점지>_점수.csv 에 문항별 점수
"""
import csv
import re
import sys
from pathlib import Path

MARK = {"O": 1.0, "o": 1.0, "○": 1.0, "X": 0.0, "x": 0.0, "×": 0.0, "△": 0.5, "▲": 0.5, "0.5": 0.5}
BOX = r"\[\s*(O|o|○|X|x|×|△|▲|0\.5)?\s*\]"


def parse(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"(?m)^=+\n(\d+)번 · ", text)[1:]
    rows = []
    for i in range(0, len(blocks), 2):
        n, body = blocks[i], blocks[i + 1]
        body = body.split("\n문항 목록")[0]
        judge = re.search(rf"(?m)^\s*{BOX} 판단", body)
        details = [m.group(1) for m in re.finditer(rf"(?m)^\s*{BOX} \d+\. ", body)]
        cites = [m.group(1) for m in re.finditer(rf"(?m)^\s*{BOX} \[\d+\]", body)]
        viol = re.search(r"경계 위반:\s*(\d+)\s*건", body)
        hit = re.search(rf"(?m)^\s*{BOX} 검색 적중", body)
        scope = re.search(r"\[범위 판정\] (O|X)", body)
        row = {"번호": n,
               "판단": MARK.get(judge.group(1)) if judge and judge.group(1) else None,
               "세부점수": sum(MARK[d] for d in details if d),
               "세부채점": sum(1 for d in details if d), "세부전체": len(details),
               "인용일치": sum(MARK[c] == 1.0 for c in cites if c),
               "인용채점": sum(1 for c in cites if c), "인용전체": len(cites),
               "경계위반": int(viol.group(1)) if viol else None,
               "검색적중": (MARK.get(hit.group(1)) if hit.group(1) else None) if hit else "",
               "범위판정": scope.group(1) if scope else ""}
        rows.append(row)
    return rows


def report(path, rows):
    graded = [r for r in rows if r["판단"] is not None]
    d_done = sum(r["세부채점"] for r in rows)
    d_all = sum(r["세부전체"] for r in rows)
    c_done = sum(r["인용채점"] for r in rows)
    v_rows = [r for r in rows if r["경계위반"] is not None]
    print(f"\n== {path}")
    print(f"판단 정확도      {sum(r['판단'] for r in graded):g} / {len(graded)}"
          + (f" ({sum(r['판단'] for r in graded) / len(graded):.0%})" if graded else "")
          + (f"   (미채점 {len(rows) - len(graded)}문항)" if len(graded) < len(rows) else ""))
    print(f"세부 충족률      {sum(r['세부점수'] for r in rows):g} / {d_done}"
          + (f" ({sum(r['세부점수'] for r in rows) / d_done:.0%})" if d_done else "")
          + (f"   (미채점 {d_all - d_done}개 / 전체 {d_all}개)" if d_done < d_all else ""))
    if any(r["인용전체"] for r in rows):
        print(f"인용 원문 일치   {sum(r['인용일치'] for r in rows)} / {c_done}"
              + (f" ({sum(r['인용일치'] for r in rows) / c_done:.0%})" if c_done else "")
              + (f"   (미채점 {sum(r['인용전체'] for r in rows) - c_done}개)" if c_done < sum(r['인용전체'] for r in rows) else ""))
    h_rows = [r for r in rows if r["검색적중"] not in ("", None)]
    if any(r["검색적중"] != "" for r in rows):
        print(f"검색 적중(조각)  {sum(r['검색적중'] == 1.0 for r in h_rows)} / {len(h_rows)}"
              + (f" ({sum(r['검색적중'] == 1.0 for r in h_rows) / len(h_rows):.0%})" if h_rows else "")
              + (f"   (미채점 {sum(r['검색적중'] is None for r in rows)}문항)" if any(r['검색적중'] is None for r in rows) else ""))
    if v_rows:
        print(f"경계 위반        {sum(r['경계위반'] > 0 for r in v_rows)}문항 / {len(v_rows)}문항 (총 {sum(r['경계위반'] for r in v_rows)}건)")
    if any(r["범위판정"] for r in rows):
        print(f"범위 판정        {sum(r['범위판정'] == 'O' for r in rows)} / {len(rows)}")
    out = Path(path).with_name(Path(path).stem + "_점수.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"문항별: {out}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p, parse(p))


if __name__ == "__main__":
    main()
