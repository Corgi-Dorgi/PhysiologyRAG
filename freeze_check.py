#!/usr/bin/env python3
"""
동결 확인: 홀드아웃을 돌리기 전에 프롬프트·검색 설정·데이터·인덱스가 동결 때와 같은지 지문(SHA256 앞 12자리)으로 확인한다.

실행: python freeze_check.py            # eval/freeze.json 과 비교 (다르면 종료 코드 1)
      python freeze_check.py --write    # 지금 상태를 동결 기준으로 저장 (동결할 때 한 번만)
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import rag
import rag_test
from blank_test import MODEL, SYSTEM_PROMPT as BLANK_PROMPT, TEMPERATURE

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "eval" / "freeze.json"
FILES = ["data/chunks/fixed.jsonl", "index/fixed/vectors.npy", "data/manifest.csv"]


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def snapshot():
    defaults = inspect.signature(rag_test.run).parameters
    return {
        "모델": {"답변": MODEL, "온도": TEMPERATURE, "임베딩": rag.EMBED_MODEL},
        "검색": {"청킹": "fixed", **{k: defaults[k].default for k in ("k", "per_doc", "public", "public_per_doc", "translate")},
                 "질문 끝 공통 요청 제거": h(rag.GENERIC_ASK)},
        "프롬프트": {"답변": h(rag_test.SYSTEM_PROMPT), "범위 판정": h(rag_test.SCOPE_PROMPT),
                     "진단 요청": h(rag_test.DIAG_RULE), "약 조절 판정": h(rag_test.MED_CHECK_PROMPT),
                     "조건 누락": h(rag_test.ASKBACK_RULE), "빈손": h(BLANK_PROMPT),
                     "번역": h(inspect.getsource(rag.translate_query))},
        "규칙": {"범위 단어": h(rag_test.SCOPE_TERMS), "범위 밖 질환": h(rag_test.OUT_DISEASES),
                 "생애 단계": h(rag_test.LIFE_STAGES), "회원 조건": h(rag_test.COND_TERMS),
                 "진단 요청": h(rag_test.DIAG_TERMS), "판정 표현": h(rag_test.DIAG_BANNED)},
        "파일": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:12] for f in FILES},
    }


def diff(a, b, path=""):
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += diff(x, y, f"{path}{k} > ")
        elif x != y:
            out.append(f"{path}{k}: 동결 {x} → 지금 {y}")
    return out


def main():
    now = snapshot()
    if "--write" in sys.argv:
        FREEZE.write_text(json.dumps(now, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"동결 기준 저장: {FREEZE.relative_to(ROOT)}")
        return 0
    if not FREEZE.exists():
        print("동결 기준이 없음 → python freeze_check.py --write")
        return 1
    changes = diff(json.loads(FREEZE.read_text(encoding="utf-8")), now)
    if changes:
        print("동결 이후 바뀐 것:")
        print("\n".join("  " + c for c in changes))
        return 1
    print("동결 상태 그대로입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
