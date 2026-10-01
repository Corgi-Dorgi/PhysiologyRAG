#!/usr/bin/env python3
"""
청킹 전략별 검색 인덱스 만들기 (임베딩: OpenAI text-embedding-3-small)

실행: python index.py                   # data/chunks/ 의 모든 전략
      python index.py --only section_ctx
필요: python chunk.py 를 먼저 실행, .env 에 OPENAI_API_KEY
"""
import argparse

from rag import CHUNKS, EmbedCache, build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="이 전략만 (쉼표로 여러 개)")
    args = ap.parse_args()
    names = sorted(p.stem for p in CHUNKS.glob("*.jsonl"))
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    if not names:
        print("청크 파일 없음: python chunk.py 먼저 실행")
        return
    cache = EmbedCache()
    for name in names:
        print(f"[{name}]")
        n = build_index(name, cache)
        print(f"  청크 {n}개 인덱스 저장")
    print(f"새로 임베딩한 토큰: {cache.new_tokens:,} (나머지는 캐시)")


if __name__ == "__main__":
    main()
