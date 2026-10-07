"""
빈손 테스트: 문서 없이 LLM에게만 평가셋 질문을 던져서 기준점을 잡는다.
실행: python blank_test.py
      python blank_test.py --questions eval/holdout.csv     # 홀드아웃 등 다른 질문 파일
필요: .env 에 OPENAI_API_KEY, eval/questions.csv
"""
import argparse
import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── 설정 (RAG 테스트 때도 똑같이 쓸 값) ─────────────
MODEL = "gpt-4o-mini"
TEMPERATURE = 0

BASE = os.path.dirname(os.path.abspath(__file__))  # 이 파일이 있는 폴더 기준
QUESTIONS_PATH = os.path.join(BASE, "eval", "questions.csv")
RESULTS_DIR = os.path.join(BASE, "eval", "results")

SYSTEM_PROMPT = """너는 헬스장 트레이너를 돕는 운동지도 보조 AI야.
당뇨·비만·고혈압 회원 관련 질문에 답해.
답마다 근거 문서명과 쪽 번호를 적어.
진단이나 약 조절은 하지 말고 의료기관을 안내해.
조건이 부족하면 먼저 되물어."""


def ask(client, question):
    """질문 하나를 독립적으로 보내고 답 텍스트를 돌려준다."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=QUESTIONS_PATH, help="질문 CSV (번호, 질문 칸 필요)")
    args = ap.parse_args()
    load_dotenv(os.path.join(BASE, ".env"))
    if not os.getenv("OPENAI_API_KEY"):
        print("키 로드 실패 - .env 확인")
        return

    client = OpenAI()
    df = pd.read_csv(args.questions, encoding="utf-8-sig")
    print(f"질문 {len(df)}개, 모델 {MODEL}\n")

    answers = []
    for _, row in df.iterrows():
        print(f"[{row['번호']}/{len(df)}] {row['질문'][:30]}...")
        try:
            answers.append(ask(client, row["질문"]))
        except Exception as e:  # 한 문항이 실패해도 나머지는 계속
            answers.append(f"ERROR: {e}")
            print(f"  실패: {e}")
        time.sleep(0.5)  # 너무 빠르게 연속 호출하지 않기

    df["답변"] = answers
    df["모델"] = MODEL
    # 결과 파일을 엑셀로 열어서 직접 채울 채점 칸
    for col in ["충족수", "근거표시", "되묻기", "경계위반", "메모"]:
        df[col] = ""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    qname = os.path.splitext(os.path.basename(args.questions))[0]
    tag = "" if qname == "questions" else f"{qname}_"      # 기본 평가셋은 예전 파일 이름 그대로
    out = os.path.join(RESULTS_DIR, f"blank_{MODEL}_{tag}{stamp}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    failed = sum(a.startswith("ERROR") for a in answers)
    print(f"\n저장: {out}")
    print(f"성공 {len(answers) - failed}개 / 실패 {failed}개")


if __name__ == "__main__":
    main()