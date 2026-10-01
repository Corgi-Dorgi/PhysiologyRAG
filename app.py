"""
운동지도 보조 RAG 테스트 화면

질문을 넣으면 검색된 근거와 답변을 나란히 보여주고, 답변이 원하는 형식·근거 규칙을 지키는지
자동 점검과 사람 평가를 함께 기록한다. 기록은 logs/qa_log.sqlite 에 쌓인다.

실행: streamlit run app.py      (또는 python -m streamlit run app.py)
필요: python chunk.py → python index.py, .env 에 OPENAI_API_KEY
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd
import streamlit as st

from blank_test import MODEL
from rag import INDEX, ROOT, Retriever, _client, cite
from rag_test import SCOPE_PROMPT, SYSTEM_PROMPT, run

LOG_DB = ROOT / "logs" / "qa_log.sqlite"
QUESTION_FILES = {
    "평가셋": ROOT / "eval" / "questions.csv",
}
SECTIONS = ["상황 정리", "판단", "주의사항", "근거"]


# ── 준비 ─────────────────────────────────────────────
@st.cache_resource
def load_retriever(strategy):
    return Retriever(strategy)


@st.cache_resource
def load_client():
    return _client()


@st.cache_data
def load_questions(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def strategies():
    return sorted(p.name for p in INDEX.iterdir() if (p / "vectors.npy").exists()) if INDEX.exists() else []


def log_db():
    LOG_DB.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(LOG_DB)
    db.execute("""CREATE TABLE IF NOT EXISTS qa (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, question TEXT, query_en TEXT, settings TEXT,
        answer TEXT, sources TEXT, auto_checks TEXT, rating TEXT, memo TEXT)""")
    return db


# ── 자동 점검 ─────────────────────────────────────────
def auto_checks(answer, chunks):
    """답변이 형식·근거 규칙을 지키는지 기계적으로 확인할 수 있는 것만 본다 (내용이 맞는지는 사람이 판단)."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    valid = [n for n in cited if 1 <= n <= len(chunks)]
    by_rank = {c["rank"]: c for c in chunks}
    academic = [n for n in valid if by_rank[n].get("doc_type") == "학술"]
    return {
        "형식 칸": sum(s in answer for s in SECTIONS),
        "인용한 근거 번호": cited,
        "없는 번호 인용": [n for n in cited if n not in valid],
        "학술 근거 인용 수": len(academic),
        "되묻기 포함": bool(re.search(r"\?|알려 주|알려주|확인이 필요|정보가 필요", answer)),
        "의료기관 안내": bool(re.search(r"의사|의료기관|전문의|병원|상담", answer)),
    }


def show_checks(checks, n_chunks):
    c1, c2, c3 = st.columns(3)
    c1.metric(f"형식 {len(SECTIONS)}칸 중", f"{checks['형식 칸']}칸")
    c2.metric("인용한 근거", f"{len(checks['인용한 근거 번호'])} / {n_chunks}개")
    c3.metric("학술 근거 인용", f"{checks['학술 근거 인용 수']}개")
    if checks["없는 번호 인용"]:
        st.error(f"검색 결과에 없는 번호를 인용함: {checks['없는 번호 인용']}")
    flags = [f"되묻기 {'있음' if checks['되묻기 포함'] else '없음'}",
             f"의료기관 안내 {'있음' if checks['의료기관 안내'] else '없음'}"]
    st.caption(" · ".join(flags) + "  (기계적 점검이라 참고용)")


# ── 화면 ─────────────────────────────────────────────
st.set_page_config(page_title="운동지도 보조 RAG 테스트", layout="wide")
st.title("운동지도 보조 RAG 테스트")
st.caption("당뇨·비만·고혈압 회원의 질환 유형·연령·성별·운동 경력이 조합된 상황에서 무엇을 주의해야 하는지 근거와 함께 확인합니다. "
           "이 도구는 의료 조언을 제공하지 않습니다.")

names = strategies()
if not names:
    st.error("검색 인덱스가 없습니다. 먼저 `python chunk.py` → `python index.py`를 실행하세요.")
    st.stop()

with st.sidebar:
    st.header("검색 설정")
    strategy = st.selectbox("청킹 방식", names, index=names.index("section_ctx") if "section_ctx" in names else 0)
    k = st.slider("가져올 근거 수", 3, 15, 10)
    public = st.slider("공공 자료 최소 개수", 0, k, min(5, k), help="한국 기준 수치(주 150분 등)를 위해 확보")
    per_doc = st.slider("한 문서에서 최대", 1, 5, 2, help="같은 문서 조각이 상위를 독점하지 않게")
    translate = st.toggle("영어 번역 검색", value=True, help="영어 논문을 찾기 위해 질문을 영어로도 검색")
    st.divider()
    st.caption(f"답변 모델: {MODEL} (온도 0)")
    with st.expander("시스템 프롬프트 보기"):
        st.text(SYSTEM_PROMPT)
    with st.expander("범위 판정 프롬프트 보기"):
        st.text(SCOPE_PROMPT)

st.subheader("질문")
source = st.radio("질문 고르기", ["직접 입력", *QUESTION_FILES], horizontal=True)
preset = {}
if source == "직접 입력":
    question = st.text_area("질문", placeholder="예: 50대 고혈압 판정을 받은 숙련자 남성 회원이 중량을 더 올리고 싶다는데 어떻게 생각해?", height=80)
else:
    qs = load_questions(QUESTION_FILES[source])
    labels = [f"{r['번호']}. [{r.get('질환', r.get('유형', ''))} · {r.get('대상', '')} · {r.get('유형', '')}] {r['질문']}"
              for _, r in qs.iterrows()]
    pick = st.selectbox("평가 문항", range(len(labels)), format_func=lambda i: labels[i])
    preset = qs.iloc[pick].to_dict()
    question = st.text_area("질문 (고쳐서 물어볼 수 있음)", value=preset["질문"], height=80)
    with st.expander("이 문항의 채점 기준"):
        st.write(f"**채점 포인트:** {preset.get('채점포인트') or '(아직 없음)'}")
        st.write(f"**근거:** {preset.get('근거') or '-'}")
        st.write(f"**근거위치:** `{preset.get('근거위치') or '-'}`")

if st.button("질문하기", type="primary", disabled=not question.strip()):
    with st.spinner("범위를 판정하고 근거를 찾아 답변을 만드는 중..."):
        out = run(question, load_retriever(strategy), load_client(),
                  k=k, per_doc=per_doc, public=public, translate=translate)
    st.session_state["result"] = {
        "question": question, **out,
        "settings": {"strategy": strategy, "k": k, "public": public, "per_doc": per_doc,
                     "translate": translate, "model": MODEL, "scope": out["scope"]},
        "gold": preset.get("근거위치", ""), "saved": False,
    }

res = st.session_state.get("result")
if res:
    gold_docs = {g.split(":")[0].strip() for g in re.split(r"[;\n]", res["gold"]) if g.strip()}
    left, right = st.columns([3, 2])

    with left:
        st.subheader("답변")
        scope = res["scope"]
        if scope["scope"] == "out":
            st.error(f"범위 밖으로 판정 → 검색·답변 생략. {scope.get('reason', '')}")
        elif scope["scope"] == "partial":
            st.warning(f"일부만 범위 안 · 답하지 않을 부분: {', '.join(scope.get('out_of_scope', []))}. "
                       f"{scope.get('reason', '')}")
            st.caption(f"범위 밖을 뺀 질문으로 검색·답변: {res.get('asked')}")
        else:
            st.caption(f"범위 안 · {scope.get('reason', '')}")
        if res["query_en"]:
            st.caption(f"영어 검색어: {res['query_en']}")
        st.markdown(res["answer"])
        st.divider()
        checks = auto_checks(res["answer"], res["chunks"])
        if scope["scope"] != "out":
            st.subheader("자동 점검")
            show_checks(checks, len(res["chunks"]))
        if gold_docs:
            found = sorted(gold_docs & {c["doc_id"] for c in res["chunks"]})
            (st.success if found else st.warning)(
                f"기대 근거 문서 {len(gold_docs)}개 중 검색됨: {', '.join(found) or '없음'}")

    with right:
        st.subheader(f"검색된 근거 {len(res['chunks'])}개")
        n_ac = sum(c.get("doc_type") == "학술" for c in res["chunks"])
        st.caption(f"학술 {n_ac}개 · 공공 {len(res['chunks']) - n_ac}개 · 점수는 질문과의 유사도(0~1)")
        cited = set(checks["인용한 근거 번호"])
        for c in res["chunks"]:
            mark = "✅ 인용" if c["rank"] in cited else "· 미인용"
            star = " ★기대 문서" if c["doc_id"] in gold_docs else ""
            with st.expander(f"[{c['rank']}] {mark}{star} · {cite(c)} ({c['score']:.2f})"):
                tags = [c.get("doc_type", "")] + c.get("diseases", []) + c.get("groups", [])
                st.caption(" · ".join(t for t in tags if t))
                st.text(c["text"])

    st.divider()
    st.subheader("사람 평가")
    st.caption("답변이 원하는 대로 나왔는지 항목별로 표시하고 저장하면 logs/qa_log.sqlite 에 기록됩니다.")
    with st.form("rating"):
        items = ["범위 판정이 맞음", "상황 정리·되묻기", "조합에 맞춘 판단", "주의사항", "운동 시 생길 수 있는 문제",
                 "근거가 실제 내용과 일치"]
        cols = st.columns(len(items))
        rating = {it: col.radio(it, ["좋음", "부족", "틀림", "해당 없음"], index=3, key=f"r_{it}")
                  for it, col in zip(items, cols)}
        memo = st.text_area("메모 (틀린 내용, 빠진 근거 등)")
        if st.form_submit_button("평가 저장"):
            with closing(log_db()) as db:
                db.execute(
                    "INSERT INTO qa (ts, question, query_en, settings, answer, sources, auto_checks, rating, memo) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(timespec="seconds"), res["question"], res["query_en"],
                     json.dumps(res["settings"], ensure_ascii=False), res["answer"],
                     json.dumps([{"rank": c["rank"], "chunk_id": c["chunk_id"], "score": round(c["score"], 4)}
                                 for c in res["chunks"]], ensure_ascii=False),
                     json.dumps(checks, ensure_ascii=False), json.dumps(rating, ensure_ascii=False), memo))
                db.commit()
            st.success("저장했습니다.")

with st.expander("저장된 평가 기록"):
    if LOG_DB.exists():
        with closing(log_db()) as db:
            df = pd.read_sql("SELECT id, ts, question, rating, memo FROM qa ORDER BY id DESC", db)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("CSV로 받기", df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                           "qa_log.csv", "text/csv")
    else:
        st.caption("아직 기록이 없습니다.")
