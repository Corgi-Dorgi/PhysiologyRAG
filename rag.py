"""
검색 공용 모듈: 임베딩 캐시, 인덱스 만들기·불러오기, 질문으로 청크 찾기

인덱스는 청킹 전략마다 index/<전략>/ 에 저장한다.
  vectors.npy   청크 임베딩 (정규화된 float32, 청크 순서와 같음)
  chunks.jsonl  청크 목록 (data/chunks/<전략>.jsonl 복사본)
  meta.json     임베딩 모델, 청크 수, 만든 날짜

청크가 수천 개 규모라 벡터 DB 없이 행렬 곱 한 번으로 전체 코사인 유사도를 구한다.
임베딩은 index/embed_cache.sqlite 에 (모델, 본문) 해시로 저장해서 다시 만들 때 비용이 들지 않는다.
"""
import hashlib
import re
import json
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data" / "chunks"
INDEX = ROOT / "index"
EMBED_MODEL = "text-embedding-3-small"
BATCH = 100


def _client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    return OpenAI()


class EmbedCache:
    def __init__(self, model=EMBED_MODEL):
        INDEX.mkdir(exist_ok=True)
        self.model = model
        # Streamlit처럼 같은 객체를 여러 스레드에서 쓰는 경우가 있어 스레드 제한을 풀고 잠금으로 보호한다
        self.db = sqlite3.connect(INDEX / "embed_cache.sqlite", check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.client = None
        self.new_tokens = 0

    def key(self, text):
        return hashlib.sha1(f"{self.model}\n{text}".encode("utf-8")).hexdigest()

    def embed(self, texts, log=None):
        keys = [self.key(t) for t in texts]
        found = {}
        for i in range(0, len(keys), 900):
            part = keys[i:i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
            with self.lock:
                found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in self.db.execute(q, part)})
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in found]
        if todo:
            self.client = self.client or _client()
            for i in range(0, len(todo), BATCH):
                part = todo[i:i + BATCH]
                resp = self._create([t for _, t in part], log)
                self.new_tokens += resp.usage.total_tokens
                rows = []
                for (k, _), d in zip(part, resp.data):
                    v = np.asarray(d.embedding, dtype=np.float32)
                    found[k] = v
                    rows.append((k, v.tobytes()))
                with self.lock:
                    self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
                    self.db.commit()
                if log:
                    log(f"  임베딩 {min(i + BATCH, len(todo))}/{len(todo)}")
        m = np.vstack([found[k] for k in keys])
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def _create(self, inputs, log=None, tries=6):
        """분당 토큰 한도(429)에 걸리면 기다렸다가 다시 요청한다."""
        from openai import RateLimitError
        for n in range(tries):
            try:
                return self.client.embeddings.create(model=self.model, input=inputs)
            except RateLimitError:
                if n == tries - 1:
                    raise
                wait = 5 * 2 ** n
                if log:
                    log(f"  요청 한도 초과 → {wait}초 대기")
                time.sleep(wait)


def build_index(strategy, cache=None, log=print):
    src = CHUNKS / f"{strategy}.jsonl"
    chunks = [json.loads(l) for l in src.open(encoding="utf-8")]
    cache = cache or EmbedCache()
    vecs = cache.embed([c["text"] for c in chunks], log=log)
    out = INDEX / strategy
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vecs)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "strategy": strategy, "model": cache.model, "chunks": len(chunks),
        "built": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return len(chunks)


class Retriever:
    def __init__(self, strategy, cache=None):
        d = INDEX / strategy
        if not (d / "vectors.npy").exists():
            raise FileNotFoundError(f"{d} 인덱스 없음: python index.py 먼저 실행")
        self.strategy = strategy
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.vecs = np.load(d / "vectors.npy")
        self.chunks = [json.loads(l) for l in (d / "chunks.jsonl").open(encoding="utf-8")]
        self.cache = cache or EmbedCache(self.meta["model"])

    def search(self, query, k=5, per_doc=None, academic=0, public=0, query_en=None, public_per_doc=None):
        """질문과 비슷한 청크 k개.
        per_doc   한 문서에서 가져올 최대 청크 수 (같은 문서·섹션 청크가 상위를 독점하지 않게)
        academic  결과 중 최소 이만큼은 [학술] 청크로 채운다 (논문을 먼저 근거로 쓰기 위해)
        public    결과 중 최소 이만큼은 [공공] 청크로 채운다. 한국어 질문 점수로 고른다
                  (영어 번역 검색을 켜면 영어 논문 점수가 높게 나와 한국 기준 자료가 밀려나기 때문)
        query_en  질문의 영어 번역. 주면 두 질문 중 더 비슷한 쪽 점수를 쓴다 (영어 논문 검색용)
        public_per_doc  공공 자료 몫을 채울 때 한 문서에서 가져올 최대 수 (한 안내서가 공공 몫을 독차지하지 않게)"""
        qs = [query] + ([query_en] if query_en else [])
        sims = self.vecs @ self.cache.embed(qs).T
        scores = sims.max(axis=1)
        order = np.argsort(-scores)
        order_ko = np.argsort(-sims[:, 0])
        used = {}

        def take(pool, n, picked, cap=per_doc):
            for i in pool:
                if len(picked) >= n:
                    break
                doc = self.chunks[i]["doc_id"]
                if i in picked or (cap and used.get(doc, 0) >= cap):
                    continue
                used[doc] = used.get(doc, 0) + 1
                picked.append(i)
            return picked

        picked = []
        if public:
            take([i for i in order_ko if self.chunks[i].get("doc_type") == "공공"], min(public, k), picked,
                 cap=public_per_doc or per_doc)
        if academic:
            take([i for i in order if self.chunks[i].get("doc_type") == "학술"], min(len(picked) + academic, k), picked)
        take(order, k, picked)
        picked.sort(key=lambda i: -scores[i])
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r}
                for r, i in enumerate(picked, 1)]


# 질문 끝의 공통 요청("지도할 때 주의해야 할 점은?")은 모든 질문에 같아서 검색 뜻만 흐리므로 빼고 검색한다
GENERIC_ASK = (r"\s*(남녀 차이를 고려해서\s*)?((지도|운동|응대)할\s*때\s*)?(오늘\s*)?"
               r"주의해야\s*할\s*점(은|을|도)?(\s*알려\s*주세요)?\s*[.?]?\s*$")


def search_text(question):
    """검색에 쓸 질문: 상황 부분만 남긴다"""
    s = re.sub(GENERIC_ASK, "", question).strip()
    return s or question


EXPAND_PROMPT = ("Translate the user's Korean question into one English search query for exercise physiology "
                 "and clinical literature. Keep medical terms precise (e.g. 원발성 고혈압 = essential hypertension, "
                 "1형 당뇨 = type 1 diabetes). Then append the specific physiological risks, complications or "
                 "mechanisms this situation implies, even if the question does not name them (e.g. "
                 "'heart rate does not rise' -> cardiac autonomic neuropathy; 'blood pressure rose too much "
                 "on a stress test' -> exaggerated blood pressure response to exercise; 'appetite increases' -> "
                 "exercise and appetite regulation). Output only the English query.")


def translate_query(question, expand=False):
    """한국어 질문 → 영어 검색어 (영어 논문을 찾기 위한 용도, 결과는 캐시)
    expand=True: 상황이 암시하는 위험·기전 이름을 검색어에 덧붙인다 (질문에 위험 이름이 없을 때 놓치는 문제 대응)
    호출마다 연결을 열고 닫아서 어느 스레드에서 불러도 안전하다."""
    from blank_test import MODEL
    INDEX.mkdir(exist_ok=True)
    with closing(sqlite3.connect(INDEX / "embed_cache.sqlite")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS tr (q TEXT PRIMARY KEY, en TEXT)")
        key = ("expand|" + question) if expand else question
        row = db.execute("SELECT en FROM tr WHERE q = ?", (key,)).fetchone()
        if row:
            return row[0]
        resp = _client().chat.completions.create(
            model=MODEL, temperature=0,
            messages=[{"role": "system", "content": EXPAND_PROMPT if expand else
                       "Translate the user's Korean question into one English search query for "
                       "exercise physiology and clinical literature. Keep medical terms precise "
                       "(e.g. 원발성 고혈압 = essential hypertension, 1형 당뇨 = type 1 diabetes). "
                       "Output only the English query."},
                      {"role": "user", "content": question}])
        en = resp.choices[0].message.content.strip()
        db.execute("INSERT OR REPLACE INTO tr VALUES (?, ?)", (key, en))
        db.commit()
    return en


def cite(chunk):
    """근거 표시 문자열: 문서명, 인쇄 쪽 번호(논문은 섹션)"""
    where = chunk["label"] if not str(chunk["label"]).replace("-", "").isdigit() else f"{chunk['label']}쪽"
    kind = f"[{chunk['doc_type']}] " if chunk.get("doc_type") else ""
    ocr = " [OCR]" if chunk.get("ocr") else ""     # 그림 글자를 읽은 쪽: 원문 대조 필요
    return f"{kind}{chunk['title']} ({chunk['doc_id']}) {where}{ocr}"
