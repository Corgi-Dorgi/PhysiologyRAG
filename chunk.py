#!/usr/bin/env python3
"""
파싱·정제·청킹 (3단계)

입력: data/manifest.csv 에서 최종 판정이 '사용'인 문서와 data/text/ 의 추출 텍스트
출력: data/chunks/<전략>.jsonl  (청크 한 줄 = 문서ID, 쪽, 인쇄 쪽 번호, 섹션, 본문)
      pages: 청크에 실제로 텍스트가 들어간 인쇄 쪽 목록 (빈 쪽을 건너뛴 범위와 구분)
      doc_type(학술/공공), diseases(문서 제목 기준 질환), groups(청크 본문 기준 대상) 꼬리표
      data/chunks/stats.csv     (전략별 청크 수·길이 분포)

정제
  - 쪽 위아래에 반복되는 머리말·꼬리말(문서명, 학회지명, 저자, 사이트 주소) 제거
  - 쪽 번호만 있는 줄 제거, 대신 인쇄된 쪽 번호를 찾아 근거 표시에 사용
    (PDF 순서와 인쇄 쪽 번호가 다른 문서가 있음: 2025 신체활동 안내서는 20번째 쪽이 '16')
  - PDF 줄바꿈 이어붙이기: 줄 끝 공백이 있으면 띄어 쓰고, 한글 단어 중간에서 끊겼으면 붙인다
  - 제어문자(\\x07 등)·글머리표 정리
  - PMC 논문: 사사·이해상충 등 부속 섹션과 범위 밖 장(약물치료·입원 치료 등) 제외

청킹 전략 (비교 실험용)
  fixed        쪽 안에서 글자 수로 자르기 (500자, 100자 겹침) — 기준선
  fixed_ctx    fixed + 청크 앞에 '[문서명 | 섹션]' 머리말 — 머리말 효과만 따로 보기
  section_ctx  제목·문단 경계로 나눈 뒤 900자까지 묶기 + 머리말 — 구조 기반

PMC 논문과 질병관리청 건강정보(.txt)는 쪽이 없어서 '## 섹션'을 쪽 대신 위치 단위로 쓴다.

실행: python chunk.py
"""
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TEXT = DATA / "text"
OUT = DATA / "chunks"
MANIFEST = DATA / "manifest.csv"

FIXED_SIZE, FIXED_OVERLAP = 500, 100
SECTION_MAX, SECTION_MIN = 900, 150
HEADER_SHARE = 0.25            # 텍스트 있는 쪽의 25% 이상 위·아래에 반복되면 머리말·꼬리말
WRAP_MIN = 20                  # 이보다 짧은 줄은 줄바꿈이 아니라 표 칸·제목으로 본다
# PMC 섹션 중 빼는 것: 논문 부속 정보, 그리고 범위 밖인 약물치료·입원 치료 장
# (2025 당뇨병 진료지침 전문은 66만 자이고 절반 가까이가 약물·입원 내용이라 검색을 흐린다)
SKIP_SECTION_RE = re.compile(
    r"acknowledg|conflicts? of interest|competing interest|funding|author.{0,5}contribution|"
    r"availability|abbreviation|declaration|ethics|consent|supplementary|"
    r"pharmacologic|antiplatelet|ketoacidosis|hospitaliz|vaccination", re.I)
STRATEGIES = ("fixed", "fixed_ctx", "section_ctx")
SECTION_DOCS = ("pmc", "kdca")  # 쪽 대신 섹션 이름을 위치로 쓰는 문서 (PMC 논문, 질병관리청 건강정보)

# 한국 공공문서·학회지의 제목 모양: Ⅰ. / 1. / 가. / 1) / ① 등 + PMC의 마크다운 제목
HEADING_RE = re.compile(
    r"^(#{1,6} |[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s|\d{1,2}\.\s|[가-하]\.\s|\d{1,2}\)\s|[①-⑳]\s?|제\s?\d+\s?[장절]\s)")
BULLET_RE = re.compile(r"^([•·▪■□○●◦※\-–»]|\d{1,2}[.)]\s|[가-하][.)]\s|[①-⑳])")
SENT_END_RE = re.compile(r"[.!?。」』:;]$")


# ── 문서 불러오기 ─────────────────────────────────────
def usable_docs():
    with open(MANIFEST, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if (r.get("판정_수동") or r.get("판정_자동")) == "사용"]


def display_title(row):
    """근거 표시에 쓸 문서 이름"""
    title = row.get("게시글제목") or row.get("원본파일명") or ""
    if row["수집방식"] == "수동" or not title:
        title = Path(row.get("원본파일명") or title).stem.replace("_", " ")
    if row["형식"] == "pdf" and row["출처ID"].startswith(("jkd", "jkma")):
        title = row["출처명"]
    if row["형식"] == "kdca":
        title = f"국가건강정보포털: {title}"
    return title.strip()


def load_pages(row):
    """[(PDF 쪽 번호 또는 섹션 순번, 원문)] 목록"""
    doc_id = row["문서ID"]
    jl = TEXT / f"{doc_id}.jsonl"
    if jl.exists():
        return [(d["page"], d["text"]) for d in map(json.loads, jl.open(encoding="utf-8"))]
    txt = TEXT / f"{doc_id}.txt"
    if not txt.exists():
        return []
    return [(0, txt.read_text(encoding="utf-8"))]


def ocr_pages(doc_id):
    """collect.py가 OCR로 채운 쪽 번호 (data/text/<문서ID>.jsonl 의 "ocr": true)"""
    jl = TEXT / f"{doc_id}.jsonl"
    if not jl.exists():
        return set()
    return {d["page"] for d in map(json.loads, jl.open(encoding="utf-8")) if d.get("ocr")}


def clean_ocr(text, edges):
    """OCR 쪽의 잡음 줄 정리: 머리말이 조금 깨져 섞인 줄, 글자·숫자가 2개 이하인 줄(선·기호 오인식)"""
    keep = []
    for line in text.split("\n"):
        if any(e in line for e in edges if len(e) >= 10):
            continue
        if len(re.findall(r"[가-힣A-Za-z0-9]", line)) <= 2:
            continue
        keep.append(line)
    return "\n".join(keep)


# ── 정제 ─────────────────────────────────────────────
def normalize(text):
    text = text.replace(" ", " ").replace("　", " ")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text.replace("\t", " "))
    text = re.sub(r"^\s*[»›]\s*", "- ", text, flags=re.M)
    return text


def repeated_edges(pages):
    """여러 쪽의 위·아래 3줄에 반복해서 나오는 줄 = 머리말·꼬리말"""
    counts, n = Counter(), 0
    for _, t in pages:
        lines = [x.strip() for x in t.split("\n") if x.strip()]
        if not lines:
            continue
        n += 1
        counts.update(set(lines[:3] + lines[-3:]))
    limit = max(3, HEADER_SHARE * n)
    return {line for line, c in counts.items() if c >= limit and not re.fullmatch(r"\d{1,4}", line)}


def page_number_offset(pages):
    """쪽 위·아래 끝에 있는 숫자 줄을 인쇄 쪽 번호로 보고, PDF 쪽 번호와의 차이(최빈값)를 구한다."""
    diffs = Counter()
    for idx, t in pages:
        lines = [x.strip() for x in t.split("\n") if x.strip()]
        for line in lines[:3] + lines[-3:]:
            if re.fullmatch(r"\d{1,4}", line):
                diffs[int(line) - idx] += 1
    if not diffs:
        return 0
    diff, c = diffs.most_common(1)[0]
    return diff if c >= 2 else 0


def join_lines(lines):
    """PDF의 강제 줄바꿈을 문단으로 되돌린다.
    - 빈 줄, 글머리표·제목으로 시작하는 줄, 앞줄이 문장부호로 끝났거나 짧은 제목 줄이면 새 문단
    - 원문 줄 끝에 공백이 있으면 띄어 쓴다
    - 한글 단어 중간에서 끊긴 긴 줄은 붙인다 ('혈\n관계' → '혈관계').
      표 칸은 짧은 줄이라 붙이지 않고 띄어 쓴다 ('운동 유형\n운동 강도' → '운동 유형 운동 강도')"""
    out, spaced, prev_len, prev_head = [], False, 0, False
    for raw in lines:
        line = raw.strip()
        if not line:
            if out and out[-1]:
                out.append("")
            continue
        if not out or not out[-1] or prev_head or BULLET_RE.match(line) or HEADING_RE.match(line) \
                or SENT_END_RE.search(out[-1]):
            out.append(line)
        elif spaced or prev_len < WRAP_MIN or not (re.search(r"[가-힣]$", out[-1]) and re.match(r"[가-힣]", line)):
            out[-1] += " " + line
        else:
            out[-1] += line
        spaced = raw.rstrip("\n").endswith(" ")
        prev_len = len(line)
        prev_head = bool(HEADING_RE.match(line)) and len(line) <= 40
    return [x for x in out if x]


def clean_page(text, edges):
    lines = normalize(text).split("\n")
    kept = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s in edges:
            continue
        if re.fullmatch(r"\d{1,4}", s) and (i < 3 or i >= len(lines) - 3):
            continue                       # 쪽 번호 줄
        if re.fullmatch(r"(https?://|www\.)\S+", s):
            continue
        kept.append(raw)
    return join_lines(kept)


def split_pmc_sections(text):
    """PMC 텍스트를 '## 섹션' 단위로 나눈다. 제목(#)과 저널 줄은 머리말로 쓰므로 뺀다."""
    sections, cur_title, cur = [], "", []
    for line in text.split("\n"):
        m = re.match(r"^(#{2,6}) (.+)", line)
        if m and len(m.group(1)) == 2:
            if cur and not SKIP_SECTION_RE.search(cur_title):
                sections.append((cur_title, cur))
            cur_title, cur = m.group(2).strip(), []
        elif line.startswith("# "):
            continue
        else:
            cur.append(line)
    if cur and not SKIP_SECTION_RE.search(cur_title):
        sections.append((cur_title, cur))
    if sections and not sections[0][0]:
        sections = sections[1:]           # '저널 연도' 줄
    return sections


def parse_doc(row):
    """문서 → 쪽(또는 섹션) 단위 정제 결과 [{page, label, section, paras}]"""
    pages = load_pages(row)
    units = []
    if row["형식"] in SECTION_DOCS or (pages and pages[0][0] == 0):
        for n, (title, lines) in enumerate(split_pmc_sections(pages[0][1]) if pages else [], 1):
            paras = [re.sub(r"\s+", " ", x).strip() for x in lines if x.strip()]
            if paras:
                units.append({"page": n, "label": title, "section": title, "paras": paras})
        return units
    edges = repeated_edges(pages)
    offset = page_number_offset(pages)
    ocr = ocr_pages(row["문서ID"])
    for idx, t in pages:
        paras = clean_page(clean_ocr(t, edges) if idx in ocr else t, edges)
        if not paras:
            continue
        units.append({"page": idx, "label": str(idx + offset), "section": "", "paras": paras,
                      **({"ocr": True} if idx in ocr else {})})
    return units


# ── 청킹 ─────────────────────────────────────────────
def window(text, size, overlap):
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                # 가능하면 문장·공백 경계에서 자른다
            cut = max(text.rfind(". ", start + size // 2, end), text.rfind("다. ", start + size // 2, end))
            cut = cut if cut > 0 else text.rfind(" ", start + size // 2, end)
            end = cut + 1 if cut > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        if len(text) - start <= overlap:  # 남은 꼬리가 겹침 구간 안에 다 들어 있으면 끝
            break
    return [x for x in out if x]


def ctx_header(title, section, label, is_pmc):
    where = section if is_pmc else (f"{section} · {label}쪽" if section else f"{label}쪽")
    return f"[{title} | {where}]\n"


def chunks_fixed(row, units, with_ctx):
    title, is_pmc = display_title(row), row["형식"] in SECTION_DOCS
    for u in units:
        for piece in window("\n".join(u["paras"]), FIXED_SIZE, FIXED_OVERLAP):
            head = ctx_header(title, u["section"], u["label"], is_pmc) if with_ctx else ""
            yield {"page": u["page"], "page_end": u["page"], "label": u["label"],
                   "pages": [u["label"]], "section": u["section"], "text": head + piece}


def chunks_section(row, units):
    """제목 줄에서 새 섹션을 시작하고, 섹션 안 문단을 SECTION_MAX 글자까지 묶는다.
    쪽을 넘어갈 수 있으며 그때는 쪽 범위를 기록한다."""
    title, is_pmc = display_title(row), row["형식"] in SECTION_DOCS
    buf, section, start, cur_label, start_label = [], "", None, "", ""

    def flush():
        nonlocal buf
        body = "\n".join(p for _, _, p in buf).strip()
        if body:
            first, last = buf[0], buf[-1]
            label = first[1] if first[1] == last[1] or is_pmc else f"{first[1]}-{last[1]}"
            pages = list(dict.fromkeys(lb for _, lb, _ in buf))   # 실제로 텍스트가 들어간 쪽만
            for piece in window(body, SECTION_MAX, FIXED_OVERLAP):
                yield {"page": first[0], "page_end": last[0], "label": label, "pages": pages,
                       "section": section, "text": ctx_header(title, section, label, is_pmc) + piece}
        buf = []

    for u in units:
        if is_pmc:
            yield from flush()
            section = u["section"]
        for p in u["paras"]:
            m = HEADING_RE.match(p)
            is_heading = bool(m) and len(p) <= 60 and not is_pmc
            n = re.match(r"(\d+)\)", p)
            if n and int(n.group(1)) > 15:
                is_heading = False         # '44) 보건복지부, 2013' 같은 각주 번호
            size = sum(len(x) for _, _, x in buf)
            if is_heading and size >= SECTION_MIN:
                yield from flush()
            if is_heading:
                section = re.sub(r"^#+\s*", "", p)[:40]
            if size + len(p) > SECTION_MAX and size >= SECTION_MIN:
                yield from flush()
            buf.append((u["page"], u["label"], p))
    yield from flush()


def build(strategy, docs):
    out = []
    for row, units in docs:
        if strategy == "section_ctx":
            gen = chunks_section(row, units)
        else:
            gen = chunks_fixed(row, units, strategy == "fixed_ctx")
        doc_tags = {"doc_type": doc_type(row), "diseases": tags(DISEASE_TAGS, row_title_text(row))}
        ocr = {u["page"] for u in units if u.get("ocr")}
        for n, c in enumerate(gen, 1):
            if any(p in ocr for p in range(c["page"], c.get("page_end", c["page"]) + 1)):
                c["ocr"] = True            # 글자 인식 오류가 있을 수 있는 근거 (인용에 'OCR' 표시)
            c.update({"chunk_id": f"{row['문서ID']}#{n:04d}", "doc_id": row["문서ID"],
                      "title": display_title(row), "lang": row.get("언어", ""), **doc_tags,
                      "groups": tags(GROUP_TAGS, c["text"])})
            out.append(c)
    return out


# ── 꼬리표 ───────────────────────────────────────────
# 학술: 학회지·PMC 논문 / 공공: 보건복지부·한국건강증진개발원·질병관리청 안내 자료
ACADEMIC_SOURCES = ("pmc", "jkd", "jkma", "ksso", "ada")
DISEASE_TAGS = {
    "당뇨": r"당뇨|혈당|diabet|glyc|insulin",
    "비만": r"비만|체중|obes|adipos|weight",
    "고혈압": r"고혈압|혈압|hypertens|blood pressure",
}
# 대상 꼬리표는 청크 본문에서 찾는다 (같은 문서 안에서도 노인·청소년 부분이 따로 있음)
GROUP_TAGS = {
    "노인": r"노인|고령|어르신|older adult|elderly|aged \d|sarcopeni|근감소",
    "청소년·소아": r"청소년|소아|아동|어린이|adolescen|child|youth|pediatric|paediatric",
    "여성": r"여성|폐경|월경|women|female|menopaus|menstrua",
    "남성": r"남성|\bmen\b|\bmale",
    "임신": r"임신|임산부|산모|pregnan|gestation",
    "1형당뇨": r"1형\s?당뇨|type 1 diabetes|T1D\b",
    "2형당뇨": r"2형\s?당뇨|type 2 diabetes|T2D\b",
    "원발성고혈압": r"원발성|본태성|일차성 고혈압|primary hypertension|essential hypertension",
    "이차성고혈압": r"이차성 고혈압|secondary hypertension",
}


def doc_type(row):
    return "학술" if row["출처ID"].startswith(ACADEMIC_SOURCES) else "공공"


def row_title_text(row):
    return f"{row.get('게시글제목', '')} {row.get('출처명', '')} {row.get('원본파일명', '')}"


def tags(table, text):
    return [name for name, pat in table.items() if re.search(pat, text, re.I)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = usable_docs()
    docs = [(r, parse_doc(r)) for r in rows]
    stats = []
    for s in STRATEGIES:
        chunks = build(s, docs)
        with open(OUT / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        lens = [len(c["text"]) for c in chunks]
        stats.append({"전략": s, "문서수": len(docs), "청크수": len(chunks),
                      "평균글자": round(statistics.mean(lens)), "중앙값": round(statistics.median(lens)),
                      "최소": min(lens), "최대": max(lens), "총글자": sum(lens)})
        print(f"{s:12s} 청크 {len(chunks):5d}개, 평균 {stats[-1]['평균글자']}자")
    with open(OUT / "stats.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    print(f"저장: {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
