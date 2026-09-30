#!/usr/bin/env python3
"""
PhysiologyRAG 문서 수집기 (2단계: 데이터 구축)

강사 규칙과의 대응
  - 코드로 수집 ................. sources.csv에 적힌 출처를 자동으로 수집
  - 재실행 가능 ................. 다시 돌리면 바뀐 문서만 갱신, 이전본은 날짜 붙여 보관
  - 스캔본·깨진 문서 걸러내기 .... 쪽별 글자 수로 텍스트 없는 쪽·이미지 PDF·깨진 파일 판정
  - 데이터코드 표 ............... data/manifest.csv (문서별) + data/summary.csv (출처별)
                                   + data/데이터코드표.xlsx (두 표를 엑셀 시트로)

출처 종류 (sources.csv의 '종류' 칸)
  pdf    파일 주소를 직접 알 때 (URL에 {page}를 넣으면 쪽범위만큼 반복)
  page   논문 소개 페이지·게시글에서 첨부 PDF 링크를 찾아 받기 (DOI 주소도 가능)
  board  게시판 목록 → 게시글 → 첨부파일
  html   웹페이지 본문 텍스트 (URL에 {page} 사용 가능)
  pmc    PubMed Central 공식 API(E-utilities)로 오픈액세스 논문 검색·수집

사용법은 docs/collect_guide.md 참고.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import unquote, urljoin, urlparse

import pymupdf
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── 경로 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
TEXT = DATA / "text"
MANIFEST = DATA / "manifest.csv"
SUMMARY = DATA / "summary.csv"
XLSX = DATA / "데이터코드표.xlsx"

# ── 설정 ──────────────────────────────────────────────
USER_AGENT = "PhysiologyRAG-collector/0.2 (educational project)"
DEFAULT_DELAY = 3.0
EMPTY_PAGE_CHARS = 20
SCAN_RATIO = 0.9
NEAR_DUP = 0.6
MIN_ARTICLE_CHARS = 1500      # pmc 본문이 이보다 짧으면 '초록만'으로 판정
DEFAULT_FILE_PATTERN = r"\.pdf|\.hwpx?|download|filedown|atchfile|attach|article-pdf"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

MANIFEST_FIELDS = [
    "문서ID", "출처ID", "출처명", "수집방식", "게시글제목", "URL", "원본파일명",
    "저장경로", "형식", "언어", "수집일", "SHA256", "쪽수", "글자수",
    "텍스트없는쪽수", "텍스트없는쪽", "판정_자동", "판정사유",
    "이용조건", "판정_수동", "비고",
]


# ── 네트워크 ──────────────────────────────────────────
class Blocked(Exception):
    """robots.txt가 금지한 주소"""


class Fetcher:
    def __init__(self, delay):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.robots = {}
        self.last_hit = {}

    @staticmethod
    def site(url):
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def robots_for(self, url):
        s = self.site(url)
        if s not in self.robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(s + "/robots.txt", timeout=15)
                if r.status_code == 404:
                    rp.allow_all = True
                elif r.status_code in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp.disallow_all = True       # 확인 실패 = 보수적으로 금지
            self.robots[s] = rp
        return self.robots[s]

    def allowed(self, url):
        return self.robots_for(url).can_fetch(USER_AGENT, url)

    def wait(self, url, min_gap=None):
        s = self.site(url)
        gap = self.delay if min_gap is None else min_gap
        if min_gap is None:
            gap = max(gap, float(self.robots_for(url).crawl_delay(USER_AGENT) or 0))
        elapsed = time.time() - self.last_hit.get(s, 0)
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self.last_hit[s] = time.time()

    def get(self, url, referer=None, params=None, check_robots=True, min_gap=None):
        if check_robots and not self.allowed(url):
            raise Blocked(url)
        self.wait(url, min_gap)
        headers = {"Referer": referer} if referer else {}
        r = self.session.get(url, headers=headers, params=params, timeout=90)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text" in ctype and "charset" not in ctype.lower():
            r.encoding = r.apparent_encoding
        return r


# ── HTML 처리 ─────────────────────────────────────────
def find_links(html, base_url, pattern, text_filter=None):
    soup = BeautifulSoup(html, "html.parser")
    found, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "#", "mailto:")):
            continue
        url = urljoin(base_url, href)
        label = a.get_text(" ", strip=True)
        if not re.search(pattern, url, re.I):
            continue
        if text_filter and not re.search(text_filter, label):
            continue
        if url not in seen:
            seen.add(url)
            found.append((url, label))
    return found


def page_title(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in ("h2", "h3", "h1", "title"):
        t = soup.find(tag)
        if t and t.get_text(strip=True):
            return t.get_text(" ", strip=True)[:120]
    return ""


def main_text(html, selector=None):
    """웹페이지에서 메뉴·꼬리말을 빼고 본문만 뽑는다.
    선택자가 없으면, 링크가 적은 문단이 가장 많이 모인 영역을 본문으로 본다."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        t.decompose()
    if selector:
        node = soup.select_one(selector)
        return (node or soup).get_text("\n", strip=True)

    def good(tag):
        txt = tag.get_text(" ", strip=True)
        if len(txt) < 20:
            return 0
        link = sum(len(a.get_text(strip=True)) for a in tag.find_all("a"))
        return 0 if link / len(txt) > 0.5 else len(txt)

    scores = {}
    for tag in soup.find_all(["p", "li", "h2", "h3", "h4", "td"]):
        n = good(tag)
        parent = tag.find_parent(["div", "article", "section", "main", "td"])
        if n and parent is not None:
            scores[parent] = scores.get(parent, 0) + n
    if not scores:
        return soup.get_text("\n", strip=True)
    best = max(scores, key=scores.get)
    up = best.find_parent(["div", "article", "section", "main"])
    if up is not None:
        up_score = sum(good(t) for t in up.find_all(["p", "li", "h2", "h3", "h4", "td"]))
        if up_score > scores[best] * 1.3:   # 제목·문단이 형제 영역에 나뉘어 있으면 한 단계 위로
            best = up
    return best.get_text("\n", strip=True)


# ── 파일 이름·형식 ─────────────────────────────────────
def filename_from(resp, url):
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;]+)", cd)
    if m:
        return unquote(m.group(1))
    m = re.search(r'filename="?([^";]+)"?', cd)
    if m:
        raw = m.group(1)
        for enc in ("utf-8", "euc-kr"):
            try:
                return unquote(raw.encode("latin-1").decode(enc))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return unquote(raw)
    return unquote(Path(urlparse(url).path).name) or "download"


def detect_type(data, name):
    head = data[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "hwp"
    if head.startswith(b"PK") and name.lower().endswith(".hwpx"):
        return "hwpx"
    if data.lstrip()[:1] == b"<":
        return "html"
    return "unknown"


def count_chars(text):
    return len(re.sub(r"\s", "", text))


def language_of(text):
    ko = len(re.findall(r"[가-힣]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if ko + en == 0:
        return ""
    return "ko" if ko / (ko + en) >= 0.3 else "en"


# ── manifest ─────────────────────────────────────────
def load_manifest():
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        return {row["URL"]: row for row in csv.DictReader(f)}


def save_manifest(rows):
    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in sorted(rows.values(), key=lambda r: r["문서ID"]):
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def next_id(rows, source_id):
    nums = [int(r["문서ID"].rsplit("-", 1)[-1]) for r in rows.values()
            if r["문서ID"].startswith(source_id + "-")]
    return f"{source_id}-{(max(nums) + 1) if nums else 1:03d}"


def register(rows, *, url, data, name, src, method, title, forced_type=None):
    """파일 하나를 저장하고 manifest에 기록한다. 내용이 같으면 '변경 없음'."""
    today = datetime.now().strftime("%Y-%m-%d")
    sha = hashlib.sha256(data).hexdigest()
    old = rows.get(url)
    if old and old["SHA256"] == sha:
        old["수집일"] = today
        return old["문서ID"], "변경 없음"

    doc_id = old["문서ID"] if old else next_id(rows, src["출처ID"])
    ftype = forced_type or detect_type(data, name)
    ext = {"pdf": ".pdf", "hwp": ".hwp", "hwpx": ".hwpx", "html": ".html",
           "pmc": ".xml", "web": ".html"}.get(ftype, ".bin")
    path = RAW / f"{doc_id}{ext}"
    note = (old or {}).get("비고", "")
    if old and path.exists():
        backup = RAW / f"{doc_id}_{old['수집일']}{ext}"
        shutil.move(path, backup)
        note = (note + f" | {today} 갱신됨(이전본: {backup.name})").strip(" |")
    path.write_bytes(data)
    rows[url] = {
        "문서ID": doc_id, "출처ID": src["출처ID"], "출처명": src["출처명"],
        "수집방식": method, "게시글제목": title, "URL": url, "원본파일명": name,
        "저장경로": str(path.relative_to(ROOT)), "형식": ftype, "수집일": today,
        "SHA256": sha, "이용조건": (old or {}).get("이용조건") or src.get("이용조건", ""),
        "판정_수동": (old or {}).get("판정_수동", ""), "비고": note,
    }
    return doc_id, "갱신" if old else "신규"


def expand_pages(spec):
    """'1~8' → [1..8], '' → [None]
    엑셀이 '1-8'을 날짜로 바꿔 저장한 경우('1월 8일')도 읽는다."""
    spec = (spec or "").strip()
    if not spec:
        return [None]
    m = re.match(r"^(\d+)\s*(?:[-~:]|월)?\s*(\d+)?", spec)
    if not m:
        raise ValueError(f"쪽범위를 읽을 수 없음: {spec!r} (예: 1~8)")
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(a, b + 1))


def expand_urls(src):
    url = src["URL"].strip()
    if "{page}" not in url:
        return [url]
    return [url.replace("{page}", str(p)) for p in expand_pages(src.get("쪽범위")) if p]


# ── PMC (E-utilities) ────────────────────────────────
def jats_text(elem):
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip() if elem is not None else ""


def parse_pmc_article(xml_bytes):
    """JATS XML에서 제목·저널·연도·라이선스·본문(섹션 제목 포함)을 뽑는다."""
    root = ET.fromstring(xml_bytes)
    art = root.find(".//article") if root.tag != "article" else root
    if art is None:
        return None
    meta = art.find("./front/article-meta")
    title = jats_text(meta.find("./title-group/article-title")) if meta is not None else ""
    journal = jats_text(art.find("./front/journal-meta/journal-title-group/journal-title"))
    year = jats_text(meta.find("./pub-date/year")) if meta is not None else ""
    lic = art.find(".//permissions/license")
    license_txt = ""
    if lic is not None:
        href = lic.get("{http://www.w3.org/1999/xlink}href", "")
        license_txt = href or jats_text(lic)[:120]
    lines = [f"# {title}", f"{journal} {year}", ""]
    abstract = meta.find("./abstract") if meta is not None else None
    if abstract is not None:
        lines += ["## Abstract", jats_text(abstract), ""]
    body = art.find("./body")

    def walk(node, depth):
        for child in node:
            tag = child.tag
            if tag == "sec":
                t = child.find("./title")
                if t is not None:
                    lines.append("#" * min(depth + 2, 6) + " " + jats_text(t))
                walk(child, depth + 1)
            elif tag == "p":
                lines.append(jats_text(child))
            elif tag == "table-wrap":
                cap = child.find("./caption")
                if cap is not None:
                    lines.append("[표] " + jats_text(cap))
                for tr in child.iter("tr"):
                    cells = [jats_text(c) for c in tr if c.tag in ("td", "th")]
                    if any(cells):
                        lines.append(" | ".join(cells))
            elif tag in ("list", "boxed-text", "disp-quote"):
                for p in child.iter("p"):
                    lines.append("- " + jats_text(p))

    has_body = body is not None
    if has_body:
        walk(body, 0)
    return {"title": title, "journal": journal, "year": year,
            "license": license_txt, "text": "\n".join(lines), "has_body": has_body}


def collect_pmc(src, fetcher, rows, dry_run, log):
    """PMC 공식 API로 오픈액세스 논문을 검색해 XML 본문을 받는다.
    E-utilities는 API라서 robots.txt 대신 NCBI 이용 규칙(초당 3회 이하, tool·email 표기)을 따른다."""
    base = {"tool": "PhysiologyRAG", "email": os.getenv("NCBI_EMAIL", "")}
    if os.getenv("NCBI_API_KEY"):
        base["api_key"] = os.getenv("NCBI_API_KEY")
    term = src.get("검색어", "").strip()
    retmax = int(src.get("최대개수") or 10)
    r = fetcher.get(f"{EUTILS}/esearch.fcgi", check_robots=False, min_gap=0.4,
                    params={**base, "db": "pmc", "term": term, "retmax": retmax,
                            "retmode": "json", "sort": "relevance"})
    res = r.json().get("esearchresult", {})
    ids = res.get("idlist", [])
    log(f"  검색 결과 {res.get('count', '?')}건 중 {len(ids)}건 수집")
    for pmcid in ids:
        url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
        if dry_run:
            log(f"  [미리보기] PMC{pmcid}")
            continue
        try:
            x = fetcher.get(f"{EUTILS}/efetch.fcgi", check_robots=False, min_gap=0.4,
                            params={**base, "db": "pmc", "id": pmcid, "retmode": "xml"})
            info = parse_pmc_article(x.content)
        except (requests.RequestException, ET.ParseError) as e:
            log(f"  PMC{pmcid} 실패: {e}")
            continue
        if info is None:
            log(f"  PMC{pmcid} 형식을 읽을 수 없음")
            continue
        doc_id, status = register(rows, url=url, data=x.content, name=f"PMC{pmcid}.xml",
                                  src=src, method="코드(API)", title=info["title"],
                                  forced_type="pmc")
        row = rows[url]
        txt = TEXT / f"{doc_id}.txt"
        txt.write_text(info["text"], encoding="utf-8")
        row["저장경로"] = str(txt.relative_to(ROOT))
        if info["license"]:
            row["이용조건"] = info["license"]
        if not info["has_body"]:
            row["비고"] = (row.get("비고", "") + " | 출판사가 본문 XML 미제공").strip(" |")
        log(f"  {doc_id} {status}: {info['title'][:50]}")


# ── 출처 종류별 수집 ─────────────────────────────────
def collect_source(src, fetcher, rows, dry_run, log):
    kind = src["종류"].strip()
    file_pat = src.get("파일링크패턴") or DEFAULT_FILE_PATTERN

    def safe(fn, *args):
        try:
            fn(*args)
        except Blocked as b:
            log(f"  robots.txt 금지로 건너뜀: {b}")
        except requests.RequestException as e:
            log(f"  요청 실패: {e}")

    def grab_file(url, title="", referer=None):
        if dry_run:
            mark = "" if fetcher.allowed(url) else " (robots.txt 금지)"
            log(f"  [미리보기] 파일: {url}{mark}")
            return
        r = fetcher.get(url, referer=referer)
        name = filename_from(r, url)
        doc_id, status = register(rows, url=url, data=r.content, name=name,
                                  src=src, method="코드", title=title)
        log(f"  {doc_id} {status}: {name}")

    def grab_page_files(page_url):
        if dry_run and not fetcher.allowed(page_url):
            log(f"  [미리보기] {page_url} (robots.txt 금지)")
            return
        r = fetcher.get(page_url)
        title = page_title(r.text)
        files = find_links(r.text, r.url, file_pat)
        if not files:
            log(f"  첨부 링크 없음: {page_url} (파일링크패턴 확인)")
        for furl, _ in files[:3]:   # 같은 PDF가 여러 링크로 걸린 경우가 많아 앞쪽 3개까지만
            safe(grab_file, furl, title, page_url)

    def grab_html(url):
        if dry_run:
            mark = "" if fetcher.allowed(url) else " (robots.txt 금지)"
            log(f"  [미리보기] 웹페이지: {url}{mark}")
            return
        r = fetcher.get(url)
        text = main_text(r.text, src.get("본문선택자") or None)
        title = page_title(r.text)
        doc_id, status = register(rows, url=url, data=r.content, name=title or url,
                                  src=src, method="코드", title=title, forced_type="web")
        txt = TEXT / f"{doc_id}.txt"
        txt.write_text(text, encoding="utf-8")
        rows[url]["저장경로"] = str(txt.relative_to(ROOT))
        log(f"  {doc_id} {status}: {title[:40] or url}")

    if kind == "pdf":
        for u in expand_urls(src):
            safe(grab_file, u)
    elif kind == "page":
        for u in expand_urls(src):
            safe(grab_page_files, u)
    elif kind == "html":
        for u in expand_urls(src):
            safe(grab_html, u)
    elif kind == "board":
        detail_pat = src.get("상세링크패턴") or r"view|detail|read|list_no=|seq=|idx="
        for pg in expand_pages(src.get("쪽범위")):
            list_url = src["URL"].replace("{page}", str(pg)) if pg else src["URL"]
            try:
                r = fetcher.get(list_url)
            except Blocked:
                log(f"  robots.txt 금지로 목록 건너뜀: {list_url}")
                continue
            except requests.RequestException as e:
                log(f"  목록 요청 실패: {e}")
                continue
            posts = find_links(r.text, r.url, detail_pat, src.get("목록필터") or None)
            log(f"  목록 {pg or ''}쪽: 게시글 {len(posts)}개")
            for purl, label in posts:
                if dry_run:
                    log(f"  [미리보기] 게시글: {label[:40]} → {purl}")
                else:
                    safe(grab_page_files, purl)
    elif kind == "pmc":
        safe(collect_pmc, src, fetcher, rows, dry_run, log)
    else:
        log(f"  알 수 없는 종류 '{kind}'")


def import_local(folder, rows, license_note, log, source_id="manual", source_name="수동 수집", note=""):
    """robots.txt 금지 등으로 직접 받은 파일을 '수동 수집'으로 등록"""
    src = {"출처ID": source_id, "출처명": source_name, "이용조건": license_note}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in (".pdf", ".hwp", ".hwpx"):
            continue
        key = f"manual://{source_id}/{p.name}"
        doc_id, status = register(rows, url=key, data=p.read_bytes(),
                                  name=p.name, src=src, method="수동", title=p.stem)
        if note and note not in rows[key].get("비고", ""):
            rows[key]["비고"] = (rows[key].get("비고", "") + " | " + note).strip(" |")
        log(f"  {doc_id} {status}: {p.name}")


# ── 검사와 판정 ──────────────────────────────────────
def inspect_pdf(path, doc_id):
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        return {"error": f"PDF 열기 실패: {e}"}
    pages = [p.get_text() for p in doc]
    counts = [count_chars(t) for t in pages]
    with open(TEXT / f"{doc_id}.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(pages, 1):
            f.write(json.dumps({"page": i, "text": t}, ensure_ascii=False) + "\n")
    return {"pages": len(pages), "chars": sum(counts), "text": "\n".join(pages),
            "empty": [i + 1 for i, c in enumerate(counts) if c < EMPTY_PAGE_CHARS]}


def doc_text(row):
    p = ROOT / row["저장경로"]
    if p.suffix == ".txt" and p.exists():
        return p.read_text(encoding="utf-8")
    jl = TEXT / f"{row['문서ID']}.jsonl"
    if jl.exists():
        return "\n".join(json.loads(l)["text"] for l in jl.open(encoding="utf-8"))
    return ""


def chunk_set(text, n=10):
    """유사 문서 비교용: 공백을 뺀 연속 10글자 조각 집합 (줄바꿈 위치에 영향 안 받음)"""
    s = re.sub(r"\s+", "", text)
    return {s[i:i + n] for i in range(0, max(len(s) - n + 1, 0))}


def judge_all(rows):
    by_sha = {}
    for row in sorted(rows.values(), key=lambda r: r["문서ID"]):
        doc_id, ftype = row["문서ID"], row["형식"]
        info = {}
        if ftype == "pdf":
            info = inspect_pdf(ROOT / row["저장경로"], doc_id)
        elif ftype in ("web", "pmc"):
            t = doc_text(row)
            info = {"pages": 1, "chars": count_chars(t), "text": t, "empty": []}
        empty = info.get("empty", [])
        row["쪽수"] = info.get("pages", "")
        row["글자수"] = info.get("chars", "")
        row["텍스트없는쪽수"] = len(empty) if info and "error" not in info else ""
        row["텍스트없는쪽"] = ",".join(map(str, empty))
        row["언어"] = language_of(info.get("text", "")[:20000])

        verdict, reason = "사용", ""
        if row["SHA256"] in by_sha:
            verdict, reason = "제외", f"완전 중복: {by_sha[row['SHA256']]}"
        elif ftype in ("hwp", "hwpx"):
            verdict, reason = "보류", "HWP 파일: PDF 변환 또는 HWP 파서 필요"
        elif ftype == "html":
            verdict, reason = "제외", "파일 대신 웹페이지가 옴(다운로드 실패 가능성)"
        elif ftype == "unknown":
            verdict, reason = "보류", "형식 알 수 없음"
        elif "error" in info:
            verdict, reason = "제외", info["error"]
        elif ftype == "web" and info["chars"] < 200:
            verdict, reason = "제외", "본문이 거의 없음(본문선택자 확인)"
        elif ftype == "pmc" and info["chars"] < MIN_ARTICLE_CHARS:
            verdict, reason = "제외", "본문 없음(초록만 제공)"
        elif ftype == "pdf" and info.get("pages"):
            ratio = len(empty) / info["pages"]
            if ratio >= SCAN_RATIO:
                verdict, reason = "제외", f"텍스트 층 없는 PDF({len(empty)}/{info['pages']}쪽) → OCR 필요"
            elif empty:
                reason = f"텍스트 없는 쪽 {len(empty)}개 → OCR 검토"
        by_sha.setdefault(row["SHA256"], doc_id)
        row["판정_자동"], row["판정사유"] = verdict, reason

    live = {r["문서ID"]: r for r in rows.values() if r["판정_자동"] != "제외"}
    sets = {i: chunk_set(doc_text(r)) for i, r in live.items()}
    for a in live:
        for b in live:
            if a >= b or not sets[a] or not sets[b] or live[a]["언어"] != live[b]["언어"]:
                continue
            inter, union = len(sets[a] & sets[b]), len(sets[a] | sets[b])
            if union and inter / union >= NEAR_DUP:
                pct = f"{100 * inter / union:.0f}%"
                for x, y in ((a, b), (b, a)):
                    live[x]["판정사유"] = (live[x]["판정사유"] + f" | 유사: {y} {pct}").strip(" |")


def final_verdict(row):
    return row.get("판정_수동") or row.get("판정_자동", "")


def write_summary(rows):
    """출처별 요약: 문서 수, 사용·제외·보류, 사용 문서 총 글자 수, 최근 수집일"""
    by = {}
    for r in rows.values():
        s = by.setdefault(r["출처ID"], {"출처ID": r["출처ID"], "출처명": r["출처명"],
                                         "수집방식": r["수집방식"], "문서수": 0, "사용": 0,
                                         "제외": 0, "보류": 0, "사용_총글자수": 0, "최근수집일": ""})
        s["문서수"] += 1
        v = final_verdict(r)
        if v in ("사용", "제외", "보류"):
            s[v] += 1
        if v == "사용":
            s["사용_총글자수"] += int(r.get("글자수") or 0)
        s["최근수집일"] = max(s["최근수집일"], r.get("수집일", ""))
    table = sorted(by.values(), key=lambda s: s["출처ID"])
    total = {"출처ID": "합계", "출처명": "", "수집방식": "",
             **{k: sum(s[k] for s in table) for k in ("문서수", "사용", "제외", "보류", "사용_총글자수")},
             "최근수집일": max((s["최근수집일"] for s in table), default="")}
    fields = list(total.keys())
    with open(SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table + [total])
    try:
        import pandas as pd
        with pd.ExcelWriter(XLSX) as xw:
            pd.DataFrame([{k: r.get(k, "") for k in MANIFEST_FIELDS}
                          for r in sorted(rows.values(), key=lambda r: r["문서ID"])]
                         ).to_excel(xw, sheet_name="문서목록", index=False)
            pd.DataFrame(table + [total]).to_excel(xw, sheet_name="출처별요약", index=False)
    except ImportError:
        pass
    return total


# ── 실행 ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PhysiologyRAG 문서 수집기")
    ap.add_argument("--sources", default=str(ROOT / "sources.csv"))
    ap.add_argument("--only", help="이 출처ID만 (쉼표로 여러 개)")
    ap.add_argument("--dry-run", action="store_true", help="받지 않고 대상만 확인")
    ap.add_argument("--import-dir", help="직접 받은 파일 폴더를 수동 수집으로 등록")
    ap.add_argument("--license", default="", help="--import-dir 파일의 이용조건")
    ap.add_argument("--source-id", default="manual", help="--import-dir 파일의 출처ID (예: mohw)")
    ap.add_argument("--source-name", default="수동 수집", help="--import-dir 파일의 출처명")
    ap.add_argument("--note", default="", help="--import-dir 파일의 비고 (예: 규칙 문서)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    TEXT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    log = print

    if args.import_dir:
        log(f"[수동 등록] {args.import_dir}")
        import_local(args.import_dir, rows, args.license, log,
                     args.source_id, args.source_name, args.note)
    else:
        with open(args.sources, encoding="utf-8-sig") as f:
            sources = [s for s in csv.DictReader(f)
                       if s.get("출처ID") and not s["출처ID"].startswith("#")]
        if args.only:
            wanted = set(args.only.split(","))
            sources = [s for s in sources if s["출처ID"] in wanted]
        fetcher = Fetcher(args.delay)
        for src in sources:
            log(f"[{src['출처ID']}] {src['출처명']} ({src['종류']})")
            collect_source(src, fetcher, rows, args.dry_run, log)

    if args.dry_run:
        log("\n미리보기라서 파일은 받지 않았어.")
        return

    judge_all(rows)
    save_manifest(rows)
    total = write_summary(rows)
    log(f"\n완료: 문서 {total['문서수']}개 (사용 {total['사용']} / 제외 {total['제외']} / 보류 {total['보류']})")
    log(f"사용 문서 총 글자 수: {total['사용_총글자수']:,}")
    log(f"표: {MANIFEST.relative_to(ROOT)}, {SUMMARY.relative_to(ROOT)}"
        + (f", {XLSX.relative_to(ROOT)}" if XLSX.exists() else ""))
    for r in sorted(rows.values(), key=lambda r: r["문서ID"]):
        if r["판정_자동"] != "사용" or r["판정사유"]:
            log(f"  {r['문서ID']} [{r['판정_자동']}] {r['판정사유']}")


if __name__ == "__main__":
    sys.exit(main())