# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "markdownify>=0.14",
#   "html2text>=2024.2.26",
#   "pymupdf4llm>=0.0.17",
# ]
# ///
"""Evaluate several HTML/PDF → markdown converters on representative arxiv papers.

Run: `uv run leaderboard/experiments/md_converter_eval/eval.py`

For each arxiv ID we fetch:
  - HTML from https://arxiv.org/html/{id}
  - PDF  from https://arxiv.org/pdf/{id}

and pipe through each converter, writing outputs to
leaderboard/experiments/md_converter_eval/outputs/{arxiv_id}/{converter}.md.

Quality signals per output (cheap, automatable):
  - bytes                 size of markdown
  - has_bib               does a References / Bibliography heading survive?
  - table_lines           number of `|`-delimited markdown table rows
  - math_lines            lines that contain `$...$` or `\\(...\\)` math
  - bibkeys_resolved      citation anchors like `(author2024foo)` / `[bib1]`
  - crashed               the converter raised / returned empty

After the run, prints a per-paper × per-converter summary.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative papers — pick a mix of known edge cases.
SAMPLES = [
    ("2506.06677", "robocerebra — ltx span tables (failed today)"),
    ("2508.19236", "MemoryVLA v2 — late revision added MIKASA section"),
    ("2502.10550", "MIKASA paper — results only in figures"),
    ("2510.07151", "ELMUR — clean tabular results (works today)"),
    ("2511.11478", "LIBERO-Mem paper — small paper, v2"),
    ("2410.24164", "π0 — widely cited baseline paper"),
    ("2601.07060", "random recent (calvin+libero citer)"),
]


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


def _fetch(url: str, timeout: int = 60) -> bytes | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VLA-Leaderboard-Eval/1.0 (+https://github.com/allenai/vla-evaluation-harness)",
            "Accept": "text/html,application/pdf,*/*;q=0.8",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt < 2:
                time.sleep(5)
                continue
            return None
    return None


def fetch_html(arxiv_id: str) -> str | None:
    data = _fetch(f"https://arxiv.org/html/{arxiv_id}")
    return data.decode("utf-8", errors="replace") if data else None


def fetch_pdf(arxiv_id: str, cache: Path) -> Path | None:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "paper.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    data = _fetch(f"https://arxiv.org/pdf/{arxiv_id}")
    if not data:
        return None
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# converters
# ---------------------------------------------------------------------------


_CELL_INNER_RE = re.compile(r"<t[dh][^>]*>.*?</t[dh]>", re.DOTALL | re.IGNORECASE)


def _flatten_cell_inner(match: re.Match[str]) -> str:
    cell = match.group(0)
    cell = re.sub(r"<(p|div|br|li)[^>]*>", " ", cell, flags=re.IGNORECASE)
    cell = re.sub(r"</(p|div|li)>", " ", cell, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cell)


def convert_custom(html: str) -> str:
    """Current extract.py _html_to_markdown — baseline."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    for n in range(6, 0, -1):
        text = re.sub(
            rf"<h{n}[^>]*>(.*?)</h{n}>",
            lambda m: "\n" + "#" * n + " " + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    text = _CELL_INNER_RE.sub(_flatten_cell_inner, text)
    text = re.sub(r"<tr[^>]*>", "\n| ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", " |", text, flags=re.IGNORECASE)
    text = re.sub(r"<t[dh][^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<(p|br|div|li)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for old, new in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def convert_pandoc(html: str) -> str:
    """pandoc html → gfm with pipe tables."""
    proc = subprocess.run(
        ["pandoc", "-f", "html", "-t", "gfm+pipe_tables", "--wrap=none"],
        input=html,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pandoc: {proc.stderr[:400]}")
    return proc.stdout


def convert_markdownify(html: str) -> str:
    import markdownify

    return markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])


# LaTeXML class → standard HTML tag mapping. arxiv.org/html renders
# tables with <span class="ltx_tabular/ltx_tr/ltx_td"> instead of the
# standard <table>/<tr>/<td>; lifting them back lets any HTML-aware
# converter see a real table.
_LTX_TAG_MAP = [
    ("ltx_tabular", "table"),
    ("ltx_thead", "thead"),
    ("ltx_tbody", "tbody"),
    ("ltx_tr", "tr"),
    ("ltx_th", "th"),
    ("ltx_td", "td"),
]


def _lift_ltx_spans(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("span"):
        classes = tag.get("class") or []
        for ltx_cls, standard in _LTX_TAG_MAP:
            if ltx_cls in classes:
                tag.name = standard
                break
    return str(soup)


def convert_lifted_markdownify(html: str) -> str:
    """Lift LaTeXML ltx_* spans to standard table tags, then markdownify."""
    import markdownify

    lifted = _lift_ltx_spans(html)
    return markdownify.markdownify(lifted, heading_style="ATX", strip=["script", "style"])


def convert_lifted_pandoc(html: str) -> str:
    lifted = _lift_ltx_spans(html)
    proc = subprocess.run(
        ["pandoc", "-f", "html", "-t", "gfm+pipe_tables", "--wrap=none"],
        input=lifted,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pandoc(lifted): {proc.stderr[:400]}")
    return proc.stdout


def convert_html2text(html: str) -> str:
    import html2text

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html)


def convert_pymupdf4llm(pdf_path: Path) -> str:
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf_path))


# ---------------------------------------------------------------------------
# quality signals
# ---------------------------------------------------------------------------

_BIB_RE = re.compile(r"^\s{0,3}#{1,6}\s*(References|Bibliography)\s*$", re.MULTILINE | re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_MATH_INLINE_RE = re.compile(r"\$[^\n$]+\$|\\\(.+?\\\)")


@dataclass
class Report:
    bytes: int
    has_bib: bool
    table_lines: int
    math_lines: int
    crashed: bool = False
    note: str = ""


def score(md: str) -> Report:
    if not md:
        return Report(bytes=0, has_bib=False, table_lines=0, math_lines=0, crashed=True, note="empty output")
    return Report(
        bytes=len(md),
        has_bib=bool(_BIB_RE.search(md)),
        table_lines=len(_TABLE_ROW_RE.findall(md)),
        math_lines=len(_MATH_INLINE_RE.findall(md)),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


CONVERTERS_HTML = [
    ("custom", convert_custom),
    ("pandoc", convert_pandoc),
    ("markdownify", convert_markdownify),
    ("html2text", convert_html2text),
    ("lifted_markdownify", convert_lifted_markdownify),
    ("lifted_pandoc", convert_lifted_pandoc),
]
CONVERTERS_PDF = [
    ("pymupdf4llm", convert_pymupdf4llm),
]


def run_one(arxiv_id: str, label: str) -> dict[str, Report]:
    print(f"\n=== {arxiv_id} · {label} ===")
    paper_dir = OUT_DIR / arxiv_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Report] = {}

    html = fetch_html(arxiv_id)
    if html is None:
        print(f"  HTML fetch failed for {arxiv_id}")
    else:
        (paper_dir / "source.html").write_text(html, encoding="utf-8")
        for name, fn in CONVERTERS_HTML:
            try:
                md = fn(html)
            except Exception as e:
                print(f"  {name}: FAIL — {e}")
                reports[name] = Report(0, False, 0, 0, crashed=True, note=str(e)[:120])
                continue
            (paper_dir / f"{name}.md").write_text(md, encoding="utf-8")
            r = score(md)
            reports[name] = r
            print(
                f"  {name:12} bytes={r.bytes:>7} bib={r.has_bib!s:5} tables={r.table_lines:>5} math={r.math_lines:>5}"
            )

    pdf_path = fetch_pdf(arxiv_id, paper_dir)
    if pdf_path is None:
        print(f"  PDF fetch failed for {arxiv_id}")
    else:
        for name, fn in CONVERTERS_PDF:
            try:
                md = fn(pdf_path)
            except Exception as e:
                print(f"  {name}: FAIL — {e}")
                reports[name] = Report(0, False, 0, 0, crashed=True, note=str(e)[:120])
                continue
            (paper_dir / f"{name}.md").write_text(md, encoding="utf-8")
            r = score(md)
            reports[name] = r
            print(
                f"  {name:12} bytes={r.bytes:>7} bib={r.has_bib!s:5} tables={r.table_lines:>5} math={r.math_lines:>5}"
            )

    return reports


def main() -> int:
    if not shutil.which("pandoc"):
        print("pandoc not found on PATH; install pandoc first")
        return 1

    all_reports: dict[str, dict[str, Report]] = {}
    for aid, label in SAMPLES:
        all_reports[aid] = run_one(aid, label)

    # summary
    names = ["custom", "pandoc", "markdownify", "html2text", "lifted_markdownify", "lifted_pandoc", "pymupdf4llm"]
    print("\n\n=== summary (bytes / tables / bib / math) ===\n")
    header = "paper        " + "  ".join(f"{n:>20}" for n in names)
    print(header)
    for aid, _ in SAMPLES:
        row = [f"{aid:12}"]
        for n in names:
            r = all_reports[aid].get(n)
            if r is None or r.crashed:
                cell = "FAIL"
            else:
                bib = "Y" if r.has_bib else "n"
                cell = f"{r.bytes:>6} t={r.table_lines:>3} b={bib} m={r.math_lines:>3}"
            row.append(f"{cell:>20}")
        print("  ".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
