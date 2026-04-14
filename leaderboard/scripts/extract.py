# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Extract benchmark scores from arxiv papers via LLM.

Two subcommands::

    uv run extract.py scan [--benchmark libero]   # discover citing papers
    uv run extract.py run 2505.05800 [--workers 4] # extract from papers

Pipeline: scan → run → build.py → validate.py → sync_external.py
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "papers"
EXTRACTIONS_DIR = ROOT / ".cache" / "extractions"
EXTRACTION_LOGS_DIR = ROOT / ".cache" / "extraction_logs"
EXTRACTIONS_JSON = DATA_DIR / "extractions.json"
BENCHMARKS_DIR = ROOT / "benchmarks"
BENCHMARKS_JSON_PATH = DATA_DIR / "benchmarks.json"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"
COVERAGE_PATH = DATA_DIR / "coverage.json"
SCAN_CACHE_PATH = ROOT / ".cache" / "scan_results.json"
FETCH_FAILURES_PATH = CACHE_DIR / "fetch_failures.json"

PAPER_BUDGET = 80_000
_ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d+\.\d+)")

# Lock for thread-safe fetch failure writes
_failures_lock = threading.Lock()


def _extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Benchmark protocol files (leaderboard/benchmarks/*.md)
# ---------------------------------------------------------------------------


def _load_benchmark_md(bm_key: str) -> str:
    path = BENCHMARKS_DIR / f"{bm_key}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3 :].strip()
    return text


def _load_all_benchmark_rules() -> str:
    """Load all benchmark protocol files into a single string for the LLM prompt."""
    parts = []
    global_md = _load_benchmark_md("_global")
    if global_md:
        parts.append(f"## Global Rules\n\n{global_md}")

    for f in sorted(BENCHMARKS_DIR.glob("*.md")):
        if f.stem == "_global":
            continue
        text = _load_benchmark_md(f.stem)
        if text:
            parts.append(f"## Benchmark: {f.stem}\n\n{text}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Paper cache (fetch + HTML→markdown)
# ---------------------------------------------------------------------------

_CELL_INNER_RE = re.compile(r"<t[dh][^>]*>.*?</t[dh]>", re.DOTALL | re.IGNORECASE)


def _flatten_cell_inner(match: re.Match[str]) -> str:
    cell = match.group(0)
    cell = re.sub(r"<(p|div|br|li)[^>]*>", " ", cell, flags=re.IGNORECASE)
    cell = re.sub(r"</(p|div|li)>", " ", cell, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cell)


def _html_to_markdown(html: str) -> str:
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
    for old, new in [
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&ndash;", "-"),
        ("&mdash;", "\u2014"),
    ]:
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _paper_md_path(arxiv_id: str) -> Path:
    return CACHE_DIR / arxiv_id / "paper.md"


def _paper_meta_path(arxiv_id: str) -> Path:
    return CACHE_DIR / arxiv_id / "meta.json"


def _is_paper_cached(arxiv_id: str) -> bool:
    return _paper_md_path(arxiv_id).exists()


def _fetch_url(url: str, timeout: int = 30) -> str | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VLA-Leaderboard-Extract/1.0 (+https://github.com/allenai/vla-evaluation-harness)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if attempt == 2:
                return None
            time.sleep(5)
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt < 2:
                time.sleep(5)
                continue
            return None
    return None


def _fetch_paper(arxiv_id: str) -> bool:
    cache_dir = CACHE_DIR / arxiv_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source, url in [
        ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"),
        ("arxiv", f"https://arxiv.org/html/{arxiv_id}"),
    ]:
        html = _fetch_url(url)
        if html is None or len(html) < 2000:
            continue
        markdown = _html_to_markdown(html)
        if len(markdown) < 1000:
            continue
        _paper_md_path(arxiv_id).write_text(markdown, encoding="utf-8")
        ph = "sha256:" + hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        _paper_meta_path(arxiv_id).write_text(
            json.dumps(
                {
                    "arxiv_id": arxiv_id,
                    "source": source,
                    "fetched_at": _now_iso(),
                    "url": url,
                    "paper_hash": ph,
                    "bytes": len(markdown),
                },
                indent=2,
            )
            + "\n"
        )
        return True
    return False


def _load_fetch_failures() -> dict[str, str]:
    return json.loads(FETCH_FAILURES_PATH.read_text()) if FETCH_FAILURES_PATH.exists() else {}


def _save_fetch_failures(failures: dict[str, str]) -> None:
    FETCH_FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FETCH_FAILURES_PATH.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")


def _record_failure(arxiv_id: str, reason: str) -> None:
    """Thread-safe: append one failure and flush to disk immediately."""
    with _failures_lock:
        failures = _load_fetch_failures()
        failures[arxiv_id] = reason
        _save_fetch_failures(failures)


def _load_paper_markdown(arxiv_id: str) -> str | None:
    p = _paper_md_path(arxiv_id)
    return p.read_text(encoding="utf-8") if p.exists() else None


def _paper_hash(arxiv_id: str) -> str | None:
    meta = _paper_meta_path(arxiv_id)
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("paper_hash")


# ---------------------------------------------------------------------------
# Extraction cache (per-paper)
# ---------------------------------------------------------------------------


def _extraction_cache_path(arxiv_id: str) -> Path:
    return EXTRACTIONS_DIR / f"{arxiv_id}.json"


@functools.lru_cache(maxsize=1)
def _current_benchmark_keys() -> frozenset[str]:
    return frozenset(f.stem for f in BENCHMARKS_DIR.glob("*.md") if f.stem != "_global")


def _load_cached_extraction(arxiv_id: str) -> dict | None:
    p = _extraction_cache_path(arxiv_id)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("paper_hash") != _paper_hash(arxiv_id):
        return None
    # Invalidate if benchmarks were added since extraction
    if set(data.get("extraction_scope", [])) != _current_benchmark_keys():
        return None
    return data


def _save_cached_extraction(arxiv_id: str, data: dict) -> None:
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _extraction_cache_path(arxiv_id).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    pass


EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "required": ["benchmarks", "confidence"],
    "properties": {
        "benchmarks": {
            "type": "array",
            "description": "One entry per benchmark found in this paper. Empty array if the paper does not evaluate any known benchmark.",
            "items": {
                "type": "object",
                "required": ["benchmark", "models"],
                "properties": {
                    "benchmark": {
                        "type": "string",
                        "description": "Benchmark key exactly as listed in the rules (e.g. 'libero', 'calvin', 'simpler_env')",
                    },
                    "models": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["label", "scores"],
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Model label as it appears in the paper's results table",
                                },
                                "label_quote": {"type": ["string", "null"]},
                                "params": {"type": ["string", "null"]},
                                "params_quote": {"type": ["string", "null"]},
                                "weight_type": {"type": "string", "enum": ["shared", "finetuned", "unknown"]},
                                "weight_type_quote": {"type": ["string", "null"]},
                                "is_score_original": {
                                    "type": "string",
                                    "enum": ["original", "cited_baseline", "reproduction", "unknown"],
                                },
                                "attribution_quote": {"type": ["string", "null"]},
                                "scores": {
                                    "type": "object",
                                    "properties": {
                                        "overall_score": {"type": ["number", "null"]},
                                        "overall_score_quote": {"type": ["string", "null"]},
                                        "suite_scores": {
                                            "type": "object",
                                            "additionalProperties": {
                                                "type": "object",
                                                "required": ["value", "quote"],
                                                "properties": {
                                                    "value": {"type": "number"},
                                                    "quote": {"type": "string"},
                                                },
                                            },
                                        },
                                        "task_scores": {
                                            "type": "object",
                                            "additionalProperties": {
                                                "type": "object",
                                                "required": ["value", "quote"],
                                                "properties": {
                                                    "value": {"type": "number"},
                                                    "quote": {"type": "string"},
                                                },
                                            },
                                        },
                                        "source_table": {"type": ["string", "null"]},
                                    },
                                },
                                "protocol": {
                                    "type": "object",
                                    "required": ["matches_standard", "rationale"],
                                    "properties": {
                                        "matches_standard": {
                                            "type": "string",
                                            "enum": ["yes", "no", "partial", "unknown"],
                                        },
                                        "rationale": {"type": "string"},
                                        "evidence_quote": {"type": ["string", "null"]},
                                    },
                                },
                            },
                        },
                    },
                    "risky_patterns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "answer"],
                            "properties": {
                                "id": {"type": "string"},
                                "answer": {"type": "string", "enum": ["yes", "no", "unknown"]},
                                "quote": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}


def _build_system_prompt(all_rules: str) -> str:
    return f"""You are curating entries for a public VLA benchmark leaderboard.

A leaderboard entry represents a distinct, publicly identifiable VLA model or
method. Your job is to extract only rows that belong on such a leaderboard,
not every row in every table.

## Inclusion criteria (ALL must hold)

A model is eligible ONLY if ALL of the following are true:

1. **Public name**: it has a specific, canonical name a reader could Google.
   Examples of ELIGIBLE names: "OpenVLA", "RT-2", "π₀", "Diffusion Policy",
   "3D Diffuser Actor", "CogACT".
   NEVER extract rows labeled: "Ours", "Our Method", "Our Model", "Proposed",
   "This Work", "Baseline", "Ablation", or anything that is only meaningful
   inside the paper. If the only label is "Ours", find the method's actual
   name from the title/abstract — and if there is none, SKIP the row.

2. **Primary configuration**: it represents a distinct method, not a minor
   variant along one axis. SKIP rows that are ablations, hyperparameter
   sweeps, training-stage snapshots, or post-processing variants of a
   primary method. Specifically skip rows whose only differentiator is:
   - quantization scheme (INT4, INT8, FP8, AWQ, PTQ, QAT, GPTQ, GGUF, ...)
   - parameter-efficient tuning (LoRA, QLoRA, adapter, prefix-tuning, ...)
   - data/training-stage variant ("w/o pretrain", "stage 1", "50% data", ...)
   - horizon/action-chunk hyperparameters ("k=1", "chunk=8", ...)
   - a minor architecture tweak marked with a suffix like "+feature X"
   Unless such a variant IS the paper's main contribution (e.g. a paper
   whose core claim is about quantization), treat it as an ablation and
   skip it.

3. **Score attribution**: the row reports a concrete numerical score on a
   listed benchmark that the paper either ran itself or cites verbatim.
   Skip rows with only qualitative notes or with no recoverable number.

## Classification

For each eligible model, set `is_score_original`:
- `original` — paper ran this model itself (new run, their proposed method
  or their re-run of a baseline)
- `cited_baseline` — number quoted from another paper, not re-run here
- `reproduction` — paper explicitly marks it as their reproduction of prior work
- `unknown` — genuinely cannot tell

And `weight_type`: `shared` (same checkpoint across benchmarks) or
`finetuned` (trained specifically on this benchmark's data).

## Hard rules

- Every extracted score MUST carry a verbatim `quote` from the paper.
- If you cannot find a value, return null. Never guess or compute.
- Use the exact benchmark key as listed (e.g. "libero", "calvin").
- For each benchmark found, answer its risky patterns.
- Be conservative. A paper whose entire contribution is a survey,
  reproduction study, or evaluation harness (not a new method) should
  usually return an empty `benchmarks` array — those rows are already
  on the leaderboard via their original papers.

{all_rules}
"""


def _call_claude_cli(
    system_prompt: str,
    user_content: str,
    json_schema: dict,
    model: str = "sonnet",
    timeout: int = 600,
    log_path: Path | None = None,
) -> dict:
    """Call claude CLI with stream-json to capture thinking and text blocks.

    When log_path is given, writes a plaintext log of thinking/text blocks there.
    """
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--json-schema",
        json.dumps(json_schema),
        "--output-format",
        "stream-json",
        "--verbose",
        "--disallowedTools",
        "Bash,Write,Edit,NotebookEdit,WebFetch,WebSearch,Read,Grep,Glob",
        "--permission-mode",
        "default",
        "--no-session-persistence",
    ]
    try:
        result = subprocess.run(cmd, input=user_content, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise LLMError("claude CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"timed out after {timeout}s") from e
    if result.returncode != 0:
        raise LLMError(f"exit {result.returncode}: {result.stderr[:500]}")

    # Parse stream-json: one JSON event per line
    structured: dict | None = None
    log_blocks: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("is_error"):
            raise LLMError(f"error: {evt.get('subtype')}")
        # Collect thinking and text blocks from assistant messages
        if evt.get("type") == "assistant":
            for block in evt.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    log_blocks.append("### thinking\n" + block.get("thinking", ""))
                elif btype == "text":
                    log_blocks.append("### text\n" + block.get("text", ""))
        # Final result event carries structured_output
        if evt.get("type") == "result":
            structured = evt.get("structured_output")

    if log_path is not None and log_blocks:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n\n".join(log_blocks) + "\n", encoding="utf-8")

    if not isinstance(structured, dict):
        raise LLMError("no structured_output in stream")
    return structured


# ---------------------------------------------------------------------------
# Core extraction (per-paper)
# ---------------------------------------------------------------------------


def extract_one(arxiv_id: str, all_rules: str, model: str, *, timeout: int = 600, resume: bool = True) -> dict | None:
    """Extract all benchmark results from one paper. Returns extraction dict or None on failure."""
    if resume:
        cached = _load_cached_extraction(arxiv_id)
        if cached is not None:
            return cached

    # Auto-fetch if not cached, then load
    paper_md = _load_paper_markdown(arxiv_id)
    if paper_md is None:
        if not _fetch_paper(arxiv_id):
            _record_failure(arxiv_id, f"HTML not available, {_now_iso()}")
            return None
        paper_md = _load_paper_markdown(arxiv_id)
        if paper_md is None:
            return None

    # Trim paper if over budget
    if len(paper_md) > PAPER_BUDGET:
        paper_md = paper_md[:PAPER_BUDGET] + f"\n\n[...truncated, original {len(paper_md)} chars...]"

    system_prompt = _build_system_prompt(all_rules)
    user_prompt = f"Extract all benchmark results from this paper.\n\n<paper>\n{paper_md}\n</paper>"

    try:
        log_path = EXTRACTION_LOGS_DIR / f"{arxiv_id}.log"
        llm_output = _call_claude_cli(
            system_prompt, user_prompt, EXTRACTION_SCHEMA, model=model, timeout=timeout, log_path=log_path
        )
    except LLMError as e:
        print(f"    LLM error for {arxiv_id}: {e}")
        return None

    result = {
        "arxiv_id": arxiv_id,
        "extracted_at": _now_iso(),
        "model_used": model,
        "paper_hash": _paper_hash(arxiv_id),
        "extraction_scope": sorted(_current_benchmark_keys()),
        "benchmarks": llm_output.get("benchmarks", []),
        "confidence": llm_output.get("confidence"),
    }
    _save_cached_extraction(arxiv_id, result)
    return result


# ---------------------------------------------------------------------------
# S2 citation API
# ---------------------------------------------------------------------------


def _fetch_s2_citations(arxiv_id: str, limit: int = 1000) -> tuple[list[dict], int]:
    """Return (arxiv-citing papers, total citation count).

    The total count includes non-arxiv citations (conference/journal papers
    without arxiv preprints) — those cannot be processed by our pipeline but
    are shown as the coverage denominator to reflect real-world citation scale.
    """
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    all_papers: list[dict] = []
    total_count = 0
    offset = 0
    while True:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/citations?fields=externalIds,title&limit={limit}&offset={offset}"
        req = urllib.request.Request(url, headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, OSError):
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise
        for item in data.get("data", []):
            total_count += 1
            paper = item.get("citingPaper", {})
            aid = (paper.get("externalIds") or {}).get("ArXiv")
            if aid:
                all_papers.append({"arxiv_id": aid, "title": paper.get("title", "")})
        if data.get("next") is None:
            break
        offset = data["next"]
        time.sleep(1)
    return all_papers, total_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Extract benchmark scores from arxiv papers via LLM.", add_completion=False)


@app.command()
def scan(
    benchmark: Annotated[Optional[str], typer.Option(help="Only scan one benchmark.")] = None,
) -> None:
    """Discover citing papers for each benchmark via Semantic Scholar."""
    if not BENCHMARKS_JSON_PATH.exists():
        print(f"{BENCHMARKS_JSON_PATH} not found.")
        raise typer.Exit(1)
    benchmarks = json.loads(BENCHMARKS_JSON_PATH.read_text())
    scan_results: dict[str, dict] = {}
    total_new = 0
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    extracted_stems = {f.stem for f in EXTRACTIONS_DIR.glob("*.json")}

    for bm_key, bm in sorted(benchmarks.items()):
        if benchmark and bm_key != benchmark:
            continue
        bm_arxiv = _extract_arxiv_id(bm.get("paper_url", ""))
        if not bm_arxiv:
            print(f"  {bm_key}: no paper_url, skipping")
            continue
        print(f"  {bm_key} ({bm_arxiv}): fetching citations...")
        try:
            citing, total_count = _fetch_s2_citations(bm_arxiv)
        except Exception as e:
            print(f"    error: {e}")
            continue
        citing_ids = {p["arxiv_id"] for p in citing}
        reviewed = extracted_stems & citing_ids
        new_ids = citing_ids - reviewed
        total_new += len(new_ids)
        scan_results[bm_key] = {
            "arxiv_id": bm_arxiv,
            "display_name": bm.get("display_name", bm_key),
            "citing_papers": total_count,
            "arxiv_citing_papers": len(citing),
            "extracted": len(reviewed),
            "all_citing_ids": sorted(citing_ids),
            "new_papers": sorted(new_ids),
            "new_paper_titles": {p["arxiv_id"]: p["title"] for p in citing if p["arxiv_id"] in new_ids},
        }
        print(f"    {total_count} citing ({len(citing)} arxiv), {len(reviewed)} extracted, {len(new_ids)} new")

    SCAN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE_PATH.write_text(
        json.dumps({"scanned_at": _now_iso(), "benchmarks": scan_results}, indent=2, ensure_ascii=False) + "\n"
    )

    _update_coverage()

    print(f"\n{total_new} new papers across {len(scan_results)} benchmarks")
    print(f"Wrote {SCAN_CACHE_PATH}")
    print(f"Wrote {COVERAGE_PATH}")


def _update_coverage() -> None:
    """Recompute coverage.json from scan_results.json + current extractions on disk.

    Counts extracted papers per benchmark by intersecting each benchmark's
    citing_ids with the set of locally extracted arxiv IDs.
    """
    if not BENCHMARKS_JSON_PATH.exists():
        return
    benchmarks = json.loads(BENCHMARKS_JSON_PATH.read_text())
    extracted_stems = {f.stem for f in EXTRACTIONS_DIR.glob("*.json")} if EXTRACTIONS_DIR.exists() else set()

    scan_data: dict = {}
    if SCAN_CACHE_PATH.exists():
        scan_data = json.loads(SCAN_CACHE_PATH.read_text()).get("benchmarks", {})

    # Also pull totals from leaderboard.json if it exists (for the summary header)
    total_results = 0
    total_models = 0
    if LEADERBOARD_PATH.exists():
        lb = json.loads(LEADERBOARD_PATH.read_text())
        results = lb.get("results", [])
        total_results = len(results)
        total_models = len({r.get("model") for r in results if r.get("model")})

    coverage = {
        "last_updated": _now_iso()[:10],
        "total_papers_reviewed": len(extracted_stems),
        "total_results": total_results,
        "total_models": total_models,
        "benchmarks": {},
    }
    for bm_key in sorted(benchmarks):
        sr = scan_data.get(bm_key, {})
        citing_ids = set(sr.get("all_citing_ids", []))
        extracted_for_bm = len(citing_ids & extracted_stems) if citing_ids else 0
        coverage["benchmarks"][bm_key] = {
            "display_name": benchmarks[bm_key].get("display_name", bm_key),
            "arxiv_id": sr.get("arxiv_id", ""),
            "citing_papers": sr.get("citing_papers", 0),
            "arxiv_citing_papers": sr.get("arxiv_citing_papers", 0),
            "papers_reviewed": extracted_for_bm,
        }
    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n")


@app.command()
def run(
    arxiv_ids: Annotated[
        Optional[list[str]], typer.Argument(help="Arxiv IDs to extract. Omit to use --from-scan.")
    ] = None,
    from_scan: Annotated[bool, typer.Option("--from-scan", help="Extract all papers from scan_results.json.")] = False,
    model: Annotated[str, typer.Option(help="Claude model alias.")] = "sonnet",
    workers: Annotated[int, typer.Option(help="Parallel workers.")] = 1,
    timeout: Annotated[int, typer.Option(help="Claude CLI timeout in seconds.")] = 600,
    resume: Annotated[bool, typer.Option(help="Skip papers with fresh cache.")] = True,
) -> None:
    """Extract benchmark results from papers via LLM (one call per paper)."""
    if from_scan:
        if not SCAN_CACHE_PATH.exists():
            print("scan_results.json not found — run scan first.")
            raise typer.Exit(1)
        scan_data = json.loads(SCAN_CACHE_PATH.read_text())
        # Collect ALL unique citing papers across all benchmarks
        all_ids: set[str] = set()
        for bm_data in scan_data.get("benchmarks", {}).values():
            all_ids.update(bm_data.get("all_citing_ids", []))
            all_ids.update(bm_data.get("new_papers", []))
        targets = sorted(all_ids)
    elif arxiv_ids:
        targets = arxiv_ids
    else:
        print("Provide arxiv IDs or use --from-scan.")
        raise typer.Exit(2)

    if resume:
        before = len(targets)
        targets = [aid for aid in targets if _load_cached_extraction(aid) is None]
        skipped = before - len(targets)
        if skipped:
            print(f"--resume: skipping {skipped} cached papers")

    print(f"Extracting {len(targets)} papers (workers={workers}, model={model}, timeout={timeout}s)...")
    all_rules = _load_all_benchmark_rules()

    def _do(arxiv_id: str) -> tuple[str, dict | None]:
        return arxiv_id, extract_one(arxiv_id, all_rules, model, timeout=timeout, resume=False)

    def _tally(aid: str, result: dict | None, counters: list[int]) -> None:
        """Update counters [ok, empty, fail] and print progress."""
        if result is None:
            counters[2] += 1
            print(f"  FAIL {aid}")
        elif not result.get("benchmarks"):
            counters[1] += 1
        else:
            n_bm = len(result["benchmarks"])
            n_models = sum(len(b.get("models", [])) for b in result["benchmarks"])
            counters[0] += 1
            print(f"  OK   {aid} ({n_bm} benchmarks, {n_models} models)")
        total = sum(counters)
        if total % 20 == 0:
            print(f"  --- {total}/{len(targets)} ---")

    counters = [0, 0, 0]  # [ok, empty, fail]

    if workers <= 1:
        for aid in targets:
            aid, result = _do(aid)
            _tally(aid, result, counters)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit in batches (2x workers) for graceful Ctrl+C
            futs: dict = {}
            target_iter = iter(targets)
            for aid in itertools.islice(target_iter, workers * 2):
                futs[executor.submit(_do, aid)] = aid
            while futs:
                for fut in as_completed(futs):
                    aid = futs.pop(fut)
                    try:
                        _, result = fut.result()
                    except Exception as exc:
                        print(f"  CRASH {aid}: {exc}")
                        counters[2] += 1
                        result = None  # skip _tally, already counted
                    else:
                        _tally(aid, result, counters)
                    # Refill from iterator
                    next_aid = next(target_iter, None)
                    if next_aid is not None:
                        futs[executor.submit(_do, next_aid)] = next_aid
                    break  # back to as_completed with updated futs

    n_ok, n_empty, n_fail = counters
    print(f"\nDone: ok={n_ok} empty={n_empty} fail={n_fail} total={len(targets)}")
    failures = _load_fetch_failures()
    if failures:
        print(f"{len(failures)} papers in fetch_failures.json")

    _update_coverage()
    print(f"Updated {COVERAGE_PATH}")


@app.command()
def coverage() -> None:
    """Recompute data/coverage.json from current .cache/extractions/ state."""
    _update_coverage()
    print(f"Wrote {COVERAGE_PATH}")


@app.command()
def pack() -> None:
    """Pack .cache/extractions/*.json → data/extractions.json for git commit."""
    files = sorted(EXTRACTIONS_DIR.glob("*.json"))
    if not files:
        print("No extractions to pack.")
        raise typer.Exit(1)
    entries = []
    for f in files:
        entries.append(json.loads(f.read_text()))
    entries.sort(key=lambda e: e.get("arxiv_id", ""))
    # Sort internal arrays for stable diffs
    for entry in entries:
        if bms := entry.get("benchmarks"):
            bms.sort(key=lambda b: b.get("benchmark", ""))
            for bm in bms:
                if models := bm.get("models"):
                    models.sort(key=lambda m: m.get("label", ""))
    EXTRACTIONS_JSON.write_text(json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    print(f"Packed {len(entries)} extractions → {EXTRACTIONS_JSON}")


@app.command()
def unpack() -> None:
    """Unpack data/extractions.json → .cache/extractions/*.json for local work."""
    if not EXTRACTIONS_JSON.exists():
        print(f"{EXTRACTIONS_JSON} not found.")
        raise typer.Exit(1)
    entries = json.loads(EXTRACTIONS_JSON.read_text())
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        aid = entry["arxiv_id"]
        path = EXTRACTIONS_DIR / f"{aid}.json"
        path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")
    print(f"Unpacked {len(entries)} extractions → {EXTRACTIONS_DIR}")


if __name__ == "__main__":
    app()
