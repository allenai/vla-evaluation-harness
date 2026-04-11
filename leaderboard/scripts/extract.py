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
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "papers"
EXTRACTIONS_DIR = ROOT / "extractions"
BENCHMARKS_DIR = ROOT / "benchmarks"
BENCHMARKS_JSON_PATH = DATA_DIR / "benchmarks.json"
LEADERBOARD_PATH = DATA_DIR / "leaderboard.json"
COVERAGE_PATH = DATA_DIR / "coverage.json"
SCAN_CACHE_PATH = ROOT / ".cache" / "scan_results.json"
FETCH_FAILURES_PATH = CACHE_DIR / "fetch_failures.json"

PAPER_BUDGET = 80_000
_ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d+\.\d+)")

# Thread-safe accumulator for fetch failures (flushed at end of run)
_pending_failures: dict[str, str] = {}


def _extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def _now_iso() -> str:
    from datetime import datetime, timezone

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


@functools.lru_cache(maxsize=None)
def _load_paper_markdown(arxiv_id: str) -> str | None:
    p = _paper_md_path(arxiv_id)
    return p.read_text(encoding="utf-8") if p.exists() else None


@functools.lru_cache(maxsize=None)
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
    return f"""You are extracting benchmark scores from an academic paper.

Below is a list of VLA benchmarks with their evaluation protocols and rules.
Your job: read the paper and extract results for ALL benchmarks that appear.
If the paper does not evaluate a benchmark, skip it (do not include it in
the output). If the paper evaluates none of the listed benchmarks, return
an empty benchmarks array.

Hard rules:
1. Every extracted value MUST have a verbatim quote from the paper.
2. If you cannot find a value, return null. Never guess or compute.
3. Extract ALL models in each benchmark's results table, not just the
   paper's proposed model.
4. Use the exact benchmark key as listed (e.g. "libero", "calvin").
5. For each model, judge weight_type and is_score_original.
6. For each benchmark found, answer its risky patterns.

{all_rules}
"""


def _call_claude_cli(
    system_prompt: str, user_content: str, json_schema: dict, model: str = "sonnet", timeout: int = 600
) -> dict:
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
        "json",
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
    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise LLMError(f"non-JSON: {e}") from e
    if envelope.get("is_error"):
        raise LLMError(f"error: {envelope.get('subtype')}")
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        raise LLMError(f"no structured_output (keys: {sorted(envelope)})")
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
            _pending_failures[arxiv_id] = f"HTML not available, {_now_iso()}"
            return None
        _load_paper_markdown.cache_clear()
        paper_md = _load_paper_markdown(arxiv_id)
        if paper_md is None:
            return None

    # Trim paper if over budget
    if len(paper_md) > PAPER_BUDGET:
        paper_md = paper_md[:PAPER_BUDGET] + f"\n\n[...truncated, original {len(paper_md)} chars...]"

    system_prompt = _build_system_prompt(all_rules)
    user_prompt = f"Extract all benchmark results from this paper.\n\n<paper>\n{paper_md}\n</paper>"

    try:
        llm_output = _call_claude_cli(system_prompt, user_prompt, EXTRACTION_SCHEMA, model=model, timeout=timeout)
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


def _fetch_s2_citations(arxiv_id: str, limit: int = 1000) -> list[dict]:
    import os

    headers = {"Accept": "application/json"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    all_papers: list[dict] = []
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
            paper = item.get("citingPaper", {})
            aid = (paper.get("externalIds") or {}).get("ArXiv")
            if aid:
                all_papers.append({"arxiv_id": aid, "title": paper.get("title", "")})
        if data.get("next") is None:
            break
        offset = data["next"]
        time.sleep(1)
    return all_papers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import typer  # noqa: E402
from typing import Annotated, Optional  # noqa: E402

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
            citing = _fetch_s2_citations(bm_arxiv)
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
            "citing_papers": len(citing),
            "extracted": len(reviewed),
            "all_citing_ids": sorted(citing_ids),
            "new_papers": sorted(new_ids),
            "new_paper_titles": {p["arxiv_id"]: p["title"] for p in citing if p["arxiv_id"] in new_ids},
        }
        print(f"    {len(citing)} citing, {len(reviewed)} extracted, {len(new_ids)} new")

    SCAN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE_PATH.write_text(
        json.dumps({"scanned_at": _now_iso(), "benchmarks": scan_results}, indent=2, ensure_ascii=False) + "\n"
    )

    # Update coverage.json
    coverage = {
        "last_updated": _now_iso()[:10],
        "total_extracted": len(extracted_stems),
        "benchmarks": {},
    }
    for bm_key in sorted(benchmarks):
        sr = scan_results.get(bm_key, {})
        coverage["benchmarks"][bm_key] = {
            "display_name": benchmarks[bm_key].get("display_name", bm_key),
            "arxiv_id": sr.get("arxiv_id", ""),
            "citing_papers": sr.get("citing_papers", 0),
            "extracted": sr.get("extracted", 0),
        }
    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{total_new} new papers across {len(scan_results)} benchmarks")
    print(f"Wrote {SCAN_CACHE_PATH}")
    print(f"Wrote {COVERAGE_PATH}")


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

    import signal
    import threading

    lock = threading.Lock()
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(130))

    def _do(arxiv_id: str) -> tuple[str, dict | None]:
        return arxiv_id, extract_one(arxiv_id, all_rules, model, timeout=timeout, resume=False)

    n_ok, n_empty, n_fail = 0, 0, 0
    if workers <= 1:
        for aid in targets:
            aid, result = _do(aid)
            if result is None:
                n_fail += 1
                print(f"  FAIL {aid}")
            elif not result.get("benchmarks"):
                n_empty += 1
            else:
                n_bm = len(result["benchmarks"])
                n_models = sum(len(b.get("models", [])) for b in result["benchmarks"])
                n_ok += 1
                print(f"  OK   {aid} ({n_bm} benchmarks, {n_models} models)")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_do, aid): aid for aid in targets}
            for fut in as_completed(futs):
                aid, result = fut.result()
                with lock:
                    if result is None:
                        n_fail += 1
                        print(f"  FAIL {aid}")
                    elif not result.get("benchmarks"):
                        n_empty += 1
                    else:
                        n_bm = len(result["benchmarks"])
                        n_models = sum(len(b.get("models", [])) for b in result["benchmarks"])
                        n_ok += 1
                        print(f"  OK   {aid} ({n_bm} benchmarks, {n_models} models)")
                    total = n_ok + n_empty + n_fail
                    if total % 20 == 0:
                        print(f"  --- {total}/{len(targets)} ---")

    # Flush accumulated fetch failures
    if _pending_failures:
        existing = _load_fetch_failures()
        existing.update(_pending_failures)
        _save_fetch_failures(existing)
        _pending_failures.clear()

    print(f"\nDone: ok={n_ok} empty={n_empty} fail={n_fail} total={len(targets)}")
    failures = _load_fetch_failures()
    if failures:
        print(f"{len(failures)} papers in fetch_failures.json")


if __name__ == "__main__":
    app()
