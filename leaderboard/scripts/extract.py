# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Extract benchmark scores from arxiv papers via LLM.

Two subcommands::

    uv run extract.py scan [--benchmark libero]   # discover citing papers
    uv run extract.py run 2505.05800 [--workers 4] # extract from papers

Pipeline: scan → run → refine.py → validate.py → sync_external.py

Output shape is defined by leaderboard/data/extraction.schema.json. This
script loads that schema at runtime — field semantics live there, not
duplicated in the prompt.
"""

from __future__ import annotations

import functools
import hashlib
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
EXTRACTION_SCHEMA_PATH = DATA_DIR / "extraction.schema.json"
SCAN_CACHE_PATH = ROOT / ".cache" / "scan_results.json"
FETCH_FAILURES_PATH = CACHE_DIR / "fetch_failures.json"

DEFAULT_MODEL = "claude-opus-4-6[1m]"
DEFAULT_TIMEOUT = 2400
DEFAULT_BATCH_SIZE = 30
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


@functools.lru_cache(maxsize=1)
def _extraction_schema() -> dict:
    """Load the authoritative per-paper extraction schema."""
    return json.loads(EXTRACTION_SCHEMA_PATH.read_text())


# The claude CLI's `--json-schema` mode does not reliably populate
# `structured_output` for schemas larger than a few fields (the LLM
# falls back to emitting a JSON code block in assistant text, which the
# CLI does not convert). Instead of relying on that path, we instruct
# the LLM to Write a partial file per paper and post-process.


def _build_system_prompt(all_rules: str) -> str:
    return f"""You are the EXTRACT stage of a two-stage VLA leaderboard pipeline.

Your objective at this stage is recall, not precision. Surface every
row that could belong on the leaderboard. A downstream PRECISION stage
applies eligibility filters, dedup, canonical-name cleanup, and notes
composition — do not pre-filter for those concerns. When uncertain,
extract; it is better to surface a row that later gets dropped than to
silently lose a real measurement.

Baseline-comparison and related-work tables in a paper often hold the
only record of a given model on a benchmark (the original paper may
never reach extraction). Extract every row in every comparison table.

Field semantics and output structure are defined by the JSON schema you
write against. The rules below cover decisions that depend on paper
context (the schema alone cannot specify them).

## Scope per benchmark

For every benchmark listed in the rules below:

1. Grep the paper for the benchmark's key name, display name, and the
   suite/task names in its Standard.
2. Paper scores it → add to `benchmarks[]`.
3. Paper mentions it without an extractable score → add to
   `benchmarks_absent` with a one-line reason.
4. Paper doesn't mention it → omit from both.

Return `benchmarks: []` only for pure theory/survey papers with no
evaluation table.

## Resolving generic labels

When a results-table label is generic ("Ours", "Our Method", "Proposed",
"Baseline", "(b)", "variant X"), look up the method's real name in the
paper's title, abstract, or method section and emit that canonical name
as `name_in_paper`. Downstream stages cannot redo this lookup.

## Resolving model_paper

For every row, set `model_paper` to the URL of the paper that introduces
the method. Find it in the paper's reference list. Any URL is valid —
arxiv, ACL Anthology, DOI, tech report. null only when the method has
no public paper.

## Resolving cited_paper

When `is_score_original='cited_baseline'`, set `cited_paper` to the URL
the score is attributed to. Resolve arxiv references via the paper's
bibliography. Non-arxiv sources (official github, tech reports, blogs)
are valid URLs too. Leave null when the paper quotes a number without
naming a source.

## Normalize scale

Emit numeric values in the benchmark's declared `metric.range`. If the
paper reports on a different scale (commonly 0–1 for a 0–100 benchmark),
convert before emitting. Quotes stay verbatim from the paper.

## Preserve non-standard tasks

A benchmark's declared task list identifies the standard protocol, not
the set of allowed keys. Tasks outside that list (non-standard protocols)
stay in `task_scores` under the paper's verbatim names; the row's
`protocol.matches_standard` becomes 'no' but the data is preserved.

## Exclude ablation variants

Skip rows whose only differentiator is quantization (INT4, AWQ, GPTQ),
parameter-efficient tuning (LoRA, adapter), training-stage variant
("w/o pretrain", "50% data"), or hyperparameter sweep ("k=1",
"chunk=8") — unless that variant is the paper's main contribution.

## Self-check before emitting

- Every numeric value has a matching *_quote from the paper. If not
  locatable, set the value to null.
- Every claim in `protocol.rationale` (task count, demo count, split,
  embodiment) has a matching evidence_quote. Unsupported claims →
  downgrade matches_standard one step toward 'unknown'.
- If the rationale describes any Checks violation, matches_standard is
  'no' — not 'yes'.

## Benchmark rules

Each block below opens with **Standard**: (the canonical protocol).
Scoring prescribes the JSON shape for scores. Checks lists yes/no
questions; failing any → matches_standard='no'. Methodology axes are
variance dimensions — differences along these still allow 'yes'.

{all_rules}
"""


EXTRACTIONS_RAW_DIR = ROOT / ".cache" / "extractions_raw"


def _call_claude_cli(
    system_prompt: str,
    user_prompt: str,
    extra_add_dirs: list[Path],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    log_path: Path | None = None,
) -> int:
    """Invoke the claude CLI. Returns n_tool_calls observed in the stream.

    Writes raw stream-json stdout to ``log_path``. Raises LLMError on
    non-zero exit or no final result event.
    """
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        # Restrict to Claude Code native tools; block MCP servers and
        # user skills that might delegate to outside knowledge sources.
        "--strict-mcp-config",
        "--disable-slash-commands",
    ]
    for d in extra_add_dirs:
        cmd += ["--add-dir", str(d.resolve())]
    try:
        result = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise LLMError("claude CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"timed out after {timeout}s") from e
    if result.returncode != 0:
        raise LLMError(f"exit {result.returncode}: {result.stderr[:500]}")

    n_tool_calls = 0
    seen_result = False
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
        if evt.get("type") == "assistant":
            for block in evt.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    n_tool_calls += 1
        if evt.get("type") == "result":
            seen_result = True

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")

    if not seen_result:
        raise LLMError("no result event in stream")
    return n_tool_calls


def _call_claude_cli_with_retry(
    system_prompt: str,
    user_prompt: str,
    extra_add_dirs: list[Path],
    model: str,
    timeout: int,
    log_path: Path | None,
    retries: int = 1,
) -> int:
    """Retry once on transient failure."""
    last_err: LLMError | None = None
    for attempt in range(retries + 1):
        try:
            return _call_claude_cli(
                system_prompt, user_prompt, extra_add_dirs, model=model, timeout=timeout, log_path=log_path
            )
        except LLMError as e:
            last_err = e
            if attempt < retries:
                time.sleep(10)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Batched extraction (file-based: LLM writes per-paper partials)
# ---------------------------------------------------------------------------


def _partial_path(arxiv_id: str) -> Path:
    return EXTRACTIONS_RAW_DIR / f"{arxiv_id}.partial.json"


def _assemble_record(aid: str, partial: dict, model: str, batch_size: int, n_tool_calls: int) -> dict:
    """Build the full per-paper extraction record from the LLM's partial.

    Matches extraction.schema.json. The partial carries arxiv_id,
    benchmarks, and benchmarks_absent; the script fills the rest.
    """
    return {
        "arxiv_id": aid,
        "extracted_at": _now_iso(),
        "model_used": model,
        "batch_size": batch_size,
        "batch_tool_calls_total": n_tool_calls,
        "paper_hash": _paper_hash(aid),
        "extraction_scope": sorted(_current_benchmark_keys()),
        "benchmarks": partial.get("benchmarks", []),
        "benchmarks_absent": partial.get("benchmarks_absent") or {},
    }


def _run_one_batch(
    todo: list[str],
    all_rules: str,
    model: str,
    timeout: int,
) -> dict[str, dict | None]:
    """Run a single claude CLI call across ``todo`` papers.

    The LLM reads each paper and writes per-paper extraction JSON to
    ``{EXTRACTIONS_RAW_DIR}/{arxiv_id}.partial.json``. The script then
    reads each partial, fills in metadata, and saves the final record
    to ``.cache/extractions/{arxiv_id}.json``.
    """
    EXTRACTIONS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale partials for this batch's ids so "file exists" =
    # "this call produced it".
    for aid in todo:
        _partial_path(aid).unlink(missing_ok=True)

    paper_lines = [
        f"- arxiv_id={aid}  paper={CACHE_DIR / aid / 'paper.md'}  output={_partial_path(aid)}" for aid in todo
    ]
    system_prompt = _build_system_prompt(all_rules)
    user_prompt = (
        "Extract benchmark results from each paper listed below.\n\n"
        "For every paper, write a JSON file to the `output=` path shown, "
        "containing EXACTLY these fields:\n"
        "  - arxiv_id (string, matches the input)\n"
        "  - benchmarks (array, per extraction.schema.json's $defs/BenchmarkEntry)\n"
        "  - benchmarks_absent (object, or omit if none)\n\n"
        "Do not include extracted_at, paper_hash, extraction_scope, "
        "model_used, or batch_* — the script fills those.\n\n"
        "Every arxiv_id below must produce a file, even if benchmarks is "
        "[]. Every quote you write must come from that paper's `paper=` "
        "file.\n\n"
        f"Field semantics: {EXTRACTION_SCHEMA_PATH}\n\n"
        "Papers:\n" + "\n".join(paper_lines)
    )
    batch_tag = "_".join(todo[:2]) + (f"+{len(todo) - 2}" if len(todo) > 2 else "")
    log_path = EXTRACTION_LOGS_DIR / f"batch_{batch_tag}.log"

    try:
        n_tool_calls = _call_claude_cli_with_retry(
            system_prompt,
            user_prompt,
            extra_add_dirs=[CACHE_DIR, EXTRACTIONS_RAW_DIR],
            model=model,
            timeout=timeout,
            log_path=log_path,
            retries=1,
        )
    except LLMError as e:
        print(f"    LLM error for batch ({len(todo)} papers): {e}")
        return {aid: None for aid in todo}

    results: dict[str, dict | None] = {}
    for aid in todo:
        partial_path = _partial_path(aid)
        if not partial_path.exists():
            print(f"    {aid}: no partial file written")
            results[aid] = None
            continue
        try:
            partial = json.loads(partial_path.read_text())
        except json.JSONDecodeError as e:
            print(f"    {aid}: invalid JSON in partial: {e}")
            results[aid] = None
            continue
        record = _assemble_record(aid, partial, model, len(todo), n_tool_calls)
        _save_cached_extraction(aid, record)
        partial_path.unlink()
        results[aid] = record
    return results


def extract_batch(
    arxiv_ids: list[str],
    all_rules: str,
    model: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    resume: bool = True,
) -> dict[str, dict | None]:
    """Extract benchmark results from N papers in batched claude CLI calls.

    On batch-level failure, falls back to per-paper calls so one stuck
    paper cannot poison the whole batch.
    """
    results: dict[str, dict | None] = {}

    todo: list[str] = []
    for aid in arxiv_ids:
        if resume:
            cached = _load_cached_extraction(aid)
            if cached is not None:
                results[aid] = cached
                continue
        paper_path = CACHE_DIR / aid / "paper.md"
        if not paper_path.exists():
            if not _fetch_paper(aid):
                _record_failure(aid, f"HTML not available, {_now_iso()}")
                results[aid] = None
                continue
        todo.append(aid)

    if not todo:
        return results

    batch_results = _run_one_batch(todo, all_rules, model, timeout)
    results.update(batch_results)

    # Fallback: if >=50% of the batch failed, retry remaining ones one-by-one.
    # Protects against a single stuck paper poisoning a whole batch.
    failed_ids = [aid for aid, r in batch_results.items() if r is None]
    if len(todo) > 1 and len(failed_ids) >= len(todo) // 2:
        print(f"    batch degraded ({len(failed_ids)}/{len(todo)} failed) — retrying per-paper")
        for aid in failed_ids:
            single = _run_one_batch([aid], all_rules, model, timeout)
            results.update(single)

    return results


# ---------------------------------------------------------------------------
# S2 citation API
# ---------------------------------------------------------------------------


def _fetch_s2_citations(arxiv_id: str, limit: int = 1000) -> tuple[list[dict], int]:
    """Return (arxiv-citing papers, total citation count).

    Total includes non-arxiv citations (conference/journal papers without
    arxiv preprints) — those cannot be processed but are shown as the
    denominator to reflect real-world citation scale.
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


def _load_scan_cache() -> dict:
    if not SCAN_CACHE_PATH.exists():
        return {"scanned_at": None, "benchmarks": {}}
    return json.loads(SCAN_CACHE_PATH.read_text())


@app.command()
def scan(
    benchmark: Annotated[Optional[str], typer.Option(help="Only scan one benchmark.")] = None,
) -> None:
    """Discover citing papers for each benchmark via Semantic Scholar.

    Results merge into the existing scan_results.json so a single-benchmark
    rescan does not wipe other benchmarks' entries.
    """
    if not BENCHMARKS_JSON_PATH.exists():
        print(f"{BENCHMARKS_JSON_PATH} not found.")
        raise typer.Exit(1)
    benchmarks = json.loads(BENCHMARKS_JSON_PATH.read_text())

    cache = _load_scan_cache()
    scan_results: dict[str, dict] = dict(cache.get("benchmarks", {}))

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

    scope = f"benchmark '{benchmark}'" if benchmark else f"{len(scan_results)} benchmarks"
    print(f"\n{total_new} new papers across {scope}")
    print(f"Wrote {SCAN_CACHE_PATH}")
    print("Run `update_coverage.py` to refresh coverage.json.")


@app.command()
def run(
    arxiv_ids: Annotated[
        Optional[list[str]], typer.Argument(help="Arxiv IDs to extract. Omit to use --from-scan.")
    ] = None,
    from_scan: Annotated[bool, typer.Option("--from-scan", help="Extract all papers from scan_results.json.")] = False,
    benchmark: Annotated[
        Optional[str],
        typer.Option("--benchmark", help="Restrict --from-scan to one benchmark's citing papers."),
    ] = None,
    model: Annotated[str, typer.Option(help="Claude model alias or full ID.")] = DEFAULT_MODEL,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Papers per claude call.")] = DEFAULT_BATCH_SIZE,
    workers: Annotated[int, typer.Option(help="Parallel batches.")] = 1,
    timeout: Annotated[int, typer.Option(help="Per-batch claude CLI timeout in seconds.")] = DEFAULT_TIMEOUT,
    resume: Annotated[bool, typer.Option(help="Skip papers with fresh cache.")] = True,
) -> None:
    """Extract benchmark results from papers in batched claude CLI calls."""
    if from_scan:
        if not SCAN_CACHE_PATH.exists():
            print("scan_results.json not found — run scan first.")
            raise typer.Exit(1)
        scan_data = json.loads(SCAN_CACHE_PATH.read_text())
        bm_data_all = scan_data.get("benchmarks", {})
        if benchmark:
            if benchmark not in bm_data_all:
                print(f"benchmark '{benchmark}' not in scan_results.json")
                raise typer.Exit(1)
            bm_data_all = {benchmark: bm_data_all[benchmark]}
        all_ids: set[str] = set()
        for bm_data in bm_data_all.values():
            all_ids.update(bm_data.get("all_citing_ids", []))
            all_ids.update(bm_data.get("new_papers", []))
        targets = sorted(all_ids)
    elif arxiv_ids:
        targets = list(arxiv_ids)
    else:
        print("Provide arxiv IDs or use --from-scan.")
        raise typer.Exit(2)

    if resume:
        before = len(targets)
        targets = [aid for aid in targets if _load_cached_extraction(aid) is None]
        skipped = before - len(targets)
        if skipped:
            print(f"--resume: skipping {skipped} cached papers")

    if not targets:
        print("Nothing to extract.")
        return

    batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]
    print(
        f"Extracting {len(targets)} papers in {len(batches)} batches "
        f"(batch_size={batch_size}, workers={workers}, model={model}, timeout={timeout}s)..."
    )
    all_rules = _load_all_benchmark_rules()

    counters = [0, 0, 0]  # [ok, empty, fail]
    counters_lock = threading.Lock()

    def _tally_batch(batch_results: dict[str, dict | None]) -> None:
        with counters_lock:
            for aid, result in batch_results.items():
                if result is None:
                    counters[2] += 1
                    print(f"  FAIL {aid}")
                elif not result.get("benchmarks"):
                    counters[1] += 1
                    print(f"  ---  {aid} (empty)")
                else:
                    n_bm = len(result["benchmarks"])
                    n_models = sum(len(b.get("models", [])) for b in result["benchmarks"])
                    counters[0] += 1
                    print(f"  OK   {aid} ({n_bm} benchmarks, {n_models} models)")

    def _do_batch(ids: list[str]) -> dict[str, dict | None]:
        return extract_batch(ids, all_rules, model, timeout=timeout, resume=False)

    if workers <= 1:
        for i, batch in enumerate(batches, 1):
            print(f"[batch {i}/{len(batches)}] {len(batch)} papers")
            _tally_batch(_do_batch(batch))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_do_batch, b): i for i, b in enumerate(batches, 1)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    batch_results = fut.result()
                except Exception as exc:
                    print(f"  CRASH batch {idx}: {exc}")
                    continue
                print(f"[batch {idx}/{len(batches)} done]")
                _tally_batch(batch_results)

    n_ok, n_empty, n_fail = counters
    print(f"\nDone: ok={n_ok} empty={n_empty} fail={n_fail} total={len(targets)}")
    failures = _load_fetch_failures()
    if failures:
        print(f"{len(failures)} papers in fetch_failures.json")

    print("Run `refine.py main` to build leaderboard.json, then `update_coverage.py` and `update_citations.py`.")


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
                    models.sort(key=lambda m: m.get("name_in_paper", ""))
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
