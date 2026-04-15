# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///
"""Extract benchmark scores from arxiv papers via LLM.

Two subcommands::

    uv run extract.py scan [--benchmark libero]   # discover citing papers
    uv run extract.py run 2505.05800 [--workers 4] # extract from papers

Pipeline: scan → run → refine.py → validate.py → sync_external.py
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
SCAN_CACHE_PATH = ROOT / ".cache" / "scan_results.json"
FETCH_FAILURES_PATH = CACHE_DIR / "fetch_failures.json"

DEFAULT_MODEL = "claude-opus-4-6[1m]"
DEFAULT_TIMEOUT = 1200
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
                                        "reported_table": {"type": ["string", "null"]},
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
    return f"""You are the EXTRACT stage of a two-stage VLA leaderboard pipeline.

## Pipeline role

Be RECALL-FIRST. Surface every row that has any chance of being a leaderboard
entry, with the evidence (verbatim quotes, protocol notes, attribution) the
next stage needs to make the cut.

A separate REFINE stage downstream handles precision — protocol gating
(dropping rows you mark `matches_standard="no"`), score arithmetic from
component suite/task scores, eligibility filtering, dedup across papers,
canonical naming, and notes. Do NOT pre-filter for any of those concerns.
When uncertain whether to extract a row, extract it.

**Cited baselines carry the same weight as the paper's own results.** Baseline
comparison tables and related-work score tables are often the ONLY recorded
source for a given model on this benchmark — the original paper may not reach
extraction for other reasons (no arxiv preprint, different citation graph,
older than our scan). Never abbreviate or summarize a baseline table; every
row in every comparison table is load-bearing.

## What belongs on the leaderboard

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
- Be conservative. A paper whose entire contribution is a survey,
  reproduction study, or evaluation harness (not a new method) should
  usually return an empty `benchmarks` array — those rows are already
  on the leaderboard via their original papers.

## Benchmark rule template

Each benchmark block below opens with a bold `**Standard**: ...` line — the
canonical protocol in one sentence. `Scoring` prescribes the JSON shape
(`overall_score` computation, canonical `suite_scores` / `task_scores` keys).
`Checks` are yes/no questions a row must pass; a failed check means the row
has `protocol.matches_standard = "no"`. `Methodology axes` are variance
dimensions to record in your extraction rationale / quotes — they are NOT
protocol violations, so a row that merely differs along these axes still has
`matches_standard = "yes"`.

{all_rules}
"""


def _call_claude_cli(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    paper_dir: Path,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    log_path: Path | None = None,
) -> tuple[dict, int]:
    """Call claude CLI in tool mode over paper_dir.

    The paper file is NOT inlined into the prompt. The model navigates it
    via tools restricted to ``paper_dir``. Returns
    ``(structured_output, n_tool_calls)``. Writes raw stream-json stdout
    to ``log_path``.
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
        "--add-dir",
        str(paper_dir),
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
    ]
    try:
        result = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise LLMError("claude CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"timed out after {timeout}s") from e
    if result.returncode != 0:
        raise LLMError(f"exit {result.returncode}: {result.stderr[:500]}")

    structured: dict | None = None
    n_tool_calls = 0
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
            structured = evt.get("structured_output")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")

    if not isinstance(structured, dict):
        raise LLMError("no structured_output in stream")
    return structured, n_tool_calls


# ---------------------------------------------------------------------------
# Batched extraction
# ---------------------------------------------------------------------------


def _batched_schema() -> dict:
    """Wrap the single-paper EXTRACTION_SCHEMA in a ``papers`` array."""
    single = EXTRACTION_SCHEMA
    return {
        "type": "object",
        "required": ["papers"],
        "properties": {
            "papers": {
                "type": "array",
                "description": "One entry per input paper. Match arxiv_id to the path supplied.",
                "items": {
                    "type": "object",
                    "required": ["arxiv_id", "benchmarks", "confidence"],
                    "properties": {
                        "arxiv_id": {"type": "string"},
                        "benchmarks": single["properties"]["benchmarks"],
                        "confidence": single["properties"]["confidence"],
                    },
                },
            }
        },
    }


def extract_batch(
    arxiv_ids: list[str],
    all_rules: str,
    model: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    resume: bool = True,
) -> dict[str, dict | None]:
    """Extract benchmark results from N papers in a SINGLE claude CLI call.

    The model has tool access over ``CACHE_DIR`` and navigates each
    paper.md independently. Paper contents are NEVER inlined. Returns a
    dict ``{arxiv_id -> extraction | None}``. None means fetch failed or
    the model omitted the paper. Successfully-extracted rows (including
    empty `benchmarks`) are saved to the per-paper cache.
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

    paper_lines = []
    for aid in todo:
        paper_lines.append(f"- arxiv_id={aid}  path={CACHE_DIR / aid / 'paper.md'}")

    system_prompt = _build_system_prompt(all_rules)
    user_prompt = (
        "Extract benchmark results from EACH paper listed below.\n\n"
        "Return ONE entry per paper in the `papers` array, keyed by arxiv_id.\n"
        "Every arxiv_id below must appear in your output, even if its benchmarks "
        "array is empty.\n\n"
        "Use available tools to navigate each paper.md independently. "
        "Every quote you emit for a paper must come from that paper's file.\n\n"
        "Papers:\n" + "\n".join(paper_lines)
    )

    batch_tag = "_".join(todo[:2]) + (f"+{len(todo) - 2}" if len(todo) > 2 else "")
    log_path = EXTRACTION_LOGS_DIR / f"batch_{batch_tag}.log"

    try:
        llm_output, n_tool_calls = _call_claude_cli(
            system_prompt,
            user_prompt,
            _batched_schema(),
            CACHE_DIR.resolve(),
            model=model,
            timeout=timeout,
            log_path=log_path,
        )
    except LLMError as e:
        print(f"    LLM error for batch ({len(todo)} papers): {e}")
        for aid in todo:
            results[aid] = None
        return results

    scope = sorted(_current_benchmark_keys())
    by_id: dict[str, dict] = {}
    for p in llm_output.get("papers", []):
        aid = p.get("arxiv_id")
        if aid:
            by_id[aid] = p

    now = _now_iso()
    for aid in todo:
        p = by_id.get(aid)
        if p is None:
            print(f"    WARN {aid} missing from batch output")
            results[aid] = None
            continue
        record = {
            "arxiv_id": aid,
            "extracted_at": now,
            "model_used": model,
            "batch_size": len(todo),
            "batch_tool_calls_total": n_tool_calls,
            "paper_hash": _paper_hash(aid),
            "extraction_scope": scope,
            "benchmarks": p.get("benchmarks", []),
            "confidence": p.get("confidence"),
        }
        _save_cached_extraction(aid, record)
        results[aid] = record

    return results


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

    print(f"\n{total_new} new papers across {len(scan_results)} benchmarks")
    print(f"Wrote {SCAN_CACHE_PATH}")
    print("Run `update_coverage.py` to refresh coverage.json.")


DEFAULT_BATCH_SIZE = 30


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
