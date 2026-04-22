# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "typer>=0.12",
#   "beautifulsoup4>=4.12",
#   "markdownify>=0.14",
# ]
# ///
"""Extract benchmark scores from arxiv papers via LLM.

Subcommands::

    uv run extract.py run 2505.05800 [--workers 4]       # extract from papers
    uv run extract.py run --from-scan --benchmark libero # extract a benchmark's scan pool
    uv run extract.py pack                               # rewrite data/extractions.json from cache

Pipeline: scan.py → run → refine.py → validate.py → sync_external.py.
The citing-paper pools the `--from-scan` path reads from data/scan_results.json
are produced by scan.py's S2 fetch, not by this script.

`run` auto-packs on completion; `pack` is the manual entry point for the
rare case of repacking without re-extracting. data/extractions.json is
authoritative — resume reads from it first, so clones can skip already-
extracted papers without the cache.

Output shape is defined by leaderboard/data/extraction.schema.json. This
script loads that schema at runtime — field semantics live there, not
duplicated in the prompt.
"""

from __future__ import annotations

import functools
import hashlib
import json
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

import jsonschema
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
SCAN_RESULTS_PATH = DATA_DIR / "scan_results.json"
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

# LaTeXML class → standard HTML tag. arxiv.org/html emits tables as
# <span class="ltx_tabular / ltx_tr / ltx_td"> rather than real
# <table>/<tr>/<td>, so any off-the-shelf HTML→Markdown converter
# treats them as inline spans and collapses cells onto separate lines.
# Renaming the tag (without touching attributes or children) lets
# markdownify emit a proper pipe-table.
_LTX_TABLE_TAG_MAP = {
    "ltx_tabular": "table",
    "ltx_thead": "thead",
    "ltx_tbody": "tbody",
    "ltx_tr": "tr",
    "ltx_th": "th",
    "ltx_td": "td",
}


def _lift_latexml(html: str) -> str:
    """Rewrite LaTeXML table spans to standard HTML table tags; inline <math>
    blocks are replaced by their `$...$`-wrapped original LaTeX source.

    LaTeXML stores the original TeX for each math block in the `alttext`
    attribute (and as an <annotation encoding="application/x-tex"> child);
    we pull from `alttext` since it's present on every `<math>` element.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for math in soup.find_all("math"):
        # bs4's Tag.get() can return a list for multi-valued attributes; the
        # LaTeXML files we see always give a scalar, but type-wise we have
        # to coerce before calling str methods.
        alt = math.get("alttext")
        tex = alt if isinstance(alt, str) else (" ".join(alt) if alt else "")
        if not tex:
            ann = math.find("annotation", attrs={"encoding": "application/x-tex"})
            tex = ann.get_text() if ann is not None else ""
        tex = tex.strip() if tex else ""
        display = math.get("display") == "block"
        replacement = (f"$${tex}$$" if display else f"${tex}$") if tex else ""
        math.replace_with(replacement)

    for tag in soup.find_all("span"):
        classes = tag.get("class") or []
        for cls, standard in _LTX_TABLE_TAG_MAP.items():
            if cls in classes:
                tag.name = standard
                break
    return str(soup)


def _html_to_markdown(html: str) -> str:
    """arxiv.org/html → Markdown.

    Pipeline:
      1. LaTeXML lift — rename ltx_* table spans to real HTML table tags
         and inline math blocks as `$...$` from their `alttext` source.
      2. markdownify — emit ATX-style Markdown with pipe-tables; strip
         scripts/styles.
    """
    from markdownify import markdownify

    lifted = _lift_latexml(html)
    md = markdownify(lifted, heading_style="ATX", strip=["script", "style"])
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


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


_BIB_HEADING_RE = re.compile(r"^#+\s*(References|Bibliography)\s*$", re.MULTILINE)


def _has_bibliography(markdown: str) -> bool:
    return bool(_BIB_HEADING_RE.search(markdown))


def _fetch_paper(arxiv_id: str) -> bool:
    cache_dir = CACHE_DIR / arxiv_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    # arxiv.org/html is the only HTML source we trust. ar5iv was used
    # historically but has two failure modes that silently produced
    # stale or incomplete markdown:
    #   (1) it drops the rendered bibliography on some papers, so
    #       methods cited by bibkey cannot be resolved to a
    #       model_paper URL;
    #   (2) it pins to the first version it processed and does not
    #       re-render when the author posts a v2 — so a paper whose
    #       later revision adds a whole benchmark section (e.g.
    #       MemoryVLA v2 adding MIKASA-Robo) reads as if the section
    #       didn't exist. Passing ar5iv a /{id}v2 suffix doesn't help
    #       — ar5iv accepts the URL but still serves v1 content.
    # Silent data corruption beats any benefit from broader coverage,
    # so failures here become explicit fetch_failures.json entries
    # rather than quiet fallbacks.
    url = f"https://arxiv.org/html/{arxiv_id}"
    html = _fetch_url(url)
    if html is None or len(html) < 2000:
        return False
    markdown = _html_to_markdown(html)
    if len(markdown) < 1000:
        return False

    _paper_md_path(arxiv_id).write_text(markdown, encoding="utf-8")
    ph = "sha256:" + hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    _paper_meta_path(arxiv_id).write_text(
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "source": "arxiv",
                "fetched_at": _now_iso(),
                "url": url,
                "paper_hash": ph,
                "bytes": len(markdown),
                "has_bibliography": _has_bibliography(markdown),
            },
            indent=2,
        )
        + "\n"
    )
    return True


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


# ---------------------------------------------------------------------------
# Pre-Opus screen (Haiku)
# ---------------------------------------------------------------------------

SCREEN_MODEL = "claude-haiku-4-5"
SCREEN_TIMEOUT = 180
# 200K paper markdown would push Haiku near its input budget. Trim very
# long papers; benchmark evaluation sections are concentrated enough
# that the first ~120K chars are nearly always sufficient.
SCREEN_PAPER_MAX_CHARS = 120_000


@functools.lru_cache(maxsize=1)
def _screen_registry_summary() -> str:
    """One-line-per-benchmark description used to ground Haiku classifier."""
    data = json.loads(BENCHMARKS_JSON_PATH.read_text())
    lines = []
    for bm_key, entry in data.items():
        display = entry.get("display_name", bm_key)
        pu = entry.get("paper_url") or ""
        m = _ARXIV_RE.search(pu)
        arxiv_tag = f" (arxiv:{m.group(1)})" if m else ""
        # First sentence of detail_notes is usually the one-line protocol summary.
        notes = entry.get("detail_notes") or ""
        first = re.split(r"(?<=[.!?])\s+", re.sub(r"<[^>]+>", "", notes).strip())[0][:160]
        lines.append(f"- {bm_key}: {display}{arxiv_tag} — {first}")
    return "\n".join(lines)


def _screen_system_prompt() -> str:
    return (
        "You classify research papers by which benchmarks they EVALUATE their "
        "method on. Citing a benchmark in related work or references is NOT "
        "evaluation — only papers that report numeric results (success "
        "rates, scores, task completions) on a benchmark in a table or "
        "figure count.\n\n"
        "Return STRICTLY a JSON object of the shape:\n"
        '  {"evaluated_benchmarks": ["key1", "key2", ...]}\n'
        "with keys drawn from the registry below. Empty list if the paper "
        "evaluates none of them. No prose, no markdown fences, no extra "
        "keys.\n\n"
        "Benchmark registry:\n" + _screen_registry_summary()
    )


def _screen_paper_with_haiku(arxiv_id: str, paper_md: str, timeout: int = SCREEN_TIMEOUT) -> set[str] | None:
    """Ask Haiku which registered benchmarks the paper evaluates on.

    Returns a set of benchmark keys (possibly empty) on success, or
    ``None`` when the screen fails (caller should fall back to Opus —
    never skip based on a failed screen).
    """
    trimmed = paper_md[:SCREEN_PAPER_MAX_CHARS]
    user_msg = (
        f"Paper arxiv_id: {arxiv_id}\n\n"
        "Return the JSON object now, based on the full text below.\n\n"
        "-----BEGIN PAPER-----\n" + trimmed + "\n-----END PAPER-----"
    )
    cmd = [
        "claude",
        "--print",
        "--model",
        SCREEN_MODEL,
        "--system-prompt",
        _screen_system_prompt(),
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--disable-slash-commands",
        # Classifier: no tool access at all. Text-only one-shot.
        "--disallowed-tools",
        "Bash Read Write Edit Grep Glob NotebookEdit WebFetch WebSearch Agent Task TodoWrite MultiEdit",
    ]
    try:
        result = subprocess.run(cmd, input=user_msg, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    text = None
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "assistant":
            for block in evt.get("message", {}).get("content", []) or []:
                if block.get("type") == "text":
                    text = (text or "") + (block.get("text") or "")
    if not text:
        return None
    # Tolerate code fences or extra whitespace around the JSON.
    m = re.search(r"\{.*?\"evaluated_benchmarks\".*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    keys = payload.get("evaluated_benchmarks")
    if not isinstance(keys, list):
        return None
    registry = set(_benchmark_search_registry_keys())
    return {k for k in keys if isinstance(k, str) and k in registry}


@functools.lru_cache(maxsize=1)
def _benchmark_search_registry_keys() -> frozenset[str]:
    return frozenset(json.loads(BENCHMARKS_JSON_PATH.read_text()).keys())


def _extraction_cache_path(arxiv_id: str) -> Path:
    return EXTRACTIONS_DIR / f"{arxiv_id}.json"


def _load_cached_extraction(arxiv_id: str) -> dict | None:
    """Return the in-flight cache entry for ``arxiv_id`` if fresh.

    Fresh = cache's ``paper_hash`` matches ``paper.md`` on disk. The
    cache is transient (survives across runs but not across clones);
    ``data/extractions.json`` is authoritative.
    """
    p = _extraction_cache_path(arxiv_id)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if data.get("paper_hash") != _paper_hash(arxiv_id):
        return None
    return data


@functools.lru_cache(maxsize=1)
def _load_packed_extractions() -> dict[str, dict]:
    """Load ``data/extractions.json`` keyed by arxiv_id. Authoritative source."""
    if not EXTRACTIONS_JSON.exists():
        return {}
    entries = json.loads(EXTRACTIONS_JSON.read_text())
    return {e["arxiv_id"]: e for e in entries if e.get("arxiv_id")}


def _get_extracted(arxiv_id: str) -> dict | None:
    """Return the extraction record if present and fresh, else None.

    Checks authoritative ``data/extractions.json`` first (paper_hash
    must match current paper.md), then the transient cache. Either hit
    → the record; miss on both → None.

    On clean clones ``.cache/papers/<id>/meta.json`` is absent, so the
    local hash is ``None``; in that case we trust the packed record
    rather than triggering an expensive full re-extraction. A fetch
    step will repopulate the hash and re-validate on the next run.
    """
    packed = _load_packed_extractions().get(arxiv_id)
    if packed:
        local_hash = _paper_hash(arxiv_id)
        if local_hash is None or packed.get("paper_hash") == local_hash:
            return packed
    return _load_cached_extraction(arxiv_id)


def _is_extracted(arxiv_id: str) -> bool:
    return _get_extracted(arxiv_id) is not None


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

Output shape: `benchmarks` is a map {{benchmark_key → entry}}. Each
entry is either `{{status: "scored", models: [...]}}` or
`{{status: "absent", reason: "..."}}`. Include a key only when you
have something to say about it; benchmarks the paper doesn't mention
are omitted (sparse dict).

For every benchmark listed in the rules below:

1. Grep the paper for the benchmark's key name, display name, and the
   suite/task names in its Standard.
2. Paper scores it → emit `{{bm: {{status: "scored", models: [...]}}}}`.
   One entry per distinct method in the paper's results table(s).
3. Paper mentions it without an extractable score (cited-only,
   figure-only, unreadable) → emit
   `{{bm: {{status: "absent", reason: "<one line>"}}}}`.
4. Paper doesn't mention it → omit the key entirely.

Return `benchmarks: {{}}` only for pure theory/survey papers with no
evaluation table.

A `scored` entry's `models` array must carry actual numeric scores for
every row — even when `protocol.matches_standard='no'`. Non-standard
protocols still report measurements; the match verdict only affects how
refine aggregates the row, not whether the numbers exist. If the paper
presents the row in a table but the cells are blank or marked "—",
use `absent` with reason, not a scored entry with empty scores.

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
        # Block sub-agent delegation — the default Claude Code agent
        # otherwise spawns one Agent subagent per paper, which re-reads
        # the paper and re-runs extraction in its own context (2-3x
        # multiplier on tokens for no quality gain).
        "--disallowed-tools",
        "Agent",
    ]
    for d in extra_add_dirs:
        cmd += ["--add-dir", str(d.resolve())]
    try:
        result = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise LLMError("claude CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"timed out after {timeout}s") from e

    n_tool_calls = 0
    seen_result = False
    stream_success = False
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
            # Claude CLI sometimes exits non-zero (hook failure, plugin
            # teardown, etc.) even when the turn completed successfully.
            # If the stream's own result event reports success, trust it.
            stream_success = not evt.get("is_error", False) and evt.get("subtype") == "success"

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0 and not stream_success:
        raise LLMError(f"exit {result.returncode}: {result.stderr[:500]}")
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


_ARXIV_URL_RE = re.compile(r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


@functools.cache
def _benchmarks_registry() -> dict:
    """Full benchmarks.json registry, cached once per process.

    Same cache shape as the twin helper in ``refine.py`` — consistent
    decorator so readers don't have to wonder if ``maxsize=1`` meant
    something special here.
    """
    return json.loads(BENCHMARKS_JSON_PATH.read_text())


def _resolve_model_paper_from_bibliography(aid: str, benchmarks: dict) -> None:
    """Deterministic post-pass: fill ``model_paper`` from the paper's own bibliography.

    When the LLM leaves ``model_paper`` null for a cited baseline, the
    paper's reference list usually still names the origin paper (arxiv
    URL or a citation key that appears alongside the method name in
    tables and text). We scan ``paper.md`` for arxiv URLs and assign
    one when there's a unique candidate whose surrounding context
    mentions the method's ``name_in_paper``.

    Safety: only fills when the arxiv URL appears within
    :data:`_BIB_CONTEXT_WINDOW` characters of a case-sensitive
    ``name_in_paper`` occurrence, AND that candidate is unique. If the
    method name appears near several different arxiv URLs, leave it
    null — the downstream refine post-pass or the manual overrides
    file is the right place to break the tie.
    """
    paper_path = CACHE_DIR / aid / "paper.md"
    if not paper_path.exists():
        return
    try:
        md = paper_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    for bm_info in benchmarks.values():
        if not isinstance(bm_info, dict) or bm_info.get("status") != "scored":
            continue
        for m in bm_info.get("models") or []:
            if m.get("model_paper") is not None:
                continue
            name = m.get("name_in_paper") or ""
            if len(name) < 2:
                continue
            aids = _arxiv_ids_near(md, name)
            # Never assign the citing paper itself — that would make a
            # cited_baseline row masquerade as first-party.
            aids = [x for x in aids if x != aid]
            if len(aids) == 1:
                m["model_paper"] = f"https://arxiv.org/abs/{aids[0]}"


_REFERENCES_HEADING_RE = re.compile(
    r"^#{1,6}\s*(references|bibliography|reference\b)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _arxiv_ids_near(markdown: str, needle: str) -> list[str]:
    """Return unique arxiv ids whose bib entry mentions ``needle``.

    Only scans the References section (everything after the first
    ``# References`` / ``## References`` / ``# Bibliography`` heading).
    Groups entries by blank line or by leading ``[n]`` / ``n.`` markers;
    returns every arxiv URL in entries that contain ``needle``. Bounded
    by the bib-entry shape, so short papers where intro and references
    are physically close can't confuse the resolver.

    Collapse to a sorted list so callers can test for uniqueness via
    ``len()``.
    """
    m = _REFERENCES_HEADING_RE.search(markdown)
    if not m:
        return []
    refs_body = markdown[m.end() :]
    entries = _split_bib_entries(refs_body)
    found: set[str] = set()
    for entry in entries:
        if needle not in entry:
            continue
        for match in _ARXIV_URL_RE.finditer(entry):
            found.add(match.group(1))
    return sorted(found)


_ENTRY_START_RE = re.compile(r"^(?:\[\d+\]|\d+\.|\*|-)\s+", re.MULTILINE)


def _split_bib_entries(body: str) -> list[str]:
    """Split a References body into per-entry text blocks.

    Tries markdown-ish entry markers (``[1]``, ``1.``, ``*``, ``-``)
    first; falls back to blank-line splits when no markers are
    present. Empty entries are dropped.
    """
    starts = [m.start() for m in _ENTRY_START_RE.finditer(body)]
    if len(starts) >= 2:
        entries = [body[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]
        entries.append(body[starts[-1] :])
    else:
        entries = [blk for blk in re.split(r"\n\s*\n", body) if blk.strip()]
    return [e.strip() for e in entries if e.strip()]


def _validate_scale(aid: str, benchmarks: dict) -> None:
    """Sanity-check numeric scores against each benchmark's declared ``metric.range``.

    If a value is within [0, 1] but the benchmark's range is 0–100 (the
    common 0–1-scale-slip), auto-scale ×100 and leave a sentinel in
    ``notes`` noting the conversion. Values that land within the
    declared range pass through. Values that are out-of-range in any
    other direction are logged but not mutated — the LLM's output
    stands and refine/validate will surface the anomaly.
    """
    registry = _benchmarks_registry()
    for bm_key, bm_info in benchmarks.items():
        if not isinstance(bm_info, dict) or bm_info.get("status") != "scored":
            continue
        meta = registry.get(bm_key)
        if not meta:
            continue
        rng = (meta.get("metric") or {}).get("range")
        if not (isinstance(rng, list) and len(rng) == 2):
            continue
        lo, hi = float(rng[0]), float(rng[1])
        for m in bm_info.get("models") or []:
            scores = m.get("scores") or {}
            name = m.get("name_in_paper")
            _maybe_rescale_cell(scores, "overall_score", lo, hi, f"[{aid}/{bm_key}/{name}]")
            for container_key in ("suite_scores", "task_scores"):
                container = scores.get(container_key) or {}
                for cell_key, cell in container.items():
                    if isinstance(cell, dict):
                        _maybe_rescale_cell(cell, "value", lo, hi, f"[{aid}/{bm_key}/{name}/{cell_key}]")


_QUOTE_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?|\.\d+)")


def _maybe_rescale_cell(obj: dict, field: str, lo: float, hi: float, log_prefix: str) -> None:
    """Rescale 0–1 slip on 0–100 benchmarks when the quote confirms the slip.

    Only rescales when there's explicit evidence: the cell's ``*_quote``
    (or ``quote``) field parses to a number ≈ 100× the stored value.
    Without that evidence the LLM's number stands — papers do report
    legitimately low values (e.g. a 1.2 % baseline on libero) and
    blindly rescaling would corrupt them. A quote of ``.85`` reads as
    0.85 (not 85), so it matches the stored value and does NOT trigger
    a rescale.
    """
    v = obj.get(field)
    if not isinstance(v, (int, float)):
        return
    quote = obj.get(f"{field}_quote") or obj.get("quote")
    if hi >= 10 and 0 <= v <= 1 and isinstance(quote, str):
        m = _QUOTE_NUMBER_RE.search(quote)
        if m:
            try:
                quote_num = float(m.group(1))
            except ValueError:
                quote_num = None
            if quote_num is not None and abs(quote_num - v * 100) < 0.5:
                obj[field] = v * 100
                print(f"  {log_prefix} rescaled {field} {v} -> {obj[field]} (quote='{quote}')")
                return
    if not (lo <= v <= hi):
        print(f"  {log_prefix} {field}={v} out of range [{lo}, {hi}] — left as-is")


def _assemble_record(aid: str, partial: dict, model: str) -> dict:
    """Build the per-paper extraction record from the LLM's partial.

    Matches extraction.schema.json. The LLM writes ``{arxiv_id,
    benchmarks}``; this adds the metadata fields (paper_hash,
    extracted_at, model_used) so the cache entry is identical to the
    shape ``data/extractions.json`` carries after pack.

    Applies two deterministic safety nets before schema validation so
    that the common LLM misses (null model_paper on a cited row,
    0–1-scale slip on a 0–100 benchmark) don't propagate downstream:

    - :func:`_resolve_model_paper_from_bibliography` fills
      ``model_paper`` from the paper's reference list when the LLM
      left it null.
    - :func:`_validate_scale` flags out-of-range numeric scores and
      auto-scales when a paper reports 0–1 on a 0–100 metric.

    Finally validates the assembled record against
    ``extraction.schema.json``; invalid output raises
    :class:`LLMError` so the batch's retry path kicks in.
    """
    benchmarks = partial.get("benchmarks") or {}
    if not isinstance(benchmarks, dict):
        raise LLMError(f"{aid}: partial.benchmarks is {type(benchmarks).__name__}, expected dict")

    _resolve_model_paper_from_bibliography(aid, benchmarks)
    _validate_scale(aid, benchmarks)

    record = {
        "arxiv_id": aid,
        "benchmarks": benchmarks,
        "paper_hash": _paper_hash(aid) or "",
        "extracted_at": _now_iso(),
        "model_used": model,
    }
    try:
        jsonschema.validate(record, _extraction_schema())
    except jsonschema.ValidationError as e:
        # jsonschema's path gives the json pointer into the offending node;
        # hand that to LLMError so the retry path can log it cleanly.
        raise LLMError(f"{aid}: schema validation failed at {list(e.absolute_path)}: {e.message}") from e
    return record


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
        "For every paper, write a JSON file to the `output=` path shown. "
        "Each file must validate against "
        f"{EXTRACTION_SCHEMA_PATH} (write only the fields that schema "
        "defines — `arxiv_id` and `benchmarks`; the script fills the "
        "rest at pack time).\n\n"
        "Every arxiv_id below must produce a file, even if benchmarks "
        "is {}. Every quote you write must come from that paper's "
        "`paper=` file.\n\n"
        "Papers:\n" + "\n".join(paper_lines)
    )
    batch_tag = "_".join(todo[:2]) + (f"+{len(todo) - 2}" if len(todo) > 2 else "")
    log_path = EXTRACTION_LOGS_DIR / f"batch_{batch_tag}.log"

    try:
        _call_claude_cli_with_retry(
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
        try:
            record = _assemble_record(aid, partial, model)
        except LLMError as e:
            # Schema violation (or post-pass crash) on a single paper —
            # isolate it so the remaining papers in the batch still get
            # persisted. The per-paper fallback in ``extract_batch`` will
            # retry this aid alone.
            print(f"    {aid}: {e}")
            results[aid] = None
            continue
        except Exception as e:  # defensive: a post-pass bug shouldn't poison the batch
            print(f"    {aid}: unexpected post-pass error: {type(e).__name__}: {e}")
            results[aid] = None
            continue
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
            existing = _get_extracted(aid)
            if existing is not None:
                results[aid] = existing
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


def _prefilter_with_screen(arxiv_ids: list[str], workers: int = 8) -> list[str]:
    """Run the Haiku screen on each paper; drop those that evaluate no registered benchmark.

    Papers without paper.md are fetched first; fetch failures are
    logged and dropped. Screen failures (timeout, bad JSON) are
    conservative — those papers fall through to the Opus path so no
    real data is lost to a flaky screen.

    Returns the reduced list of arxiv_ids that should still go to
    Opus. Side effect: empty-screened papers are written to cache as
    ``{benchmarks: {}}`` with ``model_used=screen-haiku:...``.
    """
    print(f"Screening {len(arxiv_ids)} papers with {SCREEN_MODEL} (workers={workers})...")

    def _screen_one(aid: str) -> tuple[str, set[str] | None, str]:
        paper_path = CACHE_DIR / aid / "paper.md"
        if not paper_path.exists():
            if not _fetch_paper(aid):
                _record_failure(aid, f"HTML not available, {_now_iso()}")
                return aid, None, "fetch-fail"
        try:
            md = paper_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return aid, None, "read-fail"
        screened = _screen_paper_with_haiku(aid, md)
        return aid, screened, "ok"

    survivors: list[str] = []
    n_empty = n_keep = n_fetch_fail = n_screen_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futs = {executor.submit(_screen_one, aid): aid for aid in arxiv_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                aid, screened, status = fut.result()
            except Exception as exc:
                print(f"  SCREEN CRASH {futs[fut]}: {exc}")
                survivors.append(futs[fut])
                n_screen_fail += 1
                continue
            if status == "fetch-fail":
                n_fetch_fail += 1
                continue
            if screened is None:
                n_screen_fail += 1
                survivors.append(aid)
                continue
            if not screened:
                record = _assemble_record(aid, {"benchmarks": {}}, f"screen-haiku:{SCREEN_MODEL}")
                _save_cached_extraction(aid, record)
                n_empty += 1
            else:
                survivors.append(aid)
                n_keep += 1
            if i % 20 == 0 or i == len(arxiv_ids):
                print(
                    f"  [screen {i}/{len(arxiv_ids)}] keep={n_keep} empty={n_empty} "
                    f"fetch-fail={n_fetch_fail} screen-fail={n_screen_fail}"
                )
    print(
        f"Screen done: keep={n_keep} empty={n_empty} fetch-fail={n_fetch_fail} "
        f"screen-fail={n_screen_fail} → Opus will process {len(survivors)} papers"
    )
    return survivors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Extract benchmark scores from arxiv papers via LLM.", add_completion=False)


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
        if not SCAN_RESULTS_PATH.exists():
            print("scan_results.json not found — run scan first.")
            raise typer.Exit(1)
        scan_data = json.loads(SCAN_RESULTS_PATH.read_text())
        bm_data_all = scan_data.get("benchmarks", {})
        if benchmark:
            if benchmark not in bm_data_all:
                print(f"benchmark '{benchmark}' not in scan_results.json")
                raise typer.Exit(1)
            bm_data_all = {benchmark: bm_data_all[benchmark]}
        all_ids: set[str] = set()
        for bm_data in bm_data_all.values():
            all_ids.update(bm_data.get("all_citing_ids", []))
        targets = sorted(all_ids)
    elif arxiv_ids:
        targets = list(arxiv_ids)
    else:
        print("Provide arxiv IDs or use --from-scan.")
        raise typer.Exit(2)

    if resume:
        before = len(targets)
        targets = [aid for aid in targets if not _is_extracted(aid)]
        skipped = before - len(targets)
        if skipped:
            print(f"--resume: skipping {skipped} already-extracted papers")

    if not targets:
        print("Nothing to extract.")
        return

    # Pre-Opus screen: ask Haiku which registered benchmarks each paper
    # actually evaluates on. Papers that evaluate none are short-
    # circuited with an empty record so the expensive Opus agentic
    # loop is skipped. Empirically most papers in a benchmark's scan
    # pool cite it but don't evaluate on it — ~65% of papers screen
    # out. Screen failures fall through to Opus (never drop data).
    targets = _prefilter_with_screen(targets, workers=max(workers, 8))

    if not targets:
        print("Nothing to extract after screen.")
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
                    continue
                bms = result.get("benchmarks") or {}
                scored = {k: v for k, v in bms.items() if v.get("status") == "scored"}
                if not scored:
                    counters[1] += 1
                    n_absent = len(bms) - len(scored)
                    note = f" ({n_absent} absent)" if n_absent else ""
                    print(f"  ---  {aid} (no scored benchmarks{note})")
                else:
                    n_models = sum(len(entry.get("models") or []) for entry in scored.values())
                    counters[0] += 1
                    print(f"  OK   {aid} ({len(scored)} benchmarks, {n_models} models)")

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

    # Auto-pack: merge cache into authoritative data/extractions.json so
    # the next run's resume check picks up work from any clone, and so
    # the committed state always reflects what extract produced.
    # _load_packed_extractions is lru-cached; clear so subsequent calls
    # see the just-written file.
    n_packed = _pack_cache()
    _load_packed_extractions.cache_clear()
    print(f"Packed {n_packed} extractions → {EXTRACTIONS_JSON}")

    print("Run `refine.py main` to build leaderboard.json, then `scan.py` and `update_citations.py`.")


def _pack_cache() -> int:
    """Merge .cache/extractions/*.json into data/extractions.json. Returns entry count.

    Preserves manual edits: a packed entry whose ``paper_hash`` still
    matches the cache's is left as-is, even if the cache file was
    rewritten by a refresh. Cache is authoritative only when paper_hash
    differs (paper changed, or this is a new arxiv_id).
    """
    existing: dict[str, dict] = {}
    if EXTRACTIONS_JSON.exists():
        for e in json.loads(EXTRACTIONS_JSON.read_text()):
            aid = e.get("arxiv_id")
            if aid:
                existing[aid] = e

    for f in sorted(EXTRACTIONS_DIR.glob("*.json")):
        cache_entry = json.loads(f.read_text())
        aid = cache_entry.get("arxiv_id")
        if not aid:
            continue
        prev = existing.get(aid)
        # Preserve packed only when nothing new happened on the cache
        # side — both paper_hash and extracted_at identical. A re-
        # extraction bumps extracted_at, so its content wins even when
        # paper.md is unchanged. Manual edits on the packed file
        # survive because normal runs don't touch this paper's cache.
        if (
            prev
            and prev.get("paper_hash") == cache_entry.get("paper_hash")
            and prev.get("extracted_at") == cache_entry.get("extracted_at")
        ):
            continue
        existing[aid] = cache_entry

    entries = sorted(existing.values(), key=lambda e: e["arxiv_id"])
    # Stable diffs: sort the benchmarks dict keys and each scored entry's models list.
    for entry in entries:
        bms = entry.get("benchmarks") or {}
        entry["benchmarks"] = dict(sorted(bms.items()))
        for v in entry["benchmarks"].values():
            if isinstance(v, dict) and (models := v.get("models")):
                models.sort(key=lambda m: m.get("name_in_paper", ""))

    EXTRACTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTIONS_JSON.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return len(entries)


@app.command()
def pack() -> None:
    """Pack .cache/extractions/*.json → data/extractions.json for git commit."""
    if not sorted(EXTRACTIONS_DIR.glob("*.json")) and not EXTRACTIONS_JSON.exists():
        print("No extractions to pack.")
        raise typer.Exit(1)
    n = _pack_cache()
    print(f"Packed {n} extractions → {EXTRACTIONS_JSON}")


if __name__ == "__main__":
    app()
