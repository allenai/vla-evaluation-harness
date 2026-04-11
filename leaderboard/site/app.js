// VLA Leaderboard — app.js
// Vanilla JS, no frameworks.

(function () {
  'use strict';

  // ─── Global state ───────────────────────────────────────────────────────────
  let data = null;
  let pivotMap = {};       // model key → { benchmarkKey: resultObj }
  let modelKeys = [];      // ordered model keys (rows)
  let benchmarkKeys = [];  // ordered benchmark keys (columns with data)
  let overviewColumns = []; // expanded columns: suite-only benchmarks get one col per suite
  let selectedBenchmark = null; // null = overview, string = detail view
  let sortState = { column: null, direction: 'desc' };
  let detailSortSuite = null; // which suite column to sort by in detail view
  let coverageData = null;
  let citationData = null; // arxiv_id → citation count

  // ─── DOM refs ──────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const loadingEl = $('loading');
  const tableEl = $('leaderboard-table');
  const theadEl = tableEl ? tableEl.querySelector('thead') : null;
  const tbodyEl = tableEl ? tableEl.querySelector('tbody') : null;
  const statsEl = $('stats');
  const benchmarkFilterEl = $('benchmark-filter');
  const modelSearchEl = $('model-search');
  const dateFromEl = $('date-from');
  const dateToEl = $('date-to');
  const minCitationsEl = $('min-citations');
  const breakdownPanelEl = $('breakdown-panel');
  const coverageBarEl = $('coverage-bar');

  // ─── Bootstrap ─────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    Promise.all([
      fetch('./leaderboard.json').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
      fetch('./benchmarks.json').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
    ]).then(([leaderboard, benchmarks]) => {
      data = { ...leaderboard, benchmarks };
      init();
    }).catch(err => { if (loadingEl) loadingEl.textContent = 'Failed to load: ' + err.message; });

    fetch('./coverage.json')
      .then(r => r.ok ? r.json() : null)
      .then(json => { if (json) { coverageData = json; renderCoverage(); } })
      .catch(() => {});

    fetch('./citations.json')
      .then(r => r.ok ? r.json() : null)
      .then(json => { if (json) { citationData = json.papers || {}; renderTable(); } })
      .catch(() => {});

    if (benchmarkFilterEl) benchmarkFilterEl.addEventListener('change', onBenchmarkFilterChange);
    if (modelSearchEl) modelSearchEl.addEventListener('input', () => renderTable());
    if (dateFromEl) dateFromEl.addEventListener('change', () => renderTable());
    if (dateToEl) dateToEl.addEventListener('change', () => renderTable());
    if (minCitationsEl) minCitationsEl.addEventListener('input', () => renderTable());
    if (breakdownPanelEl) breakdownPanelEl.addEventListener('click', e => {
      if (e.target.classList.contains('breakdown-close')) closeBreakdown();
    });
  });

  function init() {
    if (loadingEl) loadingEl.style.display = 'none';
    buildPivot();
    buildBenchmarkFilter();
    renderStats();
    renderTable();
  }

  // ─── Pivot builder ─────────────────────────────────────────────────────────
  function buildPivot() {
    pivotMap = {};
    const bmSet = new Set();
    for (const r of data.results) {
      if (!pivotMap[r.model]) pivotMap[r.model] = {};
      pivotMap[r.model][r.benchmark] = r;
      bmSet.add(r.benchmark);
    }
    const seen = new Set();
    modelKeys = [];
    for (const r of data.results) {
      if (!seen.has(r.model)) { seen.add(r.model); modelKeys.push(r.model); }
    }
    const defOrder = Object.keys(data.benchmarks || {});
    benchmarkKeys = defOrder.filter(k => bmSet.has(k));
    for (const k of bmSet) { if (!benchmarkKeys.includes(k)) benchmarkKeys.push(k); }
    const countEl = document.getElementById('benchmark-count');
    if (countEl) countEl.textContent = benchmarkKeys.length + '+';
    if (!sortState.column) {
      sortState.column = '_date';
      sortState.direction = 'desc';
    }
    buildOverviewColumns();
  }

  // ─── Overview columns (expand suite-only benchmarks) ─────────────────────
  function buildOverviewColumns() {
    overviewColumns = [];
    for (const bmKey of benchmarkKeys) {
      if (shouldExpandSuites(bmKey)) {
        const bm = data.benchmarks[bmKey] || {};
        const suites = bm.suites || [];
        const bmName = bm.display_name || bmKey;
        const showAvg = !isSuiteOnlyBenchmark(bmKey);
        const avgPos = showAvg ? (bm.avg_position ?? suites.length) : -1;
        for (let i = 0; i < suites.length; i++) {
          if (i === avgPos) {
            overviewColumns.push({ bmKey, suite: '_avg', label: bmName + ' ' + (bm.avg_label || 'Avg'), colId: bmKey + ':_avg' });
          }
          overviewColumns.push({
            bmKey, suite: suites[i],
            label: bmName + ' ' + shortSuiteLabel(suites[i], bmName),
            colId: bmKey + ':' + suites[i]
          });
        }
        if (showAvg && avgPos >= suites.length) {
          overviewColumns.push({ bmKey, suite: '_avg', label: bmName + ' ' + (bm.avg_label || 'Avg'), colId: bmKey + ':_avg' });
        }
      } else {
        overviewColumns.push({
          bmKey,
          suite: null,
          label: (data.benchmarks[bmKey] || {}).display_name || bmKey,
          colId: bmKey
        });
      }
    }
  }

  function parseColId(colId) {
    const idx = colId.indexOf(':');
    if (idx === -1) return { bmKey: colId, suite: null };
    return { bmKey: colId.substring(0, idx), suite: colId.substring(idx + 1) };
  }

  // ─── Benchmark filter ──────────────────────────────────────────────────────
  function buildBenchmarkFilter() {
    if (!benchmarkFilterEl) return;
    benchmarkFilterEl.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = ''; allOpt.textContent = 'All Benchmarks (Overview)';
    benchmarkFilterEl.appendChild(allOpt);
    for (const key of benchmarkKeys) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = (data.benchmarks[key] || {}).display_name || key;
      benchmarkFilterEl.appendChild(opt);
    }
  }

  function onBenchmarkFilterChange() {
    const val = benchmarkFilterEl.value;
    selectedBenchmark = val || null;
    detailSortSuite = null; // reset suite sort when switching benchmarks
    if (val) { sortState.column = val; sortState.direction = 'desc'; }
    else { sortState.column = '_date'; sortState.direction = 'desc'; }
    closeBreakdown();
    renderTable();
  }

  // ─── Search & Filters ─────────────────────────────────────────────────────
  function searchQuery() { return modelSearchEl ? modelSearchEl.value.trim().toLowerCase() : ''; }

  /** Extract YYYY-MM publication date from an arxiv URL (YYMM.NNNNN format). */
  function extractPubMonth(url) {
    if (!url) return null;
    const m = url.match(/arxiv\.org\/abs\/(\d{2})(\d{2})\.\d+/);
    if (!m) return null;
    const yy = parseInt(m[1], 10);
    const mm = m[2];
    return (yy >= 50 ? '19' : '20') + m[1] + '-' + mm; // "2024-02"
  }

  /** Get the publication month for a model key (uses model_paper, falls back to source_paper). */
  function getModelPubMonth(mk) {
    const entries = pivotMap[mk];
    if (!entries) return null;
    for (const r of Object.values(entries)) {
      const pm = extractPubMonth(r.model_paper) || extractPubMonth(r.source_paper);
      if (pm) return pm;
    }
    return null;
  }

  /** Whether citation data has been loaded with actual entries. */
  function hasCitationData() {
    return citationData && Object.keys(citationData).length > 0;
  }

  /** Extract raw arxiv ID (e.g. "2402.10885") without prefix, for citation lookups. */
  function rawArxivId(url) {
    if (!url) return null;
    const m = url.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
    return m ? m[1] : null;
  }

  /** Get citation count for a model key. */
  function getModelCitations(mk) {
    if (!citationData) return null;
    const entries = pivotMap[mk];
    if (!entries) return null;
    for (const r of Object.values(entries)) {
      const aid = rawArxivId(r.model_paper) || rawArxivId(r.source_paper);
      if (aid && citationData[aid] != null) return citationData[aid];
    }
    return null;
  }

  function isModelVisible(mk) {
    const q = searchQuery();
    if (q && !(getModelDisplay(mk)).toLowerCase().includes(q)) return false;

    // Publication date filter
    const dateFrom = dateFromEl ? dateFromEl.value : ''; // "YYYY-MM" or ""
    const dateTo = dateToEl ? dateToEl.value : '';
    if (dateFrom || dateTo) {
      const pub = getModelPubMonth(mk);
      if (!pub) return false; // no pub date → hidden when date filter is active
      if (dateFrom && pub < dateFrom) return false;
      if (dateTo && pub > dateTo) return false;
    }

    // Citation count filter (skip entirely if citation data not loaded)
    const minCit = minCitationsEl ? parseInt(minCitationsEl.value, 10) : NaN;
    if (!isNaN(minCit) && minCit > 0 && hasCitationData()) {
      const cit = getModelCitations(mk);
      if (cit === null || cit < minCit) return false;
    }

    return true;
  }

  /** Per-result visibility for detail view (checks the result's own paper). */
  function isResultVisible(r) {
    // Text search
    const q = searchQuery();
    if (q && !(getModelDisplay(r.model)).toLowerCase().includes(q)) return false;

    // Publication date filter (per-result paper)
    const dateFrom = dateFromEl ? dateFromEl.value : '';
    const dateTo = dateToEl ? dateToEl.value : '';
    if (dateFrom || dateTo) {
      const pub = extractPubMonth(r.model_paper) || extractPubMonth(r.source_paper);
      if (!pub) return false;
      if (dateFrom && pub < dateFrom) return false;
      if (dateTo && pub > dateTo) return false;
    }

    // Citation count filter (skip entirely if citation data not loaded)
    const minCit = minCitationsEl ? parseInt(minCitationsEl.value, 10) : NaN;
    if (!isNaN(minCit) && minCit > 0 && hasCitationData()) {
      const aid = rawArxivId(r.model_paper) || rawArxivId(r.source_paper);
      const cit = aid ? (citationData[aid] ?? null) : null;
      if (cit === null || cit < minCit) return false;
    }

    return true;
  }

  // ─── Stats ─────────────────────────────────────────────────────────────────
  function renderStats() {
    if (!statsEl) return;
    statsEl.innerHTML =
      `<span class="stat"><strong>${modelKeys.length}</strong> models</span> · ` +
      `<span class="stat"><strong>${benchmarkKeys.length}</strong> benchmarks</span> · ` +
      `<span class="stat"><strong>${data.results.length}</strong> results</span> · ` +
      `Last updated: <span class="stat">${data.last_updated || '?'}</span>`;
  }

  // ─── Render dispatcher ─────────────────────────────────────────────────────
  function renderTable() {
    const noticeEl = $('official-notice');
    if (noticeEl) {
      const bm = selectedBenchmark && data.benchmarks[selectedBenchmark];
      if (bm && bm.official_leaderboard) {
        noticeEl.innerHTML =
          `This benchmark has an <a href="${escHtml(bm.official_leaderboard)}" target="_blank" rel="noopener noreferrer">official leaderboard</a>. ` +
          `Our data may be incomplete or outdated — check the official source for the latest results.`;
        noticeEl.style.display = '';
      } else {
        noticeEl.style.display = 'none';
      }
    }
    const notesEl = $('benchmark-notes');
    if (notesEl) {
      const bmNotes = selectedBenchmark && (data.benchmarks[selectedBenchmark] || {}).detail_notes;
      if (bmNotes) { notesEl.innerHTML = bmNotes; notesEl.style.display = ''; }
      else { notesEl.style.display = 'none'; }
    }

    if (selectedBenchmark) renderDetailView(selectedBenchmark);
    else renderOverviewTable();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // OVERVIEW TABLE (multi-benchmark pivot)
  // ═══════════════════════════════════════════════════════════════════════════
  function renderOverviewTable() {
    if (!theadEl || !tbodyEl) return;
    tableEl.className = 'overview-mode';

    // Header
    const htr = document.createElement('tr');
    htr.appendChild(th('Model', 'model-col'));
    htr.appendChild(th('Params', 'params-col'));
    for (const col of overviewColumns) {
      const cell = th('', 'benchmark-col');
      cell.dataset.colid = col.colId;
      cell.appendChild(el('span', col.label));
      const arrow = el('span', '', 'sort-arrow');
      updateArrow(arrow, col.colId);
      cell.appendChild(arrow);
      if (sortState.column === col.colId) cell.classList.add('sorted');
      cell.addEventListener('click', () => { toggleSort(col.colId); renderTable(); });
      htr.appendChild(cell);
    }
    theadEl.innerHTML = ''; theadEl.appendChild(htr);

    // Body
    const sorted = getSortedModels(sortState.column);
    const best = computeBestByColumn();
    tbodyEl.innerHTML = '';

    for (const mk of sorted) {
      if (!isModelVisible(mk)) continue;
      const model = Object.values(pivotMap[mk] || {})[0] || {};
      const tr = document.createElement('tr');

      // Model cell
      const mtd = document.createElement('td');
      mtd.className = 'model-col';
      if (model.model_paper) {
        const a = el('a', model.display_name || mk, 'model-name');
        a.href = model.model_paper; a.target = '_blank'; a.rel = 'noopener noreferrer';
        mtd.appendChild(a);
      } else {
        mtd.appendChild(el('span', model.display_name || mk, 'model-name'));
      }
      tr.appendChild(mtd);

      // Params cell
      const ptd = document.createElement('td');
      ptd.className = 'params-col';
      ptd.textContent = model.params || '—';
      tr.appendChild(ptd);

      // Score cells
      for (const col of overviewColumns) {
        const result = pivotMap[mk] && pivotMap[mk][col.bmKey];
        const bm = data.benchmarks[col.bmKey] || {};
        const metric = bm.metric || {};
        const td = document.createElement('td');
        td.className = 'score-cell';
        td.dataset.colid = col.colId;

        if (result) {
          if (best[col.colId] === mk) td.classList.add('best');
          const displayScore = getDisplayScore(result, col.bmKey, col.suite);
          td.appendChild(el('span', formatScore(displayScore, metric.name), 'score-value'));
          if (displayScore != null) td.dataset.score = displayScore;
          td.appendChild(buildTooltip(result));
        } else {
          td.classList.add('empty');
          td.textContent = '—';
        }
        tr.appendChild(td);
      }
      tbodyEl.appendChild(tr);
    }

    applyHeatmapColors();
  }

  function applyHeatmapColors() {
    const cells = tbodyEl.querySelectorAll('.score-cell[data-score]');
    const colScores = {};
    for (const cell of cells) {
      const colId = cell.dataset.colid;
      if (!colScores[colId]) colScores[colId] = [];
      colScores[colId].push({ cell, score: parseFloat(cell.dataset.score) });
    }
    for (const [colId, entries] of Object.entries(colScores)) {
      const scores = entries.map(e => e.score);
      const min = Math.min(...scores);
      const max = Math.max(...scores);
      if (min === max) continue;
      const { bmKey } = parseColId(colId);
      const higher = (data.benchmarks[bmKey] || {}).metric?.higher_is_better !== false;
      for (const { cell, score } of entries) {
        let norm = (score - min) / (max - min);
        if (!higher) norm = 1 - norm;
        const h = Math.round(norm * 142); // 0 (red) → 142 (green)
        cell.style.backgroundColor = `hsla(${h}, 70%, 35%, 0.3)`;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // DETAIL VIEW (single benchmark — flat table with full metadata)
  // ═══════════════════════════════════════════════════════════════════════════
  function renderDetailView(bmKey) {
    if (!theadEl || !tbodyEl) return;
    tableEl.className = 'detail-mode';
    const bm = data.benchmarks[bmKey] || {};
    const metric = bm.metric || {};
    const expandSuites = shouldExpandSuites(bmKey);
    const suites = expandSuites ? (bm.suites || []) : [];

    // Build ordered column list: suites + _avg inserted at avg_position
    const showAvg = expandSuites && !isSuiteOnlyBenchmark(bmKey);
    const avgPos = showAvg ? (bm.avg_position ?? suites.length) : -1;
    const detailColumns = [];
    for (let i = 0; i < suites.length; i++) {
      if (i === avgPos) detailColumns.push('_avg');
      detailColumns.push(suites[i]);
    }
    if (showAvg && avgPos >= suites.length) detailColumns.push('_avg');

    // Initialize detail sort suite
    if (expandSuites && (!detailSortSuite || !detailColumns.includes(detailSortSuite))) {
      detailSortSuite = showAvg ? '_avg' : (detailColumns[0] || null);
    }

    // Score accessor for a column key
    function colScore(r, col) {
      return col === '_avg' ? r.overall_score : (r.suite_scores || {})[col];
    }

    // Collect and sort results (apply date/citation filters per result)
    const results = data.results
      .filter(r => r.benchmark === bmKey && isResultVisible(r))
      .sort((a, b) => {
        if (expandSuites && detailSortSuite) {
          return (colScore(b, detailSortSuite) || 0) - (colScore(a, detailSortSuite) || 0);
        }
        return (b.overall_score || 0) - (a.overall_score || 0);
      });

    // Find best per column
    const bestByCol = {};
    if (expandSuites) {
      for (const col of detailColumns) {
        let bestVal = null, bestModel = null;
        for (const r of results) {
          const v = colScore(r, col);
          if (v != null && (bestVal === null || v > bestVal)) { bestVal = v; bestModel = r.model; }
        }
        if (bestModel) bestByCol[col] = bestModel;
      }
    }

    // Header
    const htr = document.createElement('tr');
    htr.appendChild(th('#', 'rank-col'));
    htr.appendChild(th('Model', 'model-col'));
    htr.appendChild(th('Params', 'params-col'));

    if (expandSuites) {
      for (const col of detailColumns) {
        const label = col === '_avg' ? (bm.avg_label || 'Avg') : shortSuiteLabel(col, bm.display_name);
        const cell = th('', 'score-col');
        cell.style.cursor = 'pointer';
        cell.appendChild(el('span', label));
        if (detailSortSuite === col) {
          cell.classList.add('sorted');
          cell.appendChild(el('span', ' ▼', 'sort-arrow'));
        }
        cell.addEventListener('click', ((c) => () => { detailSortSuite = c; renderTable(); })(col));
        htr.appendChild(cell);
      }
    } else {
      const scoreH = th(metric.name === 'avg_len' ? 'Avg Len' : 'Score (%)', 'score-col sorted');
      htr.appendChild(scoreH);
    }

    htr.appendChild(th('Source Paper', 'paper-col'));
    htr.appendChild(th('Table', 'table-col'));
    htr.appendChild(th('Curated By', 'curator-col'));
    htr.appendChild(th('Date Added', 'date-col'));
    htr.appendChild(th('Notes', 'notes-col'));
    theadEl.innerHTML = ''; theadEl.appendChild(htr);

    const colSpan = expandSuites ? 8 + detailColumns.length : 9;

    // Body
    tbodyEl.innerHTML = '';
    let rank = 0;
    for (const r of results) {
      rank++;
      const tr = document.createElement('tr');
      if (rank === 1) tr.classList.add('best-row');

      // Rank
      tr.appendChild(td(String(rank), 'rank-col'));

      // Model
      const mtd = document.createElement('td');
      mtd.className = 'model-col';
      if (r.model_paper) {
        const a = el('a', r.display_name || r.model, 'model-name');
        a.href = r.model_paper; a.target = '_blank'; a.rel = 'noopener noreferrer';
        mtd.appendChild(a);
      } else {
        mtd.appendChild(el('span', r.display_name || r.model, 'model-name'));
      }
      tr.appendChild(mtd);

      // Params
      tr.appendChild(td(r.params || '—', 'params-col'));

      if (expandSuites) {
        for (const col of detailColumns) {
          const v = colScore(r, col);
          const stc = td(formatScore(v, metric.name), 'score-col');
          if (v != null) {
            stc.classList.add('score-value');
            if (bestByCol[col] === r.model) stc.classList.add('best');
          } else {
            stc.classList.add('empty');
          }
          tr.appendChild(stc);
        }
      } else {
        const stc = td(formatScore(r.overall_score, metric.name), 'score-col');
        stc.classList.add('score-value');
        if (rank === 1) stc.classList.add('best');
        tr.appendChild(stc);
      }

      // Source paper
      const ptd = document.createElement('td');
      ptd.className = 'paper-col';
      if (r.source_paper) {
        const a = el('a', extractArxivId(r.source_paper) || r.source_paper, 'source-link');
        a.href = r.source_paper; a.target = '_blank'; a.rel = 'noopener noreferrer';
        ptd.appendChild(a);
      } else {
        ptd.textContent = '—';
      }
      tr.appendChild(ptd);

      // Table
      tr.appendChild(td(r.source_table || '—', 'table-col'));

      // Curator
      const ctd = document.createElement('td');
      ctd.className = 'curator-col';
      const isHuman = r.curated_by && r.curated_by.startsWith('@');
      ctd.innerHTML = `${isHuman ? '👤' : '🤖'} ${escHtml(r.curated_by || '?')}`;
      tr.appendChild(ctd);

      // Date
      tr.appendChild(td(r.date_added || '—', 'date-col'));

      // Notes
      const ntd = td(r.notes || '—', 'notes-col');
      ntd.title = r.notes || '';
      tr.appendChild(ntd);

      tbodyEl.appendChild(tr);

      // Sub-scores row: show task_scores breakdown (skip suite_scores when already shown as columns)
      const subScores = expandSuites ? r.task_scores : (r.suite_scores || r.task_scores);
      if (subScores && Object.keys(subScores).length > 0) {
        const subTr = document.createElement('tr');
        subTr.className = 'sub-scores-row';
        const subTd = document.createElement('td');
        subTd.colSpan = colSpan;
        let html = '<div class="sub-scores-grid">';
        for (const [label, val] of Object.entries(subScores)) {
          html += `<span class="sub-score-item"><span class="sub-label">${escHtml(label)}</span> `;
          html += `<span class="sub-value">${formatScore(val, metric.name)}</span></span>`;
        }
        html += '</div>';
        subTd.innerHTML = html;
        subTr.appendChild(subTd);
        tbodyEl.appendChild(subTr);
      }
    }
  }

  // ─── Score resolver (handles suite-only benchmarks) ────────────────────────
  // If suite is specified, returns that suite's score directly.
  // Otherwise returns overall_score, or first available suite score as fallback.
  function getDisplayScore(result, bmKey, suite) {
    if (suite === '_avg') return result.overall_score ?? null;
    if (suite) {
      return (result.suite_scores || {})[suite] ?? null;
    }
    if (result.overall_score != null) return result.overall_score;
    const bm = data.benchmarks[bmKey] || {};
    const suites = bm.suites || [];
    const ss = result.suite_scores || {};
    for (const s of suites) {
      if (ss[s] != null) return ss[s];
    }
    const vals = Object.values(ss);
    return vals.length > 0 ? vals[0] : null;
  }

  // Does every result for this benchmark have null overall_score?
  function isSuiteOnlyBenchmark(bmKey) {
    const results = data.results.filter(r => r.benchmark === bmKey);
    return results.length > 0 && results.every(r => r.overall_score == null)
      && (data.benchmarks[bmKey] || {}).suites && (data.benchmarks[bmKey] || {}).suites.length > 0;
  }

  // Should this benchmark expand into per-suite columns?
  function shouldExpandSuites(bmKey) {
    const bm = data.benchmarks[bmKey] || {};
    if (!bm.suites || bm.suites.length === 0) return false;
    if (bm.expand_suites) return true;
    return isSuiteOnlyBenchmark(bmKey);
  }

  // Short label for a suite, stripping redundant benchmark name prefix
  function shortSuiteLabel(suite, bmDisplayName) {
    let label = suite.replace(/_/g, ' ');
    if (bmDisplayName) {
      const prefix = bmDisplayName.toLowerCase() + ' ';
      if (label.startsWith(prefix)) label = label.substring(prefix.length);
    }
    return label.replace(/google robot/, 'GR');
  }

  // ─── Sorting ───────────────────────────────────────────────────────────────
  function toggleSort(col) {
    if (sortState.column === col) sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
    else { sortState.column = col; sortState.direction = 'desc'; }
  }

  function getLatestDate(mk) {
    let latest = '';
    const results = pivotMap[mk];
    if (results) {
      for (const r of Object.values(results)) {
        if (r.date_added && r.date_added > latest) latest = r.date_added;
      }
    }
    return latest || '—';
  }

  function getSortedModels(col) {
    const dir = sortState.direction;
    if (col === '_date') {
      return [...modelKeys].sort((a, b) => {
        const da = getLatestDate(a);
        const db = getLatestDate(b);
        if (da === db) return 0;
        return dir === 'asc' ? (da < db ? -1 : 1) : (da > db ? -1 : 1);
      });
    }
    const { bmKey, suite } = parseColId(col);
    return [...modelKeys].sort((a, b) => {
      const ra = pivotMap[a] && pivotMap[a][bmKey];
      const rb = pivotMap[b] && pivotMap[b][bmKey];
      const sa = ra ? getDisplayScore(ra, bmKey, suite) : null;
      const sb = rb ? getDisplayScore(rb, bmKey, suite) : null;
      if (sa === null && sb === null) return 0;
      if (sa === null) return 1;
      if (sb === null) return -1;
      return dir === 'asc' ? sa - sb : sb - sa;
    });
  }

  function computeBestByColumn() {
    const best = {};
    for (const col of overviewColumns) {
      const higher = (data.benchmarks[col.bmKey] || {}).metric?.higher_is_better !== false;
      let bestM = null, bestS = null;
      for (const mk of modelKeys) {
        const r = pivotMap[mk] && pivotMap[mk][col.bmKey];
        if (!r) continue;
        const s = getDisplayScore(r, col.bmKey, col.suite);
        if (s === null) continue;
        if (bestS === null || (higher ? s > bestS : s < bestS)) { bestS = s; bestM = mk; }
      }
      if (bestM) best[col.colId] = bestM;
    }
    return best;
  }

  function updateArrow(arrowEl, key) {
    if (sortState.column === key) {
      arrowEl.textContent = sortState.direction === 'asc' ? ' ▲' : ' ▼';
      arrowEl.style.opacity = '1';
    } else {
      arrowEl.textContent = ' ▼'; arrowEl.style.opacity = '0.3';
    }
  }

  // ─── Tooltip ───────────────────────────────────────────────────────────────
  function buildTooltip(result) {
    const div = document.createElement('div');
    div.className = 'tooltip-content';
    if (result.source_paper) {
      const p = document.createElement('p');
      const a = el('a', result.source_paper, '');
      a.href = result.source_paper; a.target = '_blank';
      p.appendChild(document.createTextNode('Paper: ')); p.appendChild(a);
      div.appendChild(p);
    }
    if (result.source_table) div.appendChild(el('p', 'Table: ' + result.source_table));
    div.appendChild(el('p', 'Curated by: ' + (result.curated_by || '?')));
    if (result.date_added) div.appendChild(el('p', 'Date: ' + result.date_added));
    if (result.notes) div.appendChild(el('p', 'Notes: ' + result.notes));
    return div;
  }

  // ─── Breakdown panel ───────────────────────────────────────────────────────
  function closeBreakdown() {
    if (breakdownPanelEl) { breakdownPanelEl.classList.remove('active'); breakdownPanelEl.innerHTML = ''; }
  }

  // ─── Coverage bar ─────────────────────────────────────────────────────────
  function renderCoverage() {
    if (!coverageBarEl || !coverageData) return;
    const bms = coverageData.benchmarks || {};
    const keys = Object.keys(bms).sort((a, b) => (bms[b].citing_papers || 0) - (bms[a].citing_papers || 0));

    let html = '<div class="coverage-header">';
    html += '<span class="coverage-title">Paper Coverage by Benchmark</span>';
    html += `<span class="coverage-summary">${coverageData.total_results} results from ${coverageData.total_models} models`;
    if (coverageData.total_papers_reviewed) html += ` · ${coverageData.total_papers_reviewed} papers reviewed`;
    html += '</span></div>';
    html += '<div class="coverage-explanation">Denominator = papers citing the benchmark paper (via <a href="https://www.semanticscholar.org/" target="_blank" rel="noopener">Semantic Scholar</a>). Not all citing papers report evaluation results — this shows how much of that citation pool we have covered.</div>';
    html += '<div class="coverage-grid">';

    for (const key of keys) {
      const bm = bms[key];
      const citing = bm.citing_papers;
      const reviewed = bm.papers_reviewed || 0;
      if (!citing) continue; // skip if no citation data
      const pct = Math.min(100, Math.round((reviewed / Math.max(1, citing)) * 100));
      const barColor = pct > 15 ? 'var(--accent)' : pct > 5 ? '#da9679' : '#e24a8d';
      html += `<div class="coverage-item" title="${reviewed} papers reviewed / ${citing} citing papers">`;
      html += `<div class="coverage-label"><span>${escHtml(bm.display_name)}</span><span class="coverage-nums">${reviewed}/${citing}</span></div>`;
      html += `<div class="coverage-track"><div class="coverage-fill" style="width:${Math.max(2, pct)}%;background:${barColor}"></div></div>`;
      html += '</div>';
    }
    html += '</div>';
    coverageBarEl.innerHTML = html;
  }

  // ─── Helpers ───────────────────────────────────────────────────────────────
  function formatScore(v, metricName) {
    if (v === null || v === undefined) return '—';
    const n = parseFloat(v);
    if (isNaN(n)) return String(v);
    return metricName === 'avg_len' ? n.toFixed(3) : n.toFixed(1);
  }

  function getModelDisplay(mk) {
    const r = Object.values(pivotMap[mk] || {})[0];
    return r ? (r.display_name || mk) : mk;
  }

  function extractArxivId(url) {
    if (!url) return null;
    const m = url.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
    return m ? 'arXiv:' + m[1] : null;
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  }

  // DOM helpers
  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function th(text, cls) { return el('th', text, cls); }
  function td(text, cls) { return el('td', text, cls); }
})();
