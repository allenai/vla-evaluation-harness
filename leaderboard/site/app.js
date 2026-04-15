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

  // ─── Caches (computed once in buildPivot, static until data reload) ────────
  let suiteOnlyCache = {};   // bmKey → boolean
  let modelDisplayCache = {}; // model key → display name
  let bestByColumnCache = {}; // colId → model key (best score)

  // ─── Pagination state ─────────────────────────────────────────────────────
  const PAGE_SIZE = 50;
  let currentPage = 0;
  let lastFilteredModels = []; // cached for pagination
  let lastSortCol = null;      // track whether sort/filter changed vs page-only
  let lastSortDir = null;

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
  const firstPartyOnlyEl = $('first-party-only');
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

    function resetAndRender() { currentPage = 0; renderTable(); }
    if (benchmarkFilterEl) benchmarkFilterEl.addEventListener('change', onBenchmarkFilterChange);
    for (const [elem, evt] of [
      [modelSearchEl, 'input'], [dateFromEl, 'change'], [dateToEl, 'change'],
      [minCitationsEl, 'input'], [firstPartyOnlyEl, 'change'],
    ]) {
      if (elem) elem.addEventListener(evt, resetAndRender);
    }
    if (breakdownPanelEl) breakdownPanelEl.addEventListener('click', e => {
      if (e.target.classList.contains('breakdown-close')) closeBreakdown();
    });

    // Tooltip: single delegated listener on tbody
    if (tbodyEl) {
      tbodyEl.addEventListener('mouseenter', e => {
        const cell = e.target.closest('.score-cell[data-tip-curator]');
        if (cell) showTooltip(cell);
      }, true);
      tbodyEl.addEventListener('mouseleave', e => {
        const cell = e.target.closest('.score-cell[data-tip-curator]');
        if (cell) hideTooltip();
      }, true);
    }
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

    // Cache suite-only status per benchmark
    suiteOnlyCache = {};
    for (const bmKey of benchmarkKeys) {
      const bm = data.benchmarks[bmKey] || {};
      if (!bm.suites || bm.suites.length === 0) { suiteOnlyCache[bmKey] = false; continue; }
      const bmResults = data.results.filter(r => r.benchmark === bmKey);
      suiteOnlyCache[bmKey] = bmResults.length > 0 && bmResults.every(r => r.overall_score == null);
    }

    // Cache model display names
    modelDisplayCache = {};
    for (const mk of modelKeys) {
      const r = Object.values(pivotMap[mk])[0];
      modelDisplayCache[mk] = r ? (r.display_name || mk) : mk;
    }

    // Cache best-per-column (static for all models — only changes on data reload)
    bestByColumnCache = {};

    buildOverviewColumns();
    computeBestByColumn();
  }

  // ─── Overview columns (expand suite-only benchmarks) ─────────────────────
  function buildOverviewColumns() {
    overviewColumns = [];
    for (const bmKey of benchmarkKeys) {
      if (shouldExpandSuites(bmKey)) {
        const bm = data.benchmarks[bmKey] || {};
        const suites = bm.suites || [];
        const bmName = bm.display_name || bmKey;
        const showAvg = !suiteOnlyCache[bmKey];
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
    detailSortSuite = null;
    currentPage = 0;
    if (val) { sortState.column = val; sortState.direction = 'desc'; }
    else { sortState.column = '_date'; sortState.direction = 'desc'; }
    closeBreakdown();
    renderTable();
  }

  // ─── Arxiv helpers (single parse, shared across all callers) ──────────────
  function rawArxivId(url) {
    if (!url) return null;
    const m = url.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
    return m ? m[1] : null;
  }

  function extractPubMonth(url) {
    if (!url) return null;
    const m = url.match(/arxiv\.org\/abs\/(\d{2})(\d{2})\.\d+/);
    if (!m) return null;
    const yy = parseInt(m[1], 10);
    return (yy >= 50 ? '19' : '20') + m[1] + '-' + m[2];
  }

  /** Get pub month from a result, trying model_paper then reported_paper. */
  function getResultPubMonth(r) {
    return extractPubMonth(r.model_paper) || extractPubMonth(r.reported_paper);
  }

  /** Get arxiv ID from a result, trying model_paper then reported_paper. */
  function getResultArxivId(r) {
    return rawArxivId(r.model_paper) || rawArxivId(r.reported_paper);
  }

  // ─── Search & Filters ─────────────────────────────────────────────────────
  function searchQuery() { return modelSearchEl ? modelSearchEl.value.trim().toLowerCase() : ''; }

  function getModelPubMonth(mk) {
    const entries = pivotMap[mk];
    if (!entries) return null;
    for (const r of Object.values(entries)) {
      const pm = getResultPubMonth(r);
      if (pm) return pm;
    }
    return null;
  }

  function getModelCitations(mk) {
    if (!citationData) return null;
    const entries = pivotMap[mk];
    if (!entries) return null;
    for (const r of Object.values(entries)) {
      const aid = getResultArxivId(r);
      if (aid && citationData[aid] != null) return citationData[aid];
    }
    return null;
  }

  /** Whether citation data has been loaded with actual entries. */
  function hasCitationData() {
    return citationData && Object.keys(citationData).length > 0;
  }

  /** A row is "third-party" when the paper that reported it differs from the
   *  paper that introduced the model. The bibkey-style `__` separator in the
   *  model field is a fallback signal for the same fact. */
  function isThirdParty(r) {
    if (r.reported_paper && r.model_paper && r.reported_paper !== r.model_paper) return true;
    if (typeof r.model === 'string' && r.model.includes('__')) return true;
    return false;
  }

  /** A model key is third-party when ALL its rows are third-party. */
  function isModelThirdParty(mk) {
    const entries = pivotMap[mk];
    if (!entries) return false;
    const rows = Object.values(entries);
    if (rows.length === 0) return false;
    return rows.every(isThirdParty);
  }

  /** Shared filter logic: date range + citation threshold. */
  function passesDateCitationFilter(pubMonth, arxivId) {
    const dateFrom = dateFromEl ? dateFromEl.value : '';
    const dateTo = dateToEl ? dateToEl.value : '';
    if (dateFrom || dateTo) {
      if (!pubMonth) return false;
      if (dateFrom && pubMonth < dateFrom) return false;
      if (dateTo && pubMonth > dateTo) return false;
    }
    const minCit = minCitationsEl ? parseInt(minCitationsEl.value, 10) : NaN;
    if (!isNaN(minCit) && minCit > 0 && hasCitationData()) {
      const cit = arxivId ? (citationData[arxivId] ?? null) : null;
      if (cit === null || cit < minCit) return false;
    }
    return true;
  }

  function isModelVisible(mk) {
    const q = searchQuery();
    if (q && !getModelDisplay(mk).toLowerCase().includes(q)) return false;
    if (firstPartyOnlyEl && firstPartyOnlyEl.checked && isModelThirdParty(mk)) return false;
    return passesDateCitationFilter(getModelPubMonth(mk), getResultArxivId(Object.values(pivotMap[mk])[0]));
  }

  function isResultVisible(r) {
    const q = searchQuery();
    if (q && !getModelDisplay(r.model).toLowerCase().includes(q)) return false;
    if (firstPartyOnlyEl && firstPartyOnlyEl.checked && isThirdParty(r)) return false;
    return passesDateCitationFilter(getResultPubMonth(r), getResultArxivId(r));
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
    hideTooltip();
    tableEl.className = 'overview-mode';

    // Detect whether sort/filter changed or this is a page-only change
    const sortChanged = sortState.column !== lastSortCol || sortState.direction !== lastSortDir;
    const needsRecompute = sortChanged || lastFilteredModels.length === 0;

    if (needsRecompute) {
      // Header (only rebuild when sort state changes)
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

      // Recompute filtered + sorted list
      const sorted = getSortedModels(sortState.column);
      lastFilteredModels = sorted.filter(mk => isModelVisible(mk));
      lastSortCol = sortState.column;
      lastSortDir = sortState.direction;
    }

    // Paginate
    const totalPages = Math.max(1, Math.ceil(lastFilteredModels.length / PAGE_SIZE));
    if (currentPage >= totalPages) currentPage = totalPages - 1;
    const start = currentPage * PAGE_SIZE;
    const pageModels = lastFilteredModels.slice(start, start + PAGE_SIZE);

    // Build rows in a DocumentFragment
    const frag = document.createDocumentFragment();
    for (const mk of pageModels) {
      const model = Object.values(pivotMap[mk] || {})[0] || {};
      const tr = document.createElement('tr');
      tr.appendChild(buildModelCell(model.display_name || mk, model.model_paper));

      const ptd = document.createElement('td');
      ptd.className = 'params-col';
      ptd.textContent = model.params || '—';
      tr.appendChild(ptd);

      for (const col of overviewColumns) {
        const result = pivotMap[mk] && pivotMap[mk][col.bmKey];
        const bm = data.benchmarks[col.bmKey] || {};
        const metric = bm.metric || {};
        const cell = document.createElement('td');
        cell.className = 'score-cell';
        cell.dataset.colid = col.colId;

        if (result) {
          if (bestByColumnCache[col.colId] === mk) cell.classList.add('best');
          const displayScore = getDisplayScore(result, col.bmKey, col.suite);
          cell.appendChild(el('span', formatScore(displayScore, metric.name), 'score-value'));
          if (displayScore != null) cell.dataset.score = displayScore;
          storeTooltipData(cell, result);
        } else {
          cell.classList.add('empty');
          cell.textContent = '—';
        }
        tr.appendChild(cell);
      }
      frag.appendChild(tr);
    }
    tbodyEl.innerHTML = '';
    tbodyEl.appendChild(frag);

    requestAnimationFrame(applyHeatmapColors);
    renderPagination(totalPages);
  }

  // ─── Pagination controls ─────────────────────────────────────────────────
  function renderPagination(totalPages) {
    let pager = $('pagination');
    if (totalPages <= 1) {
      if (pager) pager.style.display = 'none';
      return;
    }
    if (!pager) {
      pager = document.createElement('div');
      pager.id = 'pagination';
      pager.className = 'pagination';
      tableEl.parentNode.insertBefore(pager, tableEl.nextSibling);
    }
    pager.style.display = '';
    const total = lastFilteredModels.length;
    const s = currentPage * PAGE_SIZE + 1;
    const e = Math.min(s + PAGE_SIZE - 1, total);
    pager.innerHTML =
      `<button class="page-btn" ${currentPage === 0 ? 'disabled' : ''} data-dir="prev">\u2190 Prev</button>` +
      `<span class="page-info">${s}\u2013${e} of ${total}</span>` +
      `<button class="page-btn" ${currentPage >= totalPages - 1 ? 'disabled' : ''} data-dir="next">Next \u2192</button>`;
    pager.onclick = e => {
      const btn = e.target.closest('[data-dir]');
      if (!btn || btn.disabled) return;
      currentPage += btn.dataset.dir === 'next' ? 1 : -1;
      renderTable();
      tableEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
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
      let min = Infinity, max = -Infinity;
      for (const { score } of entries) {
        if (score < min) min = score;
        if (score > max) max = score;
      }
      if (min === max) continue;
      const { bmKey } = parseColId(colId);
      const GREEN_HUE = 142;
      const higher = (data.benchmarks[bmKey] || {}).metric?.higher_is_better !== false;
      for (const { cell, score } of entries) {
        let norm = (score - min) / (max - min);
        if (!higher) norm = 1 - norm;
        cell.style.backgroundColor = `hsla(${Math.round(norm * GREEN_HUE)}, 70%, 35%, 0.3)`;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // DETAIL VIEW (single benchmark — flat table with full metadata)
  // ═══════════════════════════════════════════════════════════════════════════
  function renderDetailView(bmKey) {
    if (!theadEl || !tbodyEl) return;
    tableEl.className = 'detail-mode';
    const pager = $('pagination');
    if (pager) pager.style.display = 'none';
    const bm = data.benchmarks[bmKey] || {};
    const metric = bm.metric || {};
    const expandSuites = shouldExpandSuites(bmKey);
    const suites = expandSuites ? (bm.suites || []) : [];

    // Build ordered column list: suites + _avg inserted at avg_position
    const showAvg = expandSuites && !suiteOnlyCache[bmKey];
    const avgPos = showAvg ? (bm.avg_position ?? suites.length) : -1;
    const detailColumns = [];
    for (let i = 0; i < suites.length; i++) {
      if (i === avgPos) detailColumns.push('_avg');
      detailColumns.push(suites[i]);
    }
    if (showAvg && avgPos >= suites.length) detailColumns.push('_avg');

    if (expandSuites && (!detailSortSuite || !detailColumns.includes(detailSortSuite))) {
      detailSortSuite = showAvg ? '_avg' : (detailColumns[0] || null);
    }

    function colScore(r, col) {
      return col === '_avg' ? r.overall_score : (r.suite_scores || {})[col];
    }

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
          cell.appendChild(el('span', ' \u25BC', 'sort-arrow'));
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

    // Body — use DocumentFragment
    const frag = document.createDocumentFragment();
    let rank = 0;
    for (const r of results) {
      rank++;
      const tr = document.createElement('tr');
      if (rank === 1) tr.classList.add('best-row');

      tr.appendChild(td(String(rank), 'rank-col'));
      tr.appendChild(buildModelCell(r.display_name || r.model, r.model_paper));
      tr.appendChild(td(r.params || '\u2014', 'params-col'));

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
      if (r.reported_paper) {
        ptd.appendChild(externalLink(r.reported_paper, extractArxivId(r.reported_paper) || r.reported_paper, 'source-link'));
      } else {
        ptd.textContent = '\u2014';
      }
      tr.appendChild(ptd);

      tr.appendChild(td(r.reported_table || '\u2014', 'table-col'));

      const ctd = document.createElement('td');
      ctd.className = 'curator-col';
      const isHuman = r.curated_by && r.curated_by.startsWith('@');
      ctd.innerHTML = `${isHuman ? '\uD83D\uDC64' : '\uD83E\uDD16'} ${escHtml(r.curated_by || '?')}`;
      tr.appendChild(ctd);

      tr.appendChild(td(r.date_added || '\u2014', 'date-col'));

      const ntd = td(r.notes || '\u2014', 'notes-col');
      ntd.title = r.notes || '';
      tr.appendChild(ntd);

      frag.appendChild(tr);

      // Sub-scores row: show task_scores breakdown (skip suite_scores when already shown as columns).
      // Use the first non-empty source — an empty `suite_scores: {}` must not mask task_scores.
      const hasKeys = o => o && Object.keys(o).length > 0;
      const subScores = expandSuites
        ? r.task_scores
        : (hasKeys(r.suite_scores) ? r.suite_scores : r.task_scores);
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
        frag.appendChild(subTr);
      }
    }
    tbodyEl.innerHTML = '';
    tbodyEl.appendChild(frag);
  }

  // ─── Score resolver ────────────────────────────────────────────────────────
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

  function shouldExpandSuites(bmKey) {
    const bm = data.benchmarks[bmKey] || {};
    if (!bm.suites || bm.suites.length === 0) return false;
    if (bm.expand_suites) return true;
    return suiteOnlyCache[bmKey];
  }

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
    currentPage = 0;
  }

  function getLatestDate(mk) {
    let latest = '';
    const results = pivotMap[mk];
    if (results) {
      for (const r of Object.values(results)) {
        if (r.date_added && r.date_added > latest) latest = r.date_added;
      }
    }
    return latest || '\u2014';
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
    bestByColumnCache = {};
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
      if (bestM) bestByColumnCache[col.colId] = bestM;
    }
  }

  function updateArrow(arrowEl, key) {
    if (sortState.column === key) {
      arrowEl.textContent = sortState.direction === 'asc' ? ' \u25B2' : ' \u25BC';
      arrowEl.style.opacity = '1';
    } else {
      arrowEl.textContent = ' \u25BC'; arrowEl.style.opacity = '0.3';
    }
  }

  // ─── Shared tooltip (single DOM element, positioned on hover) ─────────────
  let sharedTooltip = null;

  function ensureSharedTooltip() {
    if (sharedTooltip) return sharedTooltip;
    sharedTooltip = document.createElement('div');
    sharedTooltip.className = 'tooltip-content';
    sharedTooltip.style.display = 'none';
    document.body.appendChild(sharedTooltip);
    return sharedTooltip;
  }

  function storeTooltipData(td, result) {
    td.dataset.tipPaper = result.reported_paper || '';
    td.dataset.tipTable = result.reported_table || '';
    td.dataset.tipCurator = result.curated_by || '';
    td.dataset.tipDate = result.date_added || '';
    td.dataset.tipNotes = result.notes || '';
  }

  function showTooltip(td) {
    const tip = ensureSharedTooltip();
    let html = '';
    function row(label, val) { html += `<span class="tip-label">${label}</span><span>${val}</span>`; }
    if (td.dataset.tipPaper) row('Paper', `<a href="${escHtml(td.dataset.tipPaper)}" target="_blank" class="tip-link">${escHtml(td.dataset.tipPaper)}</a>`);
    if (td.dataset.tipTable) row('Table', escHtml(td.dataset.tipTable));
    row('Curated', escHtml(td.dataset.tipCurator || '?'));
    if (td.dataset.tipDate) row('Date', escHtml(td.dataset.tipDate));
    if (td.dataset.tipNotes) row('Notes', escHtml(td.dataset.tipNotes));
    tip.innerHTML = html;
    const rect = td.getBoundingClientRect();
    tip.style.visibility = 'hidden';
    tip.style.display = 'grid';
    const tipW = tip.offsetWidth;
    const tipH = tip.offsetHeight;
    tip.style.left = Math.max(0, rect.right - tipW) + window.scrollX + 'px';
    tip.style.top = (rect.top - tipH - 4) + window.scrollY + 'px';
    tip.style.visibility = '';
  }

  function hideTooltip() {
    if (sharedTooltip) sharedTooltip.style.display = 'none';
  }

  // ─── Breakdown panel ───────────────────────────────────────────────────────
  function closeBreakdown() {
    if (breakdownPanelEl) { breakdownPanelEl.classList.remove('active'); breakdownPanelEl.innerHTML = ''; }
  }

  // ─── Coverage bar ─────────────────────────────────────────────────────────
  function renderCoverage() {
    if (!coverageBarEl || !coverageData) return;
    const bms = coverageData.benchmarks || {};
    const keys = Object.keys(bms).sort(
      (a, b) => (bms[b].arxiv_citing_papers || bms[b].citing_papers || 0)
              - (bms[a].arxiv_citing_papers || bms[a].citing_papers || 0)
    );

    let html = '<div class="coverage-header">';
    html += '<span class="coverage-title">Paper Coverage by Benchmark</span>';
    html += `<span class="coverage-summary">${coverageData.total_results} results from ${coverageData.total_models} models`;
    if (coverageData.total_papers_reviewed) html += ` \xB7 ${coverageData.total_papers_reviewed} papers reviewed`;
    html += '</span></div>';
    html += '<div class="coverage-explanation">Denominator = arXiv-preprint papers citing the benchmark (via <a href="https://www.semanticscholar.org/" target="_blank" rel="noopener noreferrer">Semantic Scholar</a>). Total citations in parentheses include non-arXiv publications that cannot be reviewed via the arxiv reading pipeline. Not every citing paper reports new evaluation numbers \u2014 this shows how much of the reviewable pool we have covered.</div>';
    html += '<div class="coverage-grid">';

    for (const key of keys) {
      const bm = bms[key];
      const citingTotal = bm.citing_papers;
      const citingArxiv = bm.arxiv_citing_papers || citingTotal;
      const reviewed = bm.papers_reviewed || 0;
      if (!citingArxiv) continue;
      const pct = Math.min(100, Math.round((reviewed / Math.max(1, citingArxiv)) * 100));
      const barColor = pct > 15 ? 'var(--accent)' : pct > 5 ? '#da9679' : '#e24a8d';
      const showTotal = citingTotal && citingTotal !== citingArxiv;
      const numsText = showTotal
        ? `${reviewed}/${citingArxiv} <span class="coverage-nums-sub">(${citingTotal} total)</span>`
        : `${reviewed}/${citingArxiv}`;
      const titleText = showTotal
        ? `${reviewed} reviewed / ${citingArxiv} arXiv citing (${citingTotal} total incl. non-arXiv)`
        : `${reviewed} reviewed / ${citingArxiv} citing papers`;
      html += `<div class="coverage-item" title="${titleText}">`;
      html += `<div class="coverage-label"><span>${escHtml(bm.display_name)}</span><span class="coverage-nums">${numsText}</span></div>`;
      html += `<div class="coverage-track"><div class="coverage-fill" style="width:${Math.max(2, pct)}%;background:${barColor}"></div></div>`;
      html += '</div>';
    }
    html += '</div>';
    coverageBarEl.innerHTML = html;
  }

  // ─── Helpers ───────────────────────────────────────────────────────────────
  function formatScore(v, metricName) {
    if (v === null || v === undefined) return '\u2014';
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
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
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

  function externalLink(href, text, cls) {
    const a = el('a', text, cls);
    a.href = href; a.target = '_blank'; a.rel = 'noopener noreferrer';
    return a;
  }

  function buildModelCell(displayName, paperUrl) {
    const mtd = document.createElement('td');
    mtd.className = 'model-col';
    if (paperUrl) {
      mtd.appendChild(externalLink(paperUrl, displayName, 'model-name'));
    } else {
      mtd.appendChild(el('span', displayName, 'model-name'));
    }
    return mtd;
  }
})();
