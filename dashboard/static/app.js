window.onerror = function(msg, src, line, col, err) { document.title = 'ERR L' + line + ': ' + msg; console.error('JS ERROR line ' + line + ': ' + msg); };
let daemonAlive = false;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const tabs = ['live','graph','explorer','logs','health','traces'];
    t.classList.toggle('active', tabs[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'graph') { setTimeout(() => { if (!graph3dData) { loadGraph3D(); } else if (graph3d) { var c = document.getElementById('graph-3d'); c.style.height = 'calc(100vh - 42px)'; void c.offsetHeight; var w = c.offsetWidth || 800; var h = c.offsetHeight || 600; graph3d.width(w).height(h); graph3d.renderer().setSize(w, h); graph3d.camera().aspect = w/h; graph3d.camera().updateProjectionMatrix(); } }, 300); }
  if (name === 'explorer') searchNodes();
  if (name === 'logs') loadLogs();
  if (name === 'health') { loadHealth(); loadSystemStatus(); loadAspects(); }
  if (name === 'traces') { loadTraces(); _startTraceAutoRefresh(); } else { _stopTraceAutoRefresh(); }
}

async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    daemonAlive = d.daemon === 'alive';
    const statusClass = daemonAlive ? 'alive' : 'unavailable';
    const statusText = daemonAlive ? 'Daemon: alive' : 'Daemon: offline';
    document.getElementById('stats-bar').innerHTML =
      `<div class="stat"><span class="stat-value">${d.nodes}</span><span class="stat-label">Nodes</span></div>
       <div class="stat"><span class="stat-value">${d.edges}</span><span class="stat-label">Edges</span></div>
       <div class="stat"><span class="stat-value">${d.locked}</span><span class="stat-label">Locked</span></div>
       <div class="stat"><span class="stat-value">${d.recent_24h}</span><span class="stat-label">24h</span></div>
       <div class="stat"><span class="stat-value">${d.orphans}</span><span class="stat-label">Orphans</span></div>
       <div class="daemon-status ${statusClass}">${statusText}</div>
       <div class="daemon-status alive" style="font-size:10px;padding:3px 8px">${d.encoding ? 'Encode: ' + d.encoding.position + '/5' + (d.encoding.position === 0 ? ' ⚡' : '') : ''}</div>`;

    const banner = document.getElementById('daemon-banner');
    if (!daemonAlive) {
      banner.innerHTML = '<div class="no-daemon-banner">Daemon is not running — showing read-only data from database. Live events unavailable.</div>';
    } else {
      banner.innerHTML = '';
    }

    const sel = document.getElementById('type-filter');
    const current = sel.value;
    sel.innerHTML = '<option value="">All types</option>';
    Object.entries(d.types).forEach(([t, c]) => {
      sel.innerHTML += `<option value="${t}" ${t===current?'selected':''}>${t} (${c})</option>`;
    });
  } catch(e) {}
}
loadStats();
loadSessions();
setInterval(loadStats, 30000);
setInterval(loadSessions, 60000);

// Live feed — polls /api/recalls. Cursor is ISO timestamp (since_ts), not
// integer rowid: trace_events.id is now an 8-char hex string, so integer
// ordering no longer applies. created_at is monotonic per writer and stable
// under the schema change.
let lastRecallTs = '';
const MAX_ENTRIES = 100;

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function localTime(utcStr, mode) {
  if (!utcStr) return '';
  let s = utcStr;
  if (s.length >= 19 && !s.endsWith('Z') && !s.includes('+')) s += 'Z';
  const d = new Date(s);
  if (isNaN(d)) return utcStr;
  if (mode === 'time') return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  return d.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function toggleDetails(btn) {
  const details = btn.nextElementSibling;
  details.classList.toggle('open');
  btn.textContent = details.classList.contains('open') ? 'Hide Details' : 'Full Details';
}

function toggleHookBody(el) {
  const body = el.parentElement.querySelector('.hook-body');
  body.classList.toggle('open');
}

const SOURCE_COLORS = {
  hook: '#7eb8ff',
  mcp: '#b8ff7e',
  internal: '#888',
  unknown: '#666'
};
const SOURCE_LABELS = {
  hook: 'HOOK',
  mcp: 'ANCHOR',
  internal: 'INTERNAL',
  unknown: '?'
};

// Identity stamping landed in trace_events metadata via 75075eb / 65bf483.
// Render a compact chip like "Tom→Anchor" — the arrow direction matches the
// O/Δ pattern (human observes, agent responds). Empty strings → no chip;
// half-stamped traces (one side present, other missing) still show what we
// have so the operator can spot the gap.
function renderIdentityChip(human, agent) {
  if (!human && !agent) return '';
  const h = human ? escapeHtml(human) : '?';
  const a = agent ? escapeHtml(agent) : '?';
  return '<span class="identity-chip" title="speaker → responder">' +
    h + '<span style="color:#555;margin:0 3px">→</span>' + a + '</span>';
}

function renderRecallEntry(evt) {
  const div = document.createElement('div');
  div.className = 'hook-entry recall-entry';
  const src = evt.source || 'unknown';
  div.dataset.source = src;
  div.dataset.scale = 's1';
  div.dataset.recallId = evt.id;
  div.dataset.needsJudge = (src === 'hook' && !evt.judge_output) ? '1' : '0';
  div.dataset.ts = evt.timestamp || '';
  const t = localTime(evt.timestamp, 'time');
  const srcColor = SOURCE_COLORS[src] || '#666';
  const srcLabel = SOURCE_LABELS[src] || src.toUpperCase();
  const sid = evt.session_id ? evt.session_id.substring(0, 8) : '';
  const count = evt.returned_count || 0;
  const titles = evt.titles || {};
  const snippets = evt.snippets || {};
  const ids = evt.returned_ids || [];
  const usedIds = new Set(evt.used_ids || []);

  // Short details: judge_output = exact additionalContext sent to Claude
  // Falls back to candidate title chips if no judge data yet (MCP recalls, old data)
  let shortContent = '';
  if (evt.judge_output && evt.judge_output !== '(no selection)') {
    shortContent = '<div class="recall-judge-output"><pre>' + escapeHtml(evt.judge_output) + '</pre></div>';
  } else if (evt.judge_output === '(no selection)') {
    const titleEntries = Object.entries(titles).slice(0, 8);
    const total = Object.keys(titles).length;
    shortContent = '<div class="recall-candidates"><div class="recall-candidates-header">0 selected from ' + total + ' candidates</div>' +
      titleEntries.map(([nid, title]) => {
        return '<div class="recall-candidate" onclick="loadNodeDetail(&quot;' + nid + '&quot;)">' + escapeHtml(title) + '</div>';
      }).join('') +
      (total > 8 ? '<div class="recall-candidate more">+' + (total - 8) + ' more</div>' : '') +
      '</div>';
  } else {
    // No judge data — show candidate titles as compact list
    const titleEntries = Object.entries(titles).slice(0, 12);
    if (titleEntries.length) {
      const total = Object.keys(titles).length;
      shortContent = '<div class="recall-candidates"><div class="recall-candidates-header">' + total + ' candidates (pending judge)</div>' +
        titleEntries.map(([nid, title]) => {
          const isUsed = usedIds.has(nid);
          return '<div class="recall-candidate' + (isUsed ? ' used' : '') + '" onclick="loadNodeDetail(&quot;' + nid + '&quot;)">' +
            escapeHtml(title) + '</div>';
        }).join('') +
        (total > 12 ? '<div class="recall-candidate more">+' + (total - 12) + ' more</div>' : '') +
        '</div>';
    }
  }

  // Full details: judge_prompt = exact prompt sent to Haiku
  // Falls back to candidate list if no judge data yet
  let fullDetails = '<div class="hook-details"><pre>';
  if (evt.judge_prompt) {
    fullDetails += escapeHtml(evt.judge_prompt);
  } else {
    fullDetails += '=== ' + ids.length + ' CANDIDATES (no judge prompt stored) ===\n\n';
    for (const nid of ids) {
      const title = titles[nid] || nid.substring(0, 12);
      const snippet = snippets[nid] || '';
      fullDetails += title + '\n';
      if (snippet) fullDetails += '  ' + snippet.substring(0, 150).replace(/\n/g, ' ') + '\n';
      fullDetails += '\n';
    }
  }
  fullDetails += '</pre></div>';

  const idShort = evt.id ? String(evt.id).substring(0, 8) : '';
  const identityChip = renderIdentityChip(evt.human_identity, evt.agent_identity);
  div.innerHTML =
    '<div class="hook-header" onclick="toggleHookBody(this)">' +
      '<span class="hook-badge" style="background:' + srcColor + ';color:#000">' + srcLabel + '</span>' +
      '<span class="hook-time">' + t + '</span>' +
      (sid ? '<span class="hook-session">' + sid + '</span>' : '') +
      identityChip +
      '<span class="hook-id">#' + idShort + '</span>' +
      '<span class="hook-size">' + (evt.used_count || 0) + ' selected</span>' +
      (evt.judge_prompt ? '<button class="hook-details-btn" style="margin-left:auto" onclick="event.stopPropagation();toggleSurfacePrompt(this.parentElement.parentElement)">Show Prompt</button>' : '') +
    '</div>' +
    '<div class="hook-prompt">' + escapeHtml(evt.query || '') + '</div>' +
    '<div class="hook-body">' + shortContent + '</div>' +
    '<div class="surface-prompt-body" style="display:none"><pre>' + (evt.judge_prompt ? escapeHtml(evt.judge_prompt) : '') + '</pre></div>';
  return div;
}

function isEntryVisible(src) {
  const el = document.getElementById('scale-filter');
  const filterVal = el ? el.value : '';
  if (!filterVal) return src !== 'internal';
  if (filterVal === 's1') return src === 'hook' || src === 'mcp';
  if (filterVal === 's2') return false; // S2 entries handled separately
  return true;
}

function getSessionFilter() {
  return document.getElementById('session-filter').value || '';
}

async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    const sessions = await r.json();
    const sel = document.getElementById('session-filter');
    // Keep current selection
    const current = sel.value;
    sel.innerHTML = '<option value="">All sessions</option>';
    for (const s of sessions) {
      const label = s.short + ' (' + s.events + ' events)';
      sel.innerHTML += '<option value="' + s.id + '">' + label + '</option>';
    }
    if (current) sel.value = current;
  } catch(e) { console.error('loadSessions error:', e); }
}

function onSessionFilterChange() {
  // Reset feed and re-poll with new session filter
  lastRecallTs = '';
  document.getElementById('feed-decoding').innerHTML = '';
  pollRecallLog();
}

async function pollRecallLog() {
  try {
    let url = '/api/recalls?limit=20';
    if (lastRecallTs) url += '&since_ts=' + encodeURIComponent(lastRecallTs);
    const sf = getSessionFilter();
    if (sf) url += '&session_id=' + encodeURIComponent(sf);
    const r = await fetch(url);
    const d = await r.json();
    const feed = document.getElementById('feed-decoding');
    if (d.events && d.events.length) {
      if (feed.querySelector('.hook-placeholder')) feed.querySelector('.hook-placeholder').remove();
      const sorted = d.events.slice().reverse();
      for (const evt of sorted) {
        if (lastRecallTs && (evt.timestamp || '') <= lastRecallTs) continue;
        const el = renderRecallEntry(evt);
        // Always add to DOM, use display to filter
        if (!isEntryVisible(evt.source || 'unknown')) el.style.display = 'none';
        feed.prepend(el);
      }
      if (d.latest_ts) lastRecallTs = d.latest_ts;
      while (feed.children.length > MAX_ENTRIES) feed.removeChild(feed.lastChild);
    }
    // Async judge update: check entries missing judge data
    // Only update entries NOT currently scrolled into view or expanded
    const pending = document.querySelectorAll('#feed-decoding .recall-entry[data-needs-judge="1"]');
    if (pending.length) {
      const stamps = Array.from(pending).map(el => el.dataset.ts).filter(Boolean);
      if (stamps.length) {
        const minTs = stamps.sort()[0];
        // Step the cursor back one second to include the earliest pending row.
        const minTsBack = new Date(new Date(minTs).getTime() - 1000).toISOString();
        const jr = await fetch('/api/recalls?since_ts=' + encodeURIComponent(minTsBack) + '&limit=' + (stamps.length + 5));
        const jd = await jr.json();
        for (const evt of (jd.events || [])) {
          if (evt.judge_output) {
            const el = document.querySelector('#feed-decoding .recall-entry[data-recall-id="' + evt.id + '"][data-needs-judge="1"]');
            if (el) {
              // One-time re-render: chips → judge output. Won't fire again (needsJudge becomes 0).
              const scrollTop = feed.scrollTop;
              const newEl = renderRecallEntry(evt);
              el.replaceWith(newEl);
              feed.scrollTop = scrollTop;
            }
          }
        }
      }
    }
  } catch(e) { console.error('pollRecallLog error:', e); }
}

// Initial load
(async function() {
  const feed = document.getElementById('feed-decoding');
  feed.innerHTML = '<div class="hook-placeholder" style="color:#666;padding:20px;text-align:center">Waiting for brain activity...</div>';
  await pollRecallLog();
})();
setInterval(pollRecallLog, 2000);

// Feed toggle + encoding badge
let activeFeed = 'surface';
var encBadgeCount = 0;
function updateEncBadge(count) {
  if (activeFeed === 'encoding') return;
  encBadgeCount += count;
  var badge = document.getElementById('enc-badge');
  if (encBadgeCount > 0) { badge.textContent = encBadgeCount; badge.style.display = 'inline'; }
}
function switchFeed(name) {
  activeFeed = name;
  document.querySelectorAll('#tab-live .feed-btn').forEach(b => {
    const label = b.textContent.toLowerCase();
    b.classList.toggle('active', label.includes(name));
  });
  document.getElementById('feed-decoding').style.display = name === 'decoding' ? 'block' : 'none';
  document.getElementById('feed-encoding').style.display = name === 'encoding' ? 'block' : 'none';
  document.getElementById('scale-filter').style.display = '';
  if (name === 'decoding') loadDecodingFeed();
  if (name === 'encoding') {
    if (!encodingLoaded) loadEncodingActivity();
    encBadgeCount = 0;
    var badge = document.getElementById('enc-badge');
    badge.style.display = 'none'; badge.textContent = '';
  }
}

function loadDecodingFeed() {
  // S1 recalls auto-load via pollRecallLog interval
  // Also load S2 decode traces and append as entries
  pollRecallLog();
  loadS2DecodeEntries();
}

function filterByScale() {
  const val = document.getElementById('scale-filter').value;
  document.querySelectorAll('#feed-decoding .recall-entry, #feed-decoding .s2-entry').forEach(el => {
    const scale = el.dataset.scale || 's1';
    if (!val) { el.style.display = ''; return; }
    el.style.display = scale === val ? '' : 'none';
  });
  // Also filter encoding feed
  document.querySelectorAll('#feed-encoding .enc-entry').forEach(el => {
    const scale = el.dataset.scale || 's1';
    if (!val) { el.style.display = ''; return; }
    el.style.display = scale === val ? '' : 'none';
  });
}

// Track which S2 chain_ids we've already rendered. Polling adds new ones
// without re-rendering existing.
const s2RenderedChains = new Set();
async function loadS2DecodeEntries() {
  const container = document.getElementById('feed-decoding');
  try {
    // Only show S2 entries from last 24h in the live Decoding feed.
    // Historical S2 data lives in the Traces tab.
    const r = await fetch('/api/traces?scale=s2&hours=24');
    const events = await r.json();
    if (!Array.isArray(events) || !events.length) return;
    const chains = {};
    events.forEach(e => {
      if (!chains[e.chain_id]) chains[e.chain_id] = {events: [], chain_id: e.chain_id};
      chains[e.chain_id].events.push(e);
    });
    Object.values(chains).forEach(c => c.events.sort(
      (a, b) => (a.created_at || '').localeCompare(b.created_at || '')));
    const chainList = Object.values(chains).sort((a, b) => {
      const aMax = a.events[a.events.length - 1]?.created_at || '';
      const bMax = b.events[b.events.length - 1]?.created_at || '';
      return bMax.localeCompare(aMax);
    });

    chainList.forEach(chain => {
      if (s2RenderedChains.has(chain.chain_id)) return;  // Already in DOM
      s2RenderedChains.add(chain.chain_id);
      const el = _renderS2ChainEntry(chain);
      const chainTs = chain.events[chain.events.length - 1]?.created_at
                      || chain.events[0]?.created_at || '';
      const entries = container.querySelectorAll('.recall-entry, .s2-entry');
      let inserted = false;
      for (const entry of entries) {
        const entryTs = entry.dataset.ts || '';
        if (entryTs && entryTs < chainTs) {
          container.insertBefore(el, entry);
          inserted = true;
          break;
        }
      }
      if (!inserted) container.appendChild(el);
    });
  } catch(e) {
    console.error('S2 decode load failed:', e);
  }
}
// Refresh S2 feed every 15s while decoding tab is active. Light poll —
// the function is idempotent and only appends new chains.
setInterval(() => { if (activeFeed === 'decoding') loadS2DecodeEntries(); }, 15000);

function _renderS2ChainEntry(chain) {
  const oEvent = chain.events.find(e => e.event_type === 'O');
  const kEvent = chain.events.find(e => e.event_type === 'K');
  const deltaEvents = chain.events.filter(e => e.event_type === 'delta');

  const newestEvt = chain.events[chain.events.length - 1] || chain.events[0];
  const time = newestEvt?.created_at ? localTime(newestEvt.created_at) : '?';
  const chainShort = chain.chain_id.substring(0, 20);
  const chainTs = newestEvt?.created_at || '';

  // Unit-type table. Adding a new S2 unit = adding a row here.
  const UNIT_STYLES = {
    consolidation:    {label:'S2 CONSOLIDATION',  bg:'#1a4a2a', fg:'#33ff88'},
    community:        {label:'S2 COMMUNITY',      bg:'#1a3a4a', fg:'#45B7D1'},
    edge_family:      {label:'S2 EDGE FAMILIES',  bg:'#1a3a4a', fg:'#45B7D1'},
    healer:           {label:'S2 HEALER',         bg:'#4a1a4a', fg:'#ff66aa'},
    _default:         {label:'S2',                bg:'#1a3a4a', fg:'#45B7D1'},
  };
  let unitStyle = UNIT_STYLES._default;
  for (const [key, style] of Object.entries(UNIT_STYLES)) {
    if (key !== '_default' && chain.chain_id && chain.chain_id.includes(key)) {
      unitStyle = style;
      break;
    }
  }
  const badgeLabel = unitStyle.label;
  const badgeBg = unitStyle.bg;
  const badgeColor = unitStyle.fg;
  const borderColor = unitStyle.fg;
  const isConsolidation = badgeLabel === 'S2 CONSOLIDATION';
  const isCommunity     = badgeLabel === 'S2 COMMUNITY';
  const isHealer        = badgeLabel === 'S2 HEALER';

  let h = '';
  h += '<div class="hook-header" onclick="toggleHookBody(this)">';
  h += '<span class="hook-badge" style="background:' + badgeBg + ';color:' + badgeColor + '">' + badgeLabel + '</span>';
  h += '<span class="hook-time">' + time + '</span>';
  h += '<span class="hook-id">' + chainShort + '</span>';

  try {
    if (isConsolidation) {
      const consolidated = deltaEvents.find(d => d.ref_type === 'consolidated');
      if (consolidated) {
        h += '<span class="hook-size" style="color:' + badgeColor + '">' + escapeHtml((consolidated.summary || '').substring(0, 60)) + '</span>';
      } else if (kEvent) {
        h += '<span class="hook-size">' + escapeHtml((kEvent.summary || '').substring(0, 60)) + '</span>';
      }
    } else if (isHealer) {
      const generated = deltaEvents.find(d => d.ref_type === 'healer_generated');
      if (generated) {
        h += '<span class="hook-size" style="color:' + badgeColor + '">' + escapeHtml((generated.summary || '').substring(0, 80)) + '</span>';
      } else if (kEvent) {
        h += '<span class="hook-size">' + escapeHtml((kEvent.summary || '').substring(0, 80)) + '</span>';
      } else if (oEvent) {
        h += '<span class="hook-size" style="color:#888">' + escapeHtml((oEvent.summary || '').substring(0, 80)) + '</span>';
      }
    } else if (isCommunity) {
      const created = deltaEvents.filter(d => d.ref_type === 'community_created');
      const enriched = deltaEvents.find(d => d.ref_type === 'community_enriched');
      if (created.length) {
        h += '<span class="hook-size" style="color:' + badgeColor + '">' + created.length + ' communities created</span>';
      } else if (enriched) {
        h += '<span class="hook-size">' + escapeHtml((enriched.summary || '').substring(0, 60)) + '</span>';
      }
    } else {
      const firstSummary = (deltaEvents[0] && deltaEvents[0].summary) || (kEvent && kEvent.summary) || (oEvent && oEvent.summary) || '';
      if (firstSummary) h += '<span class="hook-size">' + escapeHtml(firstSummary.substring(0, 60)) + '</span>';
    }
  } catch (summaryErr) {
    console.warn('S2 summary render failed for', chain.chain_id, summaryErr);
  }
  h += '</div>';

  h += '<div class="hook-body" style="padding:4px 12px">';
  if (oEvent) {
    h += '<div style="padding:2px 0;color:#888;font-size:11px">';
    h += '<strong style="color:' + badgeColor + '">O (observed):</strong> ' + escapeHtml(oEvent.summary || '') + '</div>';
  }
  if (kEvent) {
    h += '<div style="padding:2px 0;color:#888;font-size:11px">';
    h += '<strong style="color:#ffaa33">K (proposals):</strong> ' + escapeHtml(kEvent.summary || '') + '</div>';

    if (isConsolidation && kEvent.metadata) {
      try {
        const meta = typeof kEvent.metadata === 'string' ? JSON.parse(kEvent.metadata) : kEvent.metadata;
        const clusters = meta.clusters || [];
        const shown = clusters.slice(0, 15);
        shown.forEach((c, i) => {
          const preClass = c.pre_class || 'needs_judgment';
          const preColor = preClass === 'likely_consolidate' ? '#33ff88' :
                           preClass === 'likely_evolve' ? '#ffcc00' :
                           preClass === 'likely_keep' ? '#45B7D1' : '#888';
          const titles = c.node_titles ? Object.values(c.node_titles) : [];
          const sim = 'c=' + (c.content_cosine||0).toFixed(2) + ' t=' + (c.title_cosine||0).toFixed(2);
          let signals = [];
          if (c.co_recall_count > 0) signals.push('co_recall=' + c.co_recall_count);
          if (c.has_correction_edge) signals.push('CORRECTION');
          if (c.has_tension_edge) signals.push('TENSION');
          if (Object.values(c.catalog_blind || {}).some(v => v)) signals.push('BLIND');
          if (c.same_community) signals.push('same_comm');

          h += '<div style="margin:3px 0;padding:3px 8px;border-left:2px solid ' + preColor + '">';
          h += '<span style="color:' + preColor + ';font-size:10px;font-weight:bold">' + preClass.toUpperCase().replace('LIKELY_','') + '</span> ';
          h += '<span style="color:#666;font-size:10px">' + sim + '</span>';
          if (signals.length) h += ' <span style="color:#aa8800;font-size:10px">' + signals.join(' ') + '</span>';
          titles.forEach(t => {
            h += '<div style="color:#ccc;font-size:11px;padding-left:4px">• ' + escapeHtml(t) + '</div>';
          });
          h += '</div>';
        });
        if (clusters.length > 15) {
          h += '<div style="color:#555;font-size:10px;padding:2px 8px">+' + (clusters.length - 15) + ' more clusters</div>';
        }
      } catch(e) {}
    }
  }
  deltaEvents.forEach(d => {
    const color = d.ref_type === 'community_created' ? '#33ff88' :
                   d.ref_type === 'community_enriched' ? '#aa66ff' :
                   d.ref_type === 'consolidated' ? '#33ff88' :
                   d.ref_type === 'recall_quality_signal' ? '#ff6666' : '#888';
    h += '<div style="padding:2px 0;color:#888;font-size:11px">';
    h += '<strong style="color:' + color + '">Δ ' + escapeHtml(d.ref_type || '') + ':</strong> ';
    h += escapeHtml(d.summary || '') + '</div>';
  });
  h += '</div>';

  const div = document.createElement('div');
  div.className = 'hook-entry s2-entry';
  div.dataset.scale = 's2';
  div.dataset.ts = chainTs;
  div.style.borderLeftColor = borderColor;
  div.innerHTML = h;
  return div;
}

function filterEncoding() {
  const val = document.getElementById('encoding-filter').value;
  document.querySelectorAll('#feed-encoding .enc-entry').forEach(el => {
    if (!val) { el.style.display = ''; return; }
    el.style.display = el.dataset.kind === val ? '' : 'none';
  });
}

// Encoding activity feed
let encodingLoaded = false;
let lastEncodingTs = '';

async function loadEncodingActivity() {
  try {
    const container = document.getElementById('feed-encoding');
    const runsR = await fetch('/api/encoding-runs?limit=50&hours=12');
    const runsD = await runsR.json();

    if (!runsD.runs || !runsD.runs.length) {
      if (!encodingLoaded) {
        container.innerHTML = '<div style="color:#666;padding:20px;text-align:center">No recent encoding runs</div>';
      }
      encodingLoaded = true;
      return;
    }

    const totalNodes = runsD.runs.reduce((s, r) => s + (r.nodes ? r.nodes.length : 0), 0);
    const totalEdges = runsD.runs.reduce((s, r) => s + (r.edges ? r.edges.length : 0), 0);
    const latestTs = runsD.runs[0] ? runsD.runs[0].start_ts : '';
    const fingerprint = runsD.runs.length + ':' + totalNodes + ':' + totalEdges + ':' + latestTs;
    const oldFingerprint = container.dataset.fingerprint || '';
    if (encodingLoaded && fingerprint === oldFingerprint) return;
    const oldRunCount = parseInt((oldFingerprint || '0').split(':')[0]) || 0;
    if (encodingLoaded && runsD.runs.length > oldRunCount) {
      const badge = document.getElementById('enc-badge');
      badge.style.display = '';
      badge.textContent = '+' + (runsD.runs.length - oldRunCount);
      setTimeout(() => { if (activeFeed !== 'encoding') badge.style.display = ''; }, 5000);
    }
    container.dataset.fingerprint = fingerprint;
    if (!encodingLoaded) container.innerHTML = '';
    encodingLoaded = true;

    let s2Runs = [];
    try {
      const consolR = await fetch('/api/consolidation-runs?hours=12');
      const consolD = await consolR.json();
      if (consolD.runs) {
        for (const run of consolD.runs) {
          s2Runs.push({type: 'consolidation', ...run, start_ts: run.timestamp});
        }
      }
    } catch(e) { console.error('S2 consolidation load:', e); }
    try {
      const commR = await fetch('/api/community-runs?hours=12');
      const commD = await commR.json();
      if (commD.runs) {
        for (const run of commD.runs) {
          s2Runs.push({type: 'community', ...run, start_ts: run.timestamp});
        }
      }
    } catch(e) { console.error('S2 community load:', e); }
    try {
      const healR = await fetch('/api/healer-runs?hours=12');
      const healD = await healR.json();
      if (healD.runs) {
        for (const run of healD.runs) {
          s2Runs.push({type: 'healer', ...run, start_ts: run.timestamp});
        }
      }
    } catch(e) { console.error('S2 healer load:', e); }

    container.innerHTML = '';

    const allRuns = [];
    for (const run of runsD.runs) {
      allRuns.push({type: 's1e', data: run, ts: run.start_ts || ''});
    }
    for (const run of s2Runs) {
      allRuns.push({type: 's2', data: run, ts: run.start_ts || ''});
    }
    allRuns.sort((a,b) => (b.ts || '').localeCompare(a.ts || ''));

    for (const item of allRuns) {
      if (item.type === 's2') {
        const run = item.data;
        const div = document.createElement('div');
        div.className = 'hook-entry enc-entry';
        div.dataset.scale = 's2';
        const isConsol = run.type === 'consolidation';
        const isHealer = run.type === 'healer';
        const color = isConsol ? '#33ff88' : isHealer ? '#ff66aa' : '#45B7D1';
        const label = isConsol ? 'S2 CONSOLIDATE' : isHealer ? 'S2 HEALER' : 'S2 COMMUNITY';
        div.style.borderLeftColor = color;
        const t = localTime(run.start_ts, 'time');
        const actionCount = (run.synthesized||[]).length + (run.archived||[]).length + (run.kept||[]).length + (run.evolved||[]).length;

        const headerSummary = isConsol ? (actionCount + ' actions')
                              : isHealer ? escapeHtml((run.summary||'').substring(0, 80))
                              : escapeHtml((run.summary||'').substring(0, 60));
        let html = '<div class="hook-header" onclick="toggleHookBody(this)">' +
          '<span class="hook-badge" style="background:' + color + ';color:#000">' + label + '</span>' +
          '<span class="hook-time">' + t + '</span>' +
          '<span class="hook-size">' + headerSummary + '</span>' +
          (isConsol ? '<button class="hook-details-btn" style="margin-left:auto" onclick="event.stopPropagation();toggleConsolPrompt(this.parentElement.parentElement)">Show Prompt</button>' : '') +
          '</div>';

        if (run.o_summary || run.k_summary) {
          html += '<div class="hook-prompt">' + escapeHtml(run.k_summary || run.o_summary || '') + '</div>';
        }

        html += '<div class="hook-body" style="padding:4px 12px">';

        if (isConsol) {
          for (const n of (run.synthesized || [])) {
            html += '<div class="enc-entry created" data-kind="created" style="margin:2px 0;padding:4px 8px;cursor:pointer" onclick="loadNodeDetail(&quot;' + (n.id||'') + '&quot;)">' +
              '<span class="enc-kind created">SYNTHESIZED</span> ' +
              '<span class="type-badge type-' + (n.type||'') + '">' + (n.type||'') + '</span> ' +
              '<span class="enc-title">' + escapeHtml(n.title || '') + '</span>' +
              (n.content ? '<div style="color:#888;font-size:10px;margin-top:2px;padding-left:4px">' + escapeHtml((n.content||'').substring(0, 400)) + '</div>' : '') +
              '</div>';
          }
          for (const n of (run.archived || [])) {
            html += '<div class="enc-entry" style="margin:2px 0;padding:4px 8px;opacity:0.6;cursor:pointer" onclick="loadNodeDetail(&quot;' + (n.id||'') + '&quot;)">' +
              '<span class="enc-kind" style="background:#663333;color:#ff8888">ARCHIVED</span> ' +
              '<span class="type-badge type-' + (n.type||'') + '">' + (n.type||'') + '</span> ' +
              '<span class="enc-title">' + escapeHtml(n.title || '') + '</span>' +
              (n.content ? '<div style="color:#666;font-size:10px;margin-top:2px;padding-left:4px">' + escapeHtml((n.content||'').substring(0, 250)) + '</div>' : '') +
              '</div>';
          }
          for (const e of (run.evolved || [])) {
            html += '<div class="enc-entry" style="margin:2px 0;padding:4px 8px">' +
              '<span class="enc-kind" style="background:#444400;color:#ffcc00">EVOLVED</span> ' +
              escapeHtml(e.survivor || '') + ' <span style="color:#ffcc00">supersedes</span> ' +
              '<span style="opacity:0.6">' + escapeHtml(e.archived || '') + '</span></div>';
          }
          for (const e of (run.kept || [])) {
            html += '<div class="enc-entry" style="margin:2px 0;padding:4px 8px">' +
              '<span class="enc-kind" style="background:#003344;color:#45B7D1">KEPT</span> ' +
              escapeHtml(e.source || '') + ' <span style="color:#45B7D1">↔</span> ' +
              escapeHtml(e.target || '') +
              (e.description ? '<div style="color:#666;font-size:10px;margin-top:2px;padding-left:4px">' + escapeHtml(e.description.substring(0, 250)) + '</div>' : '') +
              '</div>';
          }
          if (run.journal) {
            html += '<div style="margin-top:6px;padding:4px 8px;color:#666;font-size:10px;border-top:1px solid #222">' +
              '<strong>Journal:</strong><pre style="white-space:pre-wrap;margin:4px 0;color:#888">' + escapeHtml(run.journal.substring(0, 500)) + '</pre></div>';
          }
        }
        if (isHealer) {
          // Healer: O = scan results, K = proposals, Δ = generated.
          // Concise — these are operational maintenance, not new content.
          if (run.o_summary) html += '<div style="padding:2px 0;color:#888;font-size:11px"><strong style="color:' + color + '">O ' + escapeHtml(run.o_ref_type || '') + ':</strong> ' + escapeHtml(run.o_summary) + '</div>';
          if (run.k_summary) html += '<div style="padding:2px 0;color:#888;font-size:11px"><strong style="color:#ffaa33">K ' + escapeHtml(run.k_ref_type || '') + ':</strong> ' + escapeHtml(run.k_summary) + '</div>';
          if (run.summary) html += '<div style="padding:2px 0;color:#888;font-size:11px"><strong style="color:#33ff88">Δ ' + escapeHtml(run.ref_type || 'healer_generated') + ':</strong> ' + escapeHtml(run.summary) + '</div>';
        } else if (isConsol) {
          html += '<div class="consol-prompt-body" style="display:none"><pre style="white-space:pre-wrap;color:#aaa;font-size:10px;max-height:600px;overflow-y:auto">Loading...</pre></div>';
        } else {
          if (run.o_summary) html += '<div style="padding:2px 0;color:#888;font-size:11px"><strong style="color:#45B7D1">O:</strong> ' + escapeHtml(run.o_summary) + '</div>';
          if (run.k_summary) html += '<div style="padding:2px 0;color:#888;font-size:11px"><strong style="color:#ffaa33">K:</strong> ' + escapeHtml(run.k_summary) + '</div>';
          if (run.summary) html += '<div style="padding:2px 0;color:#888;font-size:11px"><strong style="color:#33ff88">Δ:</strong> ' + escapeHtml(run.summary) + '</div>';

          for (const c of (run.communities || [])) {
            const matColor = c.maturity === 'settled' ? '#33ff88' :
                             c.maturity === 'active' ? '#ffcc00' :
                             c.maturity === 'forming' ? '#45B7D1' : '#888';
            html += '<div class="enc-entry created" style="margin:3px 0;padding:4px 8px;cursor:pointer" onclick="loadNodeDetail(&quot;' + (c.id||'') + '&quot;)">' +
              '<span class="enc-kind created">COMMUNITY</span> ' +
              '<span style="color:' + matColor + ';font-size:10px;font-weight:bold;margin-right:4px">' + (c.maturity||'?').toUpperCase() + '</span>' +
              '<span class="enc-title">' + escapeHtml(c.title || '') + '</span>' +
              '<span style="color:#666;font-size:10px;margin-left:6px">' + (c.members||0) + ' members</span>' +
              (c.narrative ? '<div style="color:#888;font-size:10px;margin-top:2px;padding-left:4px">' + escapeHtml(c.narrative) + '</div>' : '') +
              (c.content ? '<div style="color:#666;font-size:10px;margin-top:2px;padding-left:4px">' + escapeHtml((c.content||'').substring(0, 300)) + '</div>' : '') +
              (c.open_questions ? '<div style="color:#aa8800;font-size:10px;margin-top:2px;padding-left:4px">Open: ' + escapeHtml(c.open_questions) + '</div>' : '') +
              '</div>';
          }
        }

        if (!actionCount && !run.summary) {
          html += '<div style="color:#555;font-size:11px;padding:4px 8px">(no write actions)</div>';
        }
        html += '</div>';

        div.innerHTML = html;
        container.appendChild(div);
        continue;
      }
      const run = item.data;
      const div = document.createElement('div');
      div.className = 'hook-entry enc-entry';
      div.dataset.scale = 's1';
      div.style.borderLeftColor = '#aa66ff';
      const t = localTime(run.start_ts, 'time');
      const nodeCount = run.nodes ? run.nodes.length : 0;
      const edgeCount = run.edges ? run.edges.length : 0;

      const sid = run.session_id ? run.session_id.substring(0, 8) : '';
      let html = '<div class="hook-header" onclick="toggleHookBody(this)">' +
        '<span class="hook-badge" style="background:#aa66ff;color:#000">S1 ENCODE</span>' +
        '<span class="hook-time">' + t + '</span>' +
        (sid ? '<span class="hook-session">' + sid + '</span>' : '') +
        '<span class="hook-id">#' + (run.counter || '') + '</span>' +
        '<span class="hook-size">' + nodeCount + ' actions</span>' +
        '<button class="hook-details-btn" style="margin-left:auto" onclick="event.stopPropagation();toggleEncPrompt(this.parentElement.parentElement)">Show Prompt</button>' +
      '</div>';

      if (run.prompt_info) {
        html += '<div class="hook-prompt">' + escapeHtml(run.prompt_info) + '</div>';
      }

      html += '<div class="hook-body" style="padding:4px 12px">';
      for (const n of (run.nodes || [])) {
        const kind = n.kind === 'revised' ? 'REVISED' : 'CREATED';
        const kindClass = n.kind === 'revised' ? 'revised' : 'created';
        html += '<div class="enc-entry ' + kindClass + '" data-kind="' + kindClass + '" style="margin:2px 0;padding:4px 8px;cursor:pointer" onclick="loadNodeDetail(&quot;' + (n.id||'') + '&quot;)">' +
          '<span class="enc-kind ' + kindClass + '">' + kind + '</span> ' +
          '<span class="type-badge type-' + (n.type||'') + '">' + (n.type||'') + '</span> ' +
          '<span class="enc-title">' + escapeHtml(n.title || '') + '</span>' +
          (n.content ? '<div style="color:#888;font-size:10px;margin-top:2px;padding-left:4px">' + escapeHtml((n.content||'').substring(0, 150)) + '</div>' : '') +
          '</div>';
      }
      for (const e of (run.edges || []).slice(0, 8)) {
        html += '<div class="enc-entry connected" data-kind="connected" style="margin:2px 0;padding:4px 8px">' +
          '<span class="enc-kind connected">CONNECTED</span> ' +
          escapeHtml(e.source_title || '') + ' <span style="color:#aa66ff">—' + (e.relation||'') + '→</span> ' +
          escapeHtml(e.target_title || '') + '</div>';
      }
      if ((run.edges || []).length > 8) {
        html += '<div style="color:#555;font-size:10px;padding:2px 8px">+' + ((run.edges || []).length - 8) + ' more edges</div>';
      }
      if (!(run.nodes || []).length && !(run.edges || []).length) {
        html += '<div style="color:#555;font-size:11px;padding:4px 8px">(no write actions)</div>';
      }
      html += '</div>';

      html += '<div class="enc-prompt-body" style="display:none"><pre>';
      if (run.encoder_prompt) {
        html += escapeHtml(run.encoder_prompt);
      } else {
        html += '(no prompt file found — encoding ran before prompt logging was added)';
      }
      html += '</pre></div>';

      div.innerHTML = html;
      container.appendChild(div);
    }
  } catch(e) { console.error('loadEncodingActivity error:', e); }
}

function toggleSurfacePrompt(entry) {
  var prompt = entry.querySelector('.surface-prompt-body');
  if (!prompt) return;
  var btn = entry.querySelector('.hook-details-btn');
  if (prompt.style.display === 'none') {
    prompt.style.display = 'block';
    if (btn) btn.textContent = 'Hide Prompt';
  } else {
    prompt.style.display = 'none';
    if (btn) btn.textContent = 'Show Prompt';
  }
}

function toggleEncPrompt(entry) {
  var prompt = entry.querySelector('.enc-prompt-body');
  if (!prompt) return;
  var btn = entry.querySelector('.hook-details-btn');
  if (prompt.style.display === 'none') {
    prompt.style.display = 'block';
    if (btn) btn.textContent = 'Hide Prompt';
  } else {
    prompt.style.display = 'none';
    if (btn) btn.textContent = 'Show Prompt';
  }
}

async function toggleConsolPrompt(entry) {
  var prompt = entry.querySelector('.consol-prompt-body');
  if (!prompt) return;
  var btn = entry.querySelector('.hook-details-btn');
  if (prompt.style.display === 'none') {
    prompt.style.display = 'block';
    if (btn) btn.textContent = 'Hide Prompt';
    if (prompt.querySelector('pre').textContent === 'Loading...') {
      try {
        const r = await fetch('/api/consolidation-prompt?batch=1');
        const d = await r.json();
        prompt.querySelector('pre').textContent = d.user_content || d.error || '(no prompt available)';
      } catch(e) {
        prompt.querySelector('pre').textContent = '(failed to load prompt)';
      }
    }
  } else {
    prompt.style.display = 'none';
    if (btn) btn.textContent = 'Show Prompt';
  }
}

setInterval(() => { if (activeFeed === 'encoding') loadEncodingActivity(); }, 3000);
setInterval(() => { if (activeFeed !== 'encoding' && encodingLoaded) loadEncodingActivity(); }, 10000);

// Explorer
let expandedNode = null;
async function searchNodes() {
  const search = document.getElementById('search-input').value;
  const type = document.getElementById('type-filter').value;
  let url = '/api/nodes?limit=100';
  if (search) url += '&search=' + encodeURIComponent(search);
  if (type) url += '&type=' + encodeURIComponent(type);
  try {
    const r = await fetch(url);
    const d = await r.json();
    const list = document.getElementById('node-list');
    list.innerHTML = d.nodes.map(n => `
      <div class="node-card" onclick="loadNodeDetail('${n.id}')" style="cursor:pointer">
        <div class="node-title">
          <span class="type-badge type-${n.type}">${n.type}</span>
          ${n.locked ? '<span class="locked-icon">&#x1f512;</span>' : ''}
          ${n.title || '(untitled)'}
        </div>
        <div class="node-meta">
          <span>conf: ${(n.confidence||0).toFixed(2)}</span>
          <span>accessed: ${n.access_count}x</span>
          <span>${n.encoding_source || ''}</span>
          <span>${localTime(n.created_at)}</span>
        </div>
      </div>
    `).join('');
  } catch(e) {}
}
function toggleNode(id, el) {
  expandedNode = expandedNode === id ? null : id;
  el.classList.toggle('expanded');
}

// Logs tab — Errors + Daemon
let activeLogFeed = 'errors';

function switchLogFeed(name) {
  activeLogFeed = name;
  document.querySelectorAll('#tab-logs .feed-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  ['errors','daemon'].forEach(f => {
    document.getElementById('feed-' + f).style.display = f === name ? '' : 'none';
  });
  if (name === 'errors') { document.getElementById('err-badge').style.display = 'none'; }
  loadLogs();
}

async function loadLogs() {
  if (activeLogFeed === 'errors') loadErrors();
  else if (activeLogFeed === 'daemon') loadDaemonLogs();
}

async function loadErrors() {
  const hours = document.getElementById('error-hours').value;
  try {
    const r = await fetch('/api/errors?hours=' + hours + '&limit=100');
    const d = await r.json();
    const feed = document.getElementById('feed-errors');
    document.getElementById('logs-count').textContent = d.count + ' errors';

    if (!d.errors || !d.errors.length) {
      feed.innerHTML = '<div style="color:#4a4;padding:20px;text-align:center">No errors in the last ' + hours + 'h</div>';
      return;
    }
    feed.innerHTML = '';
    for (const e of d.errors) {
      const div = document.createElement('div');
      div.dataset.source = e.source || '';
      const levelColor = {critical:'#ff4444',error:'#ff6644',warning:'#ffaa33',info:'#4a9eff'}[e.level] || '#888';
      div.style.cssText = 'padding:8px 12px;margin:4px 0;background:#111118;border-radius:6px;border-left:3px solid ' + levelColor + ';font-size:12px';
      const t = localTime(e.timestamp);
      const sessionTag = e.session_id ? '<span style="color:#555;font-size:9px;margin-left:4px">' + e.session_id.substring(0,8) + '</span>' : '';
      div.innerHTML =
        '<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:bold;text-transform:uppercase;background:' + levelColor + '22;color:' + levelColor + '">' + (e.level || 'error') + '</span> ' +
        '<span style="color:#888;font-size:10px">' + (e.source || '') + '</span> ' +
        '<span style="color:#aaa;font-weight:bold">' + escapeHtml(e.component || '') + '</span>' + sessionTag +
        '<div style="color:#ccc;margin-top:3px">' + escapeHtml(e.error || '') + '</div>' +
        (e.context ? '<div style="color:#666;font-size:10px;margin-top:2px">' + escapeHtml(e.context) + '</div>' : '') +
        '<div style="color:#555;font-size:10px;margin-top:2px">' + t + '</div>';
      feed.appendChild(div);
    }
  } catch(e) {
    document.getElementById('feed-errors').innerHTML = '<div style="color:#f66;padding:20px">Failed to load: ' + e + '</div>';
  }
}

async function loadDaemonLogs() {
  const hours = document.getElementById('error-hours').value;
  try {
    const r = await fetch('/api/errors?hours=' + hours + '&limit=200&source=daemon');
    const d = await r.json();
    const feed = document.getElementById('feed-daemon');
    const r2 = await fetch('/api/errors?hours=' + hours + '&limit=50&source=hook');
    const d2 = await r2.json();

    const all = [...(d.errors || []), ...(d2.errors || [])];
    all.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''));
    document.getElementById('logs-count').textContent = all.length + ' daemon events';

    if (!all.length) {
      feed.innerHTML = '<div style="color:#4a4;padding:20px;text-align:center">No daemon events in the last ' + hours + 'h</div>';
      return;
    }
    feed.innerHTML = '';
    for (const e of all) {
      const div = document.createElement('div');
      const levelColor = {critical:'#ff4444',error:'#ff6644',warning:'#ffaa33',info:'#4a9eff'}[e.level] || '#888';
      const isRestart = (e.error || '').includes('restart') || (e.component || '').includes('restart');
      const borderColor = isRestart ? '#4a9eff' : levelColor;
      div.style.cssText = 'padding:8px 12px;margin:4px 0;background:#111118;border-radius:6px;border-left:3px solid ' + borderColor + ';font-size:12px';
      const t = localTime(e.timestamp);
      div.innerHTML =
        '<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:bold;text-transform:uppercase;background:' + borderColor + '22;color:' + borderColor + '">' + (isRestart ? 'restart' : e.level || 'error') + '</span> ' +
        '<span style="color:#aaa;font-weight:bold">' + escapeHtml(e.component || '') + '</span>' +
        '<div style="color:#ccc;margin-top:3px">' + escapeHtml(e.error || '') + '</div>' +
        '<div style="color:#555;font-size:10px;margin-top:2px">' + t + '</div>';
      feed.appendChild(div);
    }
  } catch(e) {
    document.getElementById('feed-daemon').innerHTML = '<div style="color:#f66;padding:20px">Failed to load: ' + e + '</div>';
  }
}

// System Status
async function loadSystemStatus() {
  try {
    const r = await fetch('/api/system-status');
    const d = await r.json();
    const grid = document.getElementById('status-grid');
    grid.innerHTML = '';

    const components = [
      {key: 'daemon', label: 'Brain Daemon', icon: '🧠'},
      {key: 'brain_db', label: 'Brain DB', icon: '💾'},
      {key: 'logs_db', label: 'Logs DB', icon: '📋'},
      {key: 'judge', label: 'Haiku Judge', icon: '⚖️'},
      {key: 'embedder', label: 'Embedder', icon: '🔮'},
    ];

    for (const comp of components) {
      const s = d.status[comp.key] || {alive: false, error: 'unknown'};
      const alive = s.alive;
      const card = document.createElement('div');
      card.style.cssText = 'background:#111118;border-radius:8px;padding:12px 16px;border:1px solid ' + (alive ? '#1a3a1a' : '#3a1a1a');

      let details = '';
      if (comp.key === 'daemon' && alive) {
        details = 'PID: ' + (s.pid || '?') + ' · Uptime: ' + Math.round((s.uptime || 0) / 60) + 'min';
      } else if (comp.key === 'brain_db' && alive) {
        details = s.nodes + ' nodes · ' + (s.size_mb || '?') + 'MB';
      } else if (comp.key === 'logs_db' && alive) {
        details = (s.size_mb || '?') + 'MB';
      } else if (comp.key === 'dashboard_db' && alive) {
        details = (s.size_mb || '?') + 'MB · Last: ' + localTime(s.last_entry);
      } else if (comp.key === 'embedder' && alive) {
        details = s.model || '?';
      } else if (!alive) {
        details = s.error || 'unreachable';
      }

      const pathLine = s.path ? '<div style="font-size:9px;color:#444;margin-top:4px;word-break:break-all">' + escapeHtml(s.path) + '</div>' : '';
      card.innerHTML =
        '<div style="display:flex;align-items:center;gap:8px">' +
          '<span style="font-size:20px">' + comp.icon + '</span>' +
          '<div>' +
            '<div style="color:#ccc;font-weight:bold;font-size:13px">' + comp.label + '</div>' +
            '<div style="font-size:11px;margin-top:2px;color:' + (alive ? '#4a4' : '#f44') + '">' +
              (alive ? '● Live' : '● Down') +
            '</div>' +
          '</div>' +
          '<div style="margin-left:auto;font-size:10px;color:#666;text-align:right;max-width:200px;overflow:hidden;text-overflow:ellipsis">' + escapeHtml(details) + '</div>' +
        '</div>' + pathLine;
      grid.appendChild(card);
    }
  } catch(e) {
    document.getElementById('status-grid').innerHTML = '<div style="color:#f66;padding:20px">Failed to load status: ' + e + '</div>';
  }
}

setInterval(() => {
  const statusTab = document.getElementById('tab-status');
  if (statusTab && statusTab.classList.contains('active')) loadSystemStatus();
}, 5000);

let lastSeenErrorCount = -1;

setInterval(async () => {
  const logsTab = document.getElementById('tab-logs');
  const isViewing = logsTab && logsTab.classList.contains('active');

  try {
    const r = await fetch('/api/errors?hours=1&limit=1');
    const d = await r.json();
    const errBadge = document.getElementById('err-badge');
    const logsBadge = document.getElementById('logs-badge');
    if (lastSeenErrorCount < 0) lastSeenErrorCount = d.count;
    if (isViewing && activeLogFeed === 'errors') {
      lastSeenErrorCount = d.count;
      errBadge.style.display = 'none';
      loadErrors();
    } else if (d.count > lastSeenErrorCount) {
      const diff = d.count - lastSeenErrorCount;
      errBadge.textContent = diff; errBadge.style.display = '';
      logsBadge.textContent = diff; logsBadge.style.display = '';
    } else {
      errBadge.style.display = 'none';
      logsBadge.style.display = 'none';
    }
  } catch(e) {}
}, 10000);

// Aspect taxonomy — 14 aspects classifying node_types + edge_relations.
// Source: aspects_v1.json (live at $BRAIN_DB_DIR/aspects_v1.json, seed in
// servers/scales/s2/aspects_v1.json). Counts come from brain.db.
async function loadAspects() {
  try {
    const r = await fetch('/api/aspects');
    const d = await r.json();
    const grid = document.getElementById('aspects-grid');
    if (!grid) return;
    const aspects = d.aspects || [];
    if (!aspects.length) {
      grid.innerHTML = '<div style="color:#666;padding:12px">aspects_v1.json not found or empty</div>';
      return;
    }
    let html = '';
    for (const a of aspects) {
      const lockBadge = a.locked ? ' <span style="color:#ffaa33;font-size:9px">🔒</span>' : '';
      const dim = a.dimension ? '<span style="color:#555;font-size:10px;margin-left:6px">' + escapeHtml(a.dimension) + '</span>' : '';
      // Member chips — type vs relation. Cap shown to top 12; rest behind a +N.
      const topTypes = a.node_types.slice(0, 12);
      const moreTypes = a.node_types.length > 12 ? a.node_types.length - 12 : 0;
      const topRels = a.edge_relations.slice(0, 12);
      const moreRels = a.edge_relations.length > 12 ? a.edge_relations.length - 12 : 0;
      const chip = (label, count, color) =>
        '<span style="display:inline-block;background:#1a1a2a;border:1px solid #2a2a3a;color:' + color + ';padding:1px 6px;border-radius:3px;font-size:10px;margin:2px 3px 0 0">' +
        escapeHtml(label) + '<span style="color:#555;margin-left:4px">' + count + '</span></span>';
      html += '<div style="background:#111118;border-radius:8px;padding:12px;margin:6px 0;border-left:3px solid #45B7D1">';
      html += '<div style="display:flex;justify-content:space-between;align-items:baseline">';
      html += '<div><span style="color:#7eb8ff;font-weight:bold;font-size:13px">' + escapeHtml(a.name) + '</span>' + lockBadge + dim + '</div>';
      html += '<div style="color:#666;font-size:10px">' + a.totals.nodes + ' nodes · ' + a.totals.edges + ' edges</div>';
      html += '</div>';
      if (a.meaning) html += '<div style="color:#888;font-size:11px;margin-top:6px;line-height:1.4">' + escapeHtml(a.meaning) + '</div>';
      if (topTypes.length) {
        html += '<div style="margin-top:6px"><div style="color:#555;font-size:9px;text-transform:uppercase;letter-spacing:0.5px">Node types</div><div>';
        for (const t of topTypes) html += chip(t.name, t.count, '#ccc');
        if (moreTypes) html += '<span style="color:#555;font-size:10px;margin-left:4px">+' + moreTypes + '</span>';
        html += '</div></div>';
      }
      if (topRels.length) {
        html += '<div style="margin-top:6px"><div style="color:#555;font-size:9px;text-transform:uppercase;letter-spacing:0.5px">Edge relations</div><div>';
        for (const r2 of topRels) html += chip(r2.name, r2.count, '#aa66ff');
        if (moreRels) html += '<span style="color:#555;font-size:10px;margin-left:4px">+' + moreRels + '</span>';
        html += '</div></div>';
      }
      html += '</div>';
    }
    grid.innerHTML = html;
  } catch(e) { console.error('loadAspects error:', e); }
}

// Health
async function loadHealth() {
  try {
    const statsR = await fetch('/api/stats');
    const insightsR = await fetch('/api/insights');
    const d = await statsR.json();
    const ins = await insightsR.json();
    const hc = document.getElementById('health-content');
    const orphanClass = d.orphans > 20 ? 'bad' : d.orphans > 5 ? 'warn' : 'ok';
    const sevColors = {high: '#ff6666', medium: '#ffaa33', low: '#7eb8ff'};
    const insightsHtml = (ins.insights || []).map(i => `
      <div style="background:#111118;border-radius:8px;padding:14px;margin:8px 0;border-left:4px solid ${sevColors[i.severity] || '#555'}">
        <div style="font-size:15px;font-weight:bold;color:${sevColors[i.severity]}">${i.icon} ${i.title}</div>
        <div style="color:#999;margin-top:6px;font-size:12px;line-height:1.5">${i.detail}</div>
        ${i.nodes ? '<div style="margin-top:8px;font-size:11px;color:#666">' + i.nodes.map(n =>
          '<div style="padding:2px 0">&#8226; ' + (n.title||'').substring(0,80) + ' <span style="color:#555">(' + (n.type||n.count||'') + ')</span></div>'
        ).join('') + '</div>' : ''}
      </div>
    `).join('');
    hc.innerHTML = `
      <div class="health-grid">
        <div class="health-card ok"><div class="hc-value">${d.nodes}</div><div class="hc-label">Total Nodes</div></div>
        <div class="health-card ok"><div class="hc-value">${d.edges}</div><div class="hc-label">Total Edges</div></div>
        <div class="health-card ok"><div class="hc-value">${d.locked}</div><div class="hc-label">Locked</div></div>
        <div class="health-card ${d.recent_24h > 0 ? 'ok' : 'warn'}"><div class="hc-value">${d.recent_24h}</div><div class="hc-label">Last 24h</div></div>
        <div class="health-card ${orphanClass}"><div class="hc-value">${d.orphans}</div><div class="hc-label">Orphans</div></div>
      </div>
      ${insightsHtml ? '<h3 style="color:#ccc;margin:20px 0 8px">Anchor Insights</h3>' + insightsHtml : '<div style="color:#33ff88;padding:20px;text-align:center;font-size:16px">No issues detected</div>'}
      <h3 style="color:#888;margin:20px 0 8px">Node Types</h3>
      <div class="health-grid">
        ${Object.entries(d.types).map(([t,c]) => `
          <div class="health-card ok" style="padding:10px">
            <span class="type-badge type-${t}">${t}</span>
            <span style="float:right;font-size:18px;font-weight:bold;color:#7eb8ff">${c}</span>
          </div>
        `).join('')}
      </div>
    `;
  } catch(e) { console.error(e); }
}

// Traces
let _traceChainEntries = [];
let _traceRendered = 0;
const _TRACE_BATCH = 30;

function onTraceScaleChange() {
  const scale = document.getElementById('trace-scale-filter').value;
  const hoursEl = document.getElementById('trace-hours-filter');
  if (scale && scale >= 's2' && parseInt(hoursEl.value) < 168) {
    hoursEl.value = '168';
  }
  loadTraces();
}

async function loadTraces() {
  try {
    const scaleFilter = document.getElementById('trace-scale-filter').value;
    const hours = document.getElementById('trace-hours-filter').value;
    const sessionFilter = document.getElementById('trace-session-filter').value;
    let url = '/api/traces?hours=' + hours;
    if (scaleFilter) url += '&scale=' + scaleFilter;
    if (sessionFilter) url += '&session=' + sessionFilter;
    const r = await fetch(url);
    const traces = await r.json();
    const el = document.getElementById('traces-content');
    const label = hours <= 1 ? '1h' : hours <= 6 ? '6h' : hours <= 24 ? '24h' : '7d';
    document.getElementById('trace-count').textContent = traces.length + ' events (' + label + ')';

    const sessSelect = document.getElementById('trace-session-filter');
    const prevVal = sessSelect.value;
    try {
      const sr = await fetch('/api/sessions');
      const sessions = await sr.json();
      const opts = '<option value="">All sessions</option>' + sessions.map(s =>
        '<option value="' + s.id + '"' + (s.id === prevVal ? ' selected' : '') + '>' + s.short + ' (' + s.events + ' events)</option>'
      ).join('');
      sessSelect.innerHTML = opts;
    } catch(e) { /* keep existing options */ }

    if (!traces.length) {
      el.innerHTML = '<div style="color:#888;text-align:center;padding:40px">No trace events yet. Traces will appear after your next prompt.</div>';
      _traceChainEntries = [];
      return;
    }

    const chains = {};
    traces.forEach(t => {
      if (!chains[t.chain_id]) chains[t.chain_id] = [];
      chains[t.chain_id].push(t);
    });
    const chainEntries = Object.entries(chains);
    chainEntries.forEach(([_, events]) => events.sort(
      (a, b) => (a.created_at || '').localeCompare(b.created_at || '')));
    chainEntries.sort((a, b) => {
      const aMax = a[1][a[1].length - 1]?.created_at || '';
      const bMax = b[1][b[1].length - 1]?.created_at || '';
      return bMax.localeCompare(aMax);
    });
    _traceChainEntries = chainEntries;
    _traceRendered = 0;
    el.innerHTML = '';
    _renderTracesBatch(el);
  } catch(e) { console.error('loadTraces', e); }
}

function _traceChainLabel(chainId) {
  // Map chain IDs to readable labels. Chain IDs remain string-prefixed
  // (s0-/s1r-/s1e-/s2-) even after trace_events.id became hex — those are
  // different identifiers.
  if (chainId.startsWith('s0-')) { const p = chainId.split('-'); return 'S0 Exchange #' + (p[2] || '?'); }
  if (chainId.startsWith('s1r-')) { const p = chainId.split('-'); return 'S1 Recall (Surface) #' + (p[2] || '?'); }
  if (chainId.startsWith('s1e-')) { const p = chainId.split('-'); return 'S1 Encode #' + (p[2] || '?'); }
  if (chainId.startsWith('s2-')) {
    const op = chainId.split('-').slice(2).join('-');
    const labels = {community_detection:'S2 Community Detection', consolidation:'S2 Consolidation', edge_family_integration:'S2 Edge Families', healer:'S2 Healer', relation_reclassify:'S2 Edge Reclassify'};
    return labels[op] || 'S2 ' + op.replace(/_/g, ' ');
  }
  if (chainId.startsWith('s3-')) return 'S3 ' + chainId.split('-').slice(2).join(' ');
  return chainId;
}

function _renderTracesBatch(el) {
  const scaleColors = {s0:'#888', s1:'#7eb8ff', s2:'#ffaa33', s3:'#33ff88', s4:'#ff66aa'};
  const typeLabels = {O:'Observed', K:'Selected', delta:'Changed', outcome:'Outcome'};
  const typeColors = {O:'#45B7D1', K:'#ffaa33', delta:'#33ff88', outcome:'#aa66ff'};
  const end = Math.min(_traceRendered + _TRACE_BATCH, _traceChainEntries.length);

  let html = '';
  for (let i = _traceRendered; i < end; i++) {
    const [chainId, events] = _traceChainEntries[i];
    const firstTime = events[0].created_at;
    const chainScale = events[0].scale;
    const color = scaleColors[chainScale] || '#666';
    const label = _traceChainLabel(chainId);
    const sessionId = events[0].session_id || '';
    const sessionTag = sessionId ? '<span style="color:#444;font-size:9px;margin-left:6px">' + sessionId.substring(0,8) + '</span>' : '';
    // Identity per chain: first event with non-empty identity wins. Chains
    // often span multiple writes from the same speaker pair so one rendering
    // per chain is enough; a per-event chip would just repeat ad nauseam.
    let chainHi = '', chainAi = '';
    for (const ev of events) {
      if (ev.human_identity || ev.agent_identity) {
        chainHi = ev.human_identity; chainAi = ev.agent_identity; break;
      }
    }
    const identityTag = (chainHi || chainAi)
      ? '<span style="margin-left:6px">' + renderIdentityChip(chainHi, chainAi) + '</span>'
      : '';

    html += '<div style="background:#0a0a12;border-radius:8px;margin:6px 0;border-left:3px solid ' + color + '">';
    html += '<div style="padding:8px 12px;display:flex;justify-content:space-between;align-items:center">';
    html += '<div><span style="color:' + color + ';font-size:12px;font-weight:bold">' + label + '</span>' + sessionTag + identityTag + '</div>';
    html += '<span style="color:#555;font-size:10px">' + localTime(firstTime) + '</span>';
    html += '</div>';

    events.forEach(ev => {
      const tColor = typeColors[ev.event_type] || '#666';
      const tLabel = typeLabels[ev.event_type] || ev.event_type;
      html += '<div style="padding:4px 12px 4px 20px;border-top:1px solid #111;display:flex;gap:8px;align-items:flex-start">';
      html += '<span style="flex-shrink:0;font-size:10px;font-weight:bold;color:' + tColor + ';min-width:55px">' + tLabel + '</span>';
      html += '<div style="flex:1;min-width:0">';
      if (ev.ref_type) html += '<span style="color:#666;font-size:10px;background:#1a1a2a;padding:1px 4px;border-radius:2px;margin-right:4px">' + ev.ref_type + '</span>';
      html += '<div style="color:#ccc;font-size:12px;margin-top:2px;white-space:pre-wrap;word-break:break-word">' + escapeHtml((ev.summary || '').substring(0, 300)) + '</div>';
      html += '</div>';
      html += '<span style="color:#444;font-size:9px;flex-shrink:0;white-space:nowrap">' + localTime(ev.created_at, 'time') + '</span>';
      html += '</div>';
    });

    html += '</div>';
  }
  el.insertAdjacentHTML('beforeend', html);
  _traceRendered = end;

  if (_traceRendered < _traceChainEntries.length) {
    el.insertAdjacentHTML('beforeend', '<div id="trace-load-more" style="text-align:center;padding:12px"><button onclick="_loadMoreTraces()" style="background:#1a1a2a;color:#7eb8ff;border:1px solid #3a3a5a;padding:4px 16px;border-radius:4px;cursor:pointer">Load more (' + (_traceChainEntries.length - _traceRendered) + ' remaining)</button></div>');
  }
}

function _loadMoreTraces() {
  const btn = document.getElementById('trace-load-more');
  if (btn) btn.remove();
  _renderTracesBatch(document.getElementById('traces-content'));
}

let _traceAutoRefresh = null;
function _startTraceAutoRefresh() {
  _stopTraceAutoRefresh();
  _traceAutoRefresh = setInterval(() => {
    const tab = document.getElementById('tab-traces');
    if (tab && tab.classList.contains('active')) loadTraces();
  }, 5000);
}
function _stopTraceAutoRefresh() {
  if (_traceAutoRefresh) { clearInterval(_traceAutoRefresh); _traceAutoRefresh = null; }
}

// 3D Graph
let graph3d = null, graph3dData = null, legendVisible = false;

const TYPE_COLORS = {
  lesson: '#4a9eff', correction: '#ff6666', interaction: '#33ff88',
  rule: '#ffaa33', decision: '#aa66ff', mental_model: '#33dddd',
  mechanism: '#dddd33', vocabulary: '#666', context: '#555',
  bug_lesson: '#ff8866', pattern: '#ff66aa', boot: '#888',
  tension: '#ff4444', uncertainty: '#aaaaff', constraint: '#ff8833',
  impact: '#ff6644', convention: '#66aaff',
};

async function loadNodeDetail(nodeId) {
  var panel = document.getElementById('node-detail');
  panel.style.display = 'block';
  panel.innerHTML = '<div style="color:#666;padding:20px">Loading...</div>';
  try {
    // Fan out three calls in parallel:
    //   /api/node/{id}             — base node + connections (direct SQL,
    //                                works without daemon)
    //   /api/node/{id}/corrections — aspect-edge walk (via daemon)
    //   /api/node/{id}/source-refs — episodic refs from node_source_refs (v27)
    var [r, cr, srr] = await Promise.all([
      fetch('/api/node/' + nodeId),
      fetch('/api/node/' + nodeId + '/corrections').catch(() => null),
      fetch('/api/node/' + nodeId + '/source-refs').catch(() => null),
    ]);
    var d = await r.json();
    var corrections = [];
    if (cr && cr.ok) {
      try { corrections = (await cr.json()).corrections || []; } catch(e) { corrections = []; }
    }
    var sourceRefs = [];
    if (srr && srr.ok) {
      try { sourceRefs = (await srr.json()).refs || []; } catch(e) { sourceRefs = []; }
    }
    var n = d.node;
    var conns = d.connections || [];
    var meta = n.metadata || {};
    var h = '';
    h += '<div class="nd-close" onclick="document.getElementById(&quot;node-detail&quot;).style.display=&quot;none&quot;">&times;</div>';
    h += '<div class="nd-title"><span class="type-badge type-' + (n.type||'') + '">' + (n.type||'') + '</span> ';
    if (n.locked) h += '&#x1f512; ';
    if (n.critical) h += '⚠️ ';
    h += escapeHtml(n.title || '') + '</div>';
    h += '<div class="nd-meta">';
    h += '<span>id: ' + (n.id||'').substring(0,8) + '</span>';
    h += '<span>accessed: ' + n.access_count + 'x</span>';
    h += '<span>conf: ' + (n.confidence||0).toFixed(2) + '</span>';
    h += '<span>source: ' + (n.encoding_source||'?') + '</span>';
    h += '<span>' + localTime(n.created_at) + '</span>';
    h += '</div>';
    h += '<div class="nd-section">Content</div>';
    h += '<div class="nd-content">' + escapeHtml(n.content || '(empty)') + '</div>';
    var fields = [];
    if (n.situation) fields.push('<div class="nd-field"><span class="nd-fk">situation:</span> ' + escapeHtml(n.situation) + '</div>');
    if (meta.reasoning) fields.push('<div class="nd-field"><span class="nd-fk">reasoning:</span> ' + escapeHtml(meta.reasoning) + '</div>');
    if (meta.user_raw_quote) fields.push('<div class="nd-field"><span class="nd-fk">user_raw_quote:</span> <em>"' + escapeHtml(meta.user_raw_quote) + '"</em></div>');
    if (meta.correction_of) fields.push('<div class="nd-field"><span class="nd-fk">correction_of:</span> <a style="color:#7eb8ff;cursor:pointer" onclick="loadNodeDetail(&quot;' + meta.correction_of + '&quot;)">' + meta.correction_of + '</a></div>');
    if (meta.correction_pattern) fields.push('<div class="nd-field"><span class="nd-fk">correction_pattern:</span> ' + escapeHtml(meta.correction_pattern) + '</div>');
    if (meta.source_context) fields.push('<div class="nd-field"><span class="nd-fk">source_context:</span> ' + escapeHtml(meta.source_context) + '</div>');
    if (n.personal) fields.push('<div class="nd-field"><span class="nd-fk">personal:</span> ' + escapeHtml(n.personal) + '</div>');
    if (n.evolution_status) fields.push('<div class="nd-field"><span class="nd-fk">evolution_status:</span> ' + escapeHtml(n.evolution_status) + '</div>');
    if (n.revised_at) fields.push('<div class="nd-field"><span class="nd-fk">revised:</span> ' + localTime(n.revised_at) + '</div>');
    if (fields.length) h += '<div class="nd-section">Fields</div>' + fields.join('');

    // Episodic refs — which traces this node was encoded from. v27 substrate
    // (node_source_refs). If empty, either pre-v27 node OR no refs were
    // attached at encode time. Render with scale tag + summary preview.
    if (sourceRefs.length) {
      h += '<div class="nd-section">Encoded from ' + sourceRefs.length + ' trace(s)</div>';
      const scaleColors = {s0:'#888', s1:'#7eb8ff', s2:'#ffaa33', s3:'#33ff88'};
      for (const ref of sourceRefs) {
        if (ref.missing) {
          h += '<div class="nd-field" style="opacity:0.5">trace ' + (ref.trace_id||'') + ' (not found — log-rotated or archived)</div>';
          continue;
        }
        const sc = scaleColors[ref.scale] || '#666';
        const sess = ref.session_id ? ref.session_id.substring(0,8) : '';
        h += '<div class="nd-conn" style="border-left-color:' + sc + '">';
        h += '<div style="font-size:9px;color:' + sc + ';text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px">' + ref.scale + ' · ' + (ref.event_type || '') + (ref.ref_type ? ' · ' + ref.ref_type : '') + (sess ? ' · ' + sess : '') + '</div>';
        h += '<div style="color:#ccc;font-size:11px">' + escapeHtml((ref.summary || '').substring(0,180)) + '</div>';
        h += '<div style="color:#555;font-size:9px;margin-top:2px">trace ' + (ref.trace_id||'').substring(0,8) + ' · pos ' + (ref.position || 1) + ' · ' + localTime(ref.trace_created_at) + '</div>';
        h += '</div>';
      }
    }

    // Corrections — aspect-edge walk via daemon. Shows what corrects this
    // node and what this node corrects. Direction matters: 'corrects' means
    // the neighbor IS this node's corrector; 'corrected_by' is the inverse.
    if (corrections.length) {
      h += '<div class="nd-section">Corrections (' + corrections.length + ')</div>';
      for (const c of corrections) {
        const dirColor = c.direction === 'corrects' ? '#ff8866' : '#88ccff';
        const dirLabel = c.direction === 'corrects' ? '⤴ corrects this' : '↪ corrected by this';
        h += '<div class="nd-conn" onclick="loadNodeDetail(&quot;' + (c.id||'') + '&quot;)" style="border-left-color:' + dirColor + '">';
        h += '<div style="font-size:9px;color:' + dirColor + ';text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px">' + dirLabel + ' · ' + escapeHtml(c.relation || '') + '</div>';
        h += '<div class="nd-conn-title"><span class="type-badge type-' + (c.type||'') + '">' + (c.type||'') + '</span> ' + escapeHtml((c.title||'').substring(0,80)) + '</div>';
        if (c.edge_description) h += '<div style="color:#888;font-size:10px;margin-top:3px;font-style:italic">why: ' + escapeHtml(c.edge_description.substring(0,180)) + '</div>';
        if (c.content) h += '<div style="color:#aaa;font-size:11px;margin-top:3px">' + escapeHtml(c.content.substring(0,200)) + '</div>';
        if (c.user_raw_quote) h += '<div style="color:#7eb8ff;font-size:10px;margin-top:3px">"' + escapeHtml(c.user_raw_quote.substring(0,150)) + '"</div>';
        h += '</div>';
      }
    }

    h += '<div class="nd-section">Connections (' + conns.length + ')</div>';
    for (var i = 0; i < conns.length; i++) {
      var c = conns[i];
      h += '<div class="nd-conn" onclick="loadNodeDetail(&quot;' + c.id + '&quot;)">';
      h += '<div class="nd-conn-title"><span class="type-badge type-' + (c.type||'') + '">' + (c.type||'') + '</span> ' + escapeHtml((c.title||'').substring(0,60)) + '</div>';
      h += '<div class="nd-conn-meta">' + (c.relation||'') + ' · weight ' + (c.weight||0).toFixed(2) + '</div>';
      h += '</div>';
    }
    if (!conns.length) h += '<div style="color:#555;padding:8px">No connections</div>';
    panel.innerHTML = h;
  } catch(e) {
    panel.innerHTML = '<div style="color:#ff6666;padding:20px">Failed to load: ' + e.message + '</div>';
  }
}

async function loadGraph3D() {
  try {
    const r = await fetch('/api/graph3d');
    graph3dData = await r.json();
    if (!graph3dData.nodes || !graph3dData.nodes.length) return;

    // Filter: only show nodes IN communities + community hub nodes
    // Orphans get hidden — they clutter without adding structure
    const communityNodeIds = new Set();
    const hubIds = new Set();
    graph3dData.nodes.forEach(n => {
      if (n.hub) hubIds.add(n.id);
      if (n.community) communityNodeIds.add(n.id);
    });
    hubIds.forEach(id => communityNodeIds.add(id));

    const visibleNodes = graph3dData.nodes.filter(n => communityNodeIds.has(n.id) || hubIds.has(n.id));
    const visibleIds = new Set(visibleNodes.map(n => n.id));

    const visibleLinks = graph3dData.edges
      .filter(e => visibleIds.has(e.source) && visibleIds.has(e.target))
      .filter(e => e.relation !== 'co_accessed' && e.relation !== 'emergent_bridge')
      .map(e => ({source: e.source, target: e.target, relation: e.relation}));

    const container = document.getElementById('graph-3d');
    container.style.height = 'calc(100vh - 42px)';
    const w = container.offsetWidth || 800;
    const h = container.offsetHeight || 600;

    if (graph3d) {
      graph3d.graphData({nodes: visibleNodes, links: visibleLinks});
    } else {
      graph3d = ForceGraph3D()(container)
        .width(w).height(h)
        .graphData({nodes: visibleNodes, links: visibleLinks})
        .backgroundColor('#08080f')
        .nodeVal(n => n.hub ? n.val : 2)
        .nodeColor(n => n.color)
        .nodeOpacity(0.85)
        .nodeLabel(n => {
          if (n.hub) return '<div style="text-align:center;font-size:14px"><b>' + n.name + '</b><br><span style="color:#aaa">' + (n.val/0.8|0) + ' members</span></div>';
          const comm = n.community_title ? '<br><span style="color:#666">' + n.community_title.substring(0, 40) + '</span>' : '';
          return '<div style="text-align:center"><b>' + n.name + '</b><br><span style="color:#888">' + n.type + '</span>' + comm + '</div>';
        })
        .linkColor(l => l.relation === 'community_member' ? '#333' : '#222')
        .linkOpacity(l => l.relation === 'community_member' ? 0.15 : 0.08)
        .linkWidth(l => l.relation === 'community_member' ? 0.3 : 0.15)
        .d3AlphaDecay(0.08)
        .d3VelocityDecay(0.5)
        .warmupTicks(150)
        .cooldownTicks(300)
        .onEngineTick(() => {
          if (!graph3d._forcesConfigured) {
            var charge = graph3d.d3Force('charge');
            if (charge) { charge.strength(-15).distanceMax(200); }
            var link = graph3d.d3Force('link');
            if (link) { link.distance(l => l.relation === 'community_member' ? 3 : 40).strength(l => l.relation === 'community_member' ? 0.9 : 0.05); }
            graph3d._forcesConfigured = true;
          }
        })
        .onNodeClick(node => {
          graph3d.cameraPosition({x: node.x + 150, y: node.y + 80, z: node.z + 150}, node, 1000);
          loadNodeDetail(node.id);
        });
      var controls = graph3d.controls();
      if (controls) controls.zoomSpeed = 5.0;
    }

    const legendEl = document.getElementById('legend-items');
    if (graph3dData.communities && graph3dData.communities.length) {
      legendEl.innerHTML = graph3dData.communities.map(c =>
        '<div style="display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:4px;cursor:pointer;transition:background 0.15s" ' +
        'onclick="focusCommunity(&quot;' + (c.hub_id || '') + '&quot;)" ' +
        'onmouseover="this.style.background=`rgba(255,255,255,0.08)`" onmouseout="this.style.background=`none`">' +
        '<div style="width:10px;height:10px;border-radius:50%;flex-shrink:0;background:' + c.color + ';box-shadow:0 0 4px ' + c.color + '"></div>' +
        '<span style="color:#aaa">' + c.name + ' (' + c.count + ')</span></div>'
      ).join('');
    } else {
      legendEl.innerHTML = '<div style="color:#555;padding:8px">No communities yet</div>';
    }
  } catch(e) {
    console.error('Graph3D load failed:', e);
  }
}

function toggleLegend() {
  const el = document.getElementById('graph-legend');
  legendVisible = !legendVisible;
  el.style.transform = legendVisible ? 'translateX(0)' : 'translateX(220px)';
}

function focusCommunity(hubId) {
  if (!graph3d || !hubId) return;
  const node = graph3d.graphData().nodes.find(n => n.id === hubId);
  if (node) {
    graph3d.cameraPosition({x: node.x + 120, y: node.y + 60, z: node.z + 120}, node, 1200);
  }
}

async function loadGraph() { loadGraph3D(); }
