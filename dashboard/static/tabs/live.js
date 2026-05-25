// ===========================================================================
// tabs/live.js — activity stream: decoding (S1 recalls + S2 decode) and
// encoding (S1 scribe + S2 unit runs) sub-feeds.
// ---------------------------------------------------------------------------
// Owns: feed-decoding, feed-encoding, the feed-toggle bar, scale + session
// filters, and the polls that keep them fresh.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import bus from '/static/lib/bus.js';
import { escapeHtml, localTime, identityChipHTML } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import * as graph from './graph.js';

// ── Live feed — polls /api/recalls. ────────────────────────────────────
// Cursor is ISO timestamp (since_ts), not integer rowid: trace_events.id is
// now an 8-char hex string, so integer ordering no longer applies.
// created_at is monotonic per writer and stable under the schema change.
let lastRecallTs = '';
const MAX_ENTRIES = 100;

// Feed-toggle state. Initial value 'decoding' matches index.html's
// active-class on the Decoding button. Previously 'surface' (legacy name)
// — every interval misfired on first load.
let activeFeed = 'decoding';
let encBadgeCount = 0;

export function toggleHookBody(el) {
  const body = el.parentElement.querySelector('.hook-body');
  body.classList.toggle('open');
}

// Source tag for recall entries → drives the .badge--solid-* color class
// on the header pill and the readable label. `unknown` falls through to
// the neutral .badge variant.
const SOURCE_META = {
  hook:     { cls: 'badge badge--solid-blue',  label: 'HOOK' },
  mcp:      { cls: 'badge badge--solid-green', label: 'ANCHOR' },
  internal: { cls: 'badge',                    label: 'INTERNAL' },
  unknown:  { cls: 'badge',                    label: '?' },
};

// S2 unit metadata — single source for both the decoding feed (which
// reads it from a chain_id substring) and the encoding feed (which reads
// it from run.type). `match` is a substring tested against chain_id;
// `type` is the literal value used when iterating run records.
const S2_UNITS = [
  { type: 'consolidation', match: 'consolidation',         label: 'S2 CONSOLIDATION', bg: '#1a4a2a', fg: '#33ff88' },
  { type: 'community',     match: 'community',             label: 'S2 COMMUNITY',     bg: '#1a3a4a', fg: '#45B7D1' },
  { type: 'edge_family',   match: 'edge_family',           label: 'S2 EDGE FAMILIES', bg: '#1a3a4a', fg: '#45B7D1' },
  { type: 'healer',        match: 'healer',                label: 'S2 HEALER',        bg: '#4a1a4a', fg: '#ff66aa' },
];
const S2_UNIT_DEFAULT = { type: '_default', label: 'S2', bg: '#1a3a4a', fg: '#45B7D1' };

function s2UnitFromChainId(chainId) {
  if (!chainId) return S2_UNIT_DEFAULT;
  for (const u of S2_UNITS) {
    if (chainId.includes(u.match)) return u;
  }
  return S2_UNIT_DEFAULT;
}
function s2UnitFromType(type) {
  return S2_UNITS.find(u => u.type === type) || S2_UNIT_DEFAULT;
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
  const srcMeta = SOURCE_META[src] || { cls: 'badge', label: src.toUpperCase() };
  const sid = evt.session_id ? evt.session_id.substring(0, 8) : '';
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
  const identityChip = identityChipHTML(evt.human_identity, evt.agent_identity);
  div.innerHTML =
    '<div class="hook-header" onclick="toggleHookBody(this)">' +
      '<span class="' + srcMeta.cls + '">' + srcMeta.label + '</span>' +
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
  if (filterVal === 's2') return false;
  return true;
}

function getSessionFilter() {
  return document.getElementById('session-filter').value || '';
}

export function onSessionFilterChange() {
  lastRecallTs = '';
  document.getElementById('feed-decoding').innerHTML = '';
  pollRecallLog();
}

async function pollRecallLog() {
  try {
    const d = await api.recalls({
      limit: 20,
      since_ts: lastRecallTs || undefined,
      session_id: getSessionFilter() || undefined,
    });
    const feed = document.getElementById('feed-decoding');
    if (d.events && d.events.length) {
      if (feed.querySelector('.hook-placeholder')) feed.querySelector('.hook-placeholder').remove();
      const sorted = d.events.slice().reverse();
      for (const evt of sorted) {
        if (lastRecallTs && (evt.timestamp || '') <= lastRecallTs) continue;
        const el = renderRecallEntry(evt);
        if (!isEntryVisible(evt.source || 'unknown')) el.style.display = 'none';
        feed.prepend(el);
      }
      if (d.latest_ts) lastRecallTs = d.latest_ts;
      while (feed.children.length > MAX_ENTRIES) feed.removeChild(feed.lastChild);
    }
    // Async judge update: refresh entries missing judge data.
    const pending = document.querySelectorAll('#feed-decoding .recall-entry[data-needs-judge="1"]');
    if (pending.length) {
      const stamps = Array.from(pending).map(el => el.dataset.ts).filter(Boolean);
      if (stamps.length) {
        const minTs = stamps.sort()[0];
        // Step the cursor back one second to include the earliest pending row.
        const minTsBack = new Date(new Date(minTs).getTime() - 1000).toISOString();
        const jd = await api.recalls({
          since_ts: minTsBack,
          limit: stamps.length + 5,
        });
        for (const evt of (jd.events || [])) {
          if (evt.judge_output) {
            const el = document.querySelector('#feed-decoding .recall-entry[data-recall-id="' + evt.id + '"][data-needs-judge="1"]');
            if (el) {
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

export function switchFeed(name) {
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
    const badge = document.getElementById('enc-badge');
    badge.style.display = 'none'; badge.textContent = '';
  }
}

function loadDecodingFeed() {
  // S1 recalls auto-load via pollRecallLog interval
  // Also load S2 decode traces and append as entries
  pollRecallLog();
  loadS2DecodeEntries();
}

export function filterByScale() {
  const val = document.getElementById('scale-filter').value;
  document.querySelectorAll('#feed-decoding .recall-entry, #feed-decoding .s2-entry').forEach(el => {
    const scale = el.dataset.scale || 's1';
    if (!val) { el.style.display = ''; return; }
    el.style.display = scale === val ? '' : 'none';
  });
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
    const events = await api.traces({ scale: 's2', hours: 24 });
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
      if (s2RenderedChains.has(chain.chain_id)) return;
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

function _renderS2ChainEntry(chain) {
  const oEvent = chain.events.find(e => e.event_type === 'O');
  const kEvent = chain.events.find(e => e.event_type === 'K');
  const deltaEvents = chain.events.filter(e => e.event_type === 'delta');

  const newestEvt = chain.events[chain.events.length - 1] || chain.events[0];
  const time = newestEvt?.created_at ? localTime(newestEvt.created_at) : '?';
  const chainShort = chain.chain_id.substring(0, 20);
  const chainTs = newestEvt?.created_at || '';

  const unit = s2UnitFromChainId(chain.chain_id);
  const badgeLabel = unit.label;
  const badgeBg = unit.bg;
  const badgeColor = unit.fg;
  const borderColor = unit.fg;
  const isConsolidation = unit.type === 'consolidation';
  const isCommunity     = unit.type === 'community';
  const isHealer        = unit.type === 'healer';

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
      } catch(e) {
        console.error('[dashboard] consolidation cluster parse failed:', e);
      }
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

// ── Encoding activity feed ─────────────────────────────────────────────

let encodingLoaded = false;
let lastEncodingTs = '';

async function loadEncodingActivity() {
  try {
    const container = document.getElementById('feed-encoding');
    const runsD = await api.encodingRuns({ limit: 50, hours: 12 });

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
      const consolD = await api.consolidationRuns({ hours: 12 });
      if (consolD.runs) {
        for (const run of consolD.runs) {
          s2Runs.push({type: 'consolidation', ...run, start_ts: run.timestamp});
        }
      }
    } catch(e) { console.error('S2 consolidation load:', e); }
    try {
      const commD = await api.communityRuns({ hours: 12 });
      if (commD.runs) {
        for (const run of commD.runs) {
          s2Runs.push({type: 'community', ...run, start_ts: run.timestamp});
        }
      }
    } catch(e) { console.error('S2 community load:', e); }
    try {
      const healD = await api.healerRuns({ hours: 12 });
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
        const unit = s2UnitFromType(run.type);
        const isConsol = unit.type === 'consolidation';
        const isHealer = unit.type === 'healer';
        // Encoding feed uses the shorter "S2 CONSOLIDATE" label vs the
        // decoding feed's "S2 CONSOLIDATION" — the row is denser here.
        const color = unit.fg;
        const label = isConsol ? 'S2 CONSOLIDATE' : unit.label;
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

// One toggler — three named exports preserved as thin wrappers so the
// inline `onclick="toggleX(...)"` handlers in renderRecallEntry +
// loadEncodingActivity keep working without surgery. `lazyLoad` runs the
// first time we expand, populating the <pre> via API (consolidation only).
async function _togglePromptBody(entry, bodyClass, lazyLoad) {
  const prompt = entry.querySelector('.' + bodyClass);
  if (!prompt) return;
  const btn = entry.querySelector('.hook-details-btn');
  if (prompt.style.display === 'none') {
    prompt.style.display = 'block';
    if (btn) btn.textContent = 'Hide Prompt';
    if (lazyLoad) {
      const pre = prompt.querySelector('pre');
      if (pre && pre.textContent === 'Loading...') {
        try { pre.textContent = await lazyLoad(); }
        catch (e) { pre.textContent = '(failed to load prompt)'; }
      }
    }
  } else {
    prompt.style.display = 'none';
    if (btn) btn.textContent = 'Show Prompt';
  }
}

export function toggleSurfacePrompt(entry) { _togglePromptBody(entry, 'surface-prompt-body'); }
export function toggleEncPrompt(entry)     { _togglePromptBody(entry, 'enc-prompt-body'); }
export function toggleConsolPrompt(entry)  {
  return _togglePromptBody(entry, 'consol-prompt-body', async () => {
    const d = await api.consolidationPrompt(1);
    return d.user_content || d.error || '(no prompt available)';
  });
}

// ── Split divider drag ────────────────────────────────────────────────
// User drags the divider; we update grid-template-columns directly. The
// resulting graphPct is persisted to localStorage and published on
// `live:layout` so the graph module can re-fit its renderer.

const SPLIT_STORAGE_KEY = 'dashboard.liveSplitPct';
const SPLIT_DEFAULT_PCT = 60;

function _applySplit(pct) {
  const split = document.getElementById('live-split');
  if (!split) return;
  const clamped = Math.max(0, Math.min(100, pct));
  split.style.gridTemplateColumns = `${clamped}fr 6px ${100 - clamped}fr`;
  bus.publish('live:layout', { graphPct: clamped });
}

function _restoreSplit() {
  let pct = SPLIT_DEFAULT_PCT;
  try {
    const saved = parseFloat(localStorage.getItem(SPLIT_STORAGE_KEY));
    if (!Number.isNaN(saved) && saved >= 0 && saved <= 100) pct = saved;
  } catch (e) { /* localStorage may be blocked; fall back to default */ }
  _applySplit(pct);
}

function _setupDivider() {
  const divider = document.getElementById('live-divider');
  const split = document.getElementById('live-split');
  if (!divider || !split) return;
  let dragging = false;

  divider.addEventListener('mousedown', (e) => {
    dragging = true;
    divider.classList.add('dragging');
    split.classList.add('dragging');
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = split.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    _applySplit(pct);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    split.classList.remove('dragging');
    // Persist final value. Read it back from the inline style — _applySplit
    // already clamped it.
    const cols = split.style.gridTemplateColumns || '';
    const m = cols.match(/^(\d+(?:\.\d+)?)fr/);
    if (m) {
      try { localStorage.setItem(SPLIT_STORAGE_KEY, m[1]); } catch (e) {}
    }
  });
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // Placeholder while the feed waits for the first poll fire.
  const feed = document.getElementById('feed-decoding');
  feed.innerHTML = '<div class="hook-placeholder" style="color:#666;padding:20px;text-align:center">Waiting for brain activity...</div>';

  _restoreSplit();
  _setupDivider();

  // Recall feed — 2s cadence, only when the Live tab is open AND on the
  // decoding sub-feed. Inactive tabs get zero polls.
  poll.register({
    key: 'recalls',
    interval: 2000,
    activeWhen: () => document.getElementById('tab-live').classList.contains('active')
                      && activeFeed === 'decoding',
    fetcher: pollRecallLog,
  });

  // S2 decode chains — 15s cadence. Idempotent (only appends new chains
  // by chain_id).
  poll.register({
    key: 's2-decode',
    interval: 15000,
    activeWhen: () => document.getElementById('tab-live').classList.contains('active')
                      && activeFeed === 'decoding',
    fetcher: loadS2DecodeEntries,
  });

  // Encoding feed — fast cadence (3s) when the Encoding sub-feed is open;
  // slow cadence (10s) for badge updates when it's not.
  poll.register({
    key: 'encoding-active',
    interval: 3000,
    activeWhen: () => document.getElementById('tab-live').classList.contains('active')
                      && activeFeed === 'encoding',
    fetcher: loadEncodingActivity,
  });
  poll.register({
    key: 'encoding-background',
    interval: 10000,
    activeWhen: () => encodingLoaded
                      && !(document.getElementById('tab-live').classList.contains('active')
                           && activeFeed === 'encoding'),
    fetcher: loadEncodingActivity,
  });
}

export function activate() {
  // Live owns the graph now — drive its activate() so the 3D scene mounts
  // (first time) or resizes (subsequent). Polls auto-activate via activeWhen.
  try { graph.activate(); } catch (e) { console.error('[live] graph activate failed:', e); }
}

export function deactivate() {
  // Graph stays mounted in the hidden tab; its canvas keeps the WebGL
  // context warm so re-activating is instant.
}
