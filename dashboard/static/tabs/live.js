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
import { el, escapeHtml, localTime, identityChipHTML } from '/static/lib/dom.js';
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

// Recall-event cache (id → event payload). Populated by _renderRecallEvent
// so clicking a card's pin button can re-apply that historical recall's
// highlight to the graph without re-fetching. Bounded to MAX_ENTRIES so
// memory stays flat — once a card scrolls out of the feed and gets
// evicted from the DOM, its entry here is also pruned by the cleanup
// pass in _renderRecallEvent.
const _eventsById = new Map();

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
  // Falls back to candidate title chips if no judge data yet (MCP recalls,
  // old data). Candidate divs carry data-nid (not inline onclick) — the
  // post-innerHTML wiring below attaches one listener per candidate that
  // both opens loadNodeDetail AND stopPropagation to avoid pinning the
  // whole recall when the user wanted just one node.
  let shortContent = '';
  if (evt.judge_output && evt.judge_output !== '(no selection)') {
    shortContent = '<div class="recall-judge-output"><pre>' + escapeHtml(evt.judge_output) + '</pre></div>';
  } else if (evt.judge_output === '(no selection)') {
    const titleEntries = Object.entries(titles).slice(0, 8);
    const total = Object.keys(titles).length;
    shortContent = '<div class="recall-candidates"><div class="recall-candidates-header">0 selected from ' + total + ' candidates</div>' +
      titleEntries.map(([nid, title]) => {
        return '<div class="recall-candidate" data-nid="' + escapeHtml(nid) + '">' + escapeHtml(title) + '</div>';
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
          return '<div class="recall-candidate' + (isUsed ? ' used' : '') + '" data-nid="' + escapeHtml(nid) + '">' +
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
  // Pin behavior: the whole card is the click target. The previous design
  // had a dedicated 🎯 button next to "Show Prompt", but Tom wanted the
  // card itself to feel selectable — hover shifts the card right, click
  // pins. The `.recall-entry:hover` transform in components.css gives
  // the affordance; the container-level click handler below is the action.
  div.title = 'Click to highlight these nodes on the graph';
  div.innerHTML =
    '<div class="hook-header">' +
      '<span class="' + srcMeta.cls + '">' + srcMeta.label + '</span>' +
      '<span class="hook-time">' + t + '</span>' +
      (sid ? '<span class="hook-session">' + sid + '</span>' : '') +
      identityChip +
      '<span class="hook-id">#' + idShort + '</span>' +
      '<span class="hook-size">' + (evt.used_count || 0) + ' selected</span>' +
      (evt.judge_prompt ? '<button class="hook-details-btn hook-details-btn--right">Show Prompt</button>' : '') +
    '</div>' +
    '<div class="hook-prompt">' + escapeHtml(evt.query || '') + '</div>' +
    '<div class="hook-body">' + shortContent + '</div>' +
    '<div class="surface-prompt-body" style="display:none"><pre>' + (evt.judge_prompt ? escapeHtml(evt.judge_prompt) : '') + '</pre></div>';

  // ── Attach listeners (post-innerHTML) ───────────────────────────────
  // Header click toggles body open AND bubbles up to the container click
  // handler (which pins the recall). This dual-action — expand AND pin —
  // is intentional: clicking the header signals "I'm interested in this
  // card." No stopPropagation here.
  const header = div.querySelector('.hook-header');
  const body   = div.querySelector('.hook-body');
  if (header && body) {
    header.addEventListener('click', () => body.classList.toggle('open'));
  }
  // Show Prompt button — toggles surface-prompt-body underneath the card.
  // Reuses _wirePromptToggle from the P2.18 encoding-card migration so
  // both feeds share the same show/hide mechanic. Inline onclick was
  // ferrying event.stopPropagation; the helper handles that internally.
  const promptBtn  = div.querySelector('.hook-details-btn');
  const promptBody = div.querySelector('.surface-prompt-body');
  if (promptBtn && promptBody) _wirePromptToggle(promptBtn, promptBody, null);

  // Hover = preview, click = commit. mouseenter previews the recall's
  // nodes on the graph (saves prior state); mouseleave restores; click
  // upgrades preview to pin (graph discards the snapshot so the followup
  // mouseleave is a no-op). Same affordance as the visual hover-shift in
  // CSS: the card behaves like a button.
  div.addEventListener('mouseenter', () => {
    if (evt) graph.previewRecallOnGraph(evt);
  });
  div.addEventListener('mouseleave', () => {
    graph.clearRecallPreview();
  });
  // Container-level click → pin. Inner interactive elements that should NOT
  // also pin call event.stopPropagation in their own handlers (candidates
  // below, prompt button via _wirePromptToggle). Header bubbles up
  // intentionally — clicking the header expands AND pins, treating the
  // entire card as a selectable unit.
  div.addEventListener('click', () => {
    if (evt.id) pinRecallToGraph(evt.id);
  });
  // Candidates open node-detail and STOP the click from also pinning the
  // whole recall. Wired here as a single listener per candidate (was
  // previously inline onclick + a separate stopPropagation listener —
  // two handlers per element for the same DOM event).
  div.querySelectorAll('.recall-candidate[data-nid]').forEach(c => {
    c.addEventListener('click', (e) => {
      e.stopPropagation();
      const nid = c.dataset.nid;
      if (nid) loadNodeDetail(nid);
    });
  });
  return div;
}

// Look up a cached recall event by id and pin its highlight on the graph.
// Called from the inline onclick on each recall card's pin button. Falls
// back gracefully when the event scrolled out and got pruned — the user
// just doesn't get a visual response, no error.
export function pinRecallToGraph(eventId) {
  const evt = _eventsById.get(eventId);
  if (!evt) {
    console.warn('[live] pinRecallToGraph: event not in cache:', eventId);
    return;
  }
  graph.pinRecallToGraph(evt);
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
  // Reset both decoding + encoding feeds — until now the encoding feed
  // silently ignored the dropdown, so switching session left stale runs
  // on screen from other sessions. Both feeds now pass session_id through
  // to their respective endpoints.
  lastRecallTs = '';
  document.getElementById('feed-decoding').innerHTML = '';
  pollRecallLog();

  // Force the encoding feed to refetch under the new session filter.
  // Clear the fingerprint so loadEncodingActivity's short-circuit doesn't
  // skip the re-render even when the row count happens to match.
  encodingLoaded = false;
  const encContainer = document.getElementById('feed-encoding');
  if (encContainer) {
    encContainer.innerHTML = '';
    encContainer.dataset.fingerprint = '';
  }
  loadEncodingActivity();
}

// Recall event flow (P2.5 split): the fetcher (`pollRecallLog`) is a pure
// data fetcher — it advances the timestamp cursor and publishes each new
// or freshly-judged event on the `recall:event` bus topic. The DOM render
// + the graph pulse are independent subscribers reacting to the same
// event stream. Adding a new consumer (e.g. an insights panel) means
// subscribing in init(), not patching this fetcher.

async function pollRecallLog() {
  try {
    const d = await api.recalls({
      limit: 20,
      since_ts: lastRecallTs || undefined,
      session_id: getSessionFilter() || undefined,
    });
    if (d.events && d.events.length) {
      // Server returns newest-first; flip so subscribers see chronological
      // order (matters for the graph pulse — latest event over-stacks
      // earlier ones).
      const sorted = d.events.slice().reverse();
      for (const evt of sorted) {
        if (lastRecallTs && (evt.timestamp || '') <= lastRecallTs) continue;
        bus.publish('recall:event', { event: evt });
      }
      if (d.latest_ts) lastRecallTs = d.latest_ts;
    }

    // Judge-output backfill: hook recalls arrive without judge_output (Haiku
    // hasn't responded yet). Re-fetch the slice covering pending entries
    // and re-publish — the renderer's dedupe-by-id swaps the chip-list view
    // for the judge-output view in place.
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
          if (evt.judge_output) bus.publish('recall:event', { event: evt });
        }
      }
    }
  } catch(e) { console.error('pollRecallLog error:', e); }
}

// Insights renderer — pure subscriber on `insights:tick`. Replaces the
// panel's contents on every tick. The :empty CSS pseudo-class collapses
// padding when nothing fires, so the feed stays tight on a healthy brain.
//
// Dismissal: clicking the × on a card adds its title to `_dismissedInsights`.
// Subsequent ticks filter that title out. The Set is page-session-scoped
// only — no localStorage. Reload = fresh signal, so a real persistent
// problem can't be permanently silenced.
const _dismissedInsights = new Set();
let _lastInsights = [];

function _renderInsightsPanel({ insights }) {
  _lastInsights = insights || [];
  _redrawInsightsPanel();
}

function _redrawInsightsPanel() {
  const panel = document.getElementById('insights-panel');
  if (!panel) return;
  const visible = _lastInsights.filter(i => !_dismissedInsights.has(i.title || ''));
  if (!visible.length) {
    panel.innerHTML = '';
    return;
  }
  panel.innerHTML = visible.map(i => {
    const sev = (i.severity || 'low').toLowerCase();
    // data-title carries the RAW title for the dismiss delegate to read.
    // Escaped here as an HTML attribute value — the browser decodes it
    // back to the raw string when we read element.dataset.title.
    return '<div class="insights-card insights-card--' + escapeHtml(sev) + '">' +
      '<div class="insights-icon">' + (i.icon || '') + '</div>' +
      '<div class="insights-body">' +
        '<div class="insights-title">' + escapeHtml(i.title || '') + '</div>' +
        '<div class="insights-detail">' + escapeHtml(i.detail || '') + '</div>' +
      '</div>' +
      '<button class="insights-dismiss" title="Dismiss until reload" ' +
              'data-title="' + escapeHtml(i.title || '') + '">&times;</button>' +
    '</div>';
  }).join('');
}

// Wire dismiss delegation ONCE on the panel — adding listeners per-render
// would leak. Called from init().
function _wireInsightsDismiss() {
  const panel = document.getElementById('insights-panel');
  if (!panel) return;
  panel.addEventListener('click', (e) => {
    const btn = e.target.closest('.insights-dismiss');
    if (!btn) return;
    _dismissedInsights.add(btn.dataset.title || '');
    _redrawInsightsPanel();
  });
}

// Renderer — pure subscriber. Knows nothing about the network; just maps
// an event to a DOM node (insert new, replace existing when judge_output
// arrives for a pending entry).
function _renderRecallEvent({ event: evt }) {
  const feed = document.getElementById('feed-decoding');
  if (!feed) return;
  // Cache the event so the pin button on the card can look it up later.
  // Updates replace, so a judge_output backfill correctly overwrites the
  // pending entry.
  if (evt.id) _eventsById.set(evt.id, evt);
  const existing = feed.querySelector('.recall-entry[data-recall-id="' + evt.id + '"]');
  if (existing) {
    // Same id arrived again — only meaningful if judge_output is now set
    // and the old DOM was still in "needs judge" state.
    if (evt.judge_output && existing.dataset.needsJudge === '1') {
      const scrollTop = feed.scrollTop;
      const newEl = renderRecallEntry(evt);
      existing.replaceWith(newEl);
      feed.scrollTop = scrollTop;
    }
    return;
  }
  // New event.
  const placeholder = feed.querySelector('.hook-placeholder');
  if (placeholder) placeholder.remove();
  const el = renderRecallEntry(evt);
  if (!isEntryVisible(evt.source || 'unknown')) el.style.display = 'none';
  feed.prepend(el);
  // Eviction: keep DOM + cache in sync. When DOM drops oldest entries,
  // remove them from _eventsById too so memory stays bounded.
  while (feed.children.length > MAX_ENTRIES) {
    const removed = feed.lastChild;
    const removedId = removed?.dataset?.recallId;
    if (removedId) _eventsById.delete(removedId);
    feed.removeChild(removed);
  }
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

// Filter every top-level card in BOTH feeds by data-scale. The selector
// is `[data-scale]` (not `.enc-entry` etc) because the class names are
// overloaded: `.enc-entry` is reused on inner sub-rows (CREATED /
// REVISED / CONNECTED rows inside the encoding card body), and those
// don't carry data-scale. The previous compound-class selector caught
// those inner rows, defaulted their scale to 's1', then hid them when
// filtering by 's2' — making S2 cards visually empty. Contract: only
// top-level cards set `data-scale`; the selector now relies on that.
export function filterByScale() {
  const val = document.getElementById('scale-filter').value;
  document.querySelectorAll('#feed-decoding [data-scale], #feed-encoding [data-scale]').forEach(el => {
    if (!val) { el.style.display = ''; return; }
    el.style.display = el.dataset.scale === val ? '' : 'none';
  });
}

// Track which S2 chain_ids we've already rendered. Polling adds new ones
// without re-rendering existing. The Set is pruned on every fetch to drop
// chain_ids that fell out of the 24h window — without that pruning the
// Set grew for the page lifetime (slow leak: ~30 chars × N chains).
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
    // Prune the dedup Set to only chain_ids still in the fetch window.
    // Anything older than 24h is now stale — keeping it in the Set wastes
    // memory and would also incorrectly suppress re-renders if a chain
    // re-appeared (e.g., backfill). The corresponding DOM nodes are not
    // explicitly evicted here — they roll out naturally as new chains
    // get prepended; the bounded MAX_ENTRIES contract for recall cards
    // doesn't apply to S2 chains, but the 24h window is the natural cap.
    const liveChainIds = new Set(Object.keys(chains));
    for (const id of s2RenderedChains) {
      if (!liveChainIds.has(id)) s2RenderedChains.delete(id);
    }
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
    // Same reasoning as loadEncodingActivity — re-apply the scale filter
    // so freshly-rendered S2 chains respect the dropdown that was already
    // set before they arrived.
    filterByScale();
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

  h += '<div class="hook-body hook-body--padded">';
  if (oEvent) {
    h += '<div class="enc-tier-row">';
    h += '<strong style="color:' + badgeColor + '">O (observed):</strong> ' + escapeHtml(oEvent.summary || '') + '</div>';
  }
  if (kEvent) {
    h += '<div class="enc-tier-row">';
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
    h += '<div class="enc-tier-row">';
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
    // Honor the session-filter dropdown — without this the encoding feed
    // shows runs from every session while decoding correctly narrows.
    const sessionId = getSessionFilter();
    const runsD = await api.encodingRuns({
      limit: 50,
      hours: 12,
      ...(sessionId ? { session_id: sessionId } : {}),
    });

    if (!runsD.runs || !runsD.runs.length) {
      if (!encodingLoaded) {
        container.innerHTML = '<div class="feed-empty">No recent encoding runs</div>';
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

    // Merge S1 + S2 runs into a single list, newest-first.
    const allRuns = [];
    for (const run of runsD.runs) allRuns.push({ type: 's1e', data: run, ts: run.start_ts || '' });
    for (const run of s2Runs)     allRuns.push({ type: 's2',  data: run, ts: run.start_ts || '' });
    allRuns.sort((a, b) => (b.ts || '').localeCompare(a.ts || ''));

    for (const item of allRuns) {
      container.appendChild(
        item.type === 's2'
          ? _renderS2RunCard(item.data)
          : _renderS1EncodeCard(item.data));
    }
    // Re-apply the scale filter — entries we just appended haven't been
    // seen by filterByScale yet, so a previously-set "S2 Graph" filter
    // would leave fresh S1 cards visible until the user toggles the
    // dropdown again. Same contract as the new entries in
    // _renderRecallEvent which gates via isEntryVisible.
    filterByScale();
  } catch(e) { console.error('loadEncodingActivity error:', e); }
}

// ── Encoding-card helpers ─────────────────────────────────────────────
// Three card shapes (S1 encode / S2 consolidation / S2 healer/community),
// all sharing the same hook-header + hook-body envelope. Each helper
// returns the top-level card element with all listeners attached via
// el()'s onclick → addEventListener mapping — no inline `onclick=`
// strings, no `window.toggle*` globals, no string-concatenated HTML.

// Wire a "Show Prompt" / "Hide Prompt" button to its collapsible body.
// `lazyLoad` is an async function called the first time the body is
// expanded — its return value populates the <pre>. Used by the
// consolidation card to fetch the prompt on-demand from /api/.
function _wirePromptToggle(button, body, lazyLoad) {
  button.addEventListener('click', async (e) => {
    e.stopPropagation();
    const showing = body.style.display !== 'none';
    if (showing) {
      body.style.display = 'none';
      button.textContent = 'Show Prompt';
      return;
    }
    body.style.display = 'block';
    button.textContent = 'Hide Prompt';
    if (lazyLoad) {
      const pre = body.querySelector('pre');
      if (pre && pre.textContent === 'Loading...') {
        try { pre.textContent = await lazyLoad(); }
        catch (_) { pre.textContent = '(failed to load prompt)'; }
      }
    }
  });
}

// Build a clickable sub-row that opens a node detail panel on click.
// `kindClass` styles the left-pill (.enc-kind --modifier already exists
// for static colors; pass null for the kind-pill style if you want to
// rely on `kindLabel`'s own background via .enc-kind).
function _encSubRow({ kindClass, kindLabel, typeName, title, content, contentDim, nodeId }) {
  const kindEl = el('span', { class: ['enc-kind', kindClass].filter(Boolean) }, kindLabel);
  const typeEl = typeName ? el('span', { class: 'type-badge type-' + typeName }, typeName) : null;
  return el('div', {
    class: ['enc-entry', 'enc-sub-row', nodeId && 'enc-sub-row--clickable', kindClass].filter(Boolean),
    dataset: kindClass ? { kind: kindClass } : null,
    onclick: nodeId ? () => loadNodeDetail(nodeId) : null,
  },
    kindEl,
    ' ',
    typeEl,
    typeEl ? ' ' : null,
    el('span', { class: 'enc-title' }, title || ''),
    content ? el('div', {
      class: ['enc-meta-line', contentDim && 'enc-meta-line--dim'].filter(Boolean),
    }, (content || '').substring(0, contentDim ? 250 : 400)) : null,
  );
}

// O/K/Δ tier row. `accentClass` is one of enc-tier-label--{o,k,delta}
// for default cards; healer uses the unit color via the `accentStyle`
// override (data-driven, not design).
function _encTierRow(letter, refType, summary, { accentClass, accentStyle } = {}) {
  if (!summary) return null;
  const labelText = refType ? letter + ' ' + refType + ':' : letter + ':';
  return el('div', { class: 'enc-tier-row' },
    el('strong', {
      class: accentClass || null,
      style: accentStyle || null,
    }, labelText),
    ' ',
    summary,
  );
}

function _renderS2ConsolBody(run) {
  const out = [];
  for (const n of (run.synthesized || [])) {
    out.push(_encSubRow({
      kindClass: 'created', kindLabel: 'SYNTHESIZED',
      typeName: n.type, title: n.title, content: n.content,
      nodeId: n.id,
    }));
  }
  for (const n of (run.archived || [])) {
    out.push(_encSubRow({
      kindClass: 'enc-kind--archived enc-sub-row--archived',
      kindLabel: 'ARCHIVED',
      typeName: n.type, title: n.title, content: n.content, contentDim: true,
      nodeId: n.id,
    }));
  }
  for (const e of (run.evolved || [])) {
    out.push(el('div', { class: 'enc-entry enc-sub-row' },
      el('span', { class: 'enc-kind enc-kind--evolved' }, 'EVOLVED'),
      ' ',
      e.survivor || '',
      ' ',
      el('span', { class: 'enc-kind--evolved' }, 'supersedes'),
      ' ',
      el('span', { style: { opacity: 0.6 } }, e.archived || ''),
    ));
  }
  for (const e of (run.kept || [])) {
    out.push(el('div', { class: 'enc-entry enc-sub-row' },
      el('span', { class: 'enc-kind enc-kind--kept' }, 'KEPT'),
      ' ',
      e.source || '',
      ' ',
      el('span', { class: 'enc-kind--kept' }, '↔'),
      ' ',
      e.target || '',
      e.description ? el('div', { class: 'enc-meta-line enc-meta-line--dim' },
        e.description.substring(0, 250)) : null,
    ));
  }
  if (run.journal) {
    out.push(el('div', { class: 'enc-journal' },
      el('strong', null, 'Journal:'),
      el('pre', { class: 'enc-journal-text' }, run.journal.substring(0, 500)),
    ));
  }
  return out;
}

function _renderS2HealerBody(run, unitColor) {
  // Healer tier rows use the unit color (data-driven), not the fixed
  // O/K/Δ accents the default branch uses.
  return [
    _encTierRow('O', run.o_ref_type, run.o_summary,                   { accentStyle: { color: unitColor } }),
    _encTierRow('K', run.k_ref_type, run.k_summary,                   { accentClass: 'enc-tier-label--k' }),
    _encTierRow('Δ', run.ref_type || 'healer_generated', run.summary, { accentClass: 'enc-tier-label--delta' }),
  ];
}

function _renderS2DefaultBody(run) {
  const out = [
    _encTierRow('O', '', run.o_summary, { accentClass: 'enc-tier-label--o' }),
    _encTierRow('K', '', run.k_summary, { accentClass: 'enc-tier-label--k' }),
    _encTierRow('Δ', '', run.summary,   { accentClass: 'enc-tier-label--delta' }),
  ];
  for (const c of (run.communities || [])) {
    const matColor = c.maturity === 'settled' ? '#33ff88'
                   : c.maturity === 'active'  ? '#ffcc00'
                   : c.maturity === 'forming' ? '#45B7D1' : '#888';
    out.push(el('div', {
      class: 'enc-entry enc-community-row',
      onclick: () => loadNodeDetail(c.id || ''),
    },
      el('span', { class: 'enc-kind created' }, 'COMMUNITY'),
      ' ',
      el('span', { class: 'enc-community-maturity', style: { color: matColor } },
        (c.maturity || '?').toUpperCase()),
      el('span', { class: 'enc-title' }, c.title || ''),
      el('span', { class: 'enc-community-members' }, (c.members || 0) + ' members'),
      c.narrative ? el('div', { class: 'enc-meta-line' }, c.narrative) : null,
      c.content ? el('div', { class: 'enc-meta-line enc-meta-line--dim' },
        (c.content || '').substring(0, 300)) : null,
      c.open_questions ? el('div', { class: 'enc-meta-line enc-meta-line--warn' },
        'Open: ' + c.open_questions) : null,
    ));
  }
  return out;
}

function _renderS2RunCard(run) {
  const unit       = s2UnitFromType(run.type);
  const isConsol   = unit.type === 'consolidation';
  const isHealer   = unit.type === 'healer';
  const color      = unit.fg;
  // Encoding feed uses the shorter "S2 CONSOLIDATE" label vs the decoding
  // feed's "S2 CONSOLIDATION" — the row is denser here.
  const label      = isConsol ? 'S2 CONSOLIDATE' : unit.label;
  const t          = localTime(run.start_ts, 'time');
  const actionCount = (run.synthesized || []).length + (run.archived || []).length
                    + (run.kept || []).length + (run.evolved || []).length;
  const headerSummary = isConsol ? (actionCount + ' actions')
                      : isHealer ? (run.summary || '').substring(0, 80)
                      : (run.summary || '').substring(0, 60);

  // Build body section based on shape, plus the optional consol prompt
  // collapsible (lazy-loaded on first expand).
  const bodyRows = isConsol ? _renderS2ConsolBody(run)
                 : isHealer ? _renderS2HealerBody(run, color)
                 :            _renderS2DefaultBody(run);
  if (!actionCount && !run.summary) {
    bodyRows.push(el('div', { class: 'enc-empty-note' }, '(no write actions)'));
  }
  const body = el('div', { class: 'hook-body hook-body--padded' }, bodyRows);

  // Consolidation cards get a Show Prompt button + lazy-loaded prompt
  // body underneath. Other S2 shapes (healer, community) don't.
  let consolPromptBody = null;
  let showPromptBtn = null;
  if (isConsol) {
    consolPromptBody = el('div', { class: 'consol-prompt-body', style: { display: 'none' } },
      el('pre', { class: 'enc-prompt-pre' }, 'Loading...'),
    );
    showPromptBtn = el('button', { class: 'hook-details-btn hook-details-btn--right' }, 'Show Prompt');
  }

  const header = el('div', { class: 'hook-header' },
    el('span', { class: 'hook-badge', style: { background: color, color: '#000' } }, label),
    el('span', { class: 'hook-time' }, t),
    el('span', { class: 'hook-size' }, headerSummary),
    showPromptBtn,
  );
  header.addEventListener('click', () => body.classList.toggle('open'));

  if (isConsol && showPromptBtn && consolPromptBody) {
    _wirePromptToggle(showPromptBtn, consolPromptBody, async () => {
      const d = await api.consolidationPrompt(1);
      return d.user_content || d.error || '(no prompt available)';
    });
  }

  return el('div', {
    class: 'hook-entry enc-entry',
    dataset: { scale: 's2' },
    style: { borderLeftColor: color },
  },
    header,
    (run.o_summary || run.k_summary)
      ? el('div', { class: 'hook-prompt' }, run.k_summary || run.o_summary || '')
      : null,
    body,
    consolPromptBody,
  );
}

function _renderS1EncodeCard(run) {
  const t         = localTime(run.start_ts, 'time');
  const nodeCount = run.nodes ? run.nodes.length : 0;
  const sid       = run.session_id ? run.session_id.substring(0, 8) : '';

  // Body: created/revised nodes, then up to 8 edges, then overflow line.
  const bodyRows = [];
  for (const n of (run.nodes || [])) {
    const kindClass = n.kind === 'revised' ? 'revised' : 'created';
    const kindLabel = n.kind === 'revised' ? 'REVISED' : 'CREATED';
    bodyRows.push(_encSubRow({
      kindClass, kindLabel,
      typeName: n.type, title: n.title,
      content: n.content,
      nodeId: n.id,
    }));
  }
  for (const e of (run.edges || []).slice(0, 8)) {
    bodyRows.push(el('div', {
      class: 'enc-entry enc-sub-row connected',
      dataset: { kind: 'connected' },
    },
      el('span', { class: 'enc-kind connected' }, 'CONNECTED'),
      ' ',
      e.source_title || '',
      ' ',
      el('span', { class: 'enc-edge-arrow' }, '—' + (e.relation || '') + '→'),
      ' ',
      e.target_title || '',
    ));
  }
  if ((run.edges || []).length > 8) {
    bodyRows.push(el('div', { class: 'enc-edge-overflow' },
      '+' + ((run.edges || []).length - 8) + ' more edges'));
  }
  if (!(run.nodes || []).length && !(run.edges || []).length) {
    bodyRows.push(el('div', { class: 'enc-empty-note' }, '(no write actions)'));
  }
  const body = el('div', { class: 'hook-body hook-body--padded' }, bodyRows);

  // Prompt body — populated up-front from run.encoder_prompt (no lazy
  // load needed; the API already returned it inline).
  const encPromptBody = el('div', { class: 'enc-prompt-body', style: { display: 'none' } },
    el('pre', { class: 'enc-prompt-pre' },
      run.encoder_prompt || '(no prompt file found — encoding ran before prompt logging was added)'),
  );
  const showPromptBtn = el('button', { class: 'hook-details-btn hook-details-btn--right' }, 'Show Prompt');

  const header = el('div', { class: 'hook-header' },
    el('span', { class: 'hook-badge', style: { background: '#aa66ff', color: '#000' } }, 'S1 ENCODE'),
    el('span', { class: 'hook-time' }, t),
    sid ? el('span', { class: 'hook-session' }, sid) : null,
    el('span', { class: 'hook-id' }, '#' + (run.counter || '')),
    el('span', { class: 'hook-size' }, nodeCount + ' actions'),
    showPromptBtn,
  );
  header.addEventListener('click', () => body.classList.toggle('open'));
  _wirePromptToggle(showPromptBtn, encPromptBody, null);

  return el('div', {
    class: 'hook-entry enc-entry',
    dataset: { scale: 's1' },
    style: { borderLeftColor: '#aa66ff' },
  },
    header,
    run.prompt_info ? el('div', { class: 'hook-prompt' }, run.prompt_info) : null,
    body,
    encPromptBody,
  );
}

// toggleSurfacePrompt / _togglePromptBody removed — renderRecallEntry's
// "Show Prompt" button now wires its toggle via _wirePromptToggle with
// closure-captured element refs, same pattern as the encoding-feed
// cards. The shared helper lives at the bottom of this file.
//
// toggleHookBody remains exported below because `_renderS2ChainEntry`
// (decoding-feed S2 cards) still uses inline onclick on its header.
// Eventual migration would let us drop the export + window.* mount too.

// ── Live split layout ─────────────────────────────────────────────────
// Four orientations: graph-left / graph-right / graph-top / graph-bottom.
//
// Single state-driven function (_applyLayout) sets:
//   - grid-template-columns OR grid-template-rows on .live-split
//   - "graph-first" vs "graph-last" child order via CSS class
//   - "layout--horizontal" vs "layout--vertical" for cursor + axis
//
// The divider drag computes a new graphPct from the mouse position along
// the active axis; if graph is on the FAR side (right/bottom), the drag
// direction inverts (dragging away from graph shrinks it, toward expands).
//
// graphPct is "the graph pane's share of the split, 0-100", regardless of
// orientation. Both mode + pct persist to localStorage so the layout
// survives reloads.

const LAYOUT_MODE_KEY = 'dashboard.liveLayoutMode';
const LAYOUT_PCT_KEY  = 'dashboard.liveSplitPct';
const LAYOUT_DEFAULT_MODE = 'graph-left';
const LAYOUT_DEFAULT_PCT  = 60;

// mode → { axis: 'columns'|'rows', graphFirst: bool }
const LAYOUTS = {
  'graph-left':   { axis: 'columns', graphFirst: true  },
  'graph-right':  { axis: 'columns', graphFirst: false },
  'graph-top':    { axis: 'rows',    graphFirst: true  },
  'graph-bottom': { axis: 'rows',    graphFirst: false },
};

let _layoutMode = LAYOUT_DEFAULT_MODE;
let _graphPct = LAYOUT_DEFAULT_PCT;

// Graph visibility — orthogonal to the 4 layout orientations. The default
// is width-driven: below GRAPH_AUTOLOAD_MIN_WIDTH the dashboard is almost
// certainly in a narrow/embedded pane (e.g. viewed inside Claude), where
// the WebGL graph's continuous 60fps render loop competes with the host
// for the GPU and makes the UI sluggish — so we don't mount it at all.
// A wide browser window loads it as before. Once the operator toggles
// explicitly, that persisted choice wins over the width default.
const GRAPH_VISIBLE_KEY = 'dashboard.graphVisible';
const GRAPH_AUTOLOAD_MIN_WIDTH = 1000;
let _graphVisible = true;

function _applyLayout(mode, graphPct) {
  const split = document.getElementById('live-split');
  if (!split) return;
  const cfg = LAYOUTS[mode] || LAYOUTS[LAYOUT_DEFAULT_MODE];
  const pct = Math.max(0, Math.min(100, graphPct));

  // Cache state for the drag handler + persistence.
  _layoutMode = mode;
  _graphPct = pct;

  // Set grid template along the active axis only; the other axis fills.
  const sizing = cfg.graphFirst
    ? `${pct}fr 6px ${100 - pct}fr`
    : `${100 - pct}fr 6px ${pct}fr`;
  if (cfg.axis === 'columns') {
    split.style.gridTemplateColumns = sizing;
    split.style.gridTemplateRows = '1fr';
  } else {
    split.style.gridTemplateRows = sizing;
    split.style.gridTemplateColumns = '1fr';
  }

  // Reset then apply layout classes — drives order (which child sits in
  // which cell) and cursor (ew vs ns).
  split.classList.remove('layout--horizontal', 'layout--vertical', 'graph-first', 'graph-last');
  split.classList.add(cfg.axis === 'columns' ? 'layout--horizontal' : 'layout--vertical');
  split.classList.add(cfg.graphFirst ? 'graph-first' : 'graph-last');

  // Mark the active picker button.
  document.querySelectorAll('.live-layout-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.layout === mode);
  });

  // No bus event needed for the graph to resize — it observes its own
  // container via a ResizeObserver (see graph.js init), which catches this
  // grid-template change along with window resizes and divider drags.
}

function _restoreLayout() {
  let mode = LAYOUT_DEFAULT_MODE;
  let pct  = LAYOUT_DEFAULT_PCT;
  try {
    const savedMode = localStorage.getItem(LAYOUT_MODE_KEY);
    if (savedMode && LAYOUTS[savedMode]) mode = savedMode;
    const savedPct = parseFloat(localStorage.getItem(LAYOUT_PCT_KEY));
    if (!Number.isNaN(savedPct) && savedPct >= 0 && savedPct <= 100) pct = savedPct;
  } catch (e) { /* localStorage may be blocked; defaults are fine */ }
  _applyLayout(mode, pct);
}

function _persistLayout() {
  try {
    localStorage.setItem(LAYOUT_MODE_KEY, _layoutMode);
    localStorage.setItem(LAYOUT_PCT_KEY, String(_graphPct));
  } catch (e) { /* blocked storage → silently skip */ }
}

// ── Graph visibility ──────────────────────────────────────────────────

// Resolve the initial visibility: an explicit persisted choice wins;
// otherwise default by viewport width (narrow → off).
function _restoreGraphVisibility() {
  let visible;
  try {
    const saved = localStorage.getItem(GRAPH_VISIBLE_KEY);
    if (saved === '1') visible = true;
    else if (saved === '0') visible = false;
  } catch (e) { /* localStorage blocked → fall through to width default */ }
  if (visible === undefined) visible = window.innerWidth >= GRAPH_AUTOLOAD_MIN_WIDTH;
  _graphVisible = visible;
  _applyGraphVisibility();
}

// Reflect _graphVisible in the DOM: collapse/expand the split + sync the
// toolbar button. Does NOT mount/destroy the graph — callers do that, so
// this stays a pure view-sync (safe to call before the graph module is
// ready).
function _applyGraphVisibility() {
  const split = document.getElementById('live-split');
  if (split) split.classList.toggle('graph-hidden', !_graphVisible);
  const btn = document.getElementById('graph-toggle-btn');
  if (btn) {
    btn.classList.toggle('active', _graphVisible);
    btn.textContent = _graphVisible ? 'Hide graph' : 'Show graph';
    btn.title = _graphVisible
      ? 'Hide the 3D graph — frees the GPU render loop'
      : 'Show the 3D graph';
  }
}

// Toolbar toggle. Flips visibility, persists the explicit choice, then
// mounts or fully tears down the WebGL graph so a hidden graph costs zero
// GPU (teardown, not pause).
export function toggleGraph() {
  _graphVisible = !_graphVisible;
  try { localStorage.setItem(GRAPH_VISIBLE_KEY, _graphVisible ? '1' : '0'); } catch (e) { /* blocked */ }
  _applyGraphVisibility();
  if (_graphVisible) graph.activate();   // mounts + sizes (300ms internal defer)
  else graph.destroy();                  // releases context + render loop
}

// Public — called by the picker buttons via window.setLiveLayout.
export function setLiveLayout(mode) {
  if (!LAYOUTS[mode]) return;
  _applyLayout(mode, _graphPct);
  _persistLayout();
}

// Idempotency guard — _setupDivider attaches document-level mousemove +
// mouseup listeners. If init() ever fires twice (architecture drift, hot
// reload, future double-wire bug) we'd get N parallel handlers running
// on every mouse move. Single source of truth: once wired, never again.
let _dividerWired = false;
function _setupDivider() {
  if (_dividerWired) return;
  const divider = document.getElementById('live-divider');
  const split = document.getElementById('live-split');
  if (!divider || !split) return;
  _dividerWired = true;
  let dragging = false;

  divider.addEventListener('mousedown', (e) => {
    dragging = true;
    divider.classList.add('dragging');
    split.classList.add('dragging');
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const cfg = LAYOUTS[_layoutMode];
    const rect = split.getBoundingClientRect();
    // Mouse position along the active axis, as a 0-100 fraction.
    const raw = cfg.axis === 'columns'
      ? ((e.clientX - rect.left) / rect.width) * 100
      : ((e.clientY - rect.top)  / rect.height) * 100;
    // When graph is on the FAR side, invert — moving toward graph
    // shrinks the FIRST pane, growing graph.
    const pct = cfg.graphFirst ? raw : 100 - raw;
    _applyLayout(_layoutMode, pct);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    split.classList.remove('dragging');
    _persistLayout();
  });
}

// ── Lifecycle ─────────────────────────────────────────────────────────

export function init() {
  // Placeholder while the feed waits for the first poll fire.
  const feed = document.getElementById('feed-decoding');
  feed.innerHTML = '<div class="hook-placeholder" class="feed-empty">Waiting for brain activity...</div>';

  _restoreLayout();
  _restoreGraphVisibility();
  _setupDivider();

  // Renderer subscribes here once. The graph module subscribes to the
  // same topic in its own init() — both react to the bus event stream
  // independently.
  bus.subscribe('recall:event', _renderRecallEvent);
  bus.subscribe('insights:tick', _renderInsightsPanel);
  _wireInsightsDismiss();

  // Pinned-card highlight — graph.js publishes 'graph:pinned' when the
  // user clicks a recall card (or when pinned mode is cleared via
  // Refresh / dropdown change). We mirror the lock by adding/removing
  // .recall-entry--pinned on the matching card. Living here (the feed
  // owner) means graph.js doesn't reach across module boundaries to
  // mutate live.js DOM.
  bus.subscribe('graph:pinned', ({ eventId }) => {
    document.querySelectorAll('#feed-decoding .recall-entry--pinned')
      .forEach(el => el.classList.remove('recall-entry--pinned'));
    if (!eventId) return;
    const target = document.querySelector(
      '#feed-decoding .recall-entry[data-recall-id="' + eventId + '"]');
    if (target) target.classList.add('recall-entry--pinned');
  });

  // Insights — slow poll (60s), gated on Live tab visible. Same bus
  // pattern as `recall:event`: the fetcher only publishes; the renderer
  // is a separate subscriber. Adding a "insights count" tab-bar badge
  // later means subscribing in app.js, not editing this fetcher.
  poll.register({
    key: 'insights-live',
    interval: 60000,
    activeWhen: () => document.getElementById('tab-live').classList.contains('active'),
    fetcher: async () => {
      try {
        const env = await api.insightsLive();
        // envelope_ok: { status: 'success', data: [...] }
        // envelope_error: { status: 'error', error: '...' }
        if (env && env.status === 'success') {
          bus.publish('insights:tick', { insights: env.data || [] });
        } else if (env && env.status === 'error') {
          console.error('[live] insights endpoint error:', env.error);
        }
      } catch (e) { console.error('[live] insights fetch failed:', e); }
    },
  });

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
  // Skip entirely when the graph is hidden (narrow/embedded view) so the
  // WebGL loop never spins; toggleGraph() mounts it on demand.
  if (_graphVisible) {
    try { graph.activate(); } catch (e) { console.error('[live] graph activate failed:', e); }
  }
}

export function deactivate() {
  // Graph stays mounted in the hidden tab; its canvas keeps the WebGL
  // context warm so re-activating is instant.
}
