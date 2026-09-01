// ===========================================================================
// lib/activity.js — S1 activity: surface runs and encode runs, as themselves.
// ---------------------------------------------------------------------------
// These were briefly merged into one "moment" card on the theory that a turn
// recognizes and remembers, so both belong to one event. In practice they are
// NOT one process: recognition happens every turn, encoding happens every Nth,
// and they don't line up. The merged card therefore rendered a side-by-side
// two-pane body where one pane was empty on most turns — a layout asserting an
// alignment the data doesn't have.
//
// So: one chronological feed, two independent card kinds, each rendering only
// what it actually is. A surface run shows what surfaced and what won; an
// encode run shows the residue and the writes. Neither pretends the other
// exists. Same time axis, same stream hues — related, not conflated.
//
// Shared chrome (session chip, journal chip, prompt section, action rows) comes
// from lib/sessions.js, lib/journal.js and lib/cards.js, so the two card kinds
// stay consistent without either one owning the other's markup.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { el, localTime, relativeTime } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import { promptSection, subRow, edgeRow } from '/static/lib/cards.js';
import { journalBlock, journalChip, isStanding } from '/static/lib/journal.js';
import { sessionChip, sessionColor } from '/static/lib/sessions.js';

// Where a recall came from. `hook` is the automatic per-turn recall; `mcp` is
// Anchor deliberately reaching (recall() by hand) — a real distinction, since
// one is reflex and the other is intent.
const RECALL_SOURCE = {
  hook:     { label: 'turn',     cls: 'act-src--hook',     title: 'automatic recall on this turn' },
  mcp:      { label: 'reached',  cls: 'act-src--mcp',      title: 'recall() called deliberately' },
  internal: { label: 'internal', cls: 'act-src--internal', title: 'internal recall' },
};

/** Stop counter from an S1 chain id (`s1r-4ead64bd-12` → 12). Null when the
 *  chain isn't shaped like one, so the card just omits the turn number. */
export function stopFromChain(chainId) {
  if (!chainId) return null;
  const m = /^s1[re]-[^-]+-(\d+)$/.exec(chainId);
  return m ? parseInt(m[1], 10) : null;
}

/** Interleave surface runs and encode runs into one time-ordered list.
 *
 * No merging, no join key — each item stays itself. `kind` selects the
 * renderer; `ts` is the only thing the two share, and it's the only thing the
 * feed needs from them. */
export function assembleActivity(recallEvents, encodeRuns) {
  const items = [];
  for (const evt of (recallEvents || [])) {
    if (!evt || !evt.id) continue;
    items.push({
      kind: 'surface',
      key: 'r:' + evt.id,
      ts: evt.timestamp || '',
      session_id: evt.session_id || '',
      stop: stopFromChain(evt.chain_id),
      data: evt,
    });
  }
  for (const run of (encodeRuns || [])) {
    if (!run || !run.chain_id) continue;
    items.push({
      kind: 'encode',
      key: 'e:' + run.chain_id,
      ts: run.start_ts || '',
      session_id: run.session_id || '',
      stop: stopFromChain(run.chain_id),
      data: run,
    });
  }
  return items.sort((a, b) => (b.ts || '').localeCompare(a.ts || ''));
}

/** What must change before an item is worth re-rendering. Covers the two
 *  asynchronous fills: a surface run gaining its judge output, and an encode
 *  run's nodes/edges/journal landing after its first trace. */
export function fingerprint(item) {
  const d = item.data || {};
  return item.kind === 'surface'
    ? ['s', d.judge_output ? 1 : 0, d.used_count || 0,
       Object.keys(d.titles || {}).length].join(':')
    : ['e', (d.nodes || []).length, (d.edges || []).length,
       (d.journal_notes || []).length].join(':');
}

// ── Shared header scaffold ─────────────────────────────────────────────────
// Both kinds open with the same left-hand run: time, stream, turn. Only the
// badge and the counts differ, which is exactly the amount they have in common.
function _headBase(item, { showSession }) {
  return [
    el('span', { class: 'act-time', title: item.ts ? item.ts + '\n' + relativeTime(item.ts) : '' },
      localTime(item.ts, 'time')),
    showSession ? sessionChip(item.session_id, { compact: true }) : null,
    item.stop != null ? el('span', { class: 'act-stop', title: 'turn ' + item.stop }, '#' + item.stop) : null,
  ];
}

// ── Surface run ────────────────────────────────────────────────────────────

function _surfaceBody(evt) {
  const rows = [];
  const titles = evt.titles || {};
  const used = new Set(evt.used_ids || []);

  // What actually reached Claude — the payload that changed the turn.
  if (evt.judge_output && evt.judge_output !== '(no selection)') {
    rows.push(el('div', { class: 'act-surfaced' }, el('pre', null, evt.judge_output)));
  } else if (evt.judge_output === '(no selection)') {
    rows.push(el('div', { class: 'act-pane-empty' },
      'Nothing was selected — the surfacer saw these candidates and passed on all of them.'));
  }

  // The candidate set: chosen solid, the rest dim. The near-misses are how you
  // see whether recall had the right material and chose well, or never had it.
  const entries = Object.entries(titles);
  if (entries.length) {
    rows.push(el('div', { class: 'act-candidates' },
      ...entries.slice(0, 16).map(([nid, title]) => el('div', {
        class: ['act-candidate', used.has(nid) && 'act-candidate--used'].filter(Boolean),
        onclick: (e) => { e.stopPropagation(); loadNodeDetail(nid); },
        title: used.has(nid) ? 'surfaced to Claude' : 'candidate — not selected',
      }, title)),
      entries.length > 16
        ? el('div', { class: 'act-candidate act-candidate--more' },
            '+' + (entries.length - 16) + ' more')
        : null,
    ));
  }

  if (evt.has_prompt) {
    const p = promptSection(async () => {
      const d = await api.recallPrompt({
        pointer: evt.payload_pointer || '',
        chain_id: evt.chain_id || '',
        recall_ref: evt.recall_ref,
      });
      return d.judge_prompt || d.error || '(no prompt available)';
    }, { label: 'Surfacer prompt' });
    rows.push(el('div', { class: 'act-prompt-row' }, p.button), p.body);
  }
  if (!rows.length) {
    rows.push(el('div', { class: 'act-pane-empty' }, 'No candidate detail recorded for this recall.'));
  }
  return rows;
}

function _renderSurface(item, { showSession, hooks }) {
  const evt = item.data;
  const surfaced = Object.keys(evt.titles || {}).length || (evt.returned_ids || []).length;
  const chosen = evt.used_count || (evt.used_ids || []).length;
  const src = RECALL_SOURCE[evt.source] || { label: evt.source || '?', cls: '', title: '' };
  const color = sessionColor(item.session_id);

  const head = el('div', { class: 'act-head' },
    ..._headBase(item, { showSession }),
    el('span', { class: 'act-badge act-badge--surface', title: 'recognition — what surfaced' }, 'RECOGNIZED'),
    surfaced
      ? el('span', { class: 'act-count act-count--surface' }, chosen + '/' + surfaced)
      : null,
    el('span', { class: 'act-src ' + src.cls, title: src.title }, src.label),
    el('span', { class: 'act-head-tail' }, el('span', { class: 'act-caret' }, '▸')),
    evt.query
      ? el('div', { class: 'act-query', style: { borderLeftColor: color } }, evt.query)
      : null,
  );
  return { head, buildBody: () => _surfaceBody(evt) };
}

// ── Encode run ─────────────────────────────────────────────────────────────

function _encodeBody(run) {
  const rows = [];
  // Residue first: the encoder's mind before its hands.
  const jb = journalBlock(run.journal_notes);
  if (jb) rows.push(jb);

  for (const n of (run.nodes || [])) {
    rows.push(subRow({
      kindClass: n.kind === 'revised' ? 'revised' : 'created',
      kindLabel: n.kind === 'revised' ? 'REVISED' : 'CREATED',
      typeName: n.type, title: n.title, content: n.content, nodeId: n.id,
    }));
  }
  for (const e of (run.edges || []).slice(0, 12)) rows.push(edgeRow(e));
  if ((run.edges || []).length > 12) {
    rows.push(el('div', { class: 'enc-edge-overflow' },
      '+' + ((run.edges || []).length - 12) + ' more edges'));
  }
  if (!(run.nodes || []).length && !(run.edges || []).length && !jb) {
    rows.push(el('div', { class: 'enc-empty-note' }, '(the encoder ran and wrote nothing)'));
  }

  const p = promptSection(async () => {
    const d = await api.encodingPrompt({ chain_id: run.chain_id });
    return d.user_content || d.error || '(no prompt available)';
  }, { label: 'Encoder prompt' });
  rows.push(el('div', { class: 'act-prompt-row' }, p.button), p.body);
  return rows;
}

function _renderEncode(item, { showSession }) {
  const run = item.data;
  const nodes = run.nodes || [];
  const created = nodes.filter(n => n.kind !== 'revised').length;
  const revised = nodes.filter(n => n.kind === 'revised').length;
  const edges = (run.edges || []).length;
  const notes = run.journal_notes || [];
  const color = sessionColor(item.session_id);

  const counts = [];
  if (created) counts.push(el('span', { class: 'act-count act-count--created' }, '+' + created));
  if (revised) counts.push(el('span', { class: 'act-count act-count--revised' }, '↻' + revised));
  if (edges)   counts.push(el('span', { class: 'act-count act-count--edge' }, '—' + edges + '→'));
  if (!counts.length) counts.push(el('span', { class: 'act-count act-count--none' }, 'nothing'));

  const head = el('div', { class: 'act-head' },
    ..._headBase(item, { showSession }),
    el('span', { class: 'act-badge act-badge--encode', title: 'remembering — what got written down' }, 'REMEMBERED'),
    ...counts,
    el('span', { class: 'act-head-tail' },
      journalChip(notes, { title: 'encode · turn ' + (item.stop ?? '?') + ' · ' + localTime(item.ts, 'time') }),
      el('span', { class: 'act-caret' }, '▸'),
    ),
    run.prompt_info
      ? el('div', { class: 'act-query act-query--dim', style: { borderLeftColor: color } }, run.prompt_info)
      : null,
  );
  return { head, buildBody: () => _encodeBody(run) };
}

// ── The card ───────────────────────────────────────────────────────────────

/** One activity card. `hooks` carries the graph interactions the feed owns
 *  ({onHover, onLeave, onPin}) so this module stays pure presentation and never
 *  reaches into the graph module. Only surface cards use them — an encode run
 *  has no candidate set to light. */
export function renderActivity(item, { showSession = true, hooks = {} } = {}) {
  const isSurface = item.kind === 'surface';
  const { head, buildBody } = isSurface
    ? _renderSurface(item, { showSession, hooks })
    : _renderEncode(item, { showSession });

  // Body stays empty until first expand — eagerly building every collapsed
  // card's detail cost ~9× the DOM for content nobody had opened.
  const body = el('div', { class: 'act-body' });
  let built = false;
  const build = () => {
    if (built) return;
    built = true;
    body.replaceChildren(...buildBody().filter(Boolean));
  };

  const notes = isSurface ? [] : (item.data.journal_notes || []);
  const card = el('div', {
    class: ['act', 'act--' + item.kind,
            notes.some(isStanding) && 'card--standing'].filter(Boolean),
    dataset: {
      item: item.key,
      kind: item.kind,
      session: item.session_id || '',
      ts: item.ts || '',
      recallId: isSurface ? (item.data.id || '') : '',
    },
    style: { borderLeftColor: sessionColor(item.session_id) },
  }, head, body);

  // A re-render of an open card must repopulate immediately or the operator's
  // open card goes blank; live.js re-applies `.open` and calls this.
  card._buildActivityBody = build;

  head.addEventListener('click', () => {
    const opening = !card.classList.contains('open');
    if (opening) build();
    card.classList.toggle('open');
    if (opening && isSurface && hooks.onPin) hooks.onPin(item);
  });
  if (isSurface && hooks.onHover) card.addEventListener('mouseenter', () => hooks.onHover(item));
  if (isSurface && hooks.onLeave) card.addEventListener('mouseleave', () => hooks.onLeave(item));
  return card;
}

export default { assembleActivity, renderActivity, fingerprint, stopFromChain };
