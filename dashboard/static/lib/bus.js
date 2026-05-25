// ===========================================================================
// lib/bus.js — tiny pub/sub. Kills the 15 global vars.
// ---------------------------------------------------------------------------
// Pattern from CSS-Tricks "Build a state management system with vanilla JS"
// (Diona Rodrigues). Used for cross-cutting events between tab modules:
//
//   bus.publish('recall:event', { id, returned_ids, used_ids, ... })
//
//   bus.subscribe('recall:event', (payload) => {
//     // graph tab pulses the surfaced nodes
//   })
//
// Why not the DOM's CustomEvent + window? We avoid two pitfalls:
//   1. CustomEvent is synchronous + blocking — one slow subscriber stalls
//      every other. Bus subscribers are wrapped in try/catch and isolated.
//   2. window pollution — every tab module talking to `window` couples them
//      to global state we can't enumerate. The bus is a single import.
//
// Topic naming convention:  <source>:<event>
//   recall:event          — new S1 recall trace landed
//   graph:activation      — pulse these node ids (consumed by graph tab)
//   errors:new            — error count increased
//   health:tick           — system-status refresh fired
//   tab:active            — user switched to tab X (payload: tab name)
//
// Side benefit: subscribers don't have to be added inline in the publisher's
// code. Tab modules subscribe in their init(); the publisher knows nothing
// about who's listening.
// ===========================================================================

const _subs = new Map();  // topic → Set<handler>

export function subscribe(topic, handler) {
  if (!_subs.has(topic)) _subs.set(topic, new Set());
  _subs.get(topic).add(handler);
  // Return an unsubscribe fn — callers can store it and detach on
  // deactivate() so we don't leak subscriptions when a tab unloads.
  return () => {
    const set = _subs.get(topic);
    if (set) set.delete(handler);
  };
}

export function publish(topic, payload) {
  const set = _subs.get(topic);
  if (!set || !set.size) return;
  // Snapshot so a subscriber that unsubscribes during dispatch doesn't
  // mutate the iteration set.
  for (const handler of [...set]) {
    try {
      handler(payload);
    } catch (e) {
      // One bad subscriber should not break the dispatch. Logged loudly
      // so the bug surfaces in console.
      console.error('[bus] subscriber for "' + topic + '" threw:', e);
    }
  }
}

// Debug helper — list current subscriptions. Not used in production, useful
// from the browser devtools console when wiring a new topic.
export function _debug_topics() {
  const out = {};
  for (const [topic, set] of _subs) out[topic] = set.size;
  return out;
}

// Convenience export so tab modules can `import bus from './lib/bus.js'` and
// call `bus.publish(...)` / `bus.subscribe(...)` ergonomically.
export const bus = { subscribe, publish, _debug_topics };
export default bus;
