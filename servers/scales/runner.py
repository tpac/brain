"""Scale runner infrastructure — background thread lifecycle for scale agents.

Every scale agent (S1 encode, S2 session encode, future scales) follows
the same lifecycle:
1. Create read-only Brain instance
2. Create dispatch function (reads local, writes via TCP)
3. Call the scale's run function
4. Write delta trace
5. Release lock, close brain

This module provides the generic lifecycle. Scale-specific logic lives
in each scale's module (scales/s1/encode.py, scales/s2/encode.py, etc.).
"""

import time
import threading

from .dispatch import make_scale_dispatch, daemon_tcp_send


def run_in_background(name, brain_db_path, session_id, counter, lock,
                      run_fn, encoding_source='encoder:sonnet',
                      trace_scale='s1', trace_chain_fn=None):
    """Run a scale agent in a background thread.

    Args:
        name: Scale name for logging (e.g. 's1e', 's2')
        brain_db_path: Path to brain.db
        session_id: Session ID from SessionContext
        counter: Stop counter value
        lock: threading.Lock for mutual exclusion (one agent at a time)
        run_fn: Scale's run function: run_fn(brain, dispatch_fn, counter, session_id) -> dict
        encoding_source: encoding_source value for new nodes
        trace_scale: Scale for delta trace ('s1', 's2', etc.)
        trace_chain_fn: Function(session_id, counter) -> chain_id for delta trace.
                        If None, no delta trace is written (scale writes its own).
    """
    def _thread_fn():
        t0 = time.time()
        read_brain = None
        try:
            print("[%s] STARTING (counter=%d)" % (name, counter), flush=True)
            from servers.brain import Brain
            read_brain = Brain(brain_db_path)

            dispatch = make_scale_dispatch(read_brain, encoding_source=encoding_source)

            result = run_fn(read_brain, dispatch, counter, session_id)
            elapsed_ms = int((time.time() - t0) * 1000)
            actions = result.get('actions', 0) if isinstance(result, dict) else 0
            print("[%s] DONE: %d actions in %dms" % (name, actions, elapsed_ms), flush=True)

            # Write delta trace if chain function provided
            if trace_chain_fn:
                try:
                    chain_id = trace_chain_fn(session_id, counter)
                    action_lines = []
                    for a in (result.get('action_details', []) if isinstance(result, dict) else []):
                        action_lines.append('%s: %s' % (a.get('tool', ''), a.get('summary', '')))
                    daemon_tcp_send('trace_append', {
                        'chain_id': chain_id,
                        'scale': trace_scale,
                        'event_type': 'delta',
                        'ref_type': 'encoding_run',
                        'ref_id': str(counter),
                        'summary': '%d actions in %dms:\n%s\n---\n%s' % (
                            actions, elapsed_ms,
                            '\n'.join(action_lines) if action_lines else '(no actions)',
                            (result.get('final_text', '') or '')[:2000]),
                        'session_id': session_id,
                    })
                except Exception as e:
                    print('[%s] TRACE ERROR (delta): %s' % (name, e), flush=True)

        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            print("[%s] FAILED after %dms: %s" % (name, elapsed_ms, e), flush=True)
        finally:
            if read_brain:
                try:
                    read_brain.close()
                except Exception:
                    pass
            lock.release()

    threading.Thread(target=_thread_fn, daemon=True, name=name).start()
