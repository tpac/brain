"""S2 Coordinator — runs all S2 integration units in order.

The idle hook calls run_s2() once. S2 decides what to run.
Each unit checks its own traces to decide whether it should fire.

Ordering matters:
1. AspectIntegration — classify new node types / edge relations into the 14 aspects
2. Consolidation — clean the graph (merge convergent nodes)
3. Community detection — detect structure on clean graph
4. Healer — fill missing findability fields on nodes that lack them (uses community context)
"""


def run_s2(brain):
    """Run all S2 integration units in order.

    Args:
        brain: Brain instance (direct DB access, inline execution).

    Returns:
        dict of {unit_name: result_dict}
    """
    from .aspect_integration import AspectIntegration
    from .consolidation import Consolidation
    from .community import CommunityDetection
    from .healer import Healer

    # AspectIntegration re-wired 2026-05-10. The cascade rollback (2026-05-08)
    # was caused by the decoder writing an O trace even when nothing was
    # unclassified. Fixed in aspect_decoder.py: empty-batch early-out moved
    # BEFORE the trace write — when there's no work, the unit is a true
    # no-op (no proposals, no trace). Contract locked by
    # tests/test_aspect_decoder.py. Downstream gating concern (#2 in the
    # backlog) was already moot — none of the other decoders read
    # aspect_scan traces; they gate on their own internal state.
    units = [
        AspectIntegration(brain),
        Consolidation(brain),
        CommunityDetection(brain),
        Healer(brain),
    ]

    results = {}
    for unit in units:
        failure_key = 's2_%s_consecutive_failures' % unit.NAME
        try:
            result = unit.run()
            results[unit.NAME] = result
            # Reset failure counter on success. 'skipped' counts as neutral
            # (unit declined to run), don't reset on that.
            if not (isinstance(result, dict) and result.get('skipped')):
                prior = int(brain.get_config(failure_key) or 0)
                if prior > 0:
                    brain.set_config(failure_key, '0')
                    print('[s2:%s] Recovered after %d consecutive failures'
                          % (unit.NAME, prior), flush=True)
        except Exception as e:
            results[unit.NAME] = {'error': str(e)[:200]}
            print('[s2:%s] ERROR: %s' % (unit.NAME, e), flush=True)
            brain._log_error('s2_%s' % unit.NAME, e, 'coordinator run')

            # Persistent-failure escalation — rate-limited brain._log_error
            # suppresses repeats, hiding the fact that a unit is broken every
            # run. Track a running counter in brain_meta; each new failure
            # fires a DIFFERENT error source (encoded with the count) so the
            # rate limiter can't collapse them, and consciousness reports on
            # boot. Resets on next successful run.
            try:
                count = int(brain.get_config(failure_key) or 0) + 1
                brain.set_config(failure_key, str(count))
                if count >= 3:
                    brain._log_error(
                        's2_%s_persistent_failure' % unit.NAME,
                        RuntimeError(
                            '%s has failed %d runs in a row: %s' % (
                                unit.NAME, count, str(e)[:150])),
                        'unit is consistently broken — investigate immediately')
            except Exception as persist_err:
                # The whole point of this block is to make failures loud.
                # Swallowing the escalation is the very pattern we're fighting.
                brain._log_error(
                    's2_%s_persistent_failure_tracker_crashed' % unit.NAME,
                    persist_err,
                    'could not update failure counter for %s' % unit.NAME)

    return results
