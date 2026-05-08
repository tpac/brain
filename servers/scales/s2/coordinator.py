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
    from .consolidation import Consolidation
    from .community import CommunityDetection
    from .healer import Healer

    # AspectIntegration intentionally NOT wired here yet (2026-05-08).
    # It's been migrated and tested in eval (78.2% routing accuracy on the
    # clone), but on production it triggered a runaway S2 cascade — the
    # decoder writes an O trace even when nothing is unclassified, which
    # downstream units (community_detection, consolidation) read as "new
    # work" and re-fire on. Result: community_detection looped 224× in 3
    # minutes, daemon RSS hit 2.2GB in 51s. Rolled back pending two fixes:
    #   (1) decoder must early-out without writing the O trace when batch
    #       is empty, AND
    #   (2) downstream gating shouldn't treat aspect_scan as new s1 work.
    # AspectRegistry still reads aspects_v1.json (no rollback there) — the
    # taxonomy is shipped; only the maintenance unit is paused.
    units = [
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
