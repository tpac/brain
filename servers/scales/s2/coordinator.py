"""S2 Coordinator — runs all S2 integration units in order.

The idle hook calls run_s2() once. S2 decides what to run.
Each unit checks its own traces to decide whether it should fire.

Ordering matters:
1. Edge families — classify new relation types (other units depend on this)
2. Consolidation — clean the graph (merge convergent nodes)
3. Community detection — detect structure on clean graph
4. Enrichment — generate V5 vectors for nodes missing them (uses community context)
"""


def run_s2(brain):
    """Run all S2 integration units in order.

    Args:
        brain: Brain instance (direct DB access, inline execution).

    Returns:
        dict of {unit_name: result_dict}
    """
    from .edge_families import EdgeFamilyIntegration
    from .consolidation import Consolidation
    from .community import CommunityDetection
    from .enrichment import Enrichment

    units = [
        EdgeFamilyIntegration(brain),
        Consolidation(brain),
        CommunityDetection(brain),
        Enrichment(brain),
    ]

    results = {}
    for unit in units:
        try:
            result = unit.run()
            results[unit.NAME] = result
        except Exception as e:
            results[unit.NAME] = {'error': str(e)[:200]}
            print('[s2:%s] ERROR: %s' % (unit.NAME, e), flush=True)
            brain._log_error('s2_%s' % unit.NAME, e, 'coordinator run')

    return results
