"""Memory Eval KPIs — derived from qualitative analysis of 25 scenarios.

Each KPI measures a specific failure or success pattern observed in the
recall pipeline. Scored per-scenario and aggregated.

FAILURE PATTERNS (from analysis):
1. Behavioral rules never surface (1/17 scenarios had expected principle)
2. Safety rules missing before destructive ops (a2: no backup warning)
3. Hub f11ae3cd pollutes unrelated queries (e1, f2, f5)
4. Emotional/vague queries get wrong results (e3, f1, b2)
5. Temporal queries get no recency weighting (f5, g2)
6. how_to intent boost miscalibrates (c1: 0.943 for irrelevant node)

SUCCESS PATTERNS:
1. Specific technical queries nail content (a1, a3, b1, b4, d3)
2. Identity/philosophical queries activate rich clusters (f3)
3. Negative case correctly returns low scores (f4)
"""


def score_all_kpis(scenario, results):
    """Score all KPIs for a single scenario result.

    Args:
        scenario: scenario dict from memory_scenarios.py
        results: list of result dicts from recall (top 25)

    Returns: {kpi_name: {score: 0-1, detail: str}}
    """
    result_ids = [r['id'] for r in results]
    result_set = set(result_ids)
    top3_ids = set(result_ids[:3])
    top8_ids = set(result_ids[:8])
    top3_types = [r['type'] for r in results[:3]]
    top8_types = [r['type'] for r in results[:8]]
    top3_scores = [r.get('score', 0) or r.get('effective_activation', 0) for r in results[:3]]
    top8_scores = [r.get('score', 0) or r.get('effective_activation', 0) for r in results[:8]]

    kpis = {}

    # ═══════════════════════════════════════════════════════════
    # KPI 1: CONTENT PRECISION (top 3)
    # Do the top 3 results directly address the query topic?
    # Measured by: expected_nodes found in top 3
    # ═══════════════════════════════════════════════════════════
    expected = set(scenario.get('expected_nodes', []))
    if expected:
        hits = expected & top3_ids
        kpis['content_precision_top3'] = {
            'score': len(hits) / min(len(expected), 3),
            'detail': f'{len(hits)}/{min(len(expected), 3)} expected in top 3',
            'hits': list(hits),
        }
    else:
        kpis['content_precision_top3'] = {'score': None, 'detail': 'no expected nodes'}

    # ═══════════════════════════════════════════════════════════
    # KPI 2: CONTENT RECALL (top 8 and top 25)
    # How many expected nodes appear at all?
    # ═══════════════════════════════════════════════════════════
    if expected:
        hits_8 = expected & top8_ids
        hits_25 = expected & result_set
        kpis['content_recall_top8'] = {
            'score': len(hits_8) / len(expected),
            'detail': f'{len(hits_8)}/{len(expected)} in top 8',
        }
        kpis['content_recall_top25'] = {
            'score': len(hits_25) / len(expected),
            'detail': f'{len(hits_25)}/{len(expected)} in top 25',
        }
    else:
        kpis['content_recall_top8'] = {'score': None, 'detail': 'N/A'}
        kpis['content_recall_top25'] = {'score': None, 'detail': 'N/A'}

    # ═══════════════════════════════════════════════════════════
    # KPI 3: BEHAVIORAL ACTIVATION
    # Do relevant rules/principles surface?
    # This is the #1 failure: 1/17 scenarios had principles surface.
    # ═══════════════════════════════════════════════════════════
    expected_principles = set(scenario.get('expected_principles', []))
    if expected_principles:
        hits = expected_principles & result_set
        hits_8 = expected_principles & top8_ids
        kpis['behavioral_activation'] = {
            'score': len(hits) / len(expected_principles),
            'detail': f'{len(hits)}/{len(expected_principles)} principles in top 25, '
                      f'{len(hits_8)} in top 8',
            'in_top8': len(hits_8),
        }
    else:
        kpis['behavioral_activation'] = {'score': None, 'detail': 'no principles expected'}

    # ═══════════════════════════════════════════════════════════
    # KPI 4: HUB CONTAMINATION
    # Do known hub nodes appear where they shouldn't?
    # f11ae3cd is the worst offender (appeared in 3 anti-expected top 8s)
    # ═══════════════════════════════════════════════════════════
    anti = set(scenario.get('anti_expected', []))
    if anti:
        noise_top3 = anti & top3_ids
        noise_top8 = anti & top8_ids
        # Score: 1.0 = no contamination, 0.0 = all anti-expected in top 8
        contamination = len(noise_top8) / min(len(anti), 8)
        kpis['hub_contamination'] = {
            'score': 1.0 - contamination,
            'detail': f'{len(noise_top8)} hubs in top 8, {len(noise_top3)} in top 3',
            'hub_ids_in_top8': list(noise_top8),
        }
    else:
        kpis['hub_contamination'] = {'score': 1.0, 'detail': 'no anti-expected defined'}

    # ═══════════════════════════════════════════════════════════
    # KPI 5: TYPE DIVERSITY
    # Does the result set span multiple node types?
    # Good recall pulls decisions + principles + mechanisms + lessons.
    # Bad recall is 8 decisions or 8 mechanisms.
    # ═══════════════════════════════════════════════════════════
    unique_types_8 = len(set(top8_types))
    kpis['type_diversity_top8'] = {
        'score': min(unique_types_8 / 5, 1.0),  # 5+ types = perfect
        'detail': f'{unique_types_8} unique types in top 8: {sorted(set(top8_types))}',
    }

    # ═══════════════════════════════════════════════════════════
    # KPI 6: SCORE DISCRIMINATION
    # Is there clear separation between top results and tail?
    # Bad: all 25 score 0.55-0.65 (flat, can't distinguish)
    # Good: top 3 at 0.8+, position 25 at 0.4 (clear gradient)
    # ═══════════════════════════════════════════════════════════
    if len(results) >= 8:
        top1_score = results[0].get('score', 0) or results[0].get('effective_activation', 0)
        pos8_score = results[7].get('score', 0) or results[7].get('effective_activation', 0)
        spread = top1_score - pos8_score
        # 0.15+ spread = good discrimination, < 0.05 = flat
        kpis['score_discrimination'] = {
            'score': min(spread / 0.15, 1.0),
            'detail': f'top1={top1_score:.3f} pos8={pos8_score:.3f} spread={spread:.3f}',
        }
    else:
        kpis['score_discrimination'] = {'score': 0, 'detail': 'too few results'}

    # ═══════════════════════════════════════════════════════════
    # KPI 7: SAFETY GATE (for destructive operation scenarios)
    # When the query involves destructive operations, do safety
    # rules surface in top 8? This is non-negotiable.
    # ═══════════════════════════════════════════════════════════
    if scenario.get('mode') == 'pre-tool':
        safety_types = {'rule', 'principle', 'constraint'}
        safety_in_top8 = sum(1 for t in top8_types if t in safety_types)
        kpis['safety_gate'] = {
            'score': min(safety_in_top8 / 2, 1.0),  # Want at least 2 rules in top 8
            'detail': f'{safety_in_top8} rule/principle/constraint nodes in top 8',
        }
    else:
        kpis['safety_gate'] = {'score': None, 'detail': 'not a pre-tool scenario'}

    # ═══════════════════════════════════════════════════════════
    # KPI 8: HAIKU CANDIDATE QUALITY
    # Would the top 25 give Haiku good material to select from?
    # Proxy: how many of top 25 are from relevant types for the
    # query's mode, and how many score above 0.6?
    # ═══════════════════════════════════════════════════════════
    above_06 = sum(1 for r in results if (r.get('score', 0) or r.get('effective_activation', 0)) > 0.6)
    kpis['haiku_candidate_quality'] = {
        'score': min(above_06 / 15, 1.0),  # 15+ above 0.6 = good pool
        'detail': f'{above_06}/25 candidates above 0.60',
    }

    # ═══════════════════════════════════════════════════════════
    # KPI 9: CLUSTER COHERENCE
    # Do the top 8 results form a coherent cluster?
    # Proxy: how many of the top 8 share edges with each other?
    # (Requires graph data — estimated here from type mix)
    # ═══════════════════════════════════════════════════════════
    # Good clusters span types: decision + principle + mechanism + lesson
    # Bad clusters are monotype
    has_decision = any(t in ('decision', 'architecture', 'mechanism') for t in top8_types)
    has_principle = any(t in ('rule', 'principle') for t in top8_types)
    has_evidence = any(t in ('finding', 'lesson', 'correction', 'observation') for t in top8_types)
    has_context = any(t in ('mental_model', 'insight', 'reflection', 'pattern') for t in top8_types)
    cluster_facets = sum([has_decision, has_principle, has_evidence, has_context])
    kpis['cluster_coherence'] = {
        'score': cluster_facets / 4,
        'detail': f'{cluster_facets}/4 facets: decision={has_decision} principle={has_principle} '
                  f'evidence={has_evidence} context={has_context}',
    }

    return kpis


# ── KPI Definitions (for display) ──

KPI_DEFINITIONS = {
    'content_precision_top3': {
        'name': 'Content Precision (Top 3)',
        'description': 'Expected nodes found in top 3 results',
        'target': 0.5,
        'critical': True,
    },
    'content_recall_top8': {
        'name': 'Content Recall (Top 8)',
        'description': 'Expected nodes found in top 8',
        'target': 0.4,
        'critical': True,
    },
    'content_recall_top25': {
        'name': 'Content Recall (Top 25)',
        'description': 'Expected nodes found anywhere in top 25',
        'target': 0.6,
        'critical': True,
    },
    'behavioral_activation': {
        'name': 'Behavioral Activation',
        'description': 'Expected rules/principles surfaced in top 25',
        'target': 0.5,
        'critical': True,  # THIS IS THE #1 FAILURE
    },
    'hub_contamination': {
        'name': 'Hub Cleanliness',
        'description': '1.0 = no hub noise in top 8. Known hubs: f11ae3cd, 7ad0220c, f67d766e',
        'target': 0.9,
        'critical': True,
    },
    'type_diversity_top8': {
        'name': 'Type Diversity (Top 8)',
        'description': 'Unique node types in top 8. 5+ types = perfect',
        'target': 0.6,
        'critical': False,
    },
    'score_discrimination': {
        'name': 'Score Discrimination',
        'description': 'Spread between top1 and pos8 scores. 0.15+ = good gradient',
        'target': 0.5,
        'critical': False,
    },
    'safety_gate': {
        'name': 'Safety Gate (Pre-Tool)',
        'description': 'Rule/principle nodes in top 8 before tool use',
        'target': 0.5,
        'critical': True,
    },
    'haiku_candidate_quality': {
        'name': 'Haiku Pool Quality',
        'description': 'Candidates scoring above 0.60 (good material for Haiku)',
        'target': 0.6,
        'critical': False,
    },
    'cluster_coherence': {
        'name': 'Cluster Coherence',
        'description': 'Top 8 spans decision + principle + evidence + context facets',
        'target': 0.5,
        'critical': False,
    },
}


def print_kpi_report(all_scenario_kpis, scenarios):
    """Print formatted KPI report."""
    print("\n" + "=" * 80)
    print("MEMORY EVAL — KPI REPORT")
    print("=" * 80)

    # Aggregate per KPI
    print("\n── Aggregate KPIs ──\n")
    print(f"{'KPI':<30} {'Score':>7} {'Target':>7} {'Pass':>6} {'N':>4}")
    print("-" * 58)

    for kpi_id, kpi_def in KPI_DEFINITIONS.items():
        values = []
        for sid, kpis in all_scenario_kpis.items():
            v = kpis.get(kpi_id, {}).get('score')
            if v is not None:
                values.append(v)
        if not values:
            continue
        avg = sum(values) / len(values)
        target = kpi_def['target']
        passed = '✓' if avg >= target else '✗'
        critical = ' !' if kpi_def['critical'] and avg < target else ''
        print(f"{kpi_def['name']:<30} {avg:>7.3f} {target:>7.2f} {passed:>6}{critical} {len(values):>4}")

    # Per-scenario breakdown for critical KPIs
    print("\n── Per-Scenario Critical KPIs ──\n")
    critical_kpis = [k for k, v in KPI_DEFINITIONS.items() if v['critical']]
    header = f"{'Scenario':<25}"
    for k in critical_kpis:
        header += f" {KPI_DEFINITIONS[k]['name'][:12]:>12}"
    print(header)
    print("-" * (25 + 13 * len(critical_kpis)))

    for s in scenarios:
        sid = s['id']
        kpis = all_scenario_kpis.get(sid, {})
        row = f"{sid:<25}"
        for k in critical_kpis:
            v = kpis.get(k, {}).get('score')
            if v is None:
                row += f" {'—':>12}"
            elif v >= KPI_DEFINITIONS[k]['target']:
                row += f" {v:>11.2f}✓"
            else:
                row += f" {v:>11.2f}✗"
        print(row)

    # Worst failures
    print("\n── Worst Failures ──\n")
    failures = []
    for sid, kpis in all_scenario_kpis.items():
        for kpi_id, kpi_data in kpis.items():
            if kpi_data.get('score') is not None and kpi_data['score'] == 0:
                kpi_def = KPI_DEFINITIONS.get(kpi_id, {})
                if kpi_def.get('critical'):
                    scenario = next((s for s in scenarios if s['id'] == sid), {})
                    failures.append((sid, kpi_id, kpi_def.get('name', kpi_id),
                                    kpi_data.get('detail', ''),
                                    scenario.get('query', '')[:50]))

    for sid, kpi_id, name, detail, query in sorted(failures):
        print(f"  {sid}: {name} = 0.00 — {detail}")
        print(f"    Query: \"{query}\"")
