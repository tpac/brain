# tmemory Detailed API Reference

## Upgrade Guide for Future Claude

1. **ALWAYS back up first**: `POST /backup`
2. **Version gate**: Add `if (fromVersion < N)` in `_migrate()` in brain.js. Increment `BRAIN_VERSION`.
3. **Table rebuilds**: Disable FK with `PRAGMA foreign_keys=OFF` before DROP/RENAME. Re-enable after. This is CRITICAL — CASCADE will delete all edges otherwise.
4. **Test on backup**: Copy brain.db, test migration, then apply to real DB.
5. **Migration history**: Every migration logged in `version_history`. Never delete.
6. **Rollback**: Backups in data dir as `brain_backup_v*`. Stop server, replace brain.db, restart.
7. **DO NOT** change existing migration steps — only add new ones.
8. **DO NOT** modify locked nodes without user confirmation.
9. **Emotion calibration**: The `emotion_calibration` table contains the user's explicit feedback. Use this data to improve auto-emotion detection.

## Scoring Formula (v3)

**Regular nodes:** 35% relevance + 30% recency + 25% emotion + 10% frequency

**Locked nodes:** 50% relevance + 25% emotion + 20% recency + 5% frequency

**Emotion score:** `EMOTION_FLOOR (0.3) + raw_emotion * (1 - EMOTION_FLOOR)`

**Emotion retention boost:** If emotion > 0.5, retention multiplied by `(1 + emotion * 0.5)`

## Recency Bands

| Hours ago | Score |
|-----------|-------|
| < 1h | 1.0 |
| < 6h | 0.9 |
| < 24h | 0.75 |
| < 3 days | 0.5 |
| < 1 week | 0.3 |
| < 1 month | 0.15 |
| older | 0.05 |

## Node Type Decay Rates

| Type | Half-life | Notes |
|------|-----------|-------|
| person | 30 days | |
| project | 30 days | |
| decision | Infinity | Never decays if locked |
| rule | Infinity | Never decays |
| concept | 7 days | |
| task | 2 days | |
| file | 7 days | |
| context | 1 day | Session-specific |
| intuition | 12 hours | Dream-generated, fades unless accessed |

## Database Schema

Tables: `nodes`, `edges`, `access_log`, `brain_meta`, `summaries`, `version_history`, `emotion_calibration`, `dream_log`

Key columns on nodes: id, type, title, content, keywords, activation, stability, access_count, locked, archived, recency_score, emotion, emotion_label, emotion_source, last_accessed, created_at, updated_at

Key columns on edges: source_id, target_id, weight, relation, co_access_count, stability, last_strengthened, created_at

## Dreaming Mechanics

1. Pick 20 candidate seed nodes biased toward high-emotion, recent
2. Select 2 random seeds per dream (3 dreams per session)
3. Random walk 5 steps from each seed (weighted by edge weight)
4. If walk endpoints aren't already connected, create `intuition` node bridging them
5. Intuition nodes decay in 12h unless someone accesses them
6. Dream logged in `dream_log` table

## Hebbian Learning

Co-accessed memories get stronger connections. When multiple nodes appear in the same recall result:
- Edge weight increases by `LEARNING_RATE * 0.1`
- Co-access count increments
- Stability increases by `STABILITY_BOOST (1.5×)`

## v4: Self-Improvement Tables

### recall_log
Tracks every recall event. `returned_ids` logged automatically. `used_ids` and `precision_score` filled in by Claude via `/mark-recall-used`.

### miss_log
Tracks retrieval failures. Signals: `repetition` (worst — auto-locks node), `correction`, `explicit_miss`, `stale_recall`.

### eval_snapshots
Periodic evaluation results. Stores precision, coverage, dream hit rate, emotion accuracy, recommendations.

### tuning_log
Parameter change history. Every weight adjustment is recorded with reason and linked to the eval snapshot that triggered it.

### Metrics Definitions

- **Recall precision**: `used_count / returned_count` — what fraction of returned nodes were actually useful
- **Recall coverage**: `successful_recalls / (successful_recalls + misses)` — how often the brain finds what it should
- **Dream hit rate**: `intuitions_accessed / total_intuitions_created` — are dreams producing useful associations
- **Emotion accuracy**: `1 - avg(|auto_emotion - user_emotion|)` — how well auto-tagging matches user feedback

### Improvement Thresholds

| Metric | Good | Needs work | Critical |
|--------|------|-----------|----------|
| Recall precision | > 50% | 30-50% | < 30% |
| Recall coverage | > 80% | 60-80% | < 60% |
| Dream hit rate | > 10% | 5-10% | < 5% |
| Repetition misses | 0 | 1-2 | > 2 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| TMEMORY_PORT | 7437 | TCP port |
| TMEMORY_DB_DIR | ../data | Directory for brain.db |
| TMEMORY_SOCKET | /tmp/tmemory.sock | Unix socket path (if TCP disabled) |
| TMEMORY_TCP | 1 | Use TCP (default: yes) |
