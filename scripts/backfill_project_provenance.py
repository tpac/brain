#!/usr/bin/env python3
"""Package F — retroactive project provenance backfill.

Stamps `project` into node_metadata_kv for pre-stamping nodes by re-deriving
it from the originating session's persisted cwd, through session_env's
resolution semantics.

Linkage tiers (creation provenance, not aboutness):
  T1  node_source_refs -> trace_events.session_id (creation-anchored refs)
  T2  earliest s0/s1 delta-trace mention of the node id, accepted only when
      it falls within CREATION_WINDOW_H of nodes.created_at (a later mention
      is access, not birth)
Session -> cwd: session_state '_session_context' JSON blob (cwd persisted
since 2026-06-08; older sessions have none).
cwd -> project: live dirs run servers.session_env.detect_session_env (the
production chain: marker -> git dir name -> basename+denylist); dead dirs
(pruned worktrees) fall back to a path-prefix rule; junk anchors ($HOME,
/tmp, /bin, denylist basenames) yield no signal.

Modes:
  (default)          dry run — full coverage report, zero writes
  --dump-sessions F  dry run + write per-session classification worksheet
                     (JSON) for sessions the cwd chain can't decide
  --apply MAP.json   WRITE run. Refuses if the daemon answers PING or the
                     maintenance lock is absent. MAP.json may add
                     {"session_overrides": {session_id: project}} from the
                     operator-classified worksheet. Stamps:
                       project = <value>
                       _sys_project_by = migration:project-backfill-<date>
                     encoding_source is never touched.

Run: ./dev python3 scripts/backfill_project_provenance.py
"""
import argparse
import collections
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.daemon_client import is_daemon_responsive
from servers.session_env import detect_session_env, project_from_cwd_basename  # noqa: E402

DB_DIR = os.environ.get('BRAIN_DB_DIR') or os.path.expanduser('~/AgentsContext/brain')
BRAIN_DB = os.path.join(DB_DIR, 'brain.db')
LOGS_DB = os.path.join(DB_DIR, 'brain_logs.db')

CREATION_WINDOW_H = 48
MIGRATION_TAG = 'migration:project-backfill-%s' % datetime.now(timezone.utc).strftime('%Y%m%d')
HEX8 = re.compile(r'\b[0-9a-f]{8}\b')
HOME = os.path.realpath(os.path.expanduser('~'))
# Session cwds that assert nothing about identity (mirrors the production
# junk-anchor stance; /bin covers launcher-spawned sessions).
JUNK_CWDS = {HOME, '/', '/bin', '/usr/bin'}


def ro(path):
    return sqlite3.connect('file:%s?mode=ro' % path, uri=True)


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        return None


def resolve_cwd(cwd, cache={}):
    """cwd -> (project|None, how). Production chain for live dirs; prefix
    fallback for dead ones; junk anchors -> None."""
    if cwd in cache:
        return cache[cwd]
    p = os.path.realpath(cwd)
    if p in JUNK_CWDS or p.startswith(('/tmp', '/private/tmp', '/var/tmp')):
        out = (None, 'junk-anchor')
    elif os.path.isdir(p):
        _branch, _wt, project = detect_session_env(p)
        out = (project or None, 'live-chain') if project else (None, 'live-chain-unscoped')
    else:
        # Dead dir (e.g. pruned worktree): the tree it lived under names it.
        marker = '/.claude/worktrees/'
        if marker in p:
            root = p.split(marker)[0]
            out = (os.path.basename(root), 'dead-worktree-prefix')
        else:
            base = project_from_cwd_basename(p)
            out = (base or None, 'dead-dir-basename')
    cache[cwd] = out
    return out


def load_state():
    b, l = ro(BRAIN_DB), ro(LOGS_DB)

    stamped = {r[0] for r in b.execute(
        "SELECT DISTINCT node_id FROM node_metadata_kv WHERE key='project'")}
    nodes = {r[0]: r[1] or '' for r in b.execute(
        'SELECT id, created_at FROM nodes WHERE archived=0')}
    target = {n: ts for n, ts in nodes.items() if n not in stamped}

    # T1: source_refs
    refs = collections.defaultdict(list)
    for nid, tid in b.execute('SELECT node_id, trace_id FROM node_source_refs'):
        if nid in target:
            refs[nid].append(tid)
    tids = list({t for ts in refs.values() for t in ts})
    t2s = {}
    for i in range(0, len(tids), 400):
        ch = tids[i:i + 400]
        q = ','.join('?' * len(ch))
        for tid, sid in l.execute(
                'SELECT id, session_id FROM trace_events WHERE id IN (%s)' % q, ch):
            if sid:
                t2s[tid] = sid
    node_session = {}
    for nid, ts in refs.items():
        sids = sorted({t2s[t] for t in ts if t in t2s})
        if sids:
            node_session[nid] = (sids[0], 't1-source_refs')

    # T2: earliest delta mention within the creation window
    t2_candidates = {n for n in target if n not in node_session}
    earliest = {}  # nid -> (event_dt, session_id)
    for sid, ref, summ, meta, created in l.execute(
            "SELECT session_id, ref_id, summary, metadata, created_at "
            "FROM trace_events WHERE event_type='delta' AND scale IN ('s0','s1')"):
        if not sid:
            continue
        dt = parse_ts(created)
        if dt is None:
            continue
        blob = '%s %s %s' % (ref or '', summ or '', meta or '')
        for m in set(HEX8.findall(blob)):
            if m in t2_candidates:
                cur = earliest.get(m)
                if cur is None or dt < cur[0]:
                    earliest[m] = (dt, sid)
    t2_rejected = 0
    for nid, (dt, sid) in earliest.items():
        node_dt = parse_ts(target[nid])
        if node_dt and abs((dt - node_dt).total_seconds()) <= CREATION_WINDOW_H * 3600:
            node_session[nid] = (sid, 't2-delta-window')
        else:
            t2_rejected += 1

    # session -> cwd
    s2cwd = {}
    for sid, v in l.execute(
            "SELECT session_id, value FROM session_state WHERE key='_session_context'"):
        try:
            d = json.loads(v) if v else {}
        except ValueError:
            continue
        if d.get('cwd'):
            s2cwd[sid] = d['cwd']

    b.close()
    l.close()
    return nodes, stamped, target, node_session, s2cwd, t2_rejected


def classify(target, node_session, s2cwd, plan=None):
    """plan (operator-authored map file): node_overrides {nid: project} beat
    everything; session_overrides {sid: project} beat cwd-less fallthrough;
    session_default / orphan_default sweep the remainder. Returns
    (stamps {nid: (project, how)}, needs_operator {sid: [nids]},
    orphans [nid])."""
    plan = plan or {}
    node_ov = plan.get('node_overrides', {})
    sess_ov = plan.get('session_overrides', {})
    sess_default = plan.get('session_default', '')
    orphan_default = plan.get('orphan_default', '')
    stamps, needs_operator, orphans = {}, collections.defaultdict(list), []
    for nid in target:
        if nid in node_ov:
            stamps[nid] = (node_ov[nid], 'node-override')
            continue
        link = node_session.get(nid)
        if not link:
            orphans.append(nid)
            if orphan_default:
                stamps[nid] = (orphan_default, 'orphan-default')
            continue
        sid, how = link
        cwd = s2cwd.get(sid)
        if cwd:
            project, res = resolve_cwd(cwd)
            if project:
                stamps[nid] = (project, '%s/%s' % (how, res))
                continue
        # cwd absent or junk/unscoped — operator classification territory
        if sid in sess_ov:
            stamps[nid] = (sess_ov[sid], 'session-override')
        elif sess_default:
            stamps[nid] = (sess_default, 'session-default')
        else:
            needs_operator[sid].append(nid)
    return stamps, needs_operator, orphans


def report(nodes, stamped, target, node_session, s2cwd, t2_rejected,
           stamps, needs_operator, orphans):
    print('=' * 64)
    print('Package F backfill — DRY RUN coverage report')
    print('=' * 64)
    print('non-archived nodes: %d | already stamped: %d | target: %d'
          % (len(nodes), len(nodes.keys() & stamped), len(target)))
    tiers = collections.Counter(how for _sid, how in node_session.values())
    print('session linkage: %s | t2 rejected (mention outside %dh window): %d'
          % (dict(tiers), CREATION_WINDOW_H, t2_rejected))
    print()
    print('-- STAMPABLE: %d nodes --' % len(stamps))
    per_proj = collections.Counter(p for p, _ in stamps.values())
    for p, n in per_proj.most_common():
        print('  %5d  %s' % (n, p))
    per_how = collections.Counter(h.split('/')[-1] for _, h in stamps.values())
    print('  by method: %s' % dict(per_how.most_common()))
    print()
    n_no = sum(len(v) for v in needs_operator.values())
    print('-- NEEDS OPERATOR (session known, cwd absent/junk/unscoped): '
          '%d nodes across %d sessions --' % (n_no, len(needs_operator)))
    print()
    bym = collections.Counter((target[n] or '?')[:7] for n in orphans)
    print('-- UNRESOLVABLE orphans (no creation-anchored session): %d --'
          % len(orphans))
    print('  by month: %s' % dict(sorted(bym.items())))


def dump_sessions(path, needs_operator, s2cwd, target):
    """Worksheet: per session — cwd, node count, first user message, sample
    node titles. Operator fills 'project' per session."""
    l = ro(LOGS_DB)
    b = ro(BRAIN_DB)
    out = []
    for sid, nids in sorted(needs_operator.items(), key=lambda kv: -len(kv[1])):
        first_msgs = [r[0] for r in l.execute(
            "SELECT summary FROM trace_events WHERE session_id=? AND scale='s0' "
            "AND event_type='K' AND ref_type='user_message' "
            "ORDER BY created_at LIMIT 3", (sid,))]
        q = ','.join('?' * min(len(nids), 6))
        titles = [r[0] for r in b.execute(
            'SELECT title FROM nodes WHERE id IN (%s)' % q, nids[:6])]
        out.append({
            'session_id': sid,
            'cwd': s2cwd.get(sid, ''),
            'node_count': len(nids),
            'first_user_messages': first_msgs,
            'sample_node_titles': titles,
            'project': '',   # operator fills: brain | ex.co | personal | (skip)
        })
    with open(path, 'w') as f:
        json.dump({'session_overrides_worksheet': out}, f, indent=2)
    l.close()
    b.close()
    print('\nworksheet written: %s (%d sessions)' % (path, len(out)))


def daemon_alive():
    """The one liveness answer (daemon_config owns the address, daemon_client
    the wire). Not guarded: this module already imports from servers/ at import
    time, so a broken path would have killed it long before here."""
    return is_daemon_responsive(timeout=2)


def apply_stamps(stamps):
    lock = '/tmp/brain-maintenance-%d.lock' % os.getuid()
    if daemon_alive():
        sys.exit('REFUSING: daemon is alive. Stop it first (maintenance lock '
                 '+ launchctl unload).')
    if not os.path.exists(lock):
        sys.exit('REFUSING: maintenance lock %s absent.' % lock)
    bak = sorted(f for f in os.listdir(DB_DIR)
                 if f.startswith('brain.db.bak-')
                 or (f.startswith('brain.db.')
                     and f.endswith(('.bak', '.bak.gz'))))
    # The rolling GFS snapshots count too — any restorable image will do.
    backups_dir = os.path.join(DB_DIR, 'backups')
    if os.path.isdir(backups_dir):
        bak += sorted('backups/' + f for f in os.listdir(backups_dir)
                      if f.startswith('brain.db.') and f.endswith('.gz'))
    if not bak:
        sys.exit('REFUSING: no brain.db backup found in %s '
                 '(.bak/.bak.gz beside the DB or backups/*.gz).' % DB_DIR)
    print('newest backup: %s' % bak[-1])

    from servers.db_backends import current as db_backend
    from servers.dal_metadata import MetadataDAL
    conn = sqlite3.connect(BRAIN_DB)
    db_backend.apply_pragmas(conn)
    dal = MetadataDAL(conn)
    by_project = collections.defaultdict(dict)
    for nid, (project, _how) in stamps.items():
        by_project[project][nid] = project
    total = 0
    with conn:
        for project, nv in by_project.items():
            total += dal.bulk_set_key('project', nv)
            dal.bulk_set_key('_sys_project_by',
                             {nid: MIGRATION_TAG for nid in nv})
    conn.close()
    print('stamped project on %d nodes (%s)' % (total, MIGRATION_TAG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-sessions', metavar='FILE')
    ap.add_argument('--plan', metavar='MAP_JSON',
                    help='operator classification map (see classify())')
    ap.add_argument('--apply', action='store_true',
                    help='WRITE the stamps (requires stopped daemon, '
                         'maintenance lock, fresh backup)')
    args = ap.parse_args()

    plan = {}
    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        for row in plan.get('session_overrides_worksheet', []):
            if row.get('project'):
                plan.setdefault('session_overrides', {})[
                    row['session_id']] = row['project']

    nodes, stamped, target, node_session, s2cwd, t2rej = load_state()
    stamps, needs_operator, orphans = classify(
        target, node_session, s2cwd, plan)
    report(nodes, stamped, target, node_session, s2cwd, t2rej,
           stamps, needs_operator, orphans)

    if args.dump_sessions:
        dump_sessions(args.dump_sessions, needs_operator, s2cwd, target)
    if args.apply:
        apply_stamps(stamps)


if __name__ == '__main__':
    main()
