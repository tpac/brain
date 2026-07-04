#!/usr/bin/env python3
"""Grow the Anchor 'mind' visualization from the REAL brain graph.

Reads the three _realgraph_*.json dumps (produced via daemon eval, read-only),
maps every node/edge onto the 16-aspect taxonomy, runs an offline 3D
force-settle seeded by real community + aspect structure, and emits a
self-contained HTML artifact with the layout baked in (circles + real edges).

Re-runnable tooling — safe, touches no DB. Run: ./dev python3 build_mind.py
"""
import json, math, random, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ASPECTS_JSON = "/Users/tpac/AgentsContext/brain/aspects_v1.json"

# ── canonical aspect order (JSON order = index) ──
ASPECT_ORDER = ['identity_bearing','episodic_anchor','active_thread','lesson_insight',
    'generic_relation','noise','correction_improvement','extension_refinement',
    'explanation_causation','dependency_flow','contradiction_conflict','validation_evidence',
    'hierarchical_structure','temporal_sequence','survivor_lineage','wisdom']
AI = {n:i for i,n in enumerate(ASPECT_ORDER)}

asp = json.load(open(ASPECTS_JSON))
# first-claimant maps (JSON order)
type2asp, rel2asp = {}, {}
for name in ASPECT_ORDER:
    a = asp[name]
    for t in a.get('node_types', []):
        type2asp.setdefault(t, AI[name])
    for r in a.get('edge_relations', []):
        rel2asp.setdefault(r, AI[name])

DEFAULT_NODE_ASP = AI['lesson_insight']
DEFAULT_REL_ASP  = AI['generic_relation']

# ── load real graph ──
nd = json.load(open(HERE/"_realgraph_nodes.json"))
ed = json.load(open(HERE/"_realgraph_edges.json"))
cm = json.load(open(HERE/"_realgraph_comm.json"))

ids   = nd['ids']
rows  = nd['n']            # [type, access_count, last_accessed, created_at]
degd  = nd['deg']
titles= json.load(open(HERE/"_realgraph_titles.json"))
N = len(ids)
idx = {nid:i for i,nid in enumerate(ids)}
comm_ids = set(c[0] for c in cm['comms'])

# per-node primary aspect, degree, heat
asp_of = np.zeros(N, dtype=np.int32)
deg    = np.zeros(N, dtype=np.float32)
acc    = np.zeros(N, dtype=np.float32)
recency= np.zeros(N, dtype=np.float32)   # 0 (cold) .. 1 (recently touched)

def days_ago(ts):
    if not ts: return 9999
    s = ts.replace('Z','+00:00').replace(' ','T')
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s)
        now = datetime(2026,7,3, tzinfo=timezone.utc)
        return max(0.0,(now-dt.astimezone(timezone.utc)).total_seconds()/86400)
    except Exception:
        return 9999

for i,(t,ac,last,created) in enumerate(rows):
    asp_of[i] = type2asp.get(t, DEFAULT_NODE_ASP)
    deg[i]    = degd.get(ids[i], 0)
    acc[i]    = ac or 0
    d = days_ago(last)
    recency[i] = math.exp(-d/45.0)        # 45-day half-lifeish warmth

# heat: blend of access volume (log) and recency
heat = 0.35*np.log1p(acc)/max(1e-6,np.log1p(acc).max()) + 0.65*recency
heat = np.clip(heat,0,1)

# ── community assignment (first membership; orientation-agnostic) ──
node_comm = [-1]*N          # index into community list
clist = [c[0] for c in cm['comms']]
cidx  = {cid:k for k,cid in enumerate(clist)}
for s,tt in cm['members']:
    # member is the non-community endpoint
    member = s if s not in comm_ids else tt
    community = tt if tt in comm_ids else s
    if member in idx and community in cidx and node_comm[idx[member]] == -1:
        node_comm[idx[member]] = cidx[community]

# community -> dominant aspect + member count
C = len(clist)
comm_asp = np.full(C, DEFAULT_NODE_ASP, dtype=np.int32)
comm_cnt = np.zeros(C, dtype=np.int32)
from collections import defaultdict, Counter
cbuck = defaultdict(list)
for i in range(N):
    if node_comm[i]>=0: cbuck[node_comm[i]].append(i)
for c,mem in cbuck.items():
    comm_cnt[c] = len(mem)
    comm_asp[c] = Counter(int(asp_of[i]) for i in mem).most_common(1)[0][0]

# ── 16 aspect poles on a Fibonacci sphere ──
def fib_sphere(n):
    pts=[]; ga=math.pi*(3-math.sqrt(5))
    for i in range(n):
        y=1-(i/(n-1))*2; r=math.sqrt(max(0,1-y*y)); th=ga*i
        pts.append((math.cos(th)*r, y, math.sin(th)*r))
    return np.array(pts,dtype=np.float32)
POLE = fib_sphere(16)

def h(s):   # deterministic hash -> [0,1)
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8],16)/0xffffffff

# ── build edge arrays (index space), map relation aspect ──
NOISE_ASP = AI['noise']; GENERIC=AI['generic_relation']
E = []      # (a,b,rel_asp)
seen=set()
for s,t,rel in ed:
    if s in idx and t in idx:
        a,b = idx[s],idx[t]
        if a==b: continue
        ra = rel2asp.get(rel, DEFAULT_REL_ASP)
        E.append((a,b,ra))
        seen.add((a,b) if a<b else (b,a))

pos = np.zeros((N,3),dtype=np.float32)
rng = np.random.default_rng(42)
active = [c for c in range(C) if comm_cnt[c]>0]

# ── SPIRAL-GALAXY LAYOUT ──────────────────────────────────────────────────────
# A sphere reads as a blob from every angle; a disc with a bright bulge and trailing
# arms is instantly a galaxy. The mapping is real: RADIUS = memory age — the oldest,
# most-consolidated knowledge forms the core; recent growth spirals out on the arms.
# Communities are clumps along the arms; hubs sink coreward; low-degree memories drift
# into a diffuse halo. Brightness is still recall-heat, colour is still memory-kind.
NARM = 2
WIND = 2.75            # turns of the spiral
DISC = 0.26           # z-flattening (disc thickness vs. radius)

# node age (days); older ⇒ larger age ⇒ smaller radius
age = np.array([days_ago(rows[i][3]) for i in range(N)],dtype=np.float64)
agek = np.clip(age,0,None)
lo,hi = np.percentile(agek,3), np.percentile(agek,99)
def age_t(x):          # 0 = ancient (core) .. 1 = brand new (arm tip)
    return float(np.clip(1.0-(np.clip(x,lo,hi)-lo)/max(1e-6,hi-lo),0,1))

# community age = median member age → its galactic radius band
comm_center=np.zeros((C,3),dtype=np.float32)
def spiral_point(t, arm, jr, ja, jz):
    rad   = 0.24 + 2.4*t
    theta = arm*(2*math.pi/NARM) + t*WIND*2*math.pi + (ja-0.5)*(0.13+0.20*t)  # crisp arm, little scatter
    x = rad*math.cos(theta); y = rad*math.sin(theta)
    px,py = -math.sin(theta),math.cos(theta)         # perpendicular = arm width
    w = (jr-0.5)*(0.05+0.18*t)
    z = jz*DISC*(0.5+0.55*t)*rad                      # thin disc, puffier core & rim
    return np.array([x+px*w, y+py*w, z],dtype=np.float32)

for c in active:
    mem=cbuck[c]
    t = age_t(float(np.median(age[mem])))
    arm = 0 if h(clist[c]+'arm')<0.5 else 1
    comm_center[c]=spiral_point(t, arm, h(clist[c]+'r'), h(clist[c]+'a'), (h(clist[c]+'z')-0.5)*2)

# separate the community centroids just enough to read as DISTINCT knots — local de-overlap
# only: short-range in-plane repulsion balanced by a spring back to each clump's spiral home,
# so they nudge apart into gaps but never drift off their arm.
ac = np.array([comm_center[c] for c in active], dtype=np.float64)
home = ac.copy()
for it in range(50):
    dx = ac[:,0][:,None]-ac[:,0][None,:]; dy = ac[:,1][:,None]-ac[:,1][None,:]
    d2 = dx*dx+dy*dy; near = (d2<0.09)&(d2>1e-9)   # only push neighbours that actually overlap
    inv = np.where(near, 1.0/(d2+2e-3), 0.0)
    fx=(dx*inv).sum(1); fy=(dy*inv).sum(1)
    ac[:,0]+=np.clip(fx*0.0016,-0.03,0.03)-(ac[:,0]-home[:,0])*0.10
    ac[:,1]+=np.clip(fy*0.0016,-0.03,0.03)-(ac[:,1]-home[:,1])*0.10
for i,c in enumerate(active): comm_center[c]=ac[i].astype(np.float32)

# members: TIGHT knot around centroid; keep them in the clump (weak coreward lean only)
for c,mem in cbuck.items():
    ctr=comm_center[c]; ball=0.03+0.075*math.log1p(len(mem))
    for i in mem:
        d=rng.normal(size=3); d[2]*=DISC; d/=np.linalg.norm(d)+1e-9
        core_pull=0.16*(deg[i]/(deg[i]+15.0))
        pos[i]=ctr*(1.0-core_pull) + d*ball*(0.35+0.65*rng.random())

# free nodes: faint interstitial dust — pushed a touch outward so the knots stand clear of it
free=[i for i in range(N) if node_comm[i]<0]
for i in free:
    t=age_t(age[i]); arm=0 if rng.random()<0.5 else 1
    p=spiral_point(t, arm, rng.random(), rng.random(), rng.normal()*0.8)
    pos[i]=p*1.06

# (no cross-node edge relax — it would pull the distinct knots back into a smear)

pos -= pos.mean(0)
maxr = np.linalg.norm(pos,axis=1).max()
pos /= maxr

# ── aspect mass (real) for the HUD/labels ──
node_mass = Counter(int(a) for a in asp_of)
edge_mass = Counter(r for _,_,r in E)

# ── choose which edges to actually draw (avoid hairball) ──
# draw: all inter-community bridges (structure) + sampled intra-community, skip pure noise
draw=[]
for i,(a,b,r) in enumerate(E):
    ca,cb = node_comm[a],node_comm[b]
    bridge = (ca!=cb)
    if r in (NOISE_ASP,):        # skip co_accessed / community_member lines
        continue
    if r==GENERIC and not bridge and random.Random(i).random()>0.25:
        continue
    if not bridge and random.Random(i).random()>0.55:
        continue
    draw.append((a,b,r,1 if bridge else 0))
random.Random(7).shuffle(draw)

# ── quantize + emit compact payload ──
def q(x): return round(float(x),4)
P = [[q(p[0]),q(p[1]),q(p[2])] for p in pos]
NODES = {
  "p": P,
  "a": [int(x) for x in asp_of],
  "d": [q(min(1.0, math.sqrt(deg[i])/13.4)) for i in range(N)],   # size 0..1 (sqrt-scaled, 180->1)
  "h": [q(heat[i]) for i in range(N)],
  "ti": [(t or "")[:90] for t in titles],                         # hover label
  "dg": [int(deg[i]) for i in range(N)],                          # real connection count
  "ag": [int(round(age[i])) for i in range(N)],                   # age in days
}
EDGES = [[a,b,r,br] for (a,b,r,br) in draw]

payload = {
  "N": N, "E_total": len(E), "E_drawn": len(EDGES), "C": int((comm_cnt>0).sum()),
  "nodes": NODES, "edges": EDGES,
  "node_mass": {ASPECT_ORDER[k]:v for k,v in sorted(node_mass.items())},
  "edge_mass": {ASPECT_ORDER[k]:v for k,v in sorted(edge_mass.items())},
}
json.dump(payload, open(HERE/"mind_data.json","w"))
print(f"nodes={N} edges_total={len(E)} edges_drawn={len(EDGES)} comms={int((comm_cnt>0).sum())} free={len(free)}")
print("node_mass:", payload["node_mass"])
print("edge_mass:", {k:v for k,v in sorted(edge_mass.items(), key=lambda kv:-kv[1])[:8] and [(ASPECT_ORDER[k],v) for k,v in sorted(edge_mass.items(), key=lambda kv:-kv[1])[:8]]})
sz = (HERE/"mind_data.json").stat().st_size
print(f"mind_data.json = {sz/1024:.0f} KB")
