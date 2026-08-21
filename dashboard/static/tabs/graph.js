// ===========================================================================
// tabs/graph.js — Canvas "spiral galaxy" renderer for the brain graph.
// ---------------------------------------------------------------------------
// Replaces the former 3D-force-graph (WebGL) renderer. Canvas 2D — so there is
// NO WebGL context to exhaust (the old ~16-context-per-tab flakiness is gone).
//
// Four fundamentals, laid out in JS on load: memory AGE, COMMUNITY, memory
// KIND, and CONNECTEDNESS (degree). A SHAPE (see SHAPES below) decides which
// of them owns position; kind always keeps color and degree always keeps size,
// so switching shape re-arranges the room without changing what anything is.
// Default `galaxy`: radius = age (oldest in the bright core, recent growth out
// on the arms), knots = community. Glow = RECALL HEAT (recency + volume).
//
// Live layer: a node lights up when the brain actually touched it — a recall
// recognized it, or the operator clicked it. NOTHING lights on a timer. An
// idle brain looks idle; the slow global `breath` is the resting state and is
// the only motion that isn't caused by something real. Each activation wave is
// tinted by the STREAM that caused it (shared palette with lib/sessions.js), so
// several sessions thinking at once stay tellable apart.
//
// Lifecycle contract (unchanged, drives app.js / live.js):
//   init() activate() deactivate() destroy() resize()
//   loadGraph3D() onGraphSearch() onGraphSearchKey() onGraphRefresh()
//   setSearchQuery() previewRecallOnGraph() clearRecallPreview() pinRecallToGraph()
//   setShape() getShape()
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { escapeHtml } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import { STREAM_RGB, sessionHueIndex } from '/static/lib/sessions.js';

// ── aspect taxonomy (first-claimant, generated from aspects_v1.json) ──
const TYPE2ASP={"principle":0,"identity":0,"vision":0,"rule":0,"operator":0,"capability":0,"directive":0,"design_principle":0,"procedure":0,"philosophy":0,"framework":0,"definition":0,"preference":0,"craft_rule":0,"design_direction":0,"moment":1,"anchor_quote":1,"user_quote":1,"quote":1,"catalyst":1,"milestone":1,"event":1,"interaction":1,"context":1,"incident":1,"corridor":1,"benchmark":1,"state":1,"resolution":1,"outcome":1,"episode":1,"time_anchor":1,"personal_context":1,"profile":1,"person":1,"interest":1,"case":1,"journal":1,"experiment":1,"open":2,"tension":2,"hypothesis":2,"aspiration":2,"uncertainty":2,"gap":2,"task":2,"status":2,"product-requirement":2,"problem":2,"requirement":2,"plan":2,"performance":2,"goal":2,"feature-requirement":2,"risk":2,"backlog":2,"proposal":2,"idea":2,"todo":2,"lesson":3,"insight":3,"validation":3,"meta_learning":3,"reflection":3,"decision":3,"fact":3,"finding":3,"mechanism":3,"architecture":3,"bug":3,"concept":3,"design":3,"pattern":3,"mental_model":3,"fn_reasoning":3,"analysis":3,"constraint":3,"research":3,"diagnosis":3,"convention":3,"impact":3,"code_concept":3,"reframe":3,"reference":3,"research-finding":3,"param_influence":3,"observation":3,"artifact":3,"fix":3,"discovery":3,"arch_constraint":3,"method":3,"landscape":3,"clarification":3,"audit":3,"recommendation_set":3,"resource_list":3,"knowledge":3,"reference_cluster":3,"technique":3,"reference_list":3,"overview":3,"taxonomy":3,"process":3,"direction":3,"distinction":3,"methodology":3,"community":5,"aspect":5,"intuition":5,"thought":5,"purpose":5,"file":5,"vocabulary":5,"test":5,"project":5,"correction":6,"bug_lesson":6};
const REL2ASP={"related_to":4,"related":4,"similar_to":4,"synthesizes":4,"complements":4,"community_member":5,"corrects":6,"corrected_by":6,"reframes":6,"resolves":6,"addresses":6,"consolidated_into":6,"updates":6,"fixes":6,"extends":7,"refines":7,"implements":7,"contextualizes":7,"applies":7,"instantiates":7,"operationalizes":7,"advances":7,"explains":8,"caused_by":8,"grounds":8,"produces":8,"informs":8,"triggers":8,"motivates":8,"depends_on":9,"enables":9,"requires":9,"blocks":9,"constrains":9,"configures":9,"contradicts":10,"violates":10,"challenges":10,"contrasts_with":10,"validates":11,"confirms":11,"demonstrates":11,"strengthens":11,"supports":11,"part_of":12,"includes":12,"contains":12,"supersedes":12,"abstracts":12,"follows":13,"leads_to":13,"after":13,"before":13,"during":13,"opens":13,"completes":13,"co_anchored":13,"absorbed_into":14};
const DEF_NODE_ASP=3, DEF_REL_ASP=4, NOISE_ASP=5, GENERIC_ASP=4;

const HUE=[[255,205,110],[255,150,170],[95,208,230],[110,222,165],[110,130,175],[80,92,120],
  [255,110,95],[150,220,150],[185,140,255],[245,190,110],[245,110,180],[130,230,200],
  [140,165,240],[120,182,242],[150,150,160],[255,233,176]];
const NODE_FAMS=[0,3,1,2,6,15,5];
const NF_LABEL={0:'identity',3:'lessons',1:'moments',2:'open threads',6:'corrections',15:'wisdom',5:'scaffolding'};
const KIND_NAME={0:'identity',1:'moment',2:'open thread',3:'lesson',5:'scaffolding',6:'correction',15:'wisdom'};

// ── module state ──
let G=null;                 // laid-out galaxy { N, X,Y,Z, A,D,H, DEG, TITLE, TYPE, COMM, AGE, IDS, idIndex, adj, edges }
let raw=null;               // last /api/graph3d payload (search fallback)
let cv=null, ctx=null, host=null, tip=null, legend=null;
let raf=0, _loadGen=0, _activateTimer=null, _ro=null;
let W=0,H2=0,DPR=1,MIN=0, yaw=0, pitch=0.62, roll=0, zoom=1, panx=0, pany=0, t0=0;
let spin=true, showEdges=false, colorMode=0, hoverIdx=-1;
// Set by buildGalaxy from node count — lifts dot size on a sparse brain so a
// young graph reads as present rather than as dust. 1.0 at ~300 nodes and up.
let _sparseBoost=1;
let needSort=true, _winWired=false, breath=0;
let act=null, front=[];
let _searchQuery='';
const _highlightTier=new Map();     // id -> tier (persistent spotlight; pin/preview)
let _highlightMode='latest', _pinnedEventId=null, _previewSnapshot=null;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

const FOCAL=4.2, CX=0.5, CY=0.5, FIT=0.66;
const HTIERS={ used:[255,255,255], activation:[80,255,150], returned:[130,190,255] };

// ── sprites ──
function sprite(r,g,b){const s=48,c=document.createElement('canvas');c.width=c.height=s;
  const x=c.getContext('2d'),gd=x.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
  gd.addColorStop(0,`rgba(${Math.min(r+55,255)},${Math.min(g+55,255)},${Math.min(b+55,255)},1)`);
  gd.addColorStop(.18,`rgba(${r},${g},${b},.9)`);gd.addColorStop(.44,`rgba(${r},${g},${b},.18)`);
  gd.addColorStop(1,`rgba(${r},${g},${b},0)`);x.fillStyle=gd;x.fillRect(0,0,s,s);return c;}
let SPR=null, WHITE=null, RED=null;
function ensureSprites(){ if(SPR)return; SPR=HUE.map(h=>sprite(h[0],h[1],h[2])); WHITE=sprite(255,244,214); RED=sprite(255,104,88); }

// ── helpers ──
const NOW = Date.now();
function daysAgo(ts){ if(!ts) return 9999; const s=String(ts).replace(' ','T').replace('Z','+00:00');
  const d=Date.parse(s); if(isNaN(d)) return 9999; return Math.max(0,(NOW-d)/86400000); }
function hstr(s){ let h=2166136261>>>0; s=String(s); for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);} return (h>>>0)/4294967296; }

// ── shapes ────────────────────────────────────────────────────────────────
// Four ways to stand in the same room. The graph has exactly four
// fundamentals to spend — AGE, COMMUNITY, DEGREE, KIND — and a layout is a
// decision about which one gets the strongest visual channel (position).
// Kind always keeps color, so a shape swap never changes what a dot IS.
//
//   galaxy        radius=age, clumps=community   → history: how the brain grew
//   constellations islands=community              → structure: what belongs together
//   strata        rows=age, columns=kind          → chronology: what formed when
//   core          radius=1/degree                 → gravity: what everything hangs off
//
// Each writes X/Y/Z in place and nothing else. Adding a fifth is one entry.
const DISC=0.26;
const SHAPE_LABEL={galaxy:'galaxy', constellations:'islands', strata:'strata', core:'core'};
// Camera each shape wants. A layout that encodes meaning in a flat axis has to
// be SEEN flat — strata's whole point is reading time down the y-axis, and the
// default three-quarter tilt (with roll drifting) destroys exactly that. The
// others are volumetric and read better in motion. Applied on shape change
// only, so the operator's own rotation is never yanked mid-look.
const SHAPE_VIEW={
  galaxy:         {pitch:0.62, roll:0,    spin:true},
  constellations: {pitch:0.62, roll:0,    spin:true},
  strata:         {pitch:0,    roll:0,    spin:false},
  core:           {pitch:0.35, roll:0,    spin:true},
};
const SHAPE_ORDER=['galaxy','constellations','strata','core'];
const SHAPE_KEY='dashboard.graphShape';
let _shape='galaxy';
try { const sv=localStorage.getItem(SHAPE_KEY); if(sv&&SHAPE_ORDER.includes(sv)) _shape=sv; } catch(e){}

const SHAPES={

  // Spiral arms wound by age, each community a knot riding an arm. The
  // original — history as the organizing principle.
  galaxy({N,X,Y,Z,AGE,DEG,COMM,cmap,aget}){
    const WIND=2.75;
    function spiral(t,arm,jr,ja,jz){ const rad=0.24+2.4*t, th=arm*Math.PI+t*WIND*2*Math.PI+(ja-0.5)*(0.13+0.2*t);
      const x=rad*Math.cos(th), y=rad*Math.sin(th), px=-Math.sin(th),py=Math.cos(th), w=(jr-0.5)*(0.05+0.18*t);
      return [x+px*w, y+py*w, jz*DISC*(0.5+0.55*t)*rad]; }
    // community centroids on the spiral (by median member age)
    const cids=[...cmap.keys()], cc={}, cx=[],cy=[];
    for(const c of cids){ const mem=cmap.get(c); const med=mem.map(i=>AGE[i]).sort((a,b)=>a-b)[mem.length>>1];
      const p=spiral(aget(med), hstr(c+'m')<0.5?0:1, hstr(c+'r'), hstr(c+'a'), (hstr(c+'z')-0.5)*2);
      cc[c]=[p[0],p[1],p[2]]; cx.push(p[0]); cy.push(p[1]); }
    // local in-plane repulsion + spring home → distinct knots
    const hx=cx.slice(), hy=cy.slice(), n=cids.length;
    for(let it=0;it<45;it++){ for(let i=0;i<n;i++){ let fx=0,fy=0;
      for(let j=0;j<n;j++){ if(i===j)continue; const dx=cx[i]-cx[j],dy=cy[i]-cy[j],d2=dx*dx+dy*dy;
        if(d2<0.09&&d2>1e-9){ const inv=1/(d2+2e-3); fx+=dx*inv; fy+=dy*inv; } }
      cx[i]+=Math.max(-0.03,Math.min(0.03,fx*0.0016))-(cx[i]-hx[i])*0.1;
      cy[i]+=Math.max(-0.03,Math.min(0.03,fy*0.0016))-(cy[i]-hy[i])*0.1; } }
    for(let k=0;k<n;k++){ cc[cids[k]][0]=cx[k]; cc[cids[k]][1]=cy[k]; }
    const rnd=mulberry(12345);
    for(const c of cids){ const mem=cmap.get(c), ctr=cc[c], ball=0.03+0.075*Math.log1p(mem.length);
      for(const i of mem){ let dx=randn(rnd),dy=randn(rnd),dz=randn(rnd)*DISC; const L=Math.hypot(dx,dy,dz)||1;
        const cp=0.16*(DEG[i]/(DEG[i]+15)); const rr=ball*(0.35+0.65*rnd());
        X[i]=ctr[0]*(1-cp)+dx/L*rr; Y[i]=ctr[1]*(1-cp)+dy/L*rr; Z[i]=ctr[2]*(1-cp)+dz/L*rr; } }
    // free nodes ride the arms as faint dust
    for(let i=0;i<N;i++){ if(COMM[i]!=null) continue; const p=spiral(aget(AGE[i]), rnd()<0.5?0:1, rnd(), rnd(), randn(rnd)*0.8);
      X[i]=p[0]*1.06; Y[i]=p[1]*1.06; Z[i]=p[2]*1.06; }
  },

  // Communities as separated islands on a phyllotaxis spiral (sunflower
  // packing — even spacing at any count), sized by membership. Age is dropped
  // from position entirely, so what you see is pure belonging: how many
  // narratives there are, how big, and how much loose material floats between.
  constellations({N,X,Y,Z,DEG,COMM,cmap}){
    const cids=[...cmap.keys()], n=cids.length, rnd=mulberry(9871);
    const GA=Math.PI*(3-Math.sqrt(5));   // golden angle
    const DISC_R=2.25;
    // Island radius is derived from the ACTUAL spacing, not a fixed constant.
    // Sunflower packing puts n points ~DISC_R/sqrt(n) apart, so a fixed ball
    // size that reads well at 30 communities merges into a single cloud at
    // 726 (this brain). Tying the ball to the spacing keeps islands separate
    // at any count, and keeps the size ratio between a 160-member narrative
    // and a 6-member one visible.
    const spacing=DISC_R/Math.sqrt(Math.max(1,n));
    let maxMem=1; for(const c of cids) maxMem=Math.max(maxMem,cmap.get(c).length);
    const lm=Math.log1p(maxMem);
    const cc={};
    // Largest islands outward, where the ring has the most room for them.
    const order=cids.slice().sort((a,b)=>cmap.get(a).length-cmap.get(b).length);
    order.forEach((c,k)=>{
      const t=n>1?(k+0.5)/n:0.5, rad=0.22+(DISC_R-0.22)*Math.sqrt(t), th=k*GA;
      cc[c]=[rad*Math.cos(th), rad*Math.sin(th), (hstr(c+'z')-0.5)*DISC*1.1*rad];
    });
    for(const c of cids){ const mem=cmap.get(c), ctr=cc[c];
      const ball=spacing*(0.20+0.30*(Math.log1p(mem.length)/lm));
      for(const i of mem){ let dx=randn(rnd),dy=randn(rnd),dz=randn(rnd)*0.7; const L=Math.hypot(dx,dy,dz)||1;
        // Hubs pull toward their island's center; leaves sit on the rim.
        const cp=0.55*(DEG[i]/(DEG[i]+8)), rr=ball*(1-cp)*(0.35+0.65*rnd());
        X[i]=ctr[0]+dx/L*rr; Y[i]=ctr[1]+dy/L*rr; Z[i]=ctr[2]+dz/L*rr; } }
    // Unplaced memories form the halo — visibly OUTSIDE every narrative, and
    // clear of the outermost island, which is the honest picture of how much
    // the brain holds that no community has claimed.
    for(let i=0;i<N;i++){ if(COMM[i]!=null) continue;
      const th=rnd()*6.2832, rad=DISC_R+0.35+0.5*rnd();
      X[i]=rad*Math.cos(th); Y[i]=rad*Math.sin(th); Z[i]=randn(rnd)*DISC*0.8; }
  },

  // Time as an axis you can read: newest at the top, oldest at the bottom,
  // communities as vertical lanes. The one shape where "when did this happen"
  // is answerable by eye rather than by hover.
  strata({N,X,Y,Z,AGE,DEG,A,aget}){
    const rnd=mulberry(4242);
    // Lanes are KINDS, not communities. Community was the obvious choice and
    // the wrong one: this brain has 726 of them, which is 726 lanes across
    // four units of width — sub-pixel columns, unreadable. There are only
    // seven kinds, they're the same seven the legend already names, and
    // "which kinds of memory did I form, and when" is the question a
    // time axis can actually answer.
    const lanes=NODE_FAMS.length;
    const laneOf=new Map(); NODE_FAMS.forEach((a,k)=>laneOf.set(a,k));
    const laneX=(k)=>(lanes>1?(k/(lanes-1))*4-2:0);
    for(let i=0;i<N;i++){
      const t=aget(AGE[i]);                       // 1 = oldest, 0 = newest
      const k=laneOf.has(A[i])?laneOf.get(A[i]):lanes-1;
      // Hubs sit centered in their column, leaves spread — so the lane still
      // reads as a lane while showing where its weight is.
      const spread=0.30*(1-0.55*(DEG[i]/(DEG[i]+10)));
      X[i]=laneX(k)+randn(rnd)*spread;
      Y[i]=2.2-4.4*(1-t)+randn(rnd)*0.03;
      Z[i]=randn(rnd)*DISC*0.5;
    }
  },

  // Connectedness as gravity: the most-linked memories collapse to the core,
  // orphans drift to the shell. Answers "what is everything hanging off?" —
  // the question the age spiral can't.
  core({N,X,Y,Z,DEG,A}){
    const rnd=mulberry(777);
    let maxDeg=0; for(let i=0;i<N;i++) maxDeg=Math.max(maxDeg,DEG[i]);
    const norm=Math.max(1,Math.sqrt(maxDeg));
    for(let i=0;i<N;i++){
      const pull=Math.sqrt(DEG[i])/norm;          // 1 = biggest hub
      const rad=0.16+2.3*Math.pow(1-pull,1.35);
      // Fibonacci sphere, offset per aspect so kinds band rather than mix —
      // color still reads at the shell where the dots are densest.
      const k=i+0.5, phi=Math.acos(1-2*k/N), th=Math.PI*(1+Math.sqrt(5))*k+A[i]*0.9;
      X[i]=rad*Math.sin(phi)*Math.cos(th)+randn(rnd)*0.02;
      Y[i]=rad*Math.sin(phi)*Math.sin(th)+randn(rnd)*0.02;
      Z[i]=rad*Math.cos(phi)*(0.55+0.45*DISC)+randn(rnd)*0.02;
    }
  },
};

// ── galaxy layout (JS port of build_mind.py) ──
function buildGalaxy(data){
  const nodes=data.nodes||[], edgesRaw=data.edges||[];
  const N=nodes.length;
  const IDS=new Array(N), idIndex=new Map();
  for(let i=0;i<N;i++){ IDS[i]=nodes[i].id; idIndex.set(nodes[i].id,i); }
  const A=new Int32Array(N), DEG=new Float32Array(N), AGE=new Float64Array(N);
  const TITLE=new Array(N), TYPE=new Array(N), COMM=new Array(N);
  const acc=new Float64Array(N), rec=new Float64Array(N);
  for(let i=0;i<N;i++){ const n=nodes[i];
    A[i]=TYPE2ASP[n.type]!=null?TYPE2ASP[n.type]:DEF_NODE_ASP;
    TITLE[i]=n.name||n.id; TYPE[i]=n.type||''; COMM[i]=n.community||null;
    acc[i]=n.access_count||1; AGE[i]=daysAgo(n.created_at);
    rec[i]=Math.exp(-daysAgo(n.last_accessed||n.created_at)/45);
  }
  // degree from edges
  const E=[]; for(const e of edgesRaw){ const a=idIndex.get(e.source), b=idIndex.get(e.target);
    if(a==null||b==null||a===b) continue; DEG[a]++; DEG[b]++;
    E.push([a,b, REL2ASP[e.relation]!=null?REL2ASP[e.relation]:DEF_REL_ASP]); }
  // heat
  let maxAcc=1; for(let i=0;i<N;i++) maxAcc=Math.max(maxAcc,acc[i]); const la=Math.log1p(maxAcc);
  const H=new Float32Array(N), D=new Float32Array(N), L=new Float32Array(N);
  // Degree normalizes against THIS graph, not a constant. The old divisor
  // (13.4 ≈ sqrt(180)) was tuned to a 9k-node brain's max degree, so on a new
  // brain — where the best-connected node has a handful of edges — every D
  // landed near 0.1. Since D drives both dot SIZE and baseline BRIGHTNESS,
  // that is the whole "tiny and invisible on a fresh brain" bug: not a camera
  // problem, a normalization problem. Clamped so a big brain keeps its
  // existing look (13.4) and a 2-edge brain doesn't render every node maximal.
  let maxDeg=0; for(let i=0;i<N;i++) maxDeg=Math.max(maxDeg,DEG[i]);
  const degNorm=Math.max(3.2,Math.min(13.4,Math.sqrt(maxDeg)||1));
  // A small brain also has less TOTAL light on screen — fewer dots means the
  // whole field reads dim even with per-dot brightness fixed. Lift presence as
  // the population thins; at ~300 nodes and up this is a no-op.
  const sparse=Math.max(0,Math.min(1,(300-N)/300));
  _sparseBoost=1+1.35*sparse;
  for(let i=0;i<N;i++){ H[i]=Math.max(0,Math.min(1, 0.35*Math.log1p(acc[i])/la + 0.65*rec[i]));
    D[i]=Math.min(1, Math.sqrt(DEG[i])/degNorm);
    // baseline presence — every memory is here, lit by WHAT IT IS (kind+reach+warmth),
    // not by whether it's firing. this is the "I don't go dark" resting state.
    L[i]=Math.min(0.95, (0.30 + 0.42*H[i] + 0.34*D[i]) * (1+0.75*sparse)); }

  // communities
  const cmap=new Map();
  for(let i=0;i<N;i++){ const c=COMM[i]; if(c==null) continue; (cmap.get(c)||cmap.set(c,[]).get(c)).push(i); }
  // Radius carries AGE: oldest in the bright core, newest out on the arms.
  const ages=Array.from(AGE).filter(a=>a<9999).sort((a,b)=>a-b);
  const lo=ages.length?ages[Math.floor(ages.length*0.03)]:0, hi=ages.length?ages[Math.floor(ages.length*0.99)]:100;
  // A brain born today has NO age span: hi-lo collapses to ~0, every node maps
  // to the same t, and the spiral degenerates into a single ring. Fall back to
  // ordinal position (rank among all ages) when the span is under a day —
  // still age, just ordered rather than scaled, which is the only thing a
  // one-day-old brain can honestly say about its own history.
  const ageRank=new Map();
  if(hi-lo<1 && ages.length){ for(let k=0;k<ages.length;k++) if(!ageRank.has(ages[k])) ageRank.set(ages[k],k/Math.max(1,ages.length-1)); }
  const aget=ageRank.size
    ? (x=>1-(ageRank.has(x)?ageRank.get(x):0.5))
    : (x=>{ x=Math.max(lo,Math.min(hi,x)); return Math.max(0,Math.min(1,1-(x-lo)/Math.max(1e-6,hi-lo))); });
  const X=new Float32Array(N), Y=new Float32Array(N), Z=new Float32Array(N);
  // The shape is a CHOICE over the same four fundamentals — age, community,
  // degree, kind. Each layout below decides which of them owns radius, which
  // owns clustering, and which is left to color and size. Nothing downstream
  // (projection, picking, activation, highlight) knows which one ran.
  SHAPES[_shape] ? SHAPES[_shape]({N,X,Y,Z,AGE,DEG,COMM,A,cmap,aget})
                 : SHAPES.galaxy({N,X,Y,Z,AGE,DEG,COMM,A,cmap,aget});
  // recenter + normalize
  let mx=0,my=0,mz=0; for(let i=0;i<N;i++){mx+=X[i];my+=Y[i];mz+=Z[i];} mx/=N||1;my/=N||1;mz/=N||1;
  let mr=1e-3; for(let i=0;i<N;i++){X[i]-=mx;Y[i]-=my;Z[i]-=mz;mr=Math.max(mr,Math.hypot(X[i],Y[i],Z[i]));}
  for(let i=0;i<N;i++){X[i]/=mr;Y[i]/=mr;Z[i]/=mr;}
  // adjacency (non-noise) + drawn-edge subset
  const adj=Array.from({length:N},()=>[]);
  const draw=[]; let ei=0;
  for(const [a,b,r] of E){ if(r!==NOISE_ASP){ adj[a].push(b); adj[b].push(a); }
    if(r===NOISE_ASP) continue;
    const bridge=COMM[a]!==COMM[b]; ei++;
    // bridges always drawn; intra-community sampled (generic 1-in-4, other 1-in-2)
    if(!bridge){ if(r===GENERIC_ASP){ if(ei%4) continue; } else if(ei%2) continue; }
    draw.push([a,b,r,bridge?1:0]); }
  return { N, X,Y,Z, A, D, H, L, DEG, TITLE, TYPE, COMM, AGE, IDS, idIndex, adj, edges:draw,
           C:cmap.size, E_total:E.length };
}
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function randn(R){let u=0,v=0;while(!u)u=R();while(!v)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(6.2831853*v);}

// ── projection + render ──
let sx,sy,sp,sz,order,fired,hop,tender,streamOf;
function allocScreen(N){ sx=new Float32Array(N);sy=new Float32Array(N);sp=new Float32Array(N);sz=new Float32Array(N);
  order=new Int32Array(N); for(let i=0;i<N;i++)order[i]=i; act=new Float32Array(N); fired=new Uint8Array(N);
  hop=new Int8Array(N); hop.fill(-1); tender=new Float32Array(N);
  // Which stream of thought lit this node. Several streams think at once, and
  // an untinted bloom makes three simultaneous sessions look like one. Index
  // into the SAME palette lib/sessions.js gives the feed, so a wave in the
  // galaxy is visibly the stream whose activity is scrolling beside it.
  // 255 = "no stream" (nothing lit it, or the event carried no session).
  streamOf=new Uint8Array(N); streamOf.fill(255);
  front=[]; }
function project(){ const g=G; const cr=Math.cos(roll),sr=Math.sin(roll),cy=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),spp=Math.sin(pitch);
  const sc=(FIT*MIN)*zoom;
  for(let i=0;i<g.N;i++){ let x=g.X[i],y=g.Y[i],z=g.Z[i];
    let xr=x*cr-y*sr, yr=x*sr+y*cr; let rx=xr*cy-z*syw, rz=xr*syw+z*cy; let ry=yr*cp-rz*spp; rz=yr*spp+rz*cp;
    const pe=FOCAL/(FOCAL+rz); sx[i]=W*CX+panx+rx*sc*pe; sy[i]=H2*CY+pany+ry*sc*pe; sp[i]=pe; sz[i]=rz; } }

function matches(i){ if(!_searchQuery) return true; const q=_searchQuery;
  return (G.TITLE[i]||'').toLowerCase().includes(q) || (G.TYPE[i]||'').toLowerCase().includes(q); }

const MAXHOP=4;   // a recall lights a bounded neighbourhood, not the whole brain
function seed(i,hue){ if(i<0||i>=G.N)return; act[i]=Math.max(act[i],1); fired[i]=1; hop[i]=0;
  if(hue!=null&&hue>=0) streamOf[i]=hue;
  if(front.indexOf(i)<0)front.push(i); }
// A recall spreads like a signal firing through a network: a lit node CHARGES its
// fresh neighbours up over ~0.7s until they ignite and pass it on, out to MAXHOP
// then it stops and the whole thing fades. Each node fires once per wave (fired[]
// refractory → no feedback, no flash-all), then cools and re-arms. dt-scaled ⇒
// frame-rate independent; the front travels ~1-2 hops/s, blooms ~2s, fades by ~4s.
function stepActivation(dt){ if(!front.length)return; const nf=[], add=new Map(), hm=new Map(), sm=new Map();
  const decay=Math.pow(0.5,dt);
  for(const i of front){ const a=act[i], nb=G.adj[i], hi=hop[i];
    if(a>0.12 && hi<MAXHOP && nb.length){ const cap=Math.min(nb.length,5), h=hi+1;
      for(let k=0;k<cap;k++){ const j=nb[k]; if(!fired[j]){ add.set(j,(add.get(j)||0)+a*4.0*dt/cap);
        if(!hm.has(j)||hm.get(j)>h) hm.set(j,h);
        // The wave carries its stream outward — a spreading thought stays the
        // color of whoever is thinking it.
        if(streamOf[i]!==255) sm.set(j,streamOf[i]); } } }
    if(G.A[i]===6 && a>0.35) tender[i]=1;    // where I was wrong stays tender
    act[i]*=decay;
    if(act[i]>0.03) nf.push(i); else { fired[i]=0; hop[i]=-1; streamOf[i]=255; }   // cooled → re-armable next wave
  }
  for(const [j,v] of add){ act[j]=Math.min(1,act[j]+v); if(hop[j]<0) hop[j]=hm.get(j);
    if(sm.has(j)) streamOf[j]=sm.get(j);
    if(act[j]>0.6) fired[j]=1; if(nf.indexOf(j)<0)nf.push(j); }
  front=nf.slice(0,2500); }

function frame(now){ if(!G||!ctx){raf=0;return;}
  const dt=Math.min((now-t0)/1000,.05); t0=now;
  breath+=dt; const br=0.9+0.1*Math.sin(breath*0.9);   // a slow living breath — always present
  if(!reduce){ const tc=Math.pow(0.9993,dt*60); for(let i=0;i<G.N;i++) if(tender[i]>0.004) tender[i]*=tc; }
  if(spin&&!reduce){ roll+=dt*0.075; needSort=true; }
  stepActivation(dt);
  // NOTHING seeds activation here. A node lights up when — and only when —
  // the brain actually touched it: a recall recognized it, or an encode wrote
  // it. This used to fire a random high-heat node every 3.5–6.5s to keep the
  // galaxy looking busy, which meant the display's most load-bearing signal
  // ("something just happened") was decorative most of the time. An idle brain
  // should LOOK idle: the breath below is the resting state, and it is honest.
  project();
  ctx.clearRect(0,0,W,H2);
  ctx.globalCompositeOperation='lighter';
  const spot=_highlightTier.size>0;
  const g=G;
  if(showEdges){ const paths={},bp={};
    for(let e=0;e<g.edges.length;e++){ const a=g.edges[e][0],b=g.edges[e][1],r=g.edges[e][2],br=g.edges[e][3];
      const t=br?bp:paths; (t[r]||(t[r]=new Path2D())); const ax=sx[a],ay=sy[a],bx=sx[b],by=sy[b],mx=(ax+bx)/2,my=(ay+by)/2;
      t[r].moveTo(ax,ay); t[r].quadraticCurveTo(mx+(ay-by)*0.06,my+(bx-ax)*0.06,bx,by); }
    for(const r in paths){const h=HUE[r];ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},0.05)`;ctx.lineWidth=0.6;ctx.stroke(paths[r]);}
    for(const r in bp){const h=HUE[r];ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},0.13)`;ctx.lineWidth=0.8;ctx.stroke(bp[r]);} }

  if(needSort){ order.sort((i,j)=>sz[j]-sz[i]); needSort=false; }   // re-sort only when the view rotated
  // PASS 1 — the present body: every memory glowing by what it IS (kind+reach+warmth),
  // breathing. not heat-gated — nothing goes dark. a thought adds warmth on top.
  for(let k=0;k<g.N;k++){ const i=order[k],ac=act[i];
    const dim = (_searchQuery&&!matches(i)) || (spot&&!_highlightTier.has(i));
    const pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7));
    const s=(1.5+g.D[i]*10+ac*ac*24)*pe*1.7*_sparseBoost;
    ctx.globalAlpha=Math.min(0.4,(0.03+0.13*g.L[i])*br*(0.45+0.55*depth)+ac*0.34)*(dim?0.16:1);
    ctx.drawImage(ac>0.2?WHITE:SPR[g.A[i]],sx[i]-s,sy[i]-s,s*2,s*2); }
  // firing bloom — the crest of a thought, in the color of the stream thinking it
  for(let x=0;x<front.length;x++){const i=front[x],ac=act[i];if(ac<0.12)continue;const pe=sp[i],s=(3+ac*15)*pe;
    ctx.globalAlpha=Math.min(0.55,ac*0.65);
    ctx.drawImage(g.A[i]===6?RED:streamSprite(streamOf[i]),sx[i]-s,sy[i]-s,s*2,s*2);}
  // tenderness — where I was wrong stays a little warm, always
  for(let i=0;i<g.N;i++){ const td=tender[i]; if(td<0.02)continue; const pe=sp[i],s=(2+g.D[i]*3)*pe;
    ctx.globalAlpha=Math.min(0.2,td*0.14); ctx.drawImage(RED,sx[i]-s,sy[i]-s,s*2,s*2); }
  // PASS 2 — crisp cores: the structure, always here, source-over so it never blows out
  ctx.globalCompositeOperation='source-over';
  for(let k=0;k<g.N;k++){ const i=order[k],pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7)),heat=g.H[i],hh=heat*heat,ac=act[i];
    const dim=(_searchQuery&&!matches(i))||(spot&&!_highlightTier.has(i));
    const tier=spot?_highlightTier.get(i):null;
    const s=(1.3+g.D[i]*9+ac*4)*0.6*pe*(tier?1.5:1)*_sparseBoost;
    let spr,al;
    if(ac>0.15){spr=g.A[i]===6?RED:streamSprite(streamOf[i]);al=(0.4+0.55*ac)*(0.45+0.55*depth);}
    else if(tier){const c=HTIERS[tier]; spr=tierSprite(c); al=(0.55+0.4*depth);}
    else{spr=colorMode===0?SPR[g.A[i]]:(heat>0.62?WHITE:SPR[3]);
      al=(colorMode===0?(0.14+0.6*g.L[i])*br:(0.12+0.8*hh))*(0.45+0.55*depth);}
    if(dim && !tier && ac<0.15){ al*=0.16; }
    ctx.globalAlpha=Math.min(0.96,al); ctx.drawImage(spr,sx[i]-s,sy[i]-s,s*2,s*2); }
  // hover ring
  if(hoverIdx>=0 && hoverIdx<g.N){ const i=hoverIdx,h=HUE[g.A[i]];
    ctx.globalAlpha=0.9;ctx.strokeStyle='rgba(255,255,255,.9)';ctx.lineWidth=1.4;
    ctx.beginPath();ctx.arc(sx[i],sy[i],7+g.D[i]*10,0,6.2832);ctx.stroke();
    ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},.5)`;ctx.lineWidth=1;const nb=g.adj[i];
    for(let k=0;k<nb.length&&k<40;k++){const j=nb[k];ctx.beginPath();ctx.moveTo(sx[i],sy[i]);ctx.lineTo(sx[j],sy[j]);ctx.stroke();} }
  ctx.globalAlpha=1;
  raf=requestAnimationFrame(frame); }

const _tierSprites={};
function tierSprite(c){ const k=c.join(','); if(!_tierSprites[k])_tierSprites[k]=sprite(c[0],c[1],c[2]); return _tierSprites[k]; }
// Firing color per stream. 255 (untagged) falls back to the neutral white
// flash, so a recall with no session — an internal one — still reads as fired.
const _streamSprites={};
function streamSprite(h){ if(h===255||h>=STREAM_RGB.length) return WHITE;
  if(!_streamSprites[h]){ const c=STREAM_RGB[h]; _streamSprites[h]=sprite(c[0],c[1],c[2]); }
  return _streamSprites[h]; }

// ── picking + tooltip ──
function nearest(mx,my){ if(!G)return-1; let best=-1,bd=18*18;
  for(let i=0;i<G.N;i++){ if(sp[i]<0.4)continue; const dx=sx[i]-mx,dy=sy[i]-my,d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;} } return best; }
function ago(days){ if(days>=9999)return'unknown'; if(days<1)return'today'; if(days<30)return Math.round(days)+'d ago'; return Math.round(days/30)+'mo ago'; }
function recallWord(h){ return h>0.6?'recalled recently':h>0.3?'recalled this month':'resting'; }
function showTip(i,mx,my){ if(!tip)return; const h=HUE[G.A[i]],name=KIND_NAME[G.A[i]]||G.TYPE[i]||'memory';
  tip.querySelector('.gt-title').textContent=G.TITLE[i];
  tip.querySelector('.gt-meta').innerHTML=`<span class="gt-k" style="color:rgb(${h[0]},${h[1]},${h[2]});border-color:rgb(${h[0]},${h[1]},${h[2]})">${escapeHtml(name)}</span>`+
    `<span>${G.DEG[i]} connection${G.DEG[i]===1?'':'s'}</span><span>·</span><span>created ${ago(G.AGE[i])}</span><span>·</span><span>${recallWord(G.H[i])}</span>`;
  let x=mx+16,y=my+16; if(x>W-300)x=mx-290; if(y>H2-80)y=my-70;
  tip.style.left=x+'px'; tip.style.top=y+'px'; tip.classList.add('show'); }

// ── DOM scaffold inside #graph-3d ──
function buildScaffold(){
  host=document.getElementById('graph-3d'); if(!host) return false;
  host.innerHTML='';
  host.style.position='relative';
  cv=document.createElement('canvas'); cv.style.cssText='position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab;background:radial-gradient(140% 110% at 52% 44%,#0a0f24 0%,#06080f 44%,#04050c 100%)';
  host.appendChild(cv); ctx=cv.getContext('2d');
  tip=document.createElement('div'); tip.className='gt';
  tip.style.cssText='position:absolute;z-index:5;pointer-events:none;opacity:0;transition:opacity .12s;max-width:290px;padding:9px 11px;border-radius:9px;background:rgba(6,8,16,.94);border:1px solid #1a2138;box-shadow:0 10px 30px -12px #000';
  tip.innerHTML='<div class="gt-title" style="font:600 12.5px system-ui;color:#eef2fb;line-height:1.35;margin-bottom:6px"></div><div class="gt-meta" style="font:10.5px ui-monospace,Menlo,monospace;color:#67718e;display:flex;gap:6px;flex-wrap:wrap;align-items:center"></div>';
  host.appendChild(tip);
  legend=document.createElement('div');
  legend.style.cssText='position:absolute;left:10px;bottom:8px;z-index:4;display:flex;flex-wrap:wrap;gap:4px 8px;max-width:72%;pointer-events:none;font:10px ui-monospace,Menlo,monospace;color:#8791ab';
  legend.innerHTML=NODE_FAMS.map(a=>{const h=HUE[a];return `<span style="display:flex;align-items:center;gap:4px"><span style="width:7px;height:7px;border-radius:50%;background:rgb(${h[0]},${h[1]},${h[2]});box-shadow:0 0 5px rgb(${h[0]},${h[1]},${h[2]})"></span>${NF_LABEL[a]}</span>`;}).join('');
  host.appendChild(legend);
  // toggles overlay the canvas, banded ABOVE the legend so they never collide
  const ctl=document.createElement('div');
  ctl.style.cssText='position:absolute;right:10px;bottom:34px;z-index:4;display:flex;gap:6px';
  ctl.innerHTML=`<button data-g="shape" class="gbtn" title="How the graph is laid out — the same memories, arranged by a different fundamental">shape: ${SHAPE_LABEL[_shape]}</button><button data-g="color" class="gbtn">color: kind</button><button data-g="lines" class="gbtn">lines: off</button><button data-g="spin" class="gbtn">spin: on</button>`;
  host.appendChild(ctl);
  ctl.querySelectorAll('.gbtn').forEach(b=>{ b.style.cssText='border:1px solid #1a2138;background:rgba(8,10,20,.72);color:#c9d3e6;font:10px ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;border-radius:7px;padding:6px 9px;cursor:pointer;white-space:nowrap;backdrop-filter:blur(6px)';
    b.onclick=()=>{ const k=b.dataset.g;
      if(k==='shape'){ setShape(SHAPE_ORDER[(SHAPE_ORDER.indexOf(_shape)+1)%SHAPE_ORDER.length]); }
      if(k==='color'){colorMode^=1;b.textContent='color: '+(colorMode?'heat':'kind');}
      if(k==='lines'){showEdges=!showEdges;b.textContent='lines: '+(showEdges?'on':'off');}
      if(k==='spin'){spin=!spin;b.textContent='spin: '+(spin?'on':'off');} }; });
  wireCanvas();
  return true;
}
let drag=false,lx=0,ly=0,sh=false,moved=false;
// window-level pointer handlers: bound ONCE (in init) — they reference the
// current module cv/G/tip, which are reassigned per mount. Binding them per
// mount (the old wireCanvas) leaked a pair on every Refresh.
function _onWinMove(e){ if(!G)return;
  if(drag){const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;
    if(sh){panx+=dx;pany+=dy;}else{yaw+=dx*0.005;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.005));needSort=true;}return;}
  if(!cv)return; const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(mx<0||my<0||mx>r.width||my>r.height){hoverIdx=-1;if(tip)tip.classList.remove('show');return;}
  const i=nearest(mx,my); hoverIdx=i;
  if(i>=0){showTip(i,mx,my);cv.style.cursor='pointer';}else{if(tip)tip.classList.remove('show');cv.style.cursor='grab';} }
function _onWinUp(e){ if(!drag)return; drag=false; if(cv)cv.style.cursor='grab';
  if(!moved&&G){const r=cv.getBoundingClientRect(),i=nearest(e.clientX-r.left,e.clientY-r.top);
    if(i>=0){ seed(i); loadNodeDetail(G.IDS[i]); }} }
function wireWindowOnce(){ if(_winWired)return; _winWired=true;
  window.addEventListener('mousemove',_onWinMove); window.addEventListener('mouseup',_onWinUp); }
// canvas-scoped handlers re-bind per mount — the canvas is discarded on destroy,
// so these listeners are GC'd with it (no leak).
function wireCanvas(){
  cv.addEventListener('mousedown',e=>{drag=true;moved=false;lx=e.clientX;ly=e.clientY;sh=e.shiftKey;cv.style.cursor='grabbing';});
  cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.4,Math.min(6,zoom*(e.deltaY<0?1.12:.892)));},{passive:false});
}

function sizeCanvas(){ if(!cv||!host) return; DPR=Math.min(devicePixelRatio||1,2);
  W=host.clientWidth||800; H2=host.clientHeight||600; MIN=Math.min(W,H2);
  cv.width=W*DPR; cv.height=H2*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }

// ── load ──
export async function loadGraph3D(){
  const gen=_loadGen;
  if(!buildScaffold()) return;
  try {
    raw = await api.graph3d();
    if(gen!==_loadGen) return;
    if(!raw.nodes || !raw.nodes.length){ host.innerHTML='<div class="graph-error"><div class="graph-error-title">No graph data</div></div>'; return; }
    ensureSprites();
    G = buildGalaxy(raw);
    allocScreen(G.N); needSort=true;
    _applyShapeView();
    sizeCanvas();
    t0=performance.now();
    if(!raf) raf=requestAnimationFrame(frame);
    _updateMatchCount();
  } catch(e){ console.error('[graph] galaxy load failed:', e);
    if(host) host.innerHTML='<div class="graph-error"><div class="graph-error-title">Graph unavailable</div><div class="graph-error-msg">'+escapeHtml(String(e&&e.message||e))+'</div></div>'; }
}

// ── shape ──
/** Switch layout. Relays out from the payload already in memory — no refetch,
 *  and the highlight/activation state survives, so a shape change is a change
 *  of viewpoint on the same moment rather than a reset. */
export function setShape(name){
  if(!SHAPES[name]||name===_shape) return;
  _shape=name;
  try { localStorage.setItem(SHAPE_KEY,_shape); } catch(e){}
  const btn=host&&host.querySelector('.gbtn[data-g="shape"]');
  if(btn) btn.textContent='shape: '+SHAPE_LABEL[_shape];
  _applyShapeView();
  if(!raw) return;
  // Preserve what is currently lit: rebuilding G reallocates the activation
  // arrays, so carry the live values across by node id.
  const carried=[];
  if(G){ for(let i=0;i<G.N;i++) if(act[i]>0.02||tender[i]>0.02) carried.push([G.IDS[i],act[i],tender[i]]); }
  G=buildGalaxy(raw);
  allocScreen(G.N); needSort=true;
  for(const [id,a,td] of carried){ const i=G.idIndex.get(id); if(i!=null){ act[i]=a; tender[i]=td; if(a>0.12) front.push(i); } }
  sizeCanvas();
}
export function getShape(){ return _shape; }

/** Point the camera the way the current shape wants to be seen. Called on
 *  every load and every shape switch — a persisted `strata` must come back
 *  flat on reload, not only when you switch into it. */
function _applyShapeView(){
  const view=SHAPE_VIEW[_shape];
  if(!view) return;
  pitch=view.pitch; roll=view.roll; yaw=0; spin=view.spin; needSort=true;
  const sb=host&&host.querySelector('.gbtn[data-g="spin"]');
  if(sb) sb.textContent='spin: '+(spin?'on':'off');
}

// ── search ──
function _searchableNodes(){ if(G) return G.N; if(raw?.nodes) return raw.nodes.length; return 0; }
function _updateMatchCount(){ const el=document.getElementById('graph-search-count'); if(!el)return;
  if(!_searchQuery){el.textContent='';return;}
  let n=0; if(G){ for(let i=0;i<G.N;i++) if(matches(i)) n++; }
  else if(raw?.nodes){ const q=_searchQuery; n=raw.nodes.filter(x=>(x.name||'').toLowerCase().includes(q)||(x.type||'').toLowerCase().includes(q)).length; }
  el.textContent=n+' match'+(n===1?'':'es'); }
export function setSearchQuery(q){ _searchQuery=(q||'').toLowerCase().trim(); _updateMatchCount(); }
export function onGraphSearch(){ const i=document.getElementById('graph-search'); setSearchQuery(i?i.value:''); }
export function onGraphSearchKey(event){ if(event&&event.key==='Enter') onGraphSearch(); }

// ── recall highlight (persistent spotlight for pin/preview) + live bloom ──
function _applyEvent(event){ if(!event)return;
  const byTier={ returned:event.returned_ids||[], activation:event.activation_ids||[], used:event.used_ids||[] };
  for(const tier of ['returned','activation','used']) for(const id of byTier[tier]) _highlightTier.set(id,tier); }
// Seed ONLY the judge's picks (or a few activation seeds) and let stepActivation
// carry the wave outward — seeding the whole recall set at once made it all flash
// together instead of spreading.
function _bloomEvent(event){ if(!event||!G)return;
  const hue=sessionHueIndex(event.session_id||'');
  const src=(event.used_ids&&event.used_ids.length) ? event.used_ids : (event.activation_ids||[]).slice(0,3);
  for(const id of src){ const i=G.idIndex.get(id); if(i!=null) seed(i,hue); } }
function _onRecallEvent({event}){ if(!G||!event)return;
  _bloomEvent(event);                              // live "watch it think" bloom
  if(_highlightMode==='pinned') return; }          // pinned spotlight stays put
export function previewRecallOnGraph(event){ if(!G||!event)return; if(!_previewSnapshot) _saveSnapshot();
  _highlightTier.clear(); _applyEvent(event); }
export function clearRecallPreview(){ if(_previewSnapshot) _restoreSnapshot(); }
function _saveSnapshot(){ _previewSnapshot={tier:new Map(_highlightTier),mode:_highlightMode,pinned:_pinnedEventId}; }
function _restoreSnapshot(){ if(!_previewSnapshot)return; _highlightTier.clear();
  for(const [k,v] of _previewSnapshot.tier)_highlightTier.set(k,v); _highlightMode=_previewSnapshot.mode; _pinnedEventId=_previewSnapshot.pinned; _previewSnapshot=null; }
export function pinRecallToGraph(event){ if(!event)return; _previewSnapshot=null; _highlightMode='pinned';
  _pinnedEventId=event.id||null; _highlightTier.clear(); _applyEvent(event); _bloomEvent(event);
  bus.publish('graph:pinned',{eventId:_pinnedEventId}); }
/** Drop the spotlight without touching the layout or refetching. Split out of
 *  onGraphRefresh so a caller that only wants to stop highlighting (closing the
 *  recall panel) doesn't pay a full destroy + reload. */
export function clearHighlight(){ _highlightTier.clear(); _highlightMode='latest'; _pinnedEventId=null;
  _previewSnapshot=null; bus.publish('graph:pinned',{eventId:null}); }
export function onGraphRefresh(){ clearHighlight(); destroy(); loadGraph3D(); }

// ── lifecycle ──
export function init(){
  bus.subscribe('recall:event', _onRecallEvent);
  wireWindowOnce();
  const h=document.getElementById('graph-3d');
  if(h && 'ResizeObserver' in window){ _ro=new ResizeObserver(()=>requestAnimationFrame(resize)); _ro.observe(h); }
}
export function resize(){ if(!cv||!host) return; sizeCanvas(); needSort=true; }
export function activate(){
  if(_activateTimer) clearTimeout(_activateTimer);
  _activateTimer=setTimeout(()=>{ _activateTimer=null;
    if(!G) loadGraph3D(); else { sizeCanvas(); if(!raf){t0=performance.now();raf=requestAnimationFrame(frame);} }
    // No opening bloom. Replaying the last historical recall as a live
    // firing made every tab-switch look like the brain had just thought
    // something — the same lie as the ambient seeding, one frame later.
  }, 300);
}
export function deactivate(){ /* canvas keeps animating; cheap */ }
export function destroy(){
  if(_activateTimer){clearTimeout(_activateTimer);_activateTimer=null;}
  _loadGen++;
  if(raf){cancelAnimationFrame(raf);raf=0;}
  G=null; raw=null;
  const c=document.getElementById('graph-3d'); if(c){c.innerHTML='';} cv=null; ctx=null; host=null; tip=null;
}
