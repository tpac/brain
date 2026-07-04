// ===========================================================================
// tabs/graph.js — Canvas "spiral galaxy" renderer for the brain graph.
// ---------------------------------------------------------------------------
// Replaces the former 3D-force-graph (WebGL) renderer. Canvas 2D — so there is
// NO WebGL context to exhaust (the old ~16-context-per-tab flakiness is gone).
//
// The galaxy is grown from the real graph, laid out in JS on load:
//   • radius   = memory AGE  — oldest/most-settled knowledge in the bright core,
//                              recent growth spiralling out on the arms.
//   • knot     = COMMUNITY   — each community is a distinct clump on an arm.
//   • colour   = memory KIND — its node-type's aspect (identity/lesson/moment/…).
//   • size     = CONNECTEDNESS (degree). glow = RECALL HEAT (recency + volume).
//
// Live layer: every recall lights the galaxy — used_ids flash white, the
// activation set blooms outward, returned_ids glow faint (recall:event bus).
//
// Lifecycle contract (unchanged, drives app.js / live.js):
//   init() activate() deactivate() destroy() resize()
//   loadGraph3D() onGraphSearch() onGraphSearchKey() onGraphRefresh()
//   setSearchQuery() previewRecallOnGraph() clearRecallPreview() pinRecallToGraph()
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { escapeHtml } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

// ── aspect taxonomy (first-claimant, generated from aspects_v1.json) ──
const TYPE2ASP={"principle":0,"identity":0,"vision":0,"rule":0,"operator":0,"capability":0,"directive":0,"design_principle":0,"procedure":0,"philosophy":0,"framework":0,"definition":0,"preference":0,"craft_rule":0,"design_direction":0,"moment":1,"anchor_quote":1,"user_quote":1,"quote":1,"catalyst":1,"milestone":1,"event":1,"interaction":1,"context":1,"incident":1,"corridor":1,"benchmark":1,"state":1,"resolution":1,"outcome":1,"episode":1,"time_anchor":1,"personal_context":1,"profile":1,"person":1,"interest":1,"case":1,"journal":1,"experiment":1,"open":2,"tension":2,"hypothesis":2,"aspiration":2,"uncertainty":2,"gap":2,"task":2,"status":2,"product-requirement":2,"problem":2,"requirement":2,"plan":2,"performance":2,"goal":2,"feature-requirement":2,"risk":2,"backlog":2,"proposal":2,"idea":2,"todo":2,"lesson":3,"insight":3,"validation":3,"meta_learning":3,"reflection":3,"decision":3,"fact":3,"finding":3,"mechanism":3,"architecture":3,"bug":3,"concept":3,"design":3,"pattern":3,"mental_model":3,"fn_reasoning":3,"analysis":3,"constraint":3,"research":3,"diagnosis":3,"convention":3,"impact":3,"code_concept":3,"reframe":3,"reference":3,"research-finding":3,"param_influence":3,"observation":3,"artifact":3,"fix":3,"discovery":3,"arch_constraint":3,"method":3,"landscape":3,"clarification":3,"audit":3,"recommendation_set":3,"resource_list":3,"knowledge":3,"reference_cluster":3,"technique":3,"reference_list":3,"overview":3,"taxonomy":3,"process":3,"direction":3,"distinction":3,"methodology":3,"community":5,"aspect":5,"intuition":5,"thought":5,"purpose":5,"file":5,"vocabulary":5,"test":5,"project":5,"correction":6,"bug_lesson":6};
const REL2ASP={"related_to":4,"related":4,"similar_to":4,"synthesizes":4,"complements":4,"co_accessed":5,"emergent_bridge":5,"community_member":5,"corrects":6,"corrected_by":6,"reframes":6,"resolves":6,"addresses":6,"consolidated_into":6,"updates":6,"fixes":6,"extends":7,"refines":7,"implements":7,"contextualizes":7,"applies":7,"instantiates":7,"operationalizes":7,"advances":7,"explains":8,"caused_by":8,"grounds":8,"produces":8,"informs":8,"triggers":8,"motivates":8,"depends_on":9,"enables":9,"requires":9,"blocks":9,"constrains":9,"configures":9,"contradicts":10,"violates":10,"challenges":10,"contrasts_with":10,"validates":11,"confirms":11,"demonstrates":11,"strengthens":11,"supports":11,"part_of":12,"includes":12,"contains":12,"supersedes":12,"abstracts":12,"follows":13,"leads_to":13,"after":13,"before":13,"during":13,"opens":13,"completes":13,"co_anchored":13,"absorbed_into":14};
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
let spin=true, showEdges=false, colorMode=0, hoverIdx=-1, ambientT=0, ambientGap=4;
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
let SPR=null, WHITE=null;
function ensureSprites(){ if(SPR)return; SPR=HUE.map(h=>sprite(h[0],h[1],h[2])); WHITE=sprite(255,244,214); }

// ── helpers ──
const NOW = Date.now();
function daysAgo(ts){ if(!ts) return 9999; const s=String(ts).replace(' ','T').replace('Z','+00:00');
  const d=Date.parse(s); if(isNaN(d)) return 9999; return Math.max(0,(NOW-d)/86400000); }
function hstr(s){ let h=2166136261>>>0; s=String(s); for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);} return (h>>>0)/4294967296; }

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
  const H=new Float32Array(N), D=new Float32Array(N);
  for(let i=0;i<N;i++){ H[i]=Math.max(0,Math.min(1, 0.35*Math.log1p(acc[i])/la + 0.65*rec[i]));
    D[i]=Math.min(1, Math.sqrt(DEG[i])/13.4); }

  // communities
  const cmap=new Map();
  for(let i=0;i<N;i++){ const c=COMM[i]; if(c==null) continue; (cmap.get(c)||cmap.set(c,[]).get(c)).push(i); }
  // age percentiles for radius
  const ages=Array.from(AGE).filter(a=>a<9999).sort((a,b)=>a-b);
  const lo=ages.length?ages[Math.floor(ages.length*0.03)]:0, hi=ages.length?ages[Math.floor(ages.length*0.99)]:100;
  const aget=x=>{ x=Math.max(lo,Math.min(hi,x)); return Math.max(0,Math.min(1,1-(x-lo)/Math.max(1e-6,hi-lo))); };
  const NARM=2, WIND=2.75, DISC=0.26;
  function spiral(t,arm,jr,ja,jz){ const rad=0.24+2.4*t, th=arm*Math.PI+t*WIND*2*Math.PI+(ja-0.5)*(0.13+0.2*t);
    const x=rad*Math.cos(th), y=rad*Math.sin(th), px=-Math.sin(th),py=Math.cos(th), w=(jr-0.5)*(0.05+0.18*t);
    return [x+px*w, y+py*w, jz*DISC*(0.5+0.55*t)*rad]; }
  const X=new Float32Array(N), Y=new Float32Array(N), Z=new Float32Array(N);
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
  // place members as tight knots
  const rnd=mulberry(12345);
  for(const c of cids){ const mem=cmap.get(c), ctr=cc[c], ball=0.03+0.075*Math.log1p(mem.length);
    for(const i of mem){ let dx=randn(rnd),dy=randn(rnd),dz=randn(rnd)*DISC; const L=Math.hypot(dx,dy,dz)||1;
      const cp=0.16*(DEG[i]/(DEG[i]+15)); const rr=ball*(0.35+0.65*rnd());
      X[i]=ctr[0]*(1-cp)+dx/L*rr; Y[i]=ctr[1]*(1-cp)+dy/L*rr; Z[i]=ctr[2]*(1-cp)+dz/L*rr; } }
  // free nodes ride the arms as faint dust
  for(let i=0;i<N;i++){ if(COMM[i]!=null) continue; const p=spiral(aget(AGE[i]), rnd()<0.5?0:1, rnd(), rnd(), randn(rnd)*0.8);
    X[i]=p[0]*1.06; Y[i]=p[1]*1.06; Z[i]=p[2]*1.06; }
  // recenter + normalize
  let mx=0,my=0,mz=0; for(let i=0;i<N;i++){mx+=X[i];my+=Y[i];mz+=Z[i];} mx/=N||1;my/=N||1;mz/=N||1;
  let mr=1e-3; for(let i=0;i<N;i++){X[i]-=mx;Y[i]-=my;Z[i]-=mz;mr=Math.max(mr,Math.hypot(X[i],Y[i],Z[i]));}
  for(let i=0;i<N;i++){X[i]/=mr;Y[i]/=mr;Z[i]/=mr;}
  // adjacency (non-noise) + drawn-edge subset
  const adj=Array.from({length:N},()=>[]);
  const draw=[]; let di=0;
  for(const [a,b,r] of E){ if(r!==NOISE_ASP){ adj[a].push(b); adj[b].push(a); }
    if(r===NOISE_ASP) continue; const bridge=COMM[a]!==COMM[b];
    if(r===GENERIC_ASP && !bridge && (di++ %4)) continue;
    if(!bridge && (di++ %2)) continue;
    draw.push([a,b,r,bridge?1:0]); }
  return { N, X,Y,Z, A, D, H, DEG, TITLE, TYPE, COMM, AGE, IDS, idIndex, adj, edges:draw,
           C:cids.length, E_total:E.length };
}
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function randn(R){let u=0,v=0;while(!u)u=R();while(!v)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(6.2831853*v);}

// ── projection + render ──
let sx,sy,sp,sz,order;
function allocScreen(N){ sx=new Float32Array(N);sy=new Float32Array(N);sp=new Float32Array(N);sz=new Float32Array(N);
  order=new Int32Array(N); for(let i=0;i<N;i++)order[i]=i; act=new Float32Array(N); front=[]; }
function project(){ const g=G; const cr=Math.cos(roll),sr=Math.sin(roll),cy=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),spp=Math.sin(pitch);
  const sc=(FIT*MIN)*zoom;
  for(let i=0;i<g.N;i++){ let x=g.X[i],y=g.Y[i],z=g.Z[i];
    let xr=x*cr-y*sr, yr=x*sr+y*cr; let rx=xr*cy-z*syw, rz=xr*syw+z*cy; let ry=yr*cp-rz*spp; rz=yr*spp+rz*cp;
    const pe=FOCAL/(FOCAL+rz); sx[i]=W*CX+panx+rx*sc*pe; sy[i]=H2*CY+pany+ry*sc*pe; sp[i]=pe; sz[i]=rz; } }

function matches(i){ if(!_searchQuery) return true; const q=_searchQuery;
  return (G.TITLE[i]||'').toLowerCase().includes(q) || (G.TYPE[i]||'').toLowerCase().includes(q); }

function seed(i){ if(i<0||i>=G.N)return; act[i]=Math.max(act[i],1); if(front.indexOf(i)<0)front.push(i); }
function stepActivation(dt){ if(!front.length)return; const nf=[], add=new Map();
  for(const i of front){ const nb=G.adj[i], give=act[i]*0.26;
    if(give>0.03 && nb.length){ const per=give/Math.min(nb.length,6);
      for(let k=0;k<nb.length&&k<6;k++){const j=nb[k]; if(act[j]<0.02) add.set(j,(add.get(j)||0)+per);} }
    act[i]*=Math.pow(0.09,dt); if(act[i]>0.02) nf.push(i); }
  for(const [j,v] of add){ act[j]=Math.min(1,act[j]+v); if(nf.indexOf(j)<0)nf.push(j); }
  front=nf.slice(0,1400); }

function frame(now){ if(!G||!ctx){raf=0;return;}
  const dt=Math.min((now-t0)/1000,.05); t0=now;
  if(spin&&!reduce) roll+=dt*0.075;
  stepActivation(dt);
  if(!reduce){ ambientT+=dt; if(ambientT>ambientGap && front.length<40){ ambientT=0; ambientGap=4.5+Math.random()*3;
    let best=-1,bh=0; for(let k=0;k<34;k++){const i=(Math.random()*G.N)|0; if(G.H[i]>bh){bh=G.H[i];best=i;}} seed(best); } }
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

  order.sort((i,j)=>sz[j]-sz[i]);
  // glow underlay
  for(let k=0;k<g.N;k++){ const i=order[k],heat=g.H[i],hh=heat*heat,ac=act[i];
    const dim = (_searchQuery&&!matches(i)) || (spot&&!_highlightTier.has(i));
    if(dim && ac<0.05) continue;
    if(hh<0.14 && g.D[i]<0.14 && ac<0.05 && !_highlightTier.has(i)) continue;
    const pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7));
    const base=1.5+g.D[i]*11+hh*2.4, s=(base*1.9+ac*ac*26)*pe;
    ctx.globalAlpha=Math.min(0.34,(0.02+0.12*hh+g.D[i]*0.1)*(0.4+0.6*depth)+ac*0.28)*(dim?0.15:1);
    ctx.drawImage(SPR[g.A[i]],sx[i]-s,sy[i]-s,s*2,s*2); }
  // firing bloom
  for(let x=0;x<front.length;x++){const i=front[x],ac=act[i];if(ac<0.12)continue;const pe=sp[i],s=(3+ac*16)*pe;
    ctx.globalAlpha=Math.min(0.6,ac*0.7);ctx.drawImage(WHITE,sx[i]-s,sy[i]-s,s*2,s*2);}
  // crisp cores
  ctx.globalCompositeOperation='source-over';
  for(let k=0;k<g.N;k++){ const i=order[k],pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7)),heat=g.H[i],hh=heat*heat,ac=act[i];
    const dim=(_searchQuery&&!matches(i))||(spot&&!_highlightTier.has(i));
    const tier=spot?_highlightTier.get(i):null;
    const base=1.5+g.D[i]*11+hh*2.4+ac*4, s=base*0.62*pe*(tier?1.5:1);
    let spr,al;
    if(ac>0.15){spr=WHITE;al=(0.35+0.6*ac)*(0.4+0.6*depth);}
    else if(tier){const c=HTIERS[tier]; spr=tierSprite(c); al=(0.55+0.4*depth);}
    else{spr=colorMode===0?SPR[g.A[i]]:(heat>0.62?WHITE:SPR[3]);
      al=(colorMode===0?(0.16+0.66*hh+g.D[i]*0.34):(0.12+0.8*hh))*(0.4+0.6*depth);}
    if(dim && !tier && ac<0.15){ al*=0.12; }
    ctx.globalAlpha=Math.min(0.98,al); ctx.drawImage(spr,sx[i]-s,sy[i]-s,s*2,s*2); }
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
  ctl.innerHTML=`<button data-g="color" class="gbtn">color: kind</button><button data-g="lines" class="gbtn">lines: off</button><button data-g="spin" class="gbtn">spin: on</button>`;
  host.appendChild(ctl);
  ctl.querySelectorAll('.gbtn').forEach(b=>{ b.style.cssText='border:1px solid #1a2138;background:rgba(8,10,20,.72);color:#c9d3e6;font:10px ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;border-radius:7px;padding:6px 9px;cursor:pointer;white-space:nowrap;backdrop-filter:blur(6px)';
    b.onclick=()=>{ const k=b.dataset.g;
      if(k==='color'){colorMode^=1;b.textContent='color: '+(colorMode?'heat':'kind');}
      if(k==='lines'){showEdges=!showEdges;b.textContent='lines: '+(showEdges?'on':'off');}
      if(k==='spin'){spin=!spin;b.textContent='spin: '+(spin?'on':'off');} }; });
  wireCanvas();
  return true;
}
let drag=false,lx=0,ly=0,sh=false,moved=false;
function wireCanvas(){
  cv.addEventListener('mousedown',e=>{drag=true;moved=false;lx=e.clientX;ly=e.clientY;sh=e.shiftKey;cv.style.cursor='grabbing';});
  window.addEventListener('mousemove',e=>{ if(!G)return;
    if(drag){const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;
      if(sh){panx+=dx;pany+=dy;}else{yaw+=dx*0.005;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.005));}return;}
    if(!cv)return; const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    if(mx<0||my<0||mx>r.width||my>r.height){hoverIdx=-1;if(tip)tip.classList.remove('show');return;}
    const i=nearest(mx,my); hoverIdx=i;
    if(i>=0){showTip(i,mx,my);cv.style.cursor='pointer';}else{if(tip)tip.classList.remove('show');cv.style.cursor='grab';} });
  window.addEventListener('mouseup',e=>{ if(!drag)return; drag=false; if(cv)cv.style.cursor='grab';
    if(!moved&&G){const r=cv.getBoundingClientRect(),i=nearest(e.clientX-r.left,e.clientY-r.top);
      if(i>=0){ seed(i); loadNodeDetail(G.IDS[i]); }} });
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
    allocScreen(G.N);
    sizeCanvas();
    t0=performance.now();
    if(!raf) raf=requestAnimationFrame(frame);
    _updateMatchCount();
  } catch(e){ console.error('[graph] galaxy load failed:', e);
    if(host) host.innerHTML='<div class="graph-error"><div class="graph-error-title">Graph unavailable</div><div class="graph-error-msg">'+escapeHtml(String(e&&e.message||e))+'</div></div>'; }
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
function _bloomEvent(event){ if(!event||!G)return; let dl=0;
  for(const list of [event.used_ids||[], event.activation_ids||[]]) for(const id of list){ const i=G.idIndex.get(id); if(i!=null) seed(i); } }
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
export function onGraphRefresh(){ _highlightTier.clear(); _highlightMode='latest'; _pinnedEventId=null;
  bus.publish('graph:pinned',{eventId:null}); destroy(); loadGraph3D(); }

// ── lifecycle ──
export function init(){
  bus.subscribe('recall:event', _onRecallEvent);
  const h=document.getElementById('graph-3d');
  if(h && 'ResizeObserver' in window){ _ro=new ResizeObserver(()=>requestAnimationFrame(resize)); _ro.observe(h); }
}
export function resize(){ if(!cv||!host) return; sizeCanvas(); }
export function activate(){
  if(_activateTimer) clearTimeout(_activateTimer);
  _activateTimer=setTimeout(()=>{ _activateTimer=null;
    if(!G) loadGraph3D(); else { sizeCanvas(); if(!raf){t0=performance.now();raf=requestAnimationFrame(frame);} }
    // preload the latest recall as an opening bloom
    if(_highlightMode!=='pinned'){ api.recalls({limit:1}).then(d=>{ const e=(d.events||[])[0]; if(e) _bloomEvent(e); }).catch(()=>{}); }
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
