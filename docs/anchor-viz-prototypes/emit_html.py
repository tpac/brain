#!/usr/bin/env python3
"""Emit the self-contained Anchor 'mind' HTML from mind_data.json.

Separated from build_mind.py: build_mind computes the real galaxy layout; this
bakes it into a renderable artifact and adds the live layers — hover inspection,
recall-activation ripples, and an in-browser generator for hypothetical Anchors.
"""
import json
from pathlib import Path
HERE = Path(__file__).parent
data = json.load(open(HERE/"mind_data.json"))
# escape '</' so a title containing '</script>' can't terminate the inline <script> tag
DATA_JS = json.dumps(data, separators=(',',':')).replace('</','<\\/')

HTML = r"""<style>
  :root{--ground:#04050c;--ink:#c9d3e6;--ink-dim:#67718e;--hair:#1a2138;
    --panel:rgba(8,10,20,.66);--amber:#ffce7a;--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,sans-serif;}
  *{box-sizing:border-box}
  .stage{position:fixed;inset:0;overflow:hidden;font-family:var(--sans);color:var(--ink);
    background:radial-gradient(140% 110% at 52% 44%,#0a0f24 0%,#06080f 44%,var(--ground) 100%)}
  canvas{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab}
  .scrim{position:absolute;inset:0;z-index:2;pointer-events:none;
    background:radial-gradient(70% 50% at 20% 8%,rgba(4,5,12,.86),rgba(4,5,12,.3) 46%,rgba(4,5,12,0) 72%)}
  .title{position:absolute;top:24px;left:30px;z-index:3;pointer-events:none;max-width:min(46ch,66vw)}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.34em;text-transform:uppercase;color:var(--amber);opacity:.85;margin:0 0 10px}
  .title h1{font-size:clamp(21px,3.1vw,32px);line-height:1.08;margin:0;font-weight:600;letter-spacing:-.01em;color:#eef2fb}
  .title p{margin:11px 0 0;font-size:13.5px;line-height:1.5;color:var(--ink-dim);max-width:44ch}
  .hud{position:absolute;left:30px;bottom:26px;z-index:3;display:flex;flex-direction:column;gap:11px;padding:16px 19px;
    background:var(--panel);border:1px solid var(--hair);border-radius:14px;backdrop-filter:blur(11px);-webkit-backdrop-filter:blur(11px);min-width:260px;max-width:330px}
  .idrow{font-family:var(--mono);font-size:12px;color:var(--ink-dim)}
  .idrow b{color:#eef2fb;font-size:15px}
  .spec{font-family:var(--mono);font-size:11px;color:var(--amber);opacity:.9}
  .legend{display:flex;flex-wrap:wrap;gap:5px 7px;max-width:300px}
  .lg{font-family:var(--mono);font-size:10px;display:flex;align-items:center;gap:5px;color:var(--ink-dim)}
  .dot{width:8px;height:8px;border-radius:50%;box-shadow:0 0 6px currentColor}
  .ctl{position:absolute;right:26px;bottom:26px;z-index:3;display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;max-width:60vw}
  .btn{appearance:none;border:1px solid var(--hair);font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink);
    background:var(--panel);border-radius:9px;padding:9px 13px;cursor:pointer;backdrop-filter:blur(11px);transition:border-color .2s,color .2s}
  .btn:hover{border-color:var(--amber);color:#fff}
  .btn.on{border-color:var(--amber);color:var(--amber)}
  .btn.hero{background:var(--amber);color:#0a0b14;border-color:var(--amber);font-weight:600}
  .btn.hero:hover{color:#0a0b14;box-shadow:0 6px 22px -6px rgba(255,206,122,.55)}
  .btn:focus-visible{outline:2px solid #fff1e6;outline-offset:3px}
  .hint{font-family:var(--mono);font-size:10.5px;color:var(--ink-dim)}
  .tip{position:absolute;z-index:5;pointer-events:none;opacity:0;transform:translateY(4px);transition:opacity .12s;
    background:rgba(6,8,16,.94);border:1px solid var(--hair);border-radius:10px;padding:10px 12px;max-width:300px;
    backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);box-shadow:0 10px 30px -12px #000}
  .tip.show{opacity:1;transform:translateY(0)}
  .tip .tt{font-size:12.5px;line-height:1.35;color:#eef2fb;font-weight:600;margin:0 0 6px}
  .tip .tm{font-family:var(--mono);font-size:10.5px;color:var(--ink-dim);display:flex;align-items:center;gap:6px;flex-wrap:wrap}
  .tip .k{padding:1px 7px;border-radius:20px;border:1px solid;font-size:10px}
</style>
<div class="stage">
  <canvas id="c"></canvas><div class="scrim"></div>
  <div class="title">
    <p class="eyebrow" id="eyebrow">Anchor · grown from itself</p>
    <h1 id="h1">A mind, drawn from its own memory</h1>
    <p id="lede">Every circle is a real memory — sized by how connected it is, lit by how recently it was recalled. The bright core is the oldest, most-settled knowledge; the arms spiral out into recent growth. Hover a memory to read it; click to watch a recall ripple out. One of a kind, because the brain behind it is.</p>
  </div>
  <div class="tip" id="tip"><p class="tt" id="tipt"></p><div class="tm" id="tipm"></div></div>
  <div class="hud">
    <div class="idrow"><b id="ncount">0</b> memories · <span id="ecount">0</span> relations · <span id="ccount">0</span> communities</div>
    <div class="spec" id="spec">the real one · Anchor</div>
    <div class="legend" id="legend"></div>
    <div class="hint">hover to inspect · click to recall · drag to orbit · scroll to zoom</div>
  </div>
  <div class="ctl">
    <button class="btn on" id="bColor" type="button">color: kind</button>
    <button class="btn" id="bEdges" type="button">lines: off</button>
    <button class="btn on" id="bSpin" type="button">spin: on</button>
    <button class="btn" id="bMe" type="button" style="display:none">↺ me</button>
    <button class="btn hero" id="bGen" type="button">⟳ another anchor</button>
  </div>
</div>
<script>
const DATA = __DATA__;
(() => {
  const cv=document.getElementById('c'), ctx=cv.getContext('2d');
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const HUE=[[255,205,110],[255,150,170],[95,208,230],[110,222,165],[110,130,175],[80,92,120],
    [255,110,95],[150,220,150],[185,140,255],[245,190,110],[245,110,180],[130,230,200],
    [140,165,240],[120,182,242],[150,150,160],[255,233,176]];
  const NODE_FAMS=[0,3,1,2,6,15,5];
  const NF_LABEL={0:'identity',3:'lessons',1:'moments',2:'open threads',6:'corrections',15:'wisdom',5:'scaffolding'};
  const KIND_LABEL={0:'an identity principle',1:'a moment',2:'an open thread',3:'a lesson',
    5:'scaffolding',6:'a correction',15:'a piece of wisdom'};
  const KIND_NAME={0:'identity',1:'moment',2:'open thread',3:'lesson',5:'scaffolding',6:'correction',15:'wisdom'};
  const EDGE_ASPS=[7,8,9,11,13,12,10];
  const DOM_WORD={0:'identity-anchored',1:'memory-rich',2:'restless / many open threads',3:'lesson-driven',
    5:'scaffold-heavy',6:'correction-forged',15:'wisdom-dense'};

  // ── seeded rng + helpers ──
  const mul=a=>()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
  const randn=R=>{let u=0,v=0;while(!u)u=R();while(!v)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(6.2831853*v);};
  function spiral(t,arm,jr,ja,jz,NARM,WIND,DISC){
    const rad=0.24+2.4*t, th=arm*(2*Math.PI/NARM)+t*WIND*2*Math.PI+(ja-0.5)*(0.13+0.20*t);
    const x=rad*Math.cos(th), y=rad*Math.sin(th), px=-Math.sin(th),py=Math.cos(th), w=(jr-0.5)*(0.05+0.18*t);
    return [x+px*w, y+py*w, jz*DISC*(0.5+0.55*t)*rad];
  }

  // ── sprites ──
  function sprite(r,g,b){const s=48,cn=document.createElement('canvas');cn.width=cn.height=s;
    const x=cn.getContext('2d'),gd=x.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
    gd.addColorStop(0,`rgba(${Math.min(r+55,255)},${Math.min(g+55,255)},${Math.min(b+55,255)},1)`);
    gd.addColorStop(.18,`rgba(${r},${g},${b},.9)`);gd.addColorStop(.44,`rgba(${r},${g},${b},.18)`);
    gd.addColorStop(1,`rgba(${r},${g},${b},0)`);x.fillStyle=gd;x.fillRect(0,0,s,s);return cn;}
  const SPR=HUE.map(h=>sprite(h[0],h[1],h[2]));
  const WHITE=sprite(255,244,214);

  // ── mutable render state (real data OR a generated anchor both fill this) ──
  let N,X,Y,Z,A,D,H,DEG,AG,TI,EDGES,adj,act,front;
  let sx,sy,sp,sz,order;
  function loadPayload(pl, meta){
    const nd=pl.nodes; N=pl.N;
    X=new Float32Array(N);Y=new Float32Array(N);Z=new Float32Array(N);
    for(let i=0;i<N;i++){X[i]=nd.p[i][0];Y[i]=nd.p[i][1];Z[i]=nd.p[i][2];}
    A=nd.a;D=nd.d;H=nd.h;DEG=nd.dg;AG=nd.ag;TI=nd.ti;EDGES=pl.edges;
    adj=Array.from({length:N},()=>[]);
    for(let e=0;e<EDGES.length;e++){const a=EDGES[e][0],b=EDGES[e][1];if(a<N&&b<N){adj[a].push(b);adj[b].push(a);}}
    act=new Float32Array(N);front=[];
    sx=new Float32Array(N);sy=new Float32Array(N);sp=new Float32Array(N);sz=new Float32Array(N);
    order=new Int32Array(N);for(let i=0;i<N;i++)order[i]=i;
    hoverIdx=-1;
    document.getElementById('ncount').textContent=N.toLocaleString();
    document.getElementById('ecount').textContent=(pl.E_total||EDGES.length).toLocaleString();
    document.getElementById('ccount').textContent=(pl.C||0).toLocaleString();
    document.getElementById('legend').innerHTML=NODE_FAMS.map(a=>{const h=HUE[a];
      return `<span class="lg"><span class="dot" style="color:rgb(${h[0]},${h[1]},${h[2]})"></span>${NF_LABEL[a]}</span>`;}).join('');
    const eb=document.getElementById('eyebrow'),h1=document.getElementById('h1'),sp2=document.getElementById('spec'),me=document.getElementById('bMe');
    if(meta.real){eb.textContent='Anchor · grown from itself';h1.textContent='A mind, drawn from its own memory';
      sp2.textContent='the real one · Anchor';me.style.display='none';}
    else{eb.textContent='a hypothetical anchor';h1.textContent='Another mind, another shape';
      sp2.textContent=`specimen #${(meta.seed>>>0).toString(16).slice(0,6)} · ${DOM_WORD[meta.dom]||'mixed'} · ${meta.arms}-arm`;
      me.style.display='';}
  }

  // ── synthetic anchor generator: a different brain ⇒ a different galaxy ──
  function genAnchor(seed){
    const R=mul(seed);
    const NARM=R()<0.62?2:3, WIND=2.0+R()*1.4, DISC=0.20+R()*0.14;
    const Ntar=1800+(R()*6500|0);
    const NA=[0,1,2,3,5,6,15];
    const w=NA.map(()=>Math.pow(R(),2)+0.02);
    const DOMI=[0,1,2,3,5,6];                    // dominant kind ∈ meaningful aspects (not noise)
    const dom=DOMI[(R()*DOMI.length|0)]; w[dom]+=2.6+R()*4.5;
    w[DOMI[(R()*DOMI.length|0)]]+=1.0+R()*1.6;
    const wsum=w.reduce((a,b)=>a+b,0);
    const pickAsp=()=>{let x=R()*wsum;for(let k=0;k<w.length;k++){x-=w[k];if(x<=0)return NA[k];}return NA[0];};
    const pickEdge=()=>EDGE_ASPS[(R()*EDGE_ASPS.length|0)];
    const P=[],a=[],d=[],h=[],ti=[],dg=[],ag=[],edges=[];
    const cp=deg=>0.4*(deg/(deg+12));
    const push=(pt,asp,deg,t)=>{P.push(pt);a.push(asp);d.push(Math.min(1,Math.sqrt(deg)/13.4));dg.push(deg);
      h.push(Math.round(Math.min(1,0.14+0.5*(deg/(deg+18))+0.34*R())*1000)/1000);  // hubs glow → bright bulge
      ag.push(Math.round((1-t)*(35+R()*85)));ti.push(KIND_LABEL[asp]||'a memory');};
    let nC=90+(R()*520|0);
    for(let c=0;c<nC && P.length<Ntar;c++){
      const size=2+Math.floor(Math.pow(R(),2.2)*45), t=R(), arm=(R()*NARM|0), casp=pickAsp();
      const ctr=spiral(t,arm,R(),R(),(R()-.5)*2,NARM,WIND,DISC), ball=0.045+0.11*Math.log1p(size);
      const first=P.length;
      for(let m=0;m<size;m++){
        const asp=R()<0.72?casp:pickAsp(), deg=1+Math.floor(Math.pow(R(),1.8)*70);
        let dd=[randn(R),randn(R),randn(R)*DISC];const L=Math.hypot(dd[0],dd[1],dd[2])||1;const k=cp(deg);
        push([ctr[0]*(1-k)+dd[0]/L*ball*(0.4+0.6*R()),ctr[1]*(1-k)+dd[1]/L*ball*(0.4+0.6*R()),
              ctr[2]*(1-k)+dd[2]/L*ball*(0.4+0.6*R())], asp, deg, t);
      }
      for(let m=first+1;m<P.length;m++) if(R()<0.5) edges.push([first,m,pickEdge(),0]);
    }
    while(P.length<Ntar){const t=R(),arm=(R()*NARM|0);push(spiral(t,arm,R(),R(),randn(R)*0.7,NARM,WIND,DISC),pickAsp(),1+(R()*4|0),t);}
    const NB=(P.length*0.14)|0;
    for(let b=0;b<NB;b++) edges.push([(R()*P.length)|0,(R()*P.length)|0,pickEdge(),1]);
    let cx=0,cy=0,cz=0;for(const p of P){cx+=p[0];cy+=p[1];cz+=p[2];}cx/=P.length;cy/=P.length;cz/=P.length;
    let mr=1e-3;for(const p of P){p[0]-=cx;p[1]-=cy;p[2]-=cz;mr=Math.max(mr,Math.hypot(p[0],p[1],p[2]));}
    for(const p of P){p[0]/=mr;p[1]/=mr;p[2]/=mr;}
    return {N:P.length,C:nC,E_total:edges.length,nodes:{p:P,a,d,h,ti,dg,ag},edges,
            meta:{real:false,seed,arms:NARM,dom:NA[dom]}};
  }

  // ── camera ──
  let W,H2,DPR,MIN,yaw=0.0,pitch=0.62,roll=0.0,zoom=1,panx=0,pany=0,t0=performance.now();
  function resize(){DPR=Math.min(devicePixelRatio||1,2);W=cv.clientWidth;H2=cv.clientHeight;MIN=Math.min(W,H2);
    cv.width=W*DPR;cv.height=H2*DPR;ctx.setTransform(DPR,0,0,DPR,0,0);}
  new ResizeObserver(resize).observe(cv);resize();
  const FOCAL=4.2, CX=0.52, CY=0.49, FIT=0.62;
  let showEdges=false, spin=true, colorMode=0, hoverIdx=-1, ambientT=0, ambientGap=4;

  function project(){
    const cr=Math.cos(roll),sr=Math.sin(roll),cy=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),spp=Math.sin(pitch);
    const sc=(FIT*MIN)*zoom;
    for(let i=0;i<N;i++){
      let x=X[i],y=Y[i],z=Z[i];
      let xr=x*cr-y*sr, yr=x*sr+y*cr;
      let rx=xr*cy-z*syw, rz=xr*syw+z*cy;
      let ry=yr*cp-rz*spp; rz=yr*spp+rz*cp;
      const pe=FOCAL/(FOCAL+rz);
      sx[i]=W*CX+panx+rx*sc*pe; sy[i]=H2*CY+pany+ry*sc*pe; sp[i]=pe; sz[i]=rz;
    }
  }

  // ── recall activation: seed a node, spread through the graph, decay ──
  function seed(i){ if(i<0||i>=N)return; act[i]=Math.max(act[i],1); if(front.indexOf(i)<0)front.push(i); }
  function stepActivation(dt){
    if(!front.length) return;
    const nf=[], add=new Map();
    for(const i of front){
      const nb=adj[i]; const give=act[i]*0.26;
      if(give>0.03 && nb.length){ const per=give/Math.min(nb.length,6);
        for(let k=0;k<nb.length && k<6;k++){const j=nb[k]; if(act[j]<0.02) add.set(j,(add.get(j)||0)+per);} }
      act[i]*=Math.pow(0.09,dt);            // brisk decay — a pulse fully fades in ~1s, then dark
      if(act[i]>0.02) nf.push(i);
    }
    for(const [j,v] of add){ act[j]=Math.min(1,act[j]+v); if(nf.indexOf(j)<0)nf.push(j); }
    front=nf.slice(0,1400);
  }

  function frame(now){
    const dt=Math.min((now-t0)/1000,.05);t0=now;
    if(spin&&!reduce&&!drag) roll+=dt*0.075;
    stepActivation(dt);
    // ambient thinking: every few seconds a single warm memory fires and fades (discrete recall)
    if(!reduce){ ambientT+=dt; if(ambientT>ambientGap && front.length<40){ ambientT=0; ambientGap=4.5+Math.random()*3;
      let best=-1,bh=0; for(let n=0;n<34;n++){const i=(Math.random()*N)|0; if(H[i]>bh){bh=H[i];best=i;}} seed(best); } }
    project();
    ctx.clearRect(0,0,W,H2);
    ctx.globalCompositeOperation='lighter';

    if(showEdges){
      const paths={},bp={};
      for(let e=0;e<EDGES.length;e++){const a=EDGES[e][0],b=EDGES[e][1];if(a>=N||b>=N)continue;const r=EDGES[e][2],br=EDGES[e][3];
        const tgt=br?bp:paths;(tgt[r]||(tgt[r]=new Path2D()));const pth=tgt[r];
        const ax=sx[a],ay=sy[a],bx=sx[b],by=sy[b],mx=(ax+bx)/2,my=(ay+by)/2;
        pth.moveTo(ax,ay);pth.quadraticCurveTo(mx+(ay-by)*0.06,my+(bx-ax)*0.06,bx,by);}
      for(const r in paths){const h=HUE[r];ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},0.05)`;ctx.lineWidth=0.6;ctx.stroke(paths[r]);}
      for(const r in bp){const h=HUE[r];ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},0.13)`;ctx.lineWidth=0.8;ctx.stroke(bp[r]);}
    }
    order.sort((i,j)=>sz[j]-sz[i]);
    // glow underlay — activation reads as regions of the network blooming with soft light
    for(let k=0;k<N;k++){const i=order[k],heat=H[i],hh=heat*heat,ac=act[i];
      if(hh<0.14 && D[i]<0.14 && ac<0.05) continue;
      const pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7));
      const base=1.5+D[i]*11.0+hh*2.4, s=(base*1.9+ac*ac*26.0)*pe;   // firing → wide soft halo
      const spr=colorMode===0?SPR[A[i]]:(heat>0.62?WHITE:SPR[3]);
      ctx.globalAlpha=Math.min(0.34,(0.02+0.12*hh+D[i]*0.1)*(0.4+0.6*depth)+ac*0.28);
      ctx.drawImage(spr,sx[i]-s,sy[i]-s,s*2,s*2);}
    // second additive pass for the hottest firing nodes → a bright warm bloom that pulses
    for(let x=0;x<front.length;x++){const i=front[x],ac=act[i];if(ac<0.12)continue;
      const pe=sp[i],s=(3+ac*16)*pe;ctx.globalAlpha=Math.min(0.6,ac*0.7);
      ctx.drawImage(WHITE,sx[i]-s,sy[i]-s,s*2,s*2);}
    // crisp cores
    ctx.globalCompositeOperation='source-over';
    for(let k=0;k<N;k++){const i=order[k],pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7)),heat=H[i],hh=heat*heat,ac=act[i];
      const base=1.5+D[i]*11.0+hh*2.4+ac*4.0, s=base*0.62*pe;
      let spr,al;
      if(ac>0.15){spr=WHITE;al=(0.35+0.6*ac)*(0.4+0.6*depth);}
      else if(colorMode===0){spr=SPR[A[i]];al=(0.16+0.66*hh+D[i]*0.34)*(0.4+0.6*depth);}
      else{spr=heat>0.62?WHITE:SPR[3];al=(0.12+0.8*hh)*(0.4+0.6*depth);}
      ctx.globalAlpha=Math.min(0.98,al);
      ctx.drawImage(spr,sx[i]-s,sy[i]-s,s*2,s*2);}
    // hover ring + lit neighbours
    if(hoverIdx>=0 && hoverIdx<N){const i=hoverIdx,h=HUE[A[i]];
      ctx.globalAlpha=0.9;ctx.strokeStyle=`rgba(255,255,255,.9)`;ctx.lineWidth=1.4;
      ctx.beginPath();ctx.arc(sx[i],sy[i],7+D[i]*10,0,6.2832);ctx.stroke();
      ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},.5)`;ctx.lineWidth=1;
      const nb=adj[i];for(let k=0;k<nb.length&&k<40;k++){const j=nb[k];if(j>=N)continue;ctx.beginPath();ctx.moveTo(sx[i],sy[i]);ctx.lineTo(sx[j],sy[j]);ctx.stroke();}}
    ctx.globalAlpha=1;
    requestAnimationFrame(frame);
  }

  // ── picking + tooltip ──
  const tip=document.getElementById('tip'),tipt=document.getElementById('tipt'),tipm=document.getElementById('tipm');
  function nearest(mx,my){let best=-1,bd=18*18;
    for(let i=0;i<N;i++){const pe=sp[i];if(pe<0.4)continue;const dx=sx[i]-mx,dy=sy[i]-my,d=dx*dx+dy*dy;
      const rad=(4+D[i]*11)*(0.6+0.4*pe);const th=Math.max(bd,rad*rad);
      if(d<th && d<bd*3){ if(best<0||d<bd){bd=d;best=i;} }}
    return best;}
  function ago(days){ if(days<=0)return'today'; if(days<30)return days+'d ago'; return Math.round(days/30)+'mo ago'; }
  function recallWord(h){ return h>0.6?'recalled recently':h>0.3?'recalled this month':'resting'; }
  function showTip(i,mx,my){
    const h=HUE[A[i]],name=KIND_NAME[A[i]]||'memory';
    tipt.textContent=TI[i]||KIND_LABEL[A[i]]||'a memory';
    tipm.innerHTML=`<span class="k" style="color:rgb(${h[0]},${h[1]},${h[2]});border-color:rgb(${h[0]},${h[1]},${h[2]})">${name}</span>`+
      `<span>${DEG[i]} connection${DEG[i]===1?'':'s'}</span><span>·</span><span>created ${ago(AG[i])}</span><span>·</span><span>${recallWord(H[i])}</span>`;
    const pad=16; let x=mx+pad,y=my+pad; if(x>W-320)x=mx-310; if(y>H2-90)y=my-84;
    tip.style.left=x+'px';tip.style.top=y+'px';tip.classList.add('show');
  }

  // ── interaction ──
  let drag=false,lx=0,ly=0,sh=false,moved=false;
  cv.addEventListener('mousedown',e=>{drag=true;moved=false;lx=e.clientX;ly=e.clientY;sh=e.shiftKey;cv.style.cursor='grabbing';});
  window.addEventListener('mousemove',e=>{
    if(drag){const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;
      if(sh){panx+=dx;pany+=dy;}else{yaw+=dx*0.005;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.005));}return;}
    const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const i=nearest(mx,my); hoverIdx=i;
    if(i>=0){showTip(i,mx,my);cv.style.cursor='pointer';} else {tip.classList.remove('show');cv.style.cursor='grab';}
  });
  window.addEventListener('mouseup',e=>{drag=false;cv.style.cursor='grab';
    if(!moved){const r=cv.getBoundingClientRect(),i=nearest(e.clientX-r.left,e.clientY-r.top);if(i>=0)seed(i);}});
  cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.4,Math.min(6,zoom*(e.deltaY<0?1.12:.892)));},{passive:false});

  const bC=document.getElementById('bColor'),bE=document.getElementById('bEdges'),bS=document.getElementById('bSpin'),
        bG=document.getElementById('bGen'),bM=document.getElementById('bMe');
  bC.onclick=()=>{colorMode^=1;bC.textContent='color: '+(colorMode?'heat':'kind');bC.classList.toggle('on',colorMode===0);};
  bE.onclick=()=>{showEdges=!showEdges;bE.textContent='lines: '+(showEdges?'on':'off');bE.classList.toggle('on',showEdges);};
  bS.onclick=()=>{spin=!spin;bS.textContent='spin: '+(spin?'on':'off');bS.classList.toggle('on',spin);};
  let gseed=1;
  bG.onclick=()=>{gseed=(Math.random()*1e9)|0;const p=genAnchor(gseed);loadPayload(p,p.meta);};
  bM.onclick=()=>{loadPayload(DATA,{real:true});};

  loadPayload(DATA,{real:true});
  requestAnimationFrame(frame);
})();
</script>"""

out = HTML.replace("__DATA__", DATA_JS)
(HERE/"anchor_mind.html").write_text(out)
print("wrote anchor_mind.html", len(out)//1024, "KB")
