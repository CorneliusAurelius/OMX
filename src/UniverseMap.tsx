import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

// ===============================================================
// UniverseMap — GPU-Accelerated vΩ Explorer with Exoplanet Overlay
// WebGL2 instanced stars + GHZ shader + quadtree + real-data overlay
// Single-file TSX. Preview-safe. Deterministic seed. Snapshot + tests.
// ===============================================================

// ---------------- Types ----------------

type Vec2 = { x: number; y: number };

type Star = {
  id: number;
  R: number; // kpc radial
  z: number; // kpc vertical
  theta: number;
  pos: Vec2; // world coords (kpc) on galaxy plane
  mag: number; // base point size
  lambda: number;
  omegaA: number;
  deltaJp: number;
  Plife: number; // 0..1
  Aenv: number; // environment coherence proxy 0..1
  aInf: number; // informational albedo 0..1
  Hhuman: number; // 0..1 predictive human habitability
};

type ExoPoint = {
  name: string;
  pos: Vec2; // galactocentric world coords (kpc) on plane
  R: number;
  z: number; // kpc (vertical)
  theta: number; // rad
  Plife_obs?: number; // optional observed/derived
  Hhuman_obs?: number; // optional derived
};

// ---------------- Constants ----------------

const RMAX = 15; // kpc layout radius
const ZMAX = 1.5; // kpc vertical extent in priors
const BASE_PX_PER_KPC = 20;

// Earth reference
const EARTH_REF = { R: 8.2, z: 0.02, theta: 0 } as const;
function earthWorldPos(): Vec2 { return { x: EARTH_REF.R * Math.cos(EARTH_REF.theta), y: EARTH_REF.R * Math.sin(EARTH_REF.theta) }; }

// ---------------- Utilities ----------------

function clamp(v: number, a: number, b: number) { return Math.max(a, Math.min(b, v)); }
function norm(v: number, a: number, b: number) { return (v - a) / (b - a); }

// PRNG
function mulberry32(seed: number) { let t = seed >>> 0; return function() { t += 0x6D2B79F5; let r = Math.imul(t ^ (t >>> 15), 1 | t); r ^= r + Math.imul(r ^ (r >>> 7), 61 | r); return ((r ^ (r >>> 14)) >>> 0) / 4294967296; }; }
function hashSeed(s: string): number { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

// Deg/Rad
const DEG = Math.PI / 180;

// ---------------- vΩ fields ----------------

function fractalGHZ(R: number, z: number): number { const Df = 1.8; return Math.exp(-Math.pow(R / 8, Df) - Math.pow(Math.abs(z) / 0.3, Df)); }
function lambdaField(R: number, z: number): number { const lambda0 = 1e-36; return lambda0 * fractalGHZ(R, z) * 1.8; }
function omegaAField(): number { return 7.83; }
function aInfo(DKL: number): number { return 1 - Math.exp(-DKL); }
function deltaJPrime(Senv_over_hbar: number, DKL_atm: number, Aenv: number, lambdaVal: number, omegaA: number, GHZ: number): number {
  const Xi1 = Math.abs(Senv_over_hbar + DKL_atm - lambdaVal * Aenv) / (omegaA * Math.max(GHZ, 1e-6));
  const d = lambdaVal * 0.05 || 1e-38; const dJ = (lam: number) => Math.abs(Senv_over_hbar + DKL_atm - lam * Aenv);
  const Xi2 = 0.15 * Math.abs(dJ(lambdaVal + d) - 2 * dJ(lambdaVal) + dJ(lambdaVal - d));
  const Xi3 = (1 + 0.5 * Math.tanh(DKL_atm)) / 1.5; // smooth bridge proxy
  const hbar = 1.054e-34; const beta = 25;
  const soft = (1 / beta) * Math.log(Math.exp(beta * Xi1) + Math.exp(beta * (Xi2 / 3)) + Math.exp(beta * (Math.abs(Xi3 - 1) / 3)));
  return hbar * omegaA * soft;
}
function lifeProbability(deltaJp: number, omegaA: number, GHZ: number, curvature = 0): number {
  const hbar = 1.054e-34; const base = GHZ * Math.exp(-deltaJp / (hbar * omegaA)); const penalty = Math.exp(-0.1 * Math.abs(curvature)); return clamp(base * penalty, 0, 1);
}

// Predictive Human Habitability H_human = L* × f_env × f_temp × f_rad
function humanHabitability(Lstar: number, Aenv: number, R: number): number {
  const f_env = clamp(0.5 + 0.5 * Aenv, 0, 1);
  const f_temp = Math.exp(-Math.pow((R - 8.2) / 3.0, 2));
  const f_rad = Math.exp(-Math.max(0, 5 - R) / 5);
  return clamp(Lstar * f_env * f_temp * f_rad, 0, 1);
}

// Color maps
function colorPlife(t: number): [number, number, number] { const r = Math.floor(255 * clamp(2 * (1 - t), 0, 1)); const g = Math.floor(255 * clamp(2 * Math.min(t, 1 - Math.abs(t - 0.5) * 2), 0, 1)); const b = Math.floor(255 * clamp(2 * t, 0, 1)); return [r, g, b]; }
function colorHhuman(t: number): [number, number, number] { const a = clamp(t, 0, 1); const g = Math.floor(255 * a); const r = Math.floor(150 * a + 100 * (1 - a)); const b = Math.floor(255 * (0.6 + 0.4 * a)); return [r, g, b]; }

// ---------------- Star generation ----------------

function makeStars(count: number, seed = 12345): Star[] {
  const rnd = mulberry32(seed); const stars: Star[] = []; let id = 0;
  while (stars.length < count) {
    const R = RMAX * Math.pow(rnd(), 0.6); const z = (rnd() * 2 - 1) * ZMAX * Math.pow(rnd(), 0.5); const theta = rnd() * Math.PI * 2; const G = fractalGHZ(R, z);
    if (rnd() > G + 0.05) continue;
    const DKL = 0.002 + 0.015 * G + 0.003 * (rnd() - 0.5); const a_inf = aInfo(DKL);
    const Aenv = 1.0 * G + 0.1 * (1 - norm(R, 0, RMAX));
    const lambdaVal = lambdaField(R, z); const omegaA = omegaAField();
    const Lstar_rel = 1 + 0.5 * (rnd() - 0.5); const d_rel = 1 + R / RMAX; // demo proxy
    const Senv_over_hbar = (Lstar_rel * (1 - a_inf) / (4 * Math.PI * d_rel * d_rel)) / 1.054e-34;
    const dJp = deltaJPrime(Senv_over_hbar, DKL, Aenv, lambdaVal, omegaA, G);
    const curvature = 0.05 * (1 - G); const P = lifeProbability(dJp, omegaA, G, curvature);
    const Lstar = G * Math.exp(-dJp / (1.054e-34 * omegaA));
    const Hhuman = humanHabitability(Lstar, Aenv, R);
    const x = R * Math.cos(theta); const y = R * Math.sin(theta); const mag = 0.8 + 0.4 * rnd();
    stars.push({ id: id++, R, z, theta, pos: { x, y }, mag, lambda: lambdaVal, omegaA, deltaJp: dJp, Plife: P, Aenv, aInf: a_inf, Hhuman });
  }
  return stars;
}

// ---------------- Quadtree ----------------

class QTNode {
  bx: number; by: number; bw: number; bh: number; points: Star[] = []; div = false; nw?: QTNode; ne?: QTNode; sw?: QTNode; se?: QTNode;
  constructor(x: number, y: number, w: number, h: number, private cap = 32, private depth = 0, private maxDepth = 12) { this.bx = x; this.by = y; this.bw = w; this.bh = h; }
  contains(p: Vec2) { return p.x >= this.bx && p.x < this.bx + this.bw && p.y >= this.by && p.y < this.by + this.bh; }
  insert(s: Star): boolean { if (!this.contains(s.pos)) return false; if (this.points.length < this.cap || this.depth >= this.maxDepth) { this.points.push(s); return true; } if (!this.div) this.subdivide(); return this.nw!.insert(s) || this.ne!.insert(s) || this.sw!.insert(s) || this.se!.insert(s); }
  subdivide() { this.div = true; const hw = this.bw / 2, hh = this.bh / 2, d = this.depth + 1; this.nw = new QTNode(this.bx, this.by, hw, hh, this.cap, d, this.maxDepth); this.ne = new QTNode(this.bx + hw, this.by, hw, hh, this.cap, d, this.maxDepth); this.sw = new QTNode(this.bx, this.by + hh, hw, hh, this.cap, d, this.maxDepth); this.se = new QTNode(this.bx + hw, this.by + hh, hw, hh, this.cap, d, this.maxDepth); for (const p of this.points) { this.nw.insert(p) || this.ne!.insert(p) || this.sw!.insert(p) || this.se!.insert(p); } this.points.length = 0; }
  nearest(q: Vec2, r: number, best?: { s: Star; d2: number }): { s: Star; d2: number } | undefined {
    const distRect = (x: number, y: number, w: number, h: number) => { const dx = q.x < x ? x - q.x : q.x > x + w ? q.x - (x + w) : 0; const dy = q.y < y ? y - q.y : q.y > y + h ? q.y - (y + h) : 0; return dx * dx + dy * dy; };
    const r2 = r * r; if (best && distRect(this.bx, this.by, this.bw, this.bh) > best.d2) return best;
    if (!this.div) { for (const s of this.points) { const dx = s.pos.x - q.x, dy = s.pos.y - q.y; const d2 = dx * dx + dy * dy; if (d2 <= r2 && (!best || d2 < best.d2)) best = { s, d2 }; } return best; }
    best = this.nw!.nearest(q, r, best); best = this.ne!.nearest(q, r, best); best = this.sw!.nearest(q, r, best); best = this.se!.nearest(q, r, best); return best;
  }
}
function buildQuadtree(stars: Star[]): QTNode { const min = -RMAX, size = RMAX * 2; const root = new QTNode(min, min, size, size, 32); for (const s of stars) root.insert(s); return root; }

// ---------------- WebGL2 helpers ----------------

function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const sh = gl.createShader(type)!; gl.shaderSource(sh, source); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { throw new Error(String(gl.getShaderInfoLog(sh))); }
  return sh;
}
function createProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram { const p = gl.createProgram()!; const v = createShader(gl, gl.VERTEX_SHADER, vs); const f = createShader(gl, gl.FRAGMENT_SHADER, fs); gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { throw new Error(String(gl.getProgramInfoLog(p))); } gl.deleteShader(v); gl.deleteShader(f); return p; }

// ---------------- Exoplanet transform (RA/Dec/pc -> galactocentric kpc) ----------------

// IAU 1958 / J2000 constants
const ALPHA_G = 192.85948 * DEG;
const DELTA_G = 27.12825 * DEG;
const L_OMEGA = 32.93192 * DEG;

function radecToGalactic(raDeg: number, decDeg: number): { l: number; b: number } {
  const ra = raDeg * Math.PI / 180; const dec = decDeg * Math.PI / 180;
  const sinb = Math.sin(dec) * Math.sin(DELTA_G) + Math.cos(dec) * Math.cos(DELTA_G) * Math.cos(ra - ALPHA_G);
  const y = Math.sin(ra - ALPHA_G) * Math.cos(dec);
  const x = Math.cos(dec) * Math.sin(DELTA_G) * Math.cos(ra - ALPHA_G) - Math.sin(dec) * Math.cos(DELTA_G);
  let l = Math.atan2(y, x) + L_OMEGA; if (l < 0) l += 2 * Math.PI; if (l >= 2 * Math.PI) l -= 2 * Math.PI;
  const b = Math.asin(clamp(sinb, -1, 1));
  return { l, b };
}

function radecDistToGalactocentric(raDeg: number, decDeg: number, dist_pc: number): { x: number; y: number; z: number } {
  const { l, b } = radecToGalactic(raDeg, decDeg);
  const d = dist_pc / 1000; // kpc
  const xh = d * Math.cos(b) * Math.cos(l);
  const yh = d * Math.cos(b) * Math.sin(l);
  const zh = d * Math.sin(b);
  const x = 8.2 - xh; const y = 0 + yh; const z = 0.02 + zh; // Sun at (8.2, 0, 0.02)
  return { x, y, z };
}

function exoFromRADEC(name: string, ra_deg: number, dec_deg: number, dist_pc: number, Plife_obs?: number, Hhuman_obs?: number): ExoPoint {
  const { x, y, z } = radecDistToGalactocentric(ra_deg, dec_deg, dist_pc); const R = Math.hypot(x, y); const theta = Math.atan2(y, x);
  return { name, pos: { x, y }, R, z, theta, Plife_obs, Hhuman_obs };
}

// Built-in tiny demo overlay (approximate)
function demoExoplanets(): ExoPoint[] {
  return [
    exoFromRADEC("Proxima b", 217.4292, -62.6795, 1294, 0.3, 0.25),
    exoFromRADEC("TRAPPIST-1 e", 346.6220, -5.0413, 12, 0.4, 0.35),
    exoFromRADEC("TRAPPIST-1 f", 346.6220, -5.0413, 12, 0.42, 0.36),
    exoFromRADEC("Kepler-452 b", 289.806, 44.494, 500, 0.2, 0.2),
  ];
}

// ---------------- Component ----------------

export default function UniverseMap(): JSX.Element {
  const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [uiSize, setUiSize] = useState<{ w: number; h: number }>({ w: 900, h: 600 });

  // Camera in refs to avoid re-render on pan/zoom
  const zoomRef = useRef(1); const offsetRef = useRef<Vec2>({ x: 0, y: 0 });
  const [zoomView, setZoomView] = useState(1); const [offsetView, setOffsetView] = useState<Vec2>({ x: 0, y: 0 });

  // Toggles
  const [ghzOn, setGhzOn] = useState(true); const [ghzOpacity, setGhzOpacity] = useState(0.35);
  const [plifeMin, setPlifeMin] = useState(0.0);
  const [useHhumanColor, setUseHhumanColor] = useState(true);

  // Filters
  const [distMax, setDistMax] = useState(15); // kpc
  const [radMax, setRadMax] = useState(1.0); // proxy 0..1
  const [atmMin, setAtmMin] = useState(0.0); // aInfo 0..1

  // Data
  const stars = useMemo(() => makeStars(10000, hashSeed("map-2.0-universe")), []);
  const quadtree = useMemo(() => buildQuadtree(stars), [stars]);

  // Exoplanet overlay data + input panel
  const [exo, setExo] = useState<ExoPoint[]>(demoExoplanets());
  const [exoOpen, setExoOpen] = useState(false);
  const [exoText, setExoText] = useState<string>(JSON.stringify({
    format: "array of objects",
    fields: ["name", "ra_deg", "dec_deg", "dist_pc", "Plife_obs?", "Hhuman_obs?"],
    example: [
      { name: "Proxima b", ra_deg: 217.4292, dec_deg: -62.6795, dist_pc: 1.301, Plife_obs: 0.3, Hhuman_obs: 0.25 },
      { name: "Kepler-452 b", ra_deg: 289.806, dec_deg: 44.494, dist_pc: 500 }
    ]
  }, null, 2));

  // GPU resources
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programsRef = useRef<{ stars?: WebGLProgram; ghz?: WebGLProgram; earth?: WebGLProgram; exo?: WebGLProgram }>({});
  const buffersRef = useRef<{ pos?: WebGLBuffer; attr?: WebGLBuffer; earth?: WebGLBuffer; vao?: WebGLVertexArrayObject; exo?: WebGLBuffer }>({});

  // Hover + select
  const [hoverStar, setHoverStar] = useState<Star | null>(null); const [hoverPos, setHoverPos] = useState<Vec2 | null>(null); const [selected, setSelected] = useState<Star | null>(null);

  // Self-test
  const [selfTest, setSelfTest] = useState<{ framems: number; ok: boolean }>({ framems: 0, ok: false });

  // Persistence
  useEffect(() => { try { const raw = localStorage.getItem("umap_camera_gl"); if (raw) { const { z, ox, oy } = JSON.parse(raw); if (typeof z === "number") { zoomRef.current = clamp(z, 0.2, 5); setZoomView(zoomRef.current); } if (typeof ox === "number" && typeof oy === "number") { offsetRef.current = { x: ox, y: oy }; setOffsetView(offsetRef.current); } } } catch {} }, []);
  useEffect(() => { try { localStorage.setItem("umap_camera_gl", JSON.stringify({ z: zoomRef.current, ox: offsetRef.current.x, oy: offsetRef.current.y })); } catch {} });

  // ResizeObserver
  useLayoutEffect(() => {
    const canvas = glCanvasRef.current; if (!canvas) return; const parent = canvas.parentElement; if (!parent) return; let frame = 0;
    const ro = new ResizeObserver(entries => { const e = entries[0]; if (!e) return; cancelAnimationFrame(frame); frame = requestAnimationFrame(() => { const w = Math.max(320, Math.floor(e.contentRect.width)); const h = Math.max(240, Math.floor(e.contentRect.height)); setUiSize(prev => prev.w === w && prev.h === h ? prev : { w, h }); }); });
    ro.observe(parent); return () => { cancelAnimationFrame(frame); ro.disconnect(); };
  }, []);

  // WebGL init / render function stored on canvas
  useEffect(() => {
    const canvas = glCanvasRef.current!; const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(uiSize.w * dpr); canvas.height = Math.floor(uiSize.h * dpr); canvas.style.width = `${uiSize.w}px`; canvas.style.height = `${uiSize.h}px`;
    let gl = glRef.current; if (!gl) { gl = canvas.getContext("webgl2", { antialias: true, alpha: false }) as WebGL2RenderingContext | null; if (!gl) throw new Error("WebGL2 not available"); glRef.current = gl; }
    gl.viewport(0, 0, canvas.width, canvas.height);

    // Programs
    if (!programsRef.current.stars) {
      const vs = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_attr; // Plife, Hhuman, size, Rnorm
layout(location=2) in vec3 a_more; // Aenv, aInf, radProxy
uniform vec2 u_offset; uniform float u_scale; uniform vec2 u_viewport;
out vec4 v_attr; out vec3 v_more;
void main(){ vec2 p=(a_pos+u_offset)*u_scale; vec2 ndc=vec2(((p.x+u_viewport.x*0.5)/u_viewport.x)*2.0-1.0, ((p.y+u_viewport.y*0.5)/u_viewport.y)*2.0-1.0); gl_Position=vec4(ndc,0.0,1.0); gl_PointSize=max(1.0, a_attr.z*(0.7+0.3*(u_scale/` + BASE_PX_PER_KPC + `.0))); v_attr=a_attr; v_more=a_more; }`;
      const fs = `#version 300 es
precision highp float;
in vec4 v_attr; in vec3 v_more; out vec4 outColor;
uniform float u_plifeMin, u_distMax, u_radMax, u_atmMin; uniform bool u_useHhumanColor;
vec3 cmapPlife(float t){ float r=clamp(2.0*(1.0-t),0.0,1.0); float g=clamp(2.0*min(t,1.0-abs(t-0.5)*2.0),0.0,1.0); float b=clamp(2.0*t,0.0,1.0); return vec3(r,g,b);}
vec3 cmapH(float t){ float a=clamp(t,0.0,1.0); float g=a; float r=(150.0/255.0)*a + (100.0/255.0)*(1.0-a); float b=(0.6+0.4*a); return vec3(r,g,b);}
void main(){ float pl=v_attr.x, hh=v_attr.y, Rn=v_attr.w, rad=v_more.z, atm=v_more.y; if (pl < u_plifeMin) discard; if (Rn*` + RMAX + `.0 > u_distMax) discard; if (rad > u_radMax) discard; if (atm < u_atmMin) discard; vec2 c=gl_PointCoord-vec2(0.5); float d2=dot(c,c); if (d2>0.25) discard; float fall=smoothstep(0.25,0.0,d2); vec3 col = u_useHhumanColor ? cmapH(hh) : cmapPlife(pl); outColor=vec4(col*fall,1.0); }`;
      programsRef.current.stars = createProgram(gl, vs, fs);
    }
    if (!programsRef.current.ghz) {
      const vs = `#version 300 es
precision highp float; const vec2 v[6]=vec2[6](vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.)); out vec2 ndc; void main(){ ndc = v[gl_VertexID]; gl_Position=vec4(ndc,0.,1.);} `;
      const fs = `#version 300 es
precision highp float; in vec2 ndc; out vec4 outColor; uniform vec2 u_offset; uniform float u_scale; uniform vec2 u_viewport; uniform float u_opacity; float ghz(vec2 w){ float R=length(w); float Df=1.8; return exp(-pow(R/8.0,Df)); } void main(){ vec2 scr=((ndc+1.0)*0.5)*u_viewport; vec2 world = scr/u_scale - u_offset; float g=ghz(world); vec3 teal=vec3((10.0+30.0*g)/255.0,(120.0+120.0*g)/255.0,(140.0+100.0*g)/255.0); outColor=vec4(teal, clamp(g,0.0,1.0)*u_opacity); }`;
      programsRef.current.ghz = createProgram(gl, vs, fs);
    }
    if (!programsRef.current.earth) {
      const vs = `#version 300 es
precision highp float; layout(location=0) in vec2 a_pos; uniform vec2 u_offset; uniform float u_scale; uniform vec2 u_viewport; void main(){ vec2 p=(a_pos+u_offset)*u_scale; vec2 ndc=vec2(((p.x+u_viewport.x*0.5)/u_viewport.x)*2.0-1.0, ((p.y+u_viewport.y*0.5)/u_viewport.y)*2.0-1.0); gl_Position=vec4(ndc,0.0,1.0);} `;
      const fs = `#version 300 es
precision highp float; out vec4 outColor; void main(){ outColor=vec4(0.98,0.82,0.18,1.0);} `;
      programsRef.current.earth = createProgram(gl, vs, fs);
    }
    if (!programsRef.current.exo) {
      const vs = `#version 300 es
precision highp float; layout(location=0) in vec2 a_pos; uniform vec2 u_offset; uniform float u_scale; uniform vec2 u_viewport; void main(){ vec2 p=(a_pos+u_offset)*u_scale; vec2 ndc=vec2(((p.x+u_viewport.x*0.5)/u_viewport.x)*2.0-1.0, ((p.y+u_viewport.y*0.5)/u_viewport.y)*2.0-1.0); gl_Position=vec4(ndc,0.0,1.0); gl_PointSize=6.0;} `;
      const fs = `#version 300 es
precision highp float; out vec4 outColor; void main(){ vec2 c=gl_PointCoord-vec2(0.5); float d2=dot(c,c); if(d2>0.25) discard; float fall=smoothstep(0.25,0.0,d2); outColor=vec4(0.9,0.95,1.0,1.0)*fall; }`;
      programsRef.current.exo = createProgram(gl, vs, fs);
    }

    // Buffers/VAO for stars (static)
    if (!buffersRef.current.vao) {
      const vao = gl.createVertexArray()!; gl.bindVertexArray(vao);
      const pos = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, pos);
      const attr = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, attr);
      const N = stars.length; const posData = new Float32Array(N * 2); const attrData = new Float32Array(N * 7);
      for (let i = 0; i < N; i++) {
        const s = stars[i]; posData[i * 2] = s.pos.x; posData[i * 2 + 1] = s.pos.y;
        const Rn = s.R / RMAX; const radProxy = clamp(1 - s.R / (RMAX + 1), 0, 1);
        attrData[i * 7 + 0] = s.Plife; attrData[i * 7 + 1] = s.Hhuman; attrData[i * 7 + 2] = s.mag; attrData[i * 7 + 3] = Rn; attrData[i * 7 + 4] = s.Aenv; attrData[i * 7 + 5] = s.aInf; attrData[i * 7 + 6] = radProxy;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.bufferData(gl.ARRAY_BUFFER, posData, gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, attr); gl.bufferData(gl.ARRAY_BUFFER, attrData, gl.STATIC_DRAW); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 7 * 4, 0); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 7 * 4, 4 * 4);
      buffersRef.current = { ...buffersRef.current, vao, pos, attr };
      gl.bindVertexArray(null);
    }

    // Earth cross geometry
    if (!buffersRef.current.earth) {
      const e = earthWorldPos(); const sz = 0.3; const verts = new Float32Array([ e.x - sz, e.y, e.x + sz, e.y, e.x, e.y - sz, e.x, e.y + sz ]);
      const buf = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW); buffersRef.current.earth = buf;
    }

    // Exo buffer (dynamic)
    const exoBuf = buffersRef.current.exo || gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, exoBuf);
    const exoData = new Float32Array(exo.flatMap(p => [p.pos.x, p.pos.y])); gl.bufferData(gl.ARRAY_BUFFER, exoData, gl.DYNAMIC_DRAW); buffersRef.current.exo = exoBuf;

    renderGL(); (canvas as any)._render = renderGL;

    function renderGL() {
      const gl = glRef.current!; const canvas = gl.canvas as HTMLCanvasElement; gl.viewport(0, 0, canvas.width, canvas.height);
      const t0 = performance.now(); gl.clearColor(0.043, 0.066, 0.125, 1.0); gl.clear(gl.COLOR_BUFFER_BIT);
      // GHZ underlay
      if (ghzOn) { const p = programsRef.current.ghz!; gl.useProgram(p); gl.uniform2f(gl.getUniformLocation(p, "u_offset"), offsetRef.current.x, offsetRef.current.y); gl.uniform1f(gl.getUniformLocation(p, "u_scale"), BASE_PX_PER_KPC * zoomRef.current * (window.devicePixelRatio ? Math.min(window.devicePixelRatio, 2) : 1)); gl.uniform2f(gl.getUniformLocation(p, "u_viewport"), canvas.width, canvas.height); gl.uniform1f(gl.getUniformLocation(p, "u_opacity"), ghzOpacity); gl.drawArrays(gl.TRIANGLES, 0, 6); }
      // Stars
      { const p = programsRef.current.stars!; gl.useProgram(p); gl.bindVertexArray(buffersRef.current.vao!); gl.uniform2f(gl.getUniformLocation(p, "u_offset"), offsetRef.current.x, offsetRef.current.y); gl.uniform1f(gl.getUniformLocation(p, "u_scale"), BASE_PX_PER_KPC * zoomRef.current * (window.devicePixelRatio ? Math.min(window.devicePixelRatio, 2) : 1)); gl.uniform2f(gl.getUniformLocation(p, "u_viewport"), canvas.width, canvas.height); gl.uniform1f(gl.getUniformLocation(p, "u_plifeMin"), plifeMin); gl.uniform1f(gl.getUniformLocation(p, "u_distMax"), distMax); gl.uniform1f(gl.getUniformLocation(p, "u_radMax"), radMax); gl.uniform1f(gl.getUniformLocation(p, "u_atmMin"), atmMin); gl.uniform1i(gl.getUniformLocation(p, "u_useHhumanColor"), useHhumanColor ? 1 : 0); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE); gl.drawArrays(gl.POINTS, 0, stars.length); gl.disable(gl.BLEND); gl.bindVertexArray(null); }
      // Earth cross
      { const p = programsRef.current.earth!; gl.useProgram(p); gl.uniform2f(gl.getUniformLocation(p, "u_offset"), offsetRef.current.x, offsetRef.current.y); gl.uniform1f(gl.getUniformLocation(p, "u_scale"), BASE_PX_PER_KPC * zoomRef.current * (window.devicePixelRatio ? Math.min(window.devicePixelRatio, 2) : 1)); gl.uniform2f(gl.getUniformLocation(p, "u_viewport"), canvas.width, canvas.height); gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.earth!); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); gl.drawArrays(gl.LINES, 0, 4); gl.disableVertexAttribArray(0); }
      // Exoplanet overlay
      if (exo.length) { const p = programsRef.current.exo!; gl.useProgram(p); gl.uniform2f(gl.getUniformLocation(p, "u_offset"), offsetRef.current.x, offsetRef.current.y); gl.uniform1f(gl.getUniformLocation(p, "u_scale"), BASE_PX_PER_KPC * zoomRef.current * (window.devicePixelRatio ? Math.min(window.devicePixelRatio, 2) : 1)); gl.uniform2f(gl.getUniformLocation(p, "u_viewport"), canvas.width, canvas.height); gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.exo!); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.drawArrays(gl.POINTS, 0, exo.length); gl.disable(gl.BLEND); gl.disableVertexAttribArray(0); }
      gl.finish(); const t1 = performance.now(); setSelfTest({ framems: t1 - t0, ok: true }); setZoomView(zoomRef.current); setOffsetView({ ...offsetRef.current });
    }
  }, [uiSize, stars, ghzOn, ghzOpacity, plifeMin, distMax, radMax, atmMin, useHhumanColor, exo]);

  // Interaction
  const draggingRef = useRef(false); const dragStart = useRef<{ x: number; y: number; off: Vec2 } | null>(null);
  const onPointerDown = useCallback((e: React.PointerEvent) => { const el = e.currentTarget as HTMLCanvasElement; el.setPointerCapture(e.pointerId); draggingRef.current = true; dragStart.current = { x: e.clientX, y: e.clientY, off: { ...offsetRef.current } }; }, []);
  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const canvas = glCanvasRef.current!; const render = (canvas as any)._render as (() => void) | undefined;
    if (draggingRef.current && dragStart.current) { const scale = BASE_PX_PER_KPC * zoomRef.current; const dx = (e.clientX - dragStart.current.x) / scale; const dy = (e.clientY - dragStart.current.y) / scale; offsetRef.current = { x: dragStart.current.off.x + dx, y: dragStart.current.off.y + dy }; render && render(); }
    const rect = canvas.getBoundingClientRect(); const sx = e.clientX - rect.left, sy = e.clientY - rect.top; setHoverPos({ x: sx, y: sy });
    const scale = BASE_PX_PER_KPC * zoomRef.current; const w = { x: (sx - uiSize.w / 2) / scale - offsetRef.current.x, y: (sy - uiSize.h / 2) / scale - offsetRef.current.y };
    const hit = quadtree.nearest(w, clamp(10 / Math.max(scale, 1e-6), 0.1, 1.0)); setHoverStar(hit ? hit.s : null);
  }, [quadtree, uiSize]);
  const onPointerUp = useCallback((e: React.PointerEvent) => { const el = e.currentTarget as HTMLCanvasElement; try { el.releasePointerCapture(e.pointerId); } catch {} draggingRef.current = false; dragStart.current = null; }, []);
  const onWheel = useCallback((e: React.WheelEvent) => { e.preventDefault(); const canvas = glCanvasRef.current!; const render = (canvas as any)._render as (() => void) | undefined; const delta = -e.deltaY; const factor = delta > 0 ? 1.1 : 0.9; const rect = canvas.getBoundingClientRect(); const sx = e.clientX - rect.left, sy = e.clientY - rect.top; const scale = BASE_PX_PER_KPC * zoomRef.current; const before = { x: (sx - uiSize.w / 2) / scale - offsetRef.current.x, y: (sy - uiSize.h / 2) / scale - offsetRef.current.y }; zoomRef.current = clamp(zoomRef.current * factor, 0.2, 5); const scale2 = BASE_PX_PER_KPC * zoomRef.current; const after = { x: (sx - uiSize.w / 2) / scale2 - offsetRef.current.x, y: (sy - uiSize.h / 2) / scale2 - offsetRef.current.y }; offsetRef.current = { x: offsetRef.current.x + (before.x - after.x), y: offsetRef.current.y + (before.y - after.y) }; render && render(); }, [uiSize]);

  // Buttons
  const centerOnEarth = useCallback(() => { const e = earthWorldPos(); offsetRef.current = { x: -e.x, y: -e.y }; const canvas = glCanvasRef.current!; ((canvas as any)._render as (() => void))(); }, []);
  const resetView = useCallback(() => { zoomRef.current = 1; offsetRef.current = { x: 0, y: 0 }; const canvas = glCanvasRef.current!; ((canvas as any)._render as (() => void))(); }, []);
  const snapshotPNG = useCallback(() => { const cvs = glCanvasRef.current; if (!cvs) return; const url = cvs.toDataURL("image/png"); const a = document.createElement("a"); a.href = url; a.download = `universe-map-${Date.now()}.png`; document.body.appendChild(a); a.click(); a.remove(); }, []);

  // Life Focus (highest mean H_human tile)
  const lifeFocus = useCallback(() => {
    const tile = 1; const cols = Math.ceil((2 * RMAX) / tile); const origin = { x: -RMAX, y: -RMAX };
    const sums = new Float32Array(cols * cols); const counts = new Uint16Array(cols * cols);
    for (const s of stars) { const i = Math.floor((s.pos.x - origin.x) / tile); const j = Math.floor((s.pos.y - origin.y) / tile); if (i < 0 || j < 0 || i >= cols || j >= cols) continue; const k = j * cols + i; sums[k] += s.Hhuman; counts[k]++; }
    let bestK = 0, best = -1; for (let k = 0; k < sums.length; k++) { const m = counts[k] ? sums[k] / counts[k] : 0; if (m > best) { best = m; bestK = k; } }
    const bi = bestK % cols, bj = Math.floor(bestK / cols); const cx = origin.x + (bi + 0.5) * tile; const cy = origin.y + (bj + 0.5) * tile; offsetRef.current = { x: -cx, y: -cy }; const canvas = glCanvasRef.current!; ((canvas as any)._render as (() => void))();
  }, [stars]);

  // Keyboard
  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    const key = e.key; if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","+","=","-","_","e","E","r","R","f","F"].includes(key)) e.preventDefault();
    const pan = 0.5; if (key === "ArrowUp") offsetRef.current = { ...offsetRef.current, y: offsetRef.current.y + pan }; else if (key === "ArrowDown") offsetRef.current = { ...offsetRef.current, y: offsetRef.current.y - pan }; else if (key === "ArrowLeft") offsetRef.current = { ...offsetRef.current, x: offsetRef.current.x + pan }; else if (key === "ArrowRight") offsetRef.current = { ...offsetRef.current, x: offsetRef.current.x - pan }; else if (key === "+" || key === "=") zoomRef.current = clamp(zoomRef.current * 1.1, 0.2, 5); else if (key === "-" || key === "_") zoomRef.current = clamp(zoomRef.current * 0.9, 0.2, 5); else if (key === "e" || key === "E") centerOnEarth(); else if (key === "r" || key === "R") resetView(); else if (key === "f" || key === "F") lifeFocus(); const canvas = glCanvasRef.current!; ((canvas as any)._render as (() => void))();
  }, [centerOnEarth, resetView, lifeFocus]);

  // Inspector
  const inspector = useMemo(() => {
    if (!selected) return null; const s = selected; const e = earthWorldPos(); const d = Math.hypot(s.pos.x - e.x, s.pos.y - e.y);
    return (
      <div className="mt-2 rounded-2xl bg-slate-900/80 p-3 text-slate-100 shadow-lg">
        <div className="text-sm font-semibold">Star #{s.id}</div>
        <div className="text-xs opacity-90">R={s.R.toFixed(2)} kpc, z={s.z.toFixed(2)} kpc</div>
        <div className="text-xs opacity-90">λ={s.lambda.toExponential(2)}, Ω_A={s.omegaA.toFixed(2)} Hz</div>
        <div className="text-xs opacity-90">δJ′={s.deltaJp.toExponential(2)}, P′_life={(s.Plife).toFixed(3)}</div>
        <div className="text-xs opacity-90">H_human={(s.Hhuman).toFixed(3)}</div>
        <div className="text-xs opacity-90">Δ to Earth: {d.toFixed(2)} kpc (proj)</div>
      </div>
    );
  }, [selected]);

  // Click select
  const onClick = useCallback((e: React.MouseEvent) => {
    const canvas = glCanvasRef.current!; const rect = canvas.getBoundingClientRect(); const sx = e.clientX - rect.left, sy = e.clientY - rect.top; const scale = BASE_PX_PER_KPC * zoomRef.current; const w = { x: (sx - uiSize.w / 2) / scale - offsetRef.current.x, y: (sy - uiSize.h / 2) / scale - offsetRef.current.y }; const hit = quadtree.nearest(w, 0.5); setSelected(hit ? hit.s : null);
  }, [quadtree, uiSize]);

  // Legend gradient canvas
  const legendRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => { const c = legendRef.current; if (!c) return; const w = 160, h = 10; c.width = w; c.height = h; const ctx = c.getContext("2d"); if (!ctx) return; for (let i = 0; i < w; i++) { const t = i / (w - 1); const [r,g,b] = useHhumanColor ? colorHhuman(t) : colorPlife(t); ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(i, 0, 1, h); } }, [useHhumanColor]);

  // Update exo buffer when exo data changes
  useEffect(() => {
    const gl = glRef.current; if (!gl || !buffersRef.current.exo) return; gl.bindBuffer(gl.ARRAY_BUFFER, buffersRef.current.exo); const exoData = new Float32Array(exo.flatMap(p => [p.pos.x, p.pos.y])); gl.bufferData(gl.ARRAY_BUFFER, exoData, gl.DYNAMIC_DRAW); const canvas = glCanvasRef.current!; const r = (canvas as any)._render as (() => void) | undefined; r && r();
  }, [exo]);

  // Handlers for overlay panel
  const loadDemo = useCallback(() => { setExo(demoExoplanets()); }, []);
  const clearExo = useCallback(() => { setExo([]); }, []);
  const applyExo = useCallback(() => {
    try {
      const arr = JSON.parse(exoText);
      if (!Array.isArray(arr)) return alert("JSON must be an array of objects");
      const mapped: ExoPoint[] = arr.map((o: any) => {
        if (typeof o.R_kpc === 'number' && typeof o.z_kpc === 'number' && typeof o.theta_deg === 'number') {
          const R = o.R_kpc, z = o.z_kpc, theta = o.theta_deg * Math.PI / 180; const x = R * Math.cos(theta), y = R * Math.sin(theta);
          return { name: String(o.name || "exo"), pos: { x, y }, R, z, theta, Plife_obs: o.Plife_obs, Hhuman_obs: o.Hhuman_obs };
        }
        if (typeof o.ra_deg === 'number' && typeof o.dec_deg === 'number' && typeof o.dist_pc === 'number') {
          const ep = exoFromRADEC(String(o.name || "exo"), o.ra_deg, o.dec_deg, o.dist_pc, o.Plife_obs, o.Hhuman_obs); return ep;
        }
        throw new Error("Missing required fields");
      });
      setExo(mapped);
    } catch (e: any) { alert("Parse error: " + (e?.message || e)); }
  }, [exoText]);

  return (
    <div tabIndex={0} onKeyDown={onKeyDown} className="h-full w-full min-h-[600px] select-none bg-slate-950 text-slate-100 outline-none">
      {/* Controls */}
      <div className="absolute z-10 m-3 flex flex-wrap items-center gap-2">
        <button onClick={resetView} className="rounded-2xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-300">Reset</button>
        <button onClick={centerOnEarth} className="rounded-2xl bg-yellow-500 px-3 py-2 text-sm font-semibold text-black shadow hover:bg-yellow-400 focus:outline-none focus:ring-2 focus:ring-yellow-300">Earth</button>
        <button onClick={() => { zoomRef.current = clamp(zoomRef.current * 1.1, 0.2, 5); (glCanvasRef.current as any)._render(); }} className="rounded-2xl bg-green-600 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-green-500 focus:outline-none focus:ring-2 focus:ring-green-300">Zoom In</button>
        <button onClick={() => { zoomRef.current = clamp(zoomRef.current * 0.9, 0.2, 5); (glCanvasRef.current as any)._render(); }} className="rounded-2xl bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-red-500 focus:outline-none focus:ring-2 focus:ring-red-300">Zoom Out</button>
        <button onClick={snapshotPNG} className="rounded-2xl bg-slate-700 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-slate-600 focus:outline-none focus:ring-2 focus:ring-slate-400">Snapshot PNG</button>
        <button onClick={lifeFocus} className="rounded-2xl bg-purple-600 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-300">Life Focus</button>

        <label className="ml-2 flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><input type="checkbox" checked={ghzOn} onChange={e => { setGhzOn(e.target.checked); (glCanvasRef.current as any)._render(); }} /><span>GHZ</span></label>
        <label className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><span>Opacity</span><input type="range" min={0} max={1} step={0.05} value={ghzOpacity} onChange={e => { setGhzOpacity(parseFloat(e.target.value)); (glCanvasRef.current as any)._render(); }} /><span>{Math.round(ghzOpacity*100)}%</span></label>

        <label className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><span>P′_life ≥</span><input type="range" min={0} max={1} step={0.01} value={plifeMin} onChange={e => { setPlifeMin(parseFloat(e.target.value)); (glCanvasRef.current as any)._render(); }} /><span>{plifeMin.toFixed(2)}</span></label>

        <label className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><span>Color</span><select value={useHhumanColor ? 'H_human' : 'P_life'} onChange={e => { setUseHhumanColor(e.target.value === 'H_human'); (glCanvasRef.current as any)._render(); }} className="rounded bg-slate-800 px-2 py-1"><option value="H_human">H_human</option><option value="P_life">P′_life</option></select></label>

        <label className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><span>Dist ≤</span><input type="range" min={1} max={15} step={0.5} value={distMax} onChange={e => { setDistMax(parseFloat(e.target.value)); (glCanvasRef.current as any)._render(); }} /><span>{distMax.toFixed(1)} kpc</span></label>
        <label className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><span>Rad ≤</span><input type="range" min={0} max={1} step={0.01} value={radMax} onChange={e => { setRadMax(parseFloat(e.target.value)); (glCanvasRef.current as any)._render(); }} /><span>{radMax.toFixed(2)}</span></label>
        <label className="flex items-center gap-2 rounded-2xl bg-slate-900/80 px-3 py-2 text-xs"><span>Atm ≥</span><input type="range" min={0} max={1} step={0.01} value={atmMin} onChange={e => { setAtmMin(parseFloat(e.target.value)); (glCanvasRef.current as any)._render(); }} /><span>{atmMin.toFixed(2)}</span></label>

        <button onClick={() => setExoOpen(o => !o)} className="rounded-2xl bg-cyan-700 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-400">Exoplanets</button>

        <div className="rounded-2xl bg-slate-900/80 px-3 py-2 text-xs">Keys: ←↑→↓ pan, +/- zoom, E Earth, R reset, F Life Focus</div>
      </div>

      {/* Canvas */}
      <div className="relative h-full w-full">
        <canvas ref={glCanvasRef} className="block h-full w-full" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onWheel={onWheel} onClick={onClick} />

        {/* Hover tooltip */}
        {hoverStar && hoverPos && (
          <div className="pointer-events-none absolute z-20 rounded-xl bg-black/80 px-2 py-1 text-[11px] text-white shadow" style={{ left: Math.min(hoverPos.x + 12, uiSize.w - 200), top: Math.min(hoverPos.y + 12, uiSize.h - 80) }}>
            <div><span className="opacity-80">Hover</span> Star #{hoverStar.id}</div>
            <div className="opacity-80">P′_life {hoverStar.Plife.toFixed(3)} • H_human {hoverStar.Hhuman.toFixed(3)}</div>
            <div className="opacity-80">R {hoverStar.R.toFixed(2)} kpc</div>
          </div>
        )}

        {/* Info */}
        <div className="absolute right-3 top-3 z-10 w-96 rounded-2xl bg-slate-900/80 p-3 text-xs text-slate-100 shadow-lg">
          <div className="mb-1 text-sm font-semibold">Universe Map — GPU + Exo Overlay</div>
          <div>Stars: {stars.length} • Exo overlay: {exo.length}</div>
          <div>Zoom: {zoomView.toFixed(2)}×</div>
          <div>Offset: ({offsetView.x.toFixed(2)}, {offsetView.y.toFixed(2)}) kpc</div>
          <div>Earth: R≈{EARTH_REF.R} kpc, z≈{EARTH_REF.z} kpc</div>
          <div>Frame: {selfTest.framems.toFixed(2)} ms</div>
          <div>Self‑test: {selfTest.ok && selfTest.framems < 10 ? <span className="font-semibold text-green-400">passed</span> : <span className="font-semibold text-yellow-300">running</span>}</div>
          {inspector}
          <div className="mt-2">
            <div className="mb-1 font-semibold">Legend ({useHhumanColor ? 'H_human' : 'P′_life'})</div>
            <canvas ref={legendRef} className="h-2 w-40 rounded" />
            <div className="mt-1 flex justify-between"><span>0</span><span>0.5</span><span>1.0</span></div>
          </div>
        </div>

        {/* Exoplanet overlay panel */}
        {exoOpen && (
          <div className="absolute bottom-3 right-3 z-20 w-[520px] rounded-2xl bg-slate-900/95 p-3 text-xs text-slate-100 shadow-2xl">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm font-semibold">Exoplanet Overlay (paste JSON or use demo)</div>
              <button onClick={() => setExoOpen(false)} className="rounded bg-slate-700 px-2 py-1">Close</button>
            </div>
            <div className="mb-2 text-[11px] opacity-90">
              Accepted formats:
              <div className="mt-1">A) array like <code>{"[ { name, ra_deg, dec_deg, dist_pc, Plife_obs?, Hhuman_obs? }, ... ]"}</code></div>
              <div className="mt-1">B) array like <code>{"[ { name, R_kpc, z_kpc, theta_deg, Plife_obs?, Hhuman_obs? }, ... ]"}</code></div>
            </div>
            <textarea value={exoText} onChange={e => setExoText(e.target.value)} className="h-40 w-full rounded bg-slate-800 p-2 font-mono text-[11px]" />
            <div className="mt-2 flex gap-2">
              <button onClick={applyExo} className="rounded bg-emerald-600 px-3 py-2 text-white">Apply</button>
              <button onClick={loadDemo} className="rounded bg-indigo-600 px-3 py-2 text-white">Load Demo</button>
              <button onClick={clearExo} className="rounded bg-rose-600 px-3 py-2 text-white">Clear</button>
              <div className="flex-1 text-right opacity-80">Exoplanets draw as bright white dots on top.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

