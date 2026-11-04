// The left panel: a layer of the probe in its first principal axes, drawn on a canvas in 2D or in 3D (drag to rotate), with the
// chosen episode's path and a dot that follows the video.
import { CLASS_COLORS, cssVar } from "./colors.js";

const RING = 96; // points of one ellipse
const SPIN = 0.2; // radians per second of the slow turn of a 3D view: a full turn takes about 30 s

export class EmbeddingView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.layer = null;
    this.episode = null; // { path, alarm, onset, success }
    this.step = 0;
    this.mode3d = false;
    this.yaw = -0.6;
    this.pitch = 0.45;
    this.bounds = null;
    this.padBottom = this.padRight = 0; // the room taken by the legends that lie over the plot (see resize)
    this.dragging = false;
    this.spinning = false; // a frame loop is running
    this.visible = true;
    this.canTurn = !window.matchMedia("(prefers-reduced-motion: reduce)").matches; // visitors who ask for reduced motion get no automatic turn
    new IntersectionObserver(([entry]) => { this.visible = entry.isIntersecting; }).observe(canvas);
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener("pointerdown", (e) => this.startDrag(e));
    canvas.addEventListener("dblclick", () => { this.yaw = -0.6; this.pitch = 0.45; this.draw(); });
  }

  // Call again when a legend over the plot has changed. A legend that lies along the bottom or at the right side of the box takes
  // that room from the plot, so that the data stays clear of it.
  resize() {
    const node = this.canvas.parentElement, box = node.getBoundingClientRect();
    this.padBottom = this.padRight = 0;
    for (const legend of node.querySelectorAll(".overlay-bottom, .overlay-right")) {
      const r = legend.getBoundingClientRect();
      if (r.width > box.width * 0.6) this.padBottom = Math.max(this.padBottom, box.bottom - r.top + 8);
      else this.padRight = Math.max(this.padRight, box.right - r.left + 8);
    }
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  setLayer(layer) { this.layer = layer; this.fit(); this.draw(); }
  setEpisode(episode) { this.episode = episode; this.fit(); this.draw(); }
  setStep(step) { this.step = step; this.draw(); }
  setMode3d(on) { this.mode3d = on; this.draw(); if (on) this.turn(); }

  // The 3D view turns slowly about the vertical axis. It pauses while it is dragged, while it is off screen and while the tab is hidden,
  // and the loop ends when the view goes back to 2D.
  turn() {
    if (this.spinning || !this.canTurn) return;
    this.spinning = true;
    let last = performance.now();
    const frame = (now) => {
      if (!this.mode3d) { this.spinning = false; return; }
      if (this.visible && !this.dragging && !document.hidden) { this.yaw += ((now - last) / 1000) * SPIN; this.draw(); }
      last = now;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // The data box. Without `data-trim` it holds every sampled point, the ellipsoids and the episode's path, so that nothing is cut off.
  // With `data-trim="0.07"` (the head views) it is zoomed in on the dense core: the outermost 7% of the points on each side of every axis
  // may lie outside the view, and the episode's path always stays inside.
  fit() {
    if (!this.layer) return;
    const trim = Number(this.canvas.dataset.trim || 0);
    const core = this.layer.points; // the sampled points, whose outermost `trim` on each side may lie outside the view
    const path = this.episode ? this.episode.path : [];
    const ends = []; // the ends of the class ellipsoids' axes: inside the view unless it is zoomed
    if (!trim) for (const c of this.layer.classes) for (const s of [1, -1]) for (const a of c.axes) ends.push(c.mean.map((m, i) => m + s * a[i]));
    const mustShow = [...path, ...ends];
    const quantile = (values, q) => values.sort((a, b) => a - b)[Math.round(q * (values.length - 1))];
    const lo = [0, 1, 2].map((i) => Math.min(quantile(core.map((p) => p[i]), trim), ...mustShow.map((p) => p[i])));
    const hi = [0, 1, 2].map((i) => Math.max(quantile(core.map((p) => p[i]), 1 - trim), ...mustShow.map((p) => p[i])));
    const half = lo.map((l, i) => (hi[i] - l) / 2);
    const centre = lo.map((l, i) => (l + hi[i]) / 2);
    // radius: how far the data reaches from the centre, so that no rotation in 3D cuts it off
    const distance = (p) => Math.hypot(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
    const radius = Math.max(quantile(core.map(distance), 1 - trim), ...mustShow.map(distance));
    this.bounds = { centre, halfX: half[0], halfY: half[1], radius };
  }

  // Data point -> canvas {x, y, depth}. 2D uses the first two axes and fills the box; 3D rotates by yaw and pitch.
  project(p) {
    const { centre, halfX, halfY, radius } = this.bounds, margin = 10;
    const w = this.canvas.clientWidth - this.padRight, h = this.canvas.clientHeight - this.padBottom;
    const s = this.mode3d ? (Math.min(w, h) / 2 - margin) / radius : Math.min((w / 2 - margin) / halfX, (h / 2 - margin) / halfY);
    let [x, y, z] = p.map((v, i) => v - centre[i]);
    if (!this.mode3d) return { x: w / 2 + x * s, y: h / 2 - y * s, depth: 0 };
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    [x, z] = [x * cy + z * sy, -x * sy + z * cy];
    [y, z] = [y * cp - z * sp, y * sp + z * cp];
    const f = 1 / (1 + z / (8 * radius)); // a mild perspective
    return { x: w / 2 + x * s * f, y: h / 2 - y * s * f, depth: z };
  }

  ring(c, k) {
    // The 1-sigma ellipse of a class: in 2D the marginal in the first two axes, in 3D three rings through the ellipsoid's principal axes.
    const pad = (v) => [...v, 0, 0, 0].slice(0, 3); // the 2D axes get a zero third coordinate
    const mean = this.mode3d ? c.mean : pad(c.mean2);
    const [a, b] = this.mode3d ? [c.axes[k], c.axes[(k + 1) % 3]] : [pad(c.axes2[0]), pad(c.axes2[1])];
    const out = [];
    for (let i = 0; i <= RING; i++) {
      const t = (2 * Math.PI * i) / RING;
      out.push(mean.map((m, d) => m + Math.cos(t) * a[d] + Math.sin(t) * b[d]));
    }
    return out;
  }

  draw() {
    const { ctx, canvas } = this;
    if (!this.layer || !this.bounds) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = cssVar("--bg");
    ctx.fillRect(0, 0, w, h);
    const colour = (i) => CLASS_COLORS[this.layer.classes[i].name] || "#888";
    // sampled training points (in 3D, far points first)
    const pts = this.layer.points.map((p, i) => ({ ...this.project(p), c: this.layer.point_class[i] }));
    if (this.mode3d) pts.sort((a, b) => b.depth - a.depth);
    ctx.globalAlpha = 0.5;
    for (const p of pts) { ctx.fillStyle = colour(p.c); ctx.fillRect(p.x - 1.2, p.y - 1.2, 2.4, 2.4); }
    ctx.globalAlpha = 1;
    // class Gaussians
    ctx.lineWidth = 1.6;
    this.layer.classes.forEach((c, i) => {
      ctx.strokeStyle = colour(i);
      for (let k = 0; k < (this.mode3d ? 3 : 1); k++) {
        ctx.beginPath();
        this.ring(c, k).forEach((p, j) => { const q = this.project(p); j ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
        ctx.stroke();
      }
    });
    if (this.mode3d) this.drawAxes();
    if (this.episode) this.drawEpisode();
  }

  // A small triad at the centre of the data, labelled with the principal axes.
  drawAxes() {
    const { ctx } = this, centre = this.bounds.centre, length = this.bounds.radius * 0.35, origin = this.project(centre);
    ctx.strokeStyle = ctx.fillStyle = cssVar("--ink-soft");
    ctx.lineWidth = 1;
    ctx.font = "10px system-ui, sans-serif";
    ["PC1", "PC2", "PC3"].forEach((name, i) => {
      const tip = centre.slice();
      tip[i] += length;
      const q = this.project(tip);
      ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      ctx.fillText(name, q.x + 3, q.y - 3);
    });
  }

  drawEpisode() {
    const { ctx } = this, { path, alarm, onset, success } = this.episode;
    const line = success ? CLASS_COLORS.success : CLASS_COLORS.failure;
    const at = (i) => this.project(path[Math.min(i, path.length - 1)]);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    // the whole path faintly, the part up to the current step solid
    ctx.strokeStyle = line; ctx.globalAlpha = 0.25; ctx.lineWidth = 2;
    ctx.beginPath(); path.forEach((p, i) => { const q = this.project(p); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); }); ctx.stroke();
    ctx.globalAlpha = 1; ctx.lineWidth = 3.2;
    ctx.beginPath(); for (let i = 0; i <= this.step; i++) { const q = at(i); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); } ctx.stroke();
    const mark = (i, draw) => { if (i != null && i <= this.step) draw(at(i)); };
    mark(alarm, (q) => { ctx.strokeStyle = "#ffb000"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(q.x, q.y, 8, 0, 2 * Math.PI); ctx.stroke(); });
    mark(onset, (q) => { ctx.strokeStyle = CLASS_COLORS.failure; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(q.x - 6, q.y - 6); ctx.lineTo(q.x + 6, q.y + 6); ctx.moveTo(q.x + 6, q.y - 6); ctx.lineTo(q.x - 6, q.y + 6); ctx.stroke(); });
    const now = at(this.step);
    ctx.fillStyle = cssVar("--bg"); ctx.strokeStyle = line; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(now.x, now.y, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }

  startDrag(e) {
    if (!this.mode3d) return;
    this.canvas.setPointerCapture(e.pointerId);
    this.dragging = true; // the automatic turn waits while the view is held
    let [x, y] = [e.clientX, e.clientY];
    const move = (m) => {
      this.yaw += (m.clientX - x) * 0.01;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (m.clientY - y) * 0.01));
      [x, y] = [m.clientX, m.clientY];
      this.draw();
    };
    const up = () => {
      this.dragging = false;
      for (const type of ["pointermove", "pointerup", "pointercancel"]) this.canvas.removeEventListener(type, type === "pointermove" ? move : up);
    };
    this.canvas.addEventListener("pointermove", move);
    this.canvas.addEventListener("pointerup", up);
    this.canvas.addEventListener("pointercancel", up);
  }
}
