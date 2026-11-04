// The right panel's plot: one row per method with its score, its calibrated threshold, the alarm and the annotated failure time,
// all on the video's time axis. Click or drag on it to seek the video.
import { METHOD_COLORS, cssVar } from "./colors.js";

const ROW = 62, LEFT = 118, TOP = 6, AXIS = 22, RIGHT = 22;
const OUTCOME_TEXT = {
  early: "early alarm", on_time: "on-time alarm", late: "late alarm", missed: "no alarm (missed)",
  false_alarm: "false alarm", correct_reject: "no alarm",
};
const METHODS = ["ours", "safe_lstm", "stac"];

export class Timeline {
  constructor(canvas, onSeek) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.episode = null;
    this.time = 0;
    canvas.style.height = `${TOP + METHODS.length * ROW + AXIS}px`; // set once: the observer below must not change the size it observes
    new ResizeObserver(() => this.resize()).observe(canvas);
    const seek = (e) => {
      const rect = canvas.getBoundingClientRect(), width = rect.width - LEFT - RIGHT;
      onSeek(Math.min(1, Math.max(0, (e.clientX - rect.left - LEFT) / width)) * this.episode.video.duration);
    };
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.episode) return;
      canvas.setPointerCapture(e.pointerId);
      seek(e);
      const up = () => { canvas.removeEventListener("pointermove", seek); canvas.removeEventListener("pointerup", up); };
      canvas.addEventListener("pointermove", seek);
      canvas.addEventListener("pointerup", up);
    });
  }

  get rows() { return METHODS.length; }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.round(this.canvas.clientHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  setEpisode(episode) { this.episode = episode; this.draw(); }
  setTime(time) { this.time = time; this.draw(); }

  draw() {
    const { ctx, canvas, episode } = this;
    if (!episode) return;
    const w = canvas.clientWidth, plotW = w - LEFT - RIGHT, duration = episode.video.duration, dt = episode.step_seconds;
    const x = (t) => LEFT + (t / duration) * plotW;
    ctx.clearRect(0, 0, w, canvas.clientHeight);
    ctx.font = "12px system-ui, sans-serif";
    // annotated failure time and the on-time window before it, behind every row
    if (episode.onset_step != null) {
      const t1 = episode.onset_step * dt, t0 = Math.max(0, (episode.onset_step - episode.window_steps) * dt);
      ctx.fillStyle = "rgba(214, 67, 59, 0.14)";
      ctx.fillRect(x(t0), TOP, x(t1) - x(t0), this.rows * ROW);
      ctx.strokeStyle = "#d6433b"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(x(t1), TOP); ctx.lineTo(x(t1), TOP + this.rows * ROW); ctx.stroke(); ctx.setLineDash([]);
    }
    METHODS.forEach((name, r) => this.row(name, r, x, dt));
    // time axis
    const base = TOP + this.rows * ROW;
    ctx.strokeStyle = cssVar("--line"); ctx.fillStyle = cssVar("--ink-soft"); ctx.lineWidth = 1; ctx.textAlign = "center";
    const tick = duration > 30 ? 10 : 5;
    for (let t = 0; t <= duration + 1e-6; t += tick) {
      ctx.beginPath(); ctx.moveTo(x(t), base); ctx.lineTo(x(t), base + 4); ctx.stroke();
      ctx.fillText(`${t} s`, x(t), base + 16);
    }
    ctx.textAlign = "left";
    // playhead
    ctx.strokeStyle = cssVar("--ink"); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x(this.time), TOP); ctx.lineTo(x(this.time), base); ctx.stroke();
  }

  row(name, r, x, dt) {
    const { ctx, episode } = this, m = episode.methods[name], top = TOP + r * ROW, h = ROW - 12, bottom = top + h + 4;
    const colour = METHOD_COLORS[name];
    const thresholds = m.threshold == null ? [] : Array.isArray(m.threshold) ? m.threshold : m.scores.map(() => m.threshold); // null: not certified
    const ymax = Math.max(...m.scores, ...thresholds) * 1.08;
    const y = (v) => bottom - (v / ymax) * h;
    // frame and labels
    ctx.strokeStyle = cssVar("--line"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(LEFT, bottom); ctx.lineTo(x(episode.video.duration), bottom); ctx.stroke();
    ctx.textAlign = "left"; ctx.fillStyle = cssVar("--ink"); ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillText(m.label, 4, top + 18);
    ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = cssVar("--ink-soft");
    const alarm = m.alarm_step;
    ctx.fillText(OUTCOME_TEXT[m.outcome], 4, top + 36);
    if (alarm != null) ctx.fillText(`at ${(alarm * dt).toFixed(1)} s`, 4, top + 52);
    // calibrated threshold (dashed; it varies with the step for SAFE-LSTM), then the score
    if (thresholds.length) {
      ctx.strokeStyle = cssVar("--ink-soft"); ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]);
      stepLine(ctx, thresholds, x, y, dt);
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = colour; ctx.lineWidth = 2.2; ctx.lineJoin = "round";
    stepLine(ctx, m.scores, x, y, dt);
    // alarm marker
    if (alarm != null) {
      const px = x(alarm * dt), py = y(m.scores[alarm]);
      ctx.fillStyle = "#ffb000"; ctx.strokeStyle = cssVar("--ink"); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(px, py - 9); ctx.lineTo(px + 7, py + 4); ctx.lineTo(px - 7, py + 4); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }
}

// A line that holds each value for one policy step: horizontally to the next step's time, then vertically to its value.
function stepLine(ctx, values, x, y, dt) {
  ctx.beginPath();
  values.forEach((v, i) => {
    if (i === 0) ctx.moveTo(x(0), y(v));
    else { ctx.lineTo(x(i * dt), y(values[i - 1])); ctx.lineTo(x(i * dt), y(v)); }
  });
  ctx.lineTo(x(values.length * dt), y(values[values.length - 1]));
  ctx.stroke();
}
