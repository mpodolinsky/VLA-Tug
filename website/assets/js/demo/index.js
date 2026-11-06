// The interactive figure: choose a rollout, play it, and see the flow of the probe (three of its layers, the calibrated trigger and the predicted
// failure mode) and the scores of three methods move with the video. All data is pre-computed (python -m site_export.export) and read from data/;
// nothing runs in the browser but the drawing.
import { CLASS_COLORS, humanise } from "./colors.js";
import { EmbeddingView } from "./embedding.js";
import { Timeline } from "./timeline.js";

const $ = (id) => document.getElementById(id);
const fetchJson = async (path, revalidate = false) => {
  const response = await fetch(path, revalidate ? { cache: "no-cache" } : undefined); // index.json is always checked, so a new export is seen at once
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
};
// every other data file and video carries the content version of the export (index.json: "version"), so that new episodes with the same file names are not
// taken from the browser's cache
let version = "";
const versioned = (path) => (version ? `${path}?v=${version}` : path);
const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const percent = (r) => `${r.fpr.toFixed(1)}% / ${r.tpr_early.toFixed(1)}%`;
// a short name of a split, for the card of an episode: "New skill: place on surface", "New task signatures", "New tasks"
const splitLabel = (split) => (split.startsWith("skill_") ? `New skill: ${split.slice(6).replaceAll("_", " ")}` : split.startsWith("signature_") ? "New task signatures" : "New tasks");

async function main() {
  const root = $("demo");
  if (!root) return;
  let index;
  try {
    index = await fetchJson("data/index.json", true);
    version = index.version || "";
  } catch (error) {
    $("demo-error").hidden = false; // e.g. the page was opened from disk instead of over http
    console.error(error);
    return;
  }

  const agent = $("demo-agent"), wrist = $("demo-wrist");
  // the playback bar at the top of the demo is the only control: play and pause, the position, the clock
  const seek = $("demo-seek-top"), clock = $("demo-time-top"), playButton = $("demo-play-top");
  // the images the VLA sees at the current step, in the input row of the flow: drawn from the two videos
  const inputImages = [[agent, $("flow-agent")], [wrist, $("flow-wrist")]];
  const drawInputs = () => {
    for (const [video, canvas] of inputImages) if (video.readyState >= 2) canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  };
  for (const video of [agent, wrist]) for (const type of ["loadeddata", "seeked"]) video.addEventListener(type, drawInputs);
  const timeline = new Timeline($("demo-timeline"), (t) => goTo(t));
  const layerNames = Object.values(index.settings)[0].layers.map((l) => l.name);
  const views = new Map(layerNames.map((name) => [name, new EmbeddingView(document.querySelector(`canvas[data-layer="${name}"]`))]));
  const episodes = new Map();
  let episode = null, playing = false;

  // ---- episodes -------------------------------------------------------------------------------------------------------------
  const picker = $("demo-episodes");
  for (const e of index.episodes) {
    const button = element("button", "demo-episode");
    button.type = "button";
    button.dataset.id = e.id;
    const instruction = element("span", "demo-instruction", e.instruction);
    instruction.append(element("span", "demo-split", splitLabel(index.settings[e.setting].split)));
    button.append(element("span", `badge ${e.success ? "ok" : "fail"}`, e.success ? "Success" : `Failure: ${humanise(e.mode)}`), instruction);
    button.addEventListener("click", () => selectEpisode(e.id));
    picker.append(button);
  }

  async function selectEpisode(id) {
    if (!episodes.has(id)) episodes.set(id, await fetchJson(versioned(index.episodes.find((e) => e.id === id).file)));
    episode = episodes.get(id);
    pause();
    for (const b of picker.children) b.setAttribute("aria-pressed", String(b.dataset.id === id));
    agent.src = versioned(episode.video.agent);
    wrist.src = versioned(episode.video.wrist);
    seek.max = episode.video.duration;
    $("demo-annotation").textContent = episode.success ? "Annotated outcome: success" : `Annotated failure: ${humanise(episode.mode)}, at ${(episode.onset_step * episode.step_seconds).toFixed(1)} s`;
    await showSetting(episode.setting);
    const signature = $("demo-signature");
    signature.replaceChildren();
    if (episode.novelty) signature.append(element("b", "", episode.novelty.label), element("code", "", episode.novelty.value));
    // under the episode: its split and the calibration targets, and what the three methods do on all the unseen episodes of the split
    const setting = index.settings[episode.setting], m = setting.methods, rates = (k) => (m[k].rates ? `${m[k].label} ${percent(m[k].rates.unseen)}` : `${m[k].label} (no certified threshold)`);
    const lines = $("demo-setting");
    lines.replaceChildren(
      ...[
        ["Split: ", `${setting.split}, ${setting.description} (${setting.unseen_episodes} unseen episodes)`],
        ["Calibration targets: ", `VLA-Tug with ${m.ours.calibration}; SAFE-LSTM with ${m.safe_lstm.calibration}; STAC with ${m.stac.calibration}; fitted on the ${setting.operating_point.calibration_episodes} calibration episodes of seen tasks`],
        ["On all unseen episodes of the split (false-positive rate / early true-positive rate): ", `${rates("ours")}, ${rates("safe_lstm")}, ${rates("stac")}`],
      ].map(([label, text]) => {
        const line = element("div", "demo-setting-line");
        line.append(element("b", "", label), document.createTextNode(text));
        return line;
      }),
    );
    $("flow-instruction").textContent = `Instruction: ${episode.instruction}`;
    timeline.setEpisode(episode);
    const notes = episode.callouts || [], list = $("demo-callouts");
    list.replaceChildren(...notes.map((note, i) => {
      const item = element("li");
      item.dataset.time = note.time;
      item.append(element("span", "callout-n", String(i + 1)), element("span", "callout-t", `${note.time.toFixed(1)} s`), element("span", "callout-text", note.text));
      return item;
    }));
    list.hidden = notes.length === 0;
    for (const [name, view] of views) {
      view.setEpisode({ path: episode.embeddings[name], alarm: episode.methods.ours.alarm_step, onset: episode.onset_step, success: episode.success });
    }
    goTo(0);
  }

  // ---- the three layers of the probe of an episode's split -----------------------------------------------------------------------
  // Every episode comes from its own split, with its own probe: the three embedding spaces, their legends and the calibration are those of the split, and are
  // loaded the first time an episode of the split is shown.
  const layerCache = new Map();
  let shownSetting = null;
  async function showSetting(id) {
    if (shownSetting === id) return;
    const setting = index.settings[id];
    if (!layerCache.has(id)) layerCache.set(id, await Promise.all(setting.layers.map((l) => fetchJson(versioned(l.file)))));
    setting.layers.forEach((l, i) => {
      const layer = layerCache.get(id)[i];
      const variance = layer.explained_variance.slice(0, 3).map((x) => `${(100 * x).toFixed(0)}%`).join(", ");
      document.querySelector(`[data-legend="${l.name}"]`).replaceChildren(
        ...layer.classes.map((c) => {
          const item = element("span", "legend-item", `${humanise(c.name)} (${c.episodes})`);
          item.style.setProperty("--swatch", CLASS_COLORS[c.name] || "#888");
          return item;
        }),
        element("span", "legend-variance", `axes: ${variance} of the variance`),
      );
      views.get(l.name).setLayer(layer);
      views.get(l.name).resize(); // the legend lies over the plot: measure it, so that the data stays clear of it
    });
    $("demo-trigger-text").textContent = setting.methods.ours.calibration;
    shownSetting = id;
  }

  // 3D is the default (the page markup starts with the 3D switch pressed); 2D is one click away
  for (const view of views.values()) view.setMode3d(true);
  for (const b of document.querySelectorAll("[data-view]")) {
    b.addEventListener("click", () => {
      $("demo-flow").classList.toggle("is-3d", b.dataset.view === "3d");
      for (const view of views.values()) view.setMode3d(b.dataset.view === "3d");
      document.querySelectorAll("[data-view]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      $("demo-hint").hidden = b.dataset.view !== "3d";
    });
  }

  // ---- time -----------------------------------------------------------------------------------------------------------------
  const stepAt = (t) => Math.min(episode.n_steps - 1, Math.max(0, Math.floor(t / episode.step_seconds + 1e-9)));

  function goTo(t, fromVideo = false) {
    if (!episode) return;
    t = Math.min(Math.max(t, 0), episode.video.duration);
    if (!fromVideo) agent.currentTime = wrist.currentTime = t;
    const step = stepAt(t);
    seek.value = t;
    clock.textContent = `${t.toFixed(1)} s / ${episode.video.duration.toFixed(1)} s`;
    $("flow-step").textContent = `(policy step ${step + 1} of ${episode.n_steps})`;
    for (const view of views.values()) view.setStep(step);
    timeline.setTime(t);
    for (const item of $("demo-callouts").children) item.classList.toggle("reached", t >= Number(item.dataset.time)); // the notes the video has reached
    drawInputs();
    showTrigger(step);
    showModes(step);
  }

  // The score head's output and the calibrated trigger: VLA-Tug's score of the current step against its threshold. From the alarm step on the
  // trigger lights up and the stages that follow an alarm (mode head, predicted mode, expert model) are drawn at full strength.
  function showTrigger(step) {
    const ours = episode.methods.ours, score = ours.scores[step], alarmed = ours.alarm_step != null && step >= ours.alarm_step;
    const threshold = Array.isArray(ours.threshold) ? ours.threshold[step] : ours.threshold;
    $("demo-flow").classList.toggle("is-alarm", alarmed);
    $("demo-score-readout").textContent = `score ${score.toFixed(2)}`;
    $("demo-trigger-status").textContent = alarmed ? `ALARM at ${(ours.alarm_step * episode.step_seconds).toFixed(1)} s` : ours.alarm_step == null ? "No alarm in this rollout" : "No alarm yet";
    $("gauge-fill").style.width = `${(100 * Math.min(score, 1)).toFixed(1)}%`;
    $("gauge-mark").style.left = `${(100 * Math.min(threshold ?? 1, 1)).toFixed(1)}%`;
    $("demo-trigger-values").textContent = threshold == null ? `score ${score.toFixed(2)} (no certified threshold)` : `score ${score.toFixed(2)} · threshold ${threshold.toFixed(2)}`;
  }

  // The mode head's output at the current step (the top three modes); it is read at the alarm.
  function showModes(step) {
    const probs = episode.mode_probs[step], alarm = episode.methods.ours.alarm_step;
    const top = probs.map((p, i) => [p, index.mode_names[i]]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    $("demo-modes-rows").replaceChildren(...top.map(([p, name]) => {
      const row = element("div", "mode-row");
      const bar = element("span", "mode-bar");
      bar.style.setProperty("--w", `${(100 * p).toFixed(0)}%`);
      bar.style.setProperty("--c", CLASS_COLORS[name]);
      row.append(element("span", "mode-name", humanise(name)), bar, element("span", "mode-value", p.toFixed(2)));
      return row;
    }));
    $("demo-modes-state").textContent = alarm == null ? "No alarm, so no prediction is made." : step < alarm ? "Read at the alarm." : "Read at the alarm and kept updating.";
  }

  // ---- playback -------------------------------------------------------------------------------------------------------------
  function play() {
    if (agent.ended || agent.currentTime >= episode.video.duration - 0.05) goTo(0);
    playing = true;
    playButton.textContent = "Pause";
    agent.play(); wrist.play();
    requestAnimationFrame(tick);
  }
  function pause() {
    playing = false;
    playButton.textContent = "Play";
    agent.pause(); wrist.pause();
  }
  function tick() {
    if (!playing) return;
    if (Math.abs(wrist.currentTime - agent.currentTime) > 0.15) wrist.currentTime = agent.currentTime; // keep the two views together
    goTo(agent.currentTime, true);
    if (agent.ended) pause();
    else requestAnimationFrame(tick);
  }
  playButton.addEventListener("click", () => (playing ? pause() : play()));
  seek.addEventListener("input", () => goTo(Number(seek.value)));

  // ---- the note under the figure ---------------------------------------------------------------------------------------------
  $("demo-note").textContent =
    "Each space is drawn in its first principal axes, fitted on the training episodes of the episode's split; dots are a sample of their steps up to the failure step, outlines one standard deviation of each class, and the line is the chosen rollout. " +
    "Every episode comes from its own split of LIBERO-X and is calibrated there, at the operating point written under it, so the thresholds, the embedding spaces and the rates differ between episodes. " +
    (index.selection.criteria
      ? `The ${index.episodes.length} rollouts were selected from the unseen episodes of their splits: ${index.selection.criteria} They illustrate the signal and are not typical of all episodes.`
      : `The ${index.episodes.length} rollouts are a random draw from the unseen episodes of their splits, not hand-picked.`);

  await selectEpisode(index.episodes[0].id);
}

main();
