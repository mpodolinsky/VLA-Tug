// Colours of the interactive figure. Classes follow the paper's figures: blue for success, red for failure, one hue per failure mode.
export const CLASS_COLORS = {
  success: "#2f6fd1",
  failure: "#d6433b",
  wrong_object: "#8e5bb5",
  stuck_or_no_progress: "#7a7a7a",
  wrong_target: "#d36bb0",
  object_displacement: "#e69f00",
  placement_or_insertion_failure: "#009e73",
  grasp_failure: "#d55e00",
  open_close_failure: "#4aa8e0",
  press_failure: "#a98600",
  other: "#555555",
  timeout_or_insufficient_progress: "#8c564b",
};

// The three methods of the timeline: ours in the page's accent colour, the baselines in grey, as in the paper's figures.
export const METHOD_COLORS = { ours: "#3d3dab", safe_lstm: "#222222", stac: "#8a8a8a" };

export const humanise = (name) => name.replace(/_/g, " ");

// Reads a CSS variable of the page, so the canvases follow the light and dark theme.
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
