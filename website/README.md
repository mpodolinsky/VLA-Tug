# Project page skeleton

A static page (HTML, one CSS file, one small JS file; no build step) with the usual project-page sections: hero with link buttons, teaser, the interactive demo, abstract, method, results (with the paper's main table), dataset, code, BibTeX. The structure follows project pages such as <https://form-robots.github.io/>; the design and code are written from scratch.

```
website/
  index.html            all sections; every TODO is marked in the page or in a comment
  assets/css/style.css  light and dark theme, responsive
  assets/css/demo.css   the interactive demo only (episode picker, the flow panel (a), the runtime panel (b))
  assets/js/main.js     sticky top bar, section highlight, BibTeX copy button, the title animation (the page works without it)
  assets/js/demo/       the interactive demo: index.js (wiring, playback), embedding.js (2D/3D canvas), timeline.js (scores), colors.js
  assets/img/           favicon.svg, logo.webp, overview.png (the framework figure of the README)
  data/                 the demo's data, written by site_export (index.json, episodes/, embeddings/): do not edit by hand
  media/rollouts/       the demo's videos, written by site_export
```

The interactive demo reads `data/` and `media/rollouts/`, which `python -m site_export.export` writes from the saved runs (see `site_export/README.md`;
it is outside `website/` so that only served files are deployed). To show other episodes, run it with another `--seed` or `--split`.

## Preview

```bash
cd website && python -m http.server 8000     # then open http://localhost:8000 (the demo needs http; it does not load from a file:// address)
```

In the demo the three embedding views start in 3D and turn slowly (a full turn takes about 30 s; dragging a view pauses it, and visitors who ask for reduced motion get no automatic turn). The two head views are zoomed in on the dense core of their data (`data-trim` in `index.html`; the chosen episode's path always stays inside).

The title animation (the Tug pulls the VLA by a rope, a 5.2 s loop drawn by `assets/js/main.js`) is a pure function of time: `index.html?t=2.4` freezes it at second 2.4. It does not move with `prefers-reduced-motion`.

## Deploy (GitHub Pages)

Serve the `website/` folder: either copy it to the root of a `<name>.github.io` repository, or publish it with a GitHub Actions Pages workflow (`actions/upload-pages-artifact` with `path: website`). The page uses relative paths only, so it works from any sub-path.

## To do before it goes public

- [ ] Authors: replace "Anonymous authors" (the commented block in the hero shows the markup). Keep it anonymous while the paper is under review.
- [ ] Paper button: link the arXiv page; Data button: link the dataset. A Hub link shows the account name, so keep "soon" while the review is blind.
- [ ] Code button: now the anonymous mirror; switch to the final repository.
- [ ] Video: there is no video section for now; when there is a video, add the section again (the page's navigation and the hero button pointed to it).
- [ ] Results: write the results sentence (numbers must match the paper's tables); the result figures (key plot, calibration, failure modes, latent space) are removed for now, add them below the table when they are ready. The main table is already in the page: `python -m site_export.main_table` rewrites it from the paper's `table_main_std.tex` (means, standard deviations and the number of runs per setting) (see `site_export/README.md`).
- [ ] Dataset: the per-set episode table is removed for now; add it again with the counts from the paper's dataset table (total 3,180 simulated episodes).
- [ ] BibTeX: final entry.
- [ ] Abstract: keep in sync with the paper (the steering sentence of the draft is left out).
- [ ] Social image (`og:image` is the overview figure; a 1200x630 crop is better) and canonical URL.
