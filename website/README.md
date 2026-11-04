# Project page skeleton

A static page (HTML, one CSS file, one small JS file; no build step) with the usual project-page sections: hero with link buttons, teaser, video, abstract, method, results, dataset, code, BibTeX. The structure follows project pages such as <https://form-robots.github.io/>; the design and code are written from scratch.

```
website/
  index.html            all sections; every TODO is marked in the page or in a comment
  assets/css/style.css  light and dark theme, responsive
  assets/js/main.js     sticky top bar, section highlight, BibTeX copy button (the page works without it)
  assets/img/           favicon.svg, overview.png (the framework figure of the README)
  media/video/          put the overview video here
```

## Preview

```bash
cd website && python -m http.server 8000     # then open http://localhost:8000
```

## Deploy (GitHub Pages)

Serve the `website/` folder: either copy it to the root of a `<name>.github.io` repository, or publish it with a GitHub Actions Pages workflow (`actions/upload-pages-artifact` with `path: website`). The page uses relative paths only, so it works from any sub-path.

## To do before it goes public

- [ ] Authors: replace "Anonymous authors" (the commented block in the hero shows the markup). Keep it anonymous while the paper is under review.
- [ ] Paper button: link the arXiv page; Data button: link the dataset. A Hub link shows the account name, so keep "soon" while the review is blind.
- [ ] Code button: now the anonymous mirror; switch to the final repository.
- [ ] Video: add `media/video/overview.mp4` (and a poster) and replace the placeholder; add the three demo clips.
- [ ] Method and results figures: export from the paper (`assets/img/method.png`, the key plot, the calibration and failure-mode figures) and fill in the captions and the results sentence; numbers must match the paper's tables.
- [ ] Dataset table: per-set episode counts (total 3,180 simulated episodes).
- [ ] BibTeX: final entry.
- [ ] Abstract: keep in sync with the paper (the steering sentence of the draft is left out).
- [ ] Social image (`og:image` is the overview figure; a 1200x630 crop is better) and canonical URL.
