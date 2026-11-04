// Small enhancements; the page works without them.
(() => {
  // Top bar: appears once the hero scrolls away; the nav link of the current section is highlighted.
  const topbar = document.querySelector(".topbar");
  const hero = document.querySelector(".hero");
  if (topbar && hero && "IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      topbar.classList.toggle("is-stuck", !entry.isIntersecting);
    }, { rootMargin: "-56px 0px 0px 0px" }).observe(hero);

    const links = [...topbar.querySelectorAll("nav a")];
    const byId = new Map(links.map((a) => [a.hash.slice(1), a]));
    if (links.length) links[0].classList.add("is-active");  // the first section is highlighted until the page is scrolled
    const spy = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        links.forEach((a) => a.classList.toggle("is-active", a === byId.get(entry.target.id)));
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    byId.forEach((_, id) => {
      const section = document.getElementById(id);
      if (section) spy.observe(section);
    });
  }

  // Title: the hyphen of "VLA-Tug" is a rope and the Tug pulls on the VLA. The motion is a pure function of time,
  // so index.html?t=1.8 freezes it at second 1.8 (used to check single frames); no motion with prefers-reduced-motion.
  const mark = document.querySelector(".wordmark");
  const svg = mark && mark.querySelector(".rope");
  if (svg) {
    const vla = mark.querySelector(".w-vla");
    const tug = mark.querySelector(".w-tug");
    const base = svg.querySelector(".rope-base");
    const twist = svg.querySelector(".rope-twist");
    const CYCLE = 5.2;
    const ease = (x) => {
      x = Math.min(1, Math.max(0, x));
      return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    };
    let W = 0, H = 0, u = 16;
    const measure = () => {
      const r = svg.getBoundingClientRect();
      W = r.width; H = r.height;
      u = parseFloat(getComputedStyle(mark).fontSize);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    };
    const draw = (time) => {
      const c = time % CYCLE;
      const D = 0.30 * u;            // how far the Tug pulls out
      const slack0 = 0.14 * u;       // the rope is a little longer than the gap between the words
      const L = W + slack0;          // rope length
      const out = ease((c - 0.5) / 1.6);     // the Tug pulls out ...
      const back = ease((c - 3.0) / 1.5);    // ... and comes back
      const tugOff = D * (out - back);
      const vlaOff = Math.max(0, D * out - slack0) * (1 - ease((c - 3.5) / 1.5));  // the VLA moves only once the rope is taut, then coasts back
      const x0 = vlaOff, x1 = W + tugOff, y0 = H / 2;
      const chord = x1 - x0;
      const slack = L - chord;
      // sag of a rope of length L over a chord: arc length ~ chord + 8 h^2 / (3 chord); a taut rope only trembles
      const h = slack > 0.01 * u ? Math.sqrt((3 * chord * slack) / 8) : 0.012 * u * Math.sin(c * 38);
      const d = `M ${x0} ${y0} Q ${(x0 + x1) / 2} ${y0 + 2 * h} ${x1} ${y0}`;
      base.setAttribute("d", d);
      base.setAttribute("stroke-width", 0.085 * u);
      twist.setAttribute("d", d);
      twist.setAttribute("stroke-width", 0.045 * u);
      twist.setAttribute("stroke-dasharray", `${0.03 * u} ${0.055 * u}`);
      vla.style.transform = `translateX(${vlaOff}px)`;
      tug.style.transform = `translateX(${tugOff}px)`;
    };
    const frozen = new URLSearchParams(location.search).get("t");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    measure();
    window.addEventListener("resize", () => { measure(); if (frozen !== null) draw(Number(frozen)); });
    if (frozen !== null) draw(Number(frozen));
    else if (reduceMotion) draw(0);
    else {
      let visible = true;
      new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(mark);
      const start = performance.now();
      const frame = (now) => {
        if (visible && !document.hidden) draw((now - start) / 1000);
        requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    }
  }

  // BibTeX: copy button.
  document.querySelectorAll("button.copy[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const source = document.getElementById(button.dataset.copy);
      if (!source) return;
      try {
        await navigator.clipboard.writeText(source.innerText);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Press Ctrl+C";
        const range = document.createRange();
        range.selectNodeContents(source);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
      setTimeout(() => { button.textContent = "Copy"; }, 1600);
    });
  });
})();
