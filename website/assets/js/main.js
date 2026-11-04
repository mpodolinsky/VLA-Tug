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
