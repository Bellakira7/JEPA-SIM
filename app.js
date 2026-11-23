(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  const data = window.JEPA_DATA;
  // Reveal content once as it enters the viewport, with a small group stagger.
  const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
  const revealGroups = [
    ".intro-grid", ".principles", ".section-heading", ".method-notes",
    ".feature-comparison", ".demo-intro", ".results-heading", ".resource-grid"
  ];
  const revealItems = new Set();
  revealGroups.forEach((selector) => document.querySelectorAll(selector).forEach((group) => {
    Array.from(group.children).forEach((item, index) => {
      item.style.setProperty("--reveal-delay", Math.min(index, 3) * 90 + "ms");
      revealItems.add(item);
    });
  }));
  document.querySelectorAll(".architecture-figure, .demo-viewer, .demo-note, .benchmark, .scope-note")
    .forEach((item) => revealItems.add(item));
  if ("IntersectionObserver" in window && !motionPreference.matches) {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-revealed");
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0, rootMargin: "0px 0px -48px 0px" });
    revealItems.forEach((item) => {
      item.classList.add("scroll-reveal");
      revealObserver.observe(item);
    });
    // Keyboard focus must never land inside visually hidden content.
    document.addEventListener("focusin", (event) => {
      const item = event.target.closest(".scroll-reveal");
      if (item) item.classList.add("is-revealed");
    });
    motionPreference.addEventListener("change", (event) => {
      if (!event.matches) return;
      revealObserver.disconnect();
      revealItems.forEach((item) => item.classList.add("is-revealed"));
    });
  }
  const heroVideo = $("#hero-video");
  const heroPlayback = $("#hero-playback");
  const heroProcess = $(".hero-process");
  const processButtons = Array.from($(".process-steps").querySelectorAll("button"));
  const processStages = [
    { name: "observe", start: 0, seek: 0, title: "Observe the scene", description: "Pure camera input, before prediction." },
    { name: "predict", start: 2.5, seek: 3.15, title: "Predict in feature space", description: "Feature PCA with scene boxes and the rollout path." },
    { name: "execute", start: 5.5, seek: 6.15, title: "Execute the action", description: "Follow the path, then observe again." }
  ];
  function syncHeroPlayback() {
    heroPlayback.textContent = heroVideo.paused ? "Play video" : "Pause video";
    heroPlayback.setAttribute("aria-label", heroVideo.paused ? "Play background video" : "Pause background video");
    heroProcess.classList.toggle("paused", heroVideo.paused);
  }
  heroVideo.addEventListener("play", syncHeroPlayback);
  heroVideo.addEventListener("pause", syncHeroPlayback);
  heroPlayback.addEventListener("click", () => {
    if (heroVideo.paused) heroVideo.play().catch(syncHeroPlayback);
    else heroVideo.pause();
  });
  const latentGrid = $(".latent-grid");
  for (let i = 0; i < 24; i++) {
    const cell = document.createElement("span");
    cell.style.setProperty("--i", i);
    latentGrid.append(cell);
  }
  function syncHeroStage(time = heroVideo.currentTime) {
    const stage = time < processStages[1].start ? 0 : time < processStages[2].start ? 1 : 2;
    const current = processStages[stage];
    if (heroProcess.dataset.stage !== current.name) {
      heroProcess.dataset.stage = current.name;
      processButtons.forEach((button, i) => {
        button.classList.toggle("active", i === stage);
        button.setAttribute("aria-pressed", String(i === stage));
      });
      $("#process-title").textContent = current.title;
      $("#process-description").textContent = current.description;
    }
    $("#process-progress").style.width = ((time / (heroVideo.duration || 9.5)) * 100) + "%";
  }
  ["loadedmetadata", "timeupdate", "seeked", "pause"].forEach((event) => {
    heroVideo.addEventListener(event, () => syncHeroStage());
  });
  // Match the labels to presented video frames, including exact stage boundaries.
  if ("requestVideoFrameCallback" in heroVideo) {
    const onHeroFrame = (_, metadata) => {
      syncHeroStage(metadata.mediaTime);
      heroVideo.requestVideoFrameCallback(onHeroFrame);
    };
    heroVideo.requestVideoFrameCallback(onHeroFrame);
  }
  processButtons.forEach((button, i) => button.addEventListener("click", () => {
    heroVideo.currentTime = processStages[i].seek;
  }));
  motionPreference.addEventListener("change", (event) => {
    if (event.matches) heroVideo.pause();
  });
  heroVideo.addEventListener("canplay", () => {
    heroPlayback.disabled = false;
    processButtons.forEach((button) => { button.disabled = false; });
    syncHeroPlayback();
  });
  heroVideo.addEventListener("error", () => {
    heroPlayback.disabled = true;
    heroPlayback.textContent = "Video unavailable";
    processButtons.forEach((button) => { button.disabled = true; });
  });
  async function loadHeroVideo() {
    // A complete Blob also supports instant stage seeking on static servers
    // without HTTP Range support (including the local Python preview).
    try {
      const response = await fetch(heroVideo.dataset.src);
      if (!response.ok) throw new Error("Hero video failed to load");
      heroVideo.src = URL.createObjectURL(await response.blob());
    } catch {
      heroVideo.src = heroVideo.dataset.src;
    }
    if (!motionPreference.matches) heroVideo.play().catch(syncHeroPlayback);
  }
  loadHeroVideo();
  const menuButton = $(".menu-toggle");
  const menu = $("#mobile-menu");
  function closeMenu() {
    menu.hidden = true;
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "Open navigation");
  }
  menuButton.addEventListener("click", () => {
    const open = menu.hidden;
    menu.hidden = !open;
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  });
  menu.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(); });
  document.addEventListener("click", (event) => { if (!event.target.closest(".site-header")) closeMenu(); });
  window.matchMedia("(min-width: 801px)").addEventListener("change", (event) => { if (event.matches) closeMenu(); });

  const navLinks = Array.from(document.querySelectorAll(".section-nav a"));
  const sections = Array.from(document.querySelectorAll("main>section[id]:not(#top)"));
  let navFrame = null;
  function syncSectionNav() {
    navFrame = null;
    const active = sections.filter((section) => section.getBoundingClientRect().top <= innerHeight * .36).pop();
    navLinks.forEach((link) => {
      const selected = !!active && link.hash === "#" + active.id;
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }
  window.addEventListener("scroll", () => {
    if (navFrame === null) navFrame = requestAnimationFrame(syncSectionNav);
  }, {passive:true});
  window.addEventListener("resize", syncSectionNav);
  syncSectionNav();

  const dialog = $("#figure-dialog");
  $(".figure-button").addEventListener("click", (event) => {
    const button = event.currentTarget;
    $("#expanded-figure").src = button.dataset.image;
    $("#expanded-figure").alt = button.dataset.caption;
    $("#figure-caption").textContent = button.dataset.caption;
    dialog.showModal();
    document.body.style.overflow = "hidden";
  });
  $("#close-figure").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => { document.body.style.overflow = ""; });
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
  });

  function updateResults() {
    const i = Number($("#offset").value);
    const rows = data.results.rows;
    $("#result-rows").replaceChildren();
    $("#ade-chart").replaceChildren();
    rows.forEach((row) => {
      const ours = row.name === "JEPA-SIM";
      const tr = document.createElement("tr");
      if (ours) tr.className = "ours";
      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = row.name;
      tr.append(th);
      [row.ade[i], row.fde[i], row.latency].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value === null ? "—" : value.toFixed(3);
        tr.append(td);
      });
      $("#result-rows").append(tr);
      const bar = document.createElement("div");
      bar.className = "bar-row" + (ours ? " ours" : "");
      const label = document.createElement("span");
      label.className = "bar-name";
      label.textContent = row.name;
      const track = document.createElement("div");
      track.className = "bar-track";
      track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = (row.ade[i] / 3.2 * 100) + "%";
      track.append(fill);
      const value = document.createElement("span");
      value.className = "bar-value";
      value.textContent = row.ade[i].toFixed(3);
      bar.append(label, track, value);
      $("#ade-chart").append(bar);
    });
    $("table caption").textContent = "Planning consistency at " + data.results.offsets[i].toFixed(1) + " seconds observation offset, and total observation latency";
    $("#ade-chart").setAttribute("aria-label", "Plan ADE in meters at " + data.results.offsets[i].toFixed(1) + " seconds; lower is better");
  }
  $("#offset").addEventListener("change", updateResults);
  updateResults();
})();
