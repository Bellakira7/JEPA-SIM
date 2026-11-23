(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  // Transcribed from the supplied manuscript, Tables 1, 2 and 5.
  // null means not reported; do not extrapolate OpenScene rendering rows.
  const benchmarks = {
    nuscenes: {
      ...window.JEPA_DATA.results,
      title: "Planning consistency on nuScenes",
      subtitle: "Frozen UniAD · 150 validation scenes · 3,878 matched targets per offset",
      table: 1, yMax: 1.3, xMax: 550,
      protocol: "On-log autoregressive observations; frame-level mean agreement with real-image reference plans. All methods use the same matched targets. Observation offset is distinct from the planning horizon.",
    },
    openscene: {
      offsets: [0.5, 1.0, 1.5],
      title: "Planning consistency on OpenScene",
      subtitle: "Frozen camera-only Latent TransFuser · 131 logs · 11,774 targets · per-token mean",
      table: 2, yMax: 1.0, xMax: 400,
      protocol: "On-log autoregressive feature prediction. The final-BEV predictor uses a TF-trained checkpoint. Rendering baselines are reported only at 0.5 s: this planner has no streaming temporal history. Observation offset is distinct from the planning horizon.",
      rows: [
        { name: "SplatAD", ade: [0.807, null, null], fde: [1.938, null, null], latency: 0.112 },
        { name: "OmniRe", ade: [0.614, null, null], fde: [1.606, null, null], latency: 0.248 },
        { name: "StreetGaussian", ade: [0.923, null, null], fde: [2.403, null, null], latency: 0.362 },
        { name: "FlowSim AR", ade: [0.380, 0.490, 0.550], fde: [1.000, 1.310, 1.470], latency: null },
        { name: "FlowSim VH", ade: [0.490, 0.500, 0.510], fde: [1.310, 1.350, 1.380], latency: 0.060 },
        { name: "JEPA-SIM", ade: [0.293, 0.364, 0.421], fde: [0.743, 0.924, 1.073], latency: 0.014 },
      ],
    },
  };
  const colors = { ours: "#0866e6", render: "#aa6745", latent: "#596b80" };
  const smallScreen = matchMedia("(max-width: 540px)");
  const ns = "http://www.w3.org/2000/svg";
  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function svgIn(target, width, height, title, description) {
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
    svg.append(el("title", {}, title), el("desc", {}, description));
    $(target).replaceChildren(svg);
    return svg;
  }
  function line(svg, x1, y1, x2, y2, attrs = {}) {
    svg.append(el("line", { x1, y1, x2, y2, stroke: "#dce3e9", ...attrs }));
  }
  function label(svg, x, y, text, attrs = {}) {
    svg.append(el("text", { x, y, fill: "#5d6b78", "font-size": 13, ...attrs }, text));
  }
  function scatter(data, index) {
    const points = data.rows.filter(row => row.latency !== null && row.ade[index] !== null);
    const compact = smallScreen.matches;
    const width = compact ? 360 : 600, left = compact ? 45 : 60, right = width - (compact ? 18 : 45);
    const svg = svgIn("#efficiency-chart", width, 360,
      `${data.title}: Plan ADE versus observation latency at ${data.offsets[index]} s`,
      "Linear axes start at zero. Exact values and methods without reported timing are in the adjacent table.");
    const x = value => left + value / data.xMax * (right - left);
    const y = value => 286 - value / data.yMax * 238;
    for (let step = 0; step <= 4; step++) {
      const value = data.yMax * step / 4;
      line(svg, left, y(value), right, y(value));
      label(svg, left - 10, y(value) + 4, value.toFixed(2), { "text-anchor": "end" });
    }
    for (let value = 0; value <= data.xMax; value += 100) {
      label(svg, x(value), 310, String(value), { "text-anchor": "middle" });
    }
    label(svg, left, 23, "Plan ADE (m) ↓", { fill: "#172b3a", "font-size": 14 });
    label(svg, width / 2, 344, "Observation latency (ms/step) →", { "text-anchor": "middle", "font-size": 14 });
    points.forEach(row => {
      const ours = row.name === "JEPA-SIM";
      const color = ours ? colors.ours : row.name.startsWith("FlowSim") ? colors.latent : colors.render;
      const px = x(row.latency * 1000), py = y(row.ade[index]);
      const point = el("circle", { cx: px, cy: py, r: ours ? 7 : 5, fill: color, stroke: "white", "stroke-width": 2 });
      point.append(el("title", {}, `${row.name}: ${row.ade[index].toFixed(3)} m, ${Math.round(row.latency * 1000)} ms`));
      svg.append(point);
      const placeLeft = px > width * .67;
      // Keep the two nearby FlowSim labels on separate sides of their points.
      const dy = ours ? 24 : row.name === "FlowSim VH" ? -12 : row.name === "FlowSim AR" ? 20 : -12;
      label(svg, px + (placeLeft ? -10 : 10), py + dy, row.name, { fill: color, "text-anchor": placeLeft ? "end" : "start", "font-weight": ours ? 650 : 400, "font-size": compact ? 12 : 13 });
    });
    $("#efficiency-note").textContent = "Blue: JEPA-SIM · Copper: rendering · Gray: latent baselines. Axes stay fixed across offsets within each benchmark.";
  }
  function updateResults() {
    const data = benchmarks[$("#benchmark-select").value];
    const index = Number($("#offset").value);
    $("#benchmark-title").textContent = data.title;
    $("#benchmark-subtitle").textContent = data.subtitle;
    $("#benchmark-source").textContent = `Paper Table ${data.table} ↗`;
    $("#benchmark-protocol").textContent = data.protocol;
    $("#result-rows").replaceChildren();
    data.rows.forEach(row => {
      const tr = document.createElement("tr");
      if (row.name === "JEPA-SIM") tr.className = "ours";
      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = row.name;
      tr.append(th);
      [row.ade[index], row.fde[index], row.latency].forEach((value, i) => {
        const td = document.createElement("td");
        td.textContent = value === null ? "—" : i === 2 ? String(Math.round(value * 1000)) : value.toFixed(3);
        tr.append(td);
      });
      $("#result-rows").append(tr);
    });
    $("#planning-table caption").textContent = `${data.title} at ${data.offsets[index].toFixed(1)} s observation offset. Plan ADE/FDE in meters and observation latency in ms/step; lower is better.`;
    const ours = data.rows.find(row => row.name === "JEPA-SIM");
    const competitors = data.rows.filter(row => row !== ours && row.ade[index] !== null);
    const best = competitors.reduce((a, b) => a.ade[index] < b.ade[index] ? a : b);
    const gain = (100 * (1 - ours.ade[index] / best.ade[index])).toFixed(1);
    $("#benchmark-takeaway").textContent = `At ${data.offsets[index].toFixed(1)} s: ${ours.ade[index].toFixed(3)} m Plan ADE — ${gain}% lower than ${best.name}, the next-lowest ADE in this comparison.`;
    scatter(data, index);
  }
  function selectBenchmark() {
    const data = benchmarks[$("#benchmark-select").value];
    $("#offset").replaceChildren(...data.offsets.map((value, index) => {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `${value.toFixed(1)} s`;
      return option;
    }));
    updateResults();
  }
  $("#benchmark-select").addEventListener("change", selectBenchmark);
  $("#offset").addEventListener("change", updateResults);
  selectBenchmark();
  document.querySelectorAll("[data-benchmark]").forEach(link => link.addEventListener("click", () => {
    $("#benchmark-select").value = link.dataset.benchmark;
    selectBenchmark();
  }));

  function ablation(target, name, values, max) {
    const offsets = [0.5, 1.5, 2.5, 4];
    const full = [0.165, 0.279, 0.362, 0.454];
    const compact = smallScreen.matches;
    const width = compact ? 360 : 480, right = width - 34;
    const svg = svgIn(target, width, 305, `${name} versus Full (rollout): Plan ADE by observation offset`, "Paper Table 5. Linear axes; the full numeric values are in the expandable table below. The two charts have different vertical ranges.");
    const x = value => 54 + value / 4 * (right - 54);
    const y = value => 237 - value / max * 180;
    for (let step = 0; step <= 4; step++) {
      const value = max * step / 4;
      line(svg, 54, y(value), right, y(value));
      label(svg, 44, y(value) + 4, value.toFixed(2), { "text-anchor": "end" });
    }
    label(svg, 54, 20, "Plan ADE (m) ↓", { fill: "#172b3a" });
    label(svg, 54, 41, name, { fill: colors.render });
    label(svg, compact ? 233 : 273, 41, "Full (rollout)", { fill: colors.ours });
    offsets.forEach(t => label(svg, x(t), 259, `${t.toFixed(1)}`, { "text-anchor": "middle" }));
    label(svg, width / 2, 292, "Observation offset (s)", { "text-anchor": "middle" });
    [[values, colors.render], [full, colors.ours]].forEach(([series, color]) => {
      svg.append(el("polyline", { points: series.map((v, i) => `${x(offsets[i])},${y(v)}`).join(" "), fill: "none", stroke: color, "stroke-width": 3 }));
      series.forEach((v, i) => svg.append(el("circle", { cx: x(offsets[i]), cy: y(v), r: 4, fill: color })));
    });
  }
  ablation("#interface-chart", "Direct BEV prediction", [0.459, 1.038, 1.491, 2.059], 2.4);
  ablation("#training-chart", "TF-only training", [0.164, 0.287, 0.397, 0.530], 0.6);

  // Reuse the already published executed trajectories; this is a replay, not inference.
  let plottedCase = null;
  let markers = [];
  let lastReplay = null;
  function drawExecution(current, time) {
    lastReplay = { current, time };
    if (plottedCase !== current.id) {
      plottedCase = current.id;
      const tracks = current.tracks;
      const all = Object.values(tracks).flat();
      const xmin = Math.min(0, ...all.map(point => point.p[0]));
      const xmax = Math.max(10, ...all.map(point => point.p[0]));
      const ymin = Math.min(...all.map(point => point.p[1]));
      const ymax = Math.max(...all.map(point => point.p[1]));
      const width = smallScreen.matches ? 450 : 900;
      const scale = Math.min((width - 110) / (xmax - xmin), 150 / Math.max(4, ymax - ymin));
      const x = value => 66 + (value - xmin) * scale;
      const y = value => 102 - (value - (ymax + ymin) / 2) * scale;
      const svg = svgIn("#execution-plot", width, 230, `${current.title}: executed trajectories over six seconds`, "Equal forward and lateral spatial scale. The dashed reference is recorded road-test execution of the same frozen model. Colored dots mark replay time.");
      const step = xmax > 70 ? 20 : 10;
      for (let value = 0; value <= xmax; value += step) {
        line(svg, x(value), 28, x(value), 183, { stroke: "#263b4b" });
        label(svg, x(value), 204, value, { fill: "#9ab0c4", "text-anchor": "middle" });
      }
      line(svg, 66, y(0), width - 44, y(0), { stroke: "#475b6b", "stroke-dasharray": "3 5" });
      label(svg, 66, 18, "Left (m)", { fill: "#9ab0c4" });
      label(svg, width - 50, 224, "Forward (m)", { fill: "#9ab0c4", "text-anchor": "end" });
      [Math.floor((ymax + ymin) / 2) - 2, Math.floor((ymax + ymin) / 2) + 2].forEach(value => {
        if (y(value) > 25 && y(value) < 185) label(svg, 53, y(value) + 4, value, { fill: "#9ab0c4", "text-anchor": "end" });
      });
      markers = [];
      [["Logged ego", "#bdc9d3"], ["3DGS", "#deb394"], ["JEPA-SIM", "#91e5cc"]].forEach(([name, color]) => {
        svg.append(el("polyline", { points: tracks[name].map(point => `${x(point.p[0])},${y(point.p[1])}`).join(" "), stroke: color, "stroke-width": 2.5, fill: "none", ...(name === "Logged ego" ? { "stroke-dasharray": "6 5" } : {}) }));
        const marker = el("circle", { r: 5, fill: color, stroke: "#09131e", "stroke-width": 2 });
        svg.append(marker);
        markers.push({ marker, track: tracks[name], x, y });
      });
    }
    markers.forEach(({ marker, track, x, y }) => {
      const index = Math.min(track.length - 1, Math.round(time * 10));
      marker.setAttribute("cx", x(track[index].p[0]));
      marker.setAttribute("cy", y(track[index].p[1]));
    });
  }
  document.addEventListener("jepa-replay-frame", event => drawExecution(event.detail.current, event.detail.time));
  smallScreen.addEventListener("change", () => {
    updateResults();
    ablation("#interface-chart", "Direct BEV prediction", [0.459, 1.038, 1.491, 2.059], 2.4);
    ablation("#training-chart", "TF-only training", [0.164, 0.287, 0.397, 0.530], 0.6);
    plottedCase = null;
    if (lastReplay) drawExecution(lastReplay.current, lastReplay.time);
  });
})();
