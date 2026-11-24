# JEPA-SIM project website

Static GitHub Pages site. Preview from this directory with `python -m http.server 8765`,
then open `http://127.0.0.1:8765/`. No build step or external charting dependency is required.

## Evidence and provenance

The homepage uses the accompanying author manuscript supplied on 2026-11-23.
`assets/jepa-sim-paper.pdf` contains its first nine pages; `assets/jepa-sim-supplement.pdf`
contains the remaining eleven appendix pages. These replace the earlier anonymous PDFs.

- Table 1: nuScenes values in `data.js`, displayed by `evidence.js`.
- Table 2 and Appendix A.3: OpenScene values and evaluation population in `evidence.js`.
  Missing rendering results beyond 0.5 s and missing FlowSim AR timing remain `null`.
- Table 3: all seven in-house execution metrics in `index.html`, including the defloater's
  lower lateral error. The 60-scene aggregate is separate from the six selected replays.
- Table 5: pre-fusion versus direct BEV and rollout versus TF-only Plan ADE in `evidence.js`
  and the accessible values table in `index.html`.
- Figure 6: `assets/feature-evidence.png`, cropped from manuscript page 8. Shared error limits,
  labels, magnified regions, and the color scale are preserved. The selected examples do
  not establish an aggregate detection rate.

All latency comparisons include observation generation and image encoding, excluding
planning and control. The lead ratios name their comparators: 280/28 = 10x (SplatAD,
nuScenes), 248/14 is about 18x (OmniRe, OpenScene), and 361/45 is about 8x (plain 3DGS,
in-house). The latter two use the displayed rounded timings; the manuscript discusses
unrounded speedups. Plan errors compare against the same frozen model's real-image plans,
not human driving. Observation offset and planning horizon are distinct.

`replay.js` dispatches `jepa-replay-frame` with the selected case and shared replay time.
`evidence.js` uses the existing executed trajectories in `data.js` to draw the common-frame
plot and move its markers. It does not generate new rollouts. Spatial scale is equal on
both axes; the full six-second paths remain visible while markers track replay time.

## Checks when editing

Run `node --check app.js`, `node --check evidence.js`, `node --check replay.js` and
`git diff --check`. In a browser, verify both benchmark selectors and every offset,
missing values, all six scene switches, play/pause/seek, figure enlargement and dismissal,
mobile navigation, and desktop/mobile chart layouts. Respect reduced-motion preferences.
Check numeric changes against the cited paper tables before updating a chart or headline.
