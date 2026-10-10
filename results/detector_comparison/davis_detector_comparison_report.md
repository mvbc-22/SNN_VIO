# Stage 2B — Shi-Tomasi vs FAST on DAVIS APS

- Consecutive frame-pair starts: `[0, 338, 677, 1015, 1354]`.
- Requested feature caps: `[50, 100, 200, 300, 500, 1000]`.
- Spatial bucket grid: 8 × 6; minimum spacing: 5.0 px.
- Shi-Tomasi: quality level 0.01, block size 3 px.
- FAST: threshold 1, TYPE_9_16, nonmax suppression True.
- Repeatability: mutual nearest neighbors within 3.0 px on unwarped consecutive images.
- Runtime: median over 3 repeats; candidate detection plus per-cap spatial selection.

## Quantitative results

| Detector | Cap | Mean detected | Mean bucket coverage | Mean repeatability | Mean runtime (ms) | Mean median native response |
|---|---:|---:|---:|---:|---:|---:|
| shi tomasi | 50 | 50.0 | 99.6% | 14.0% | 6.067 | 0.000238375 |
| fast | 50 | 50.0 | 100.0% | 11.2% | 1.095 | 7.6 |
| shi tomasi | 100 | 100.0 | 99.6% | 18.0% | 6.234 | 0.000131489 |
| fast | 100 | 100.0 | 100.0% | 17.2% | 1.293 | 6.05 |
| shi tomasi | 200 | 200.0 | 99.6% | 26.3% | 6.615 | 6.98675e-05 |
| fast | 200 | 200.0 | 100.0% | 26.0% | 1.676 | 4.2 |
| shi tomasi | 300 | 300.0 | 99.6% | 31.0% | 6.963 | 4.81018e-05 |
| fast | 300 | 300.0 | 100.0% | 31.9% | 2.135 | 3.5 |
| shi tomasi | 500 | 500.0 | 99.6% | 43.1% | 7.766 | 2.77024e-05 |
| fast | 500 | 499.8 | 100.0% | 45.5% | 3.194 | 2.1 |
| shi tomasi | 1000 | 632.7 | 99.6% | 53.4% | 8.337 | 2.15896e-05 |
| fast | 1000 | 550.3 | 100.0% | 49.9% | 3.567 | 1.8 |

## Recommendation: fast

At the 500-feature operating point, fast detected 499.8 features on average, achieved 100.0% mean bucket coverage, 45.5% raw-coordinate repeatability, and took 3.194 ms per frame on average. Selected for higher raw-coordinate repeatability (>2 percentage-point difference).

Native detector-response numbers are **not directly comparable**: Shi-Tomasi reports a minimum-eigenvalue score, while FAST reports OpenCV's FAST keypoint response.

Raw-coordinate repeatability is a detector-stability proxy, not tracking accuracy. No geometric motion compensation, optical flow, or KLT is applied, so camera/object motion can lower both scores.

Visual overlays and bucket maps are illustrative only; the recommendation uses the measurements above, not visual preference.

No event association, velocity, or event-driven tracking is implemented.

Artifacts: `davis_detector_comparison_overlays.png`, `davis_detector_comparison_spatial.png`, `davis_detector_comparison_metrics.png`, `davis_detector_comparison_measurements.csv`, and `davis_detector_comparison_repeatability.csv`.
