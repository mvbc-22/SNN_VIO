# Stage 7 — Three-way feature tracking comparison

- Dataset segment: APS intervals 678 through 682.
- Initial features: 500 shared Shi-Tomasi locations.
- Frame interval indices: `[678, 679, 680, 681, 682]`.
- Reference match radius: 15 px.

| Metric | Frame-only | Event-driven | Hybrid |
|--------|------------|--------------|--------|
| EPE (px) | 4.385 | 4.209 | 4.126 |
| Track lifetime mean (μs) | 108312.9 | 122590.0 | 90863.0 |
| Successfully tracked / interval | 301.0 | 326.8 | 234.4 |
| Failure rate | 39.8000% | 34.6400% | 53.1200% |
| Processing time total (ms) | 1270.65 | 34836.73 | 65515.87 |
| Update latency mean (μs) | 1194.21 | 406.90 | 784.32 |
| Events / useful update | N/A | 49.640 | 69.208 |
| State updates | 1788 | 1791 | 6372 |
| Events processed | 0 | 81112 | 81112 |

## Measurement interpretation

EPE uses successful one-to-one greedy matches to next-frame Shi-Tomasi detections only; unmatched tracks contribute to failure rate, not EPE. Feature survival counts consecutive intervals with a successful endpoint match. Mean lifetime is measured from the first frame until the first unmatched interval and is interval-censored; it is a useful-track proxy, not identity ground truth.

Frame-only and event systems have different primitive operations. Frame-only latency is one KLT call per APS interval; event-driven and hybrid latency is association/update-loop wall time per event (including rejected events). Processing time is summed in-memory algorithm plus endpoint evaluation time, but excludes data loading and plots. Do not interpret these latency values as identical hardware operations.

Hybrid APS correction snaps each matched event-updated state to its matched next-frame Shi-Tomasi reference, advances its timestamp, and preserves velocity/confidence. Unmatched states are unchanged. Event-driven APS detections are strictly evaluation-only and do not alter state.

## Correctness checks

| Check | Result |
|---|---|
| Identical Initial Locations | PASS |
| Identical Frame Intervals | PASS |
| No Event Use By Frame Only | PASS |
| Frame Only Klt Output | PASS |
| Event Processing | PASS |
| Event State Updates | PASS |
| Matching Bounded | PASS |
| Finite Active Event States | PASS |

## Scientific scope

This segment comparison can describe the measured operating point only. It does not establish general superiority, cross-sequence performance, motion-compensated accuracy, or statistical significance. The event-driven hypothesis is evaluated by whether its states survive/match at APS boundaries and how accuracy, update counts, and runtime compare; between-frame utility is not directly ground-truthed by APS-only endpoints.
