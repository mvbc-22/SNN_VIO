# Stage 6A — Conventional frame-only KLT baseline

- APS pair: 678 → 679
- Common-axis interval: [29895515, 29939581) μs
- Initial Shi-Tomasi features: 500
- Tracked features: 463
- Reference-matched features: 409
- EPE: 3.86437 px
- Mean track lifetime: 40805 μs
- Failure rate: 0.182
- KLT-only failure rate: 0.074
- Processing time: 312.756 ms
- KLT-call latency: 3.069 ms
- Amortized latency per initial feature: 6.138 μs
- Event data: not accessed or used.

## Baseline checks

| Check | Result |
|---|---|
| Initialization | PASS |
| Klt Output | PASS |
| Reference Matching | PASS |
| Evaluation | PASS |

The endpoint evaluation reuses Stage 5's greedy one-to-one reference-feature matching with the same configured radius. Track lifetime is interval-censored: successful KLT tracks are assigned the full pair duration and failures zero lifetime.

The `davis_frame_only_comparison_ready.json` structure exposes method, event-use flag, configuration, metrics, check results and the numeric-array archive for direct comparison with Stage 5.
