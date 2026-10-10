# Stage 5 — Complete asynchronous feature tracker

- APS pair: 678 → 679
- Common-axis interval: [29895515, 29939581) μs
- Initial features: 500
- Events processed: 10879
- Successful asynchronous updates: 1555
- Events rejected: 9324
- Active features at t(k+1): 500
- Successfully predicted/reference-matched features: 354
- Endpoint EPE (matched features only): 3.9502 px
- Failure rate: 0.292
- Mean track lifetime: 44066 μs
- Events per successful feature update: 6.99614
- Total processing time: 11902.265 ms
- Event-loop time: 11523.447 ms
- Per-event update-loop latency mean / median / p95: 1057.371 / 1006.222 / 1365.573 μs

## Stage checks

| Component | Result |
|---|---|
| Event processing | PASS |
| Association | PASS |
| State update | PASS |
| Prediction | PASS |
| Frame-to-frame evaluation | PASS |

EPE is the mean Euclidean pixel error over successful, one-to-one greedy matches from predicted states to frame-k+1 Shi-Tomasi detections within the configured match radius. Unmatched predicted states are counted as failures, not included in EPE.

Track lifetime is interval-censored: because this experiment evaluates one frame pair and does not invalidate tracks, every initialized valid feature is counted for the full APS interval. Events per successful update is total processed events divided by successful state updates.

This is an algorithm-validation baseline, not a tracking-quality conclusion. No KLT, VIO, SNN, FPGA, HLS, RTL, SLAM, grid hashing, or Kalman filtering is included.
