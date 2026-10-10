# Stage 4B — Asynchronous FeatureState update

- APS interval: 678, [29895515, 29939581) μs
- Events processed: 10879
- Successful updates: 1555
- Rejected events: 9324
- Ambiguous events: 576
- Initial active features: 500
- Final active features: 500
- Features with at least one update: 132
- Mean / maximum events per updated feature: 11.780 / 44
- Mean position update magnitude: 2.02979 px
- Maximum position update magnitude: 3.99474 px
- Mean velocity update magnitude: 0.0114477 px/μs
- Maximum velocity update magnitude: 0.545827 px/μs
- Alpha / beta: 0.5 / 0.1
- Confidence penalty: 0.05
- Zero Δt: position correction is applied; velocity is unchanged.
- Positive small Δt: velocity denominator is clamped to the configured minimum.
- Only valid states are considered; invalid slots remain unchanged.
- Numerically unrepresentable updates are rejected atomically.
- Feature states contain no image patches; no Kalman filter is used.

Event pixels, pre-update predictions, and post-update positions are separately marked in the trajectory visualization.
