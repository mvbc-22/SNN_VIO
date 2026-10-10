# Stage 4A — Brute-force event-to-feature association

- Frame interval: 678, [29895515, 29939581) us
- Active feature states: 500
- Events processed: 10879
- Events associated: 10134
- Association ratio: 0.931519
- Events rejected: 745
- Ambiguous events: 6272
- Spatial threshold R: 8 px (strict squared-distance gate)
- Temporal threshold T: 50000 us (strict absolute-time gate)
- Chronological policy: stable timestamp sort, preserving source order for ties.
- No match: rejected; feature ID and slot are -1.
- Multiple matches: count as ambiguous; select minimum squared distance, then lower feature ID.
- One event can be assigned to at most one feature; a feature can receive multiple events.
- Polarity is retained for every event, visualized, and not used for association.
- Feature states are not updated by this stage.
- Distance uses squared arithmetic only; no square root is computed.

Events per feature (including zero-count active features) are included in the NPZ archive.
