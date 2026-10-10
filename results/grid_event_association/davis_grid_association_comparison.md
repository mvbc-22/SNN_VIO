# Stage 8 — Spatial grid association

- Result parity: PASS
- Exact decision fields: {'event_index': 0, 't_event': 0, 't_common_us': 0, 'x_event': 0, 'y_event': 0, 'p': 0, 'feature_id': 0, 'feature_slot': 0, 'candidate_count': 0, 'associated': 0, 'distance_squared': 0, 'events_per_feature': 0}
- Distance tolerance: abs=1e-09, rel=1e-09
- Grid cells: 16 x 16 px
- The persistent grid indexes stored feature positions. It searches cells within a conservative velocity-expanded neighborhood, then applies the exact per-feature prediction, temporal gate, and distance gate.
- Both paths use strict d2 < R2 and abs(dt) < T, and choose minimum squared distance with lower feature ID as the tie-break.
- No square root is used; the strict squared-distance gate is unchanged.

| Metric | Brute Force | Grid |
|--------|-------------|------|
| Candidate features/event | 500.000 | 36.522 |
| Distance calculations/event | 500.000 | 36.522 |
| Events processed/sec | 1090.152 | 12635.422 |
| Total runtime (s) | 9.979345 | 0.860992 |
| Association differences | 0 | 0 |
| Rejected events | 745 | 745 |
| Ambiguous associations | 6272 | 6272 |

The brute-force implementation remains the correctness reference. Performance depends on grid dimensions and feature motion. Runtime figures are software measurements, not hardware estimates.
