# Stage 3 — Persistent FeatureState memory initialization

- APS frame index: 678
- Requested Shi-Tomasi features: 500
- Detected features: 500
- Fixed state slots: 500
- Active initialized states: 500
- State timestamp: 29876317 microseconds
- State velocity: initialized to 0 pixels per microsecond.
- Confidence: initialized to 1.0 for each valid APS detection.
- Unused slots: `id=-1`, numeric fields zero, `valid=0`.
- No image patches, event association, or event processing are included.

The NPZ contains the full fixed-capacity `states` array, an `active_states` compact view, and the originating Stage 2A detections. All state fields use fixed-width numeric dtypes suitable for a BRAM-oriented layout.
