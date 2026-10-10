# Frozen Asynchronous Feature-State Tracker Specification

**Specification version:** 1.0  
**Freeze status:** Frozen research baseline; not an FPGA/HLS/RTL specification  
**Reference operating point:** DAVIS 240C, `shapes_6dof`, 500 state slots  
**Normative language:** “MUST” and “MUST NOT” describe required behavior for a conforming software reimplementation.

This document freezes the current Python asynchronous feature-state tracker
after the Stage 1–9 experiments. It does not introduce a new estimator or
claim that the selected parameters are globally optimal. Future algorithm
changes require a documented experimental failure or a later controlled
experiment that justifies reopening this specification.

## 1. Operating point and evidence

The reproducible reference configuration is:

| Parameter | Frozen value |
|---|---:|
| Sensor size | 240 × 180 pixels |
| Feature-state slots | 500 |
| APS detector | Shi–Tomasi / minimum eigenvalue |
| Detector quality level | 0.01 |
| Minimum feature separation | 5 px |
| Detector block size | 3 × 3 px |
| Detector selection buckets | 8 columns × 6 rows |
| Association grid cell size | 16 × 16 px |
| Spatial radius, `R` | 8 px |
| Temporal threshold, `T` | 50,000 μs |
| Position coefficient, `alpha` | 0.5 |
| Velocity coefficient, `beta` | 0.1 |
| Positive-`dt` denominator floor | 1 μs |
| Confidence penalty, `lambda` | 0.05 |
| Reference-match radius for evaluation only | 15 px |

These values preserve the Stage 5–8 baseline used by the three-way comparison
and Stage 8 correctness comparison. Stage 9 tested one-factor-at-a-time
variations on APS intervals 678–682. It found substantial feature-count
tradeoffs, smaller EPE changes for `R`, `T`, `alpha`, and `beta`, and no EPE
change from cell size under exact association. It did **not** establish a
unique optimum. In that sweep, online grid candidate retrieval approached the
full active feature set (about 500 candidates/event at baseline); therefore
the selected grid is frozen as the validated association organization, not
as a demonstrated latency improvement in all online workloads.

## 2. Feature representation

The feature-state memory is a fixed array of exactly 500 slots. Each slot is
a packed 33-byte record in the following order:

| Field | Type | Units / meaning |
|---|---|---|
| `id` | signed int32 | Monotonically allocated feature identifier |
| `x` | IEEE-754 float32 | Current x position, pixels |
| `y` | IEEE-754 float32 | Current y position, pixels |
| `vx` | IEEE-754 float32 | x velocity, pixels/μs |
| `vy` | IEEE-754 float32 | y velocity, pixels/μs |
| `confidence` | IEEE-754 float32 | Confidence in `[0, 1]` |
| `timestamp` | signed int64 | Last accepted state time, common-axis μs |
| `valid` | uint8 | `1` for active; `0` for inactive |

The record contains no image patch or appearance descriptor. Numeric update
intermediates are evaluated at at least float64 precision in the Python
reference and cast to float32 on state write. A hardware implementation must
not assume this reference establishes fixed-point formats or rounding rules.

An inactive slot has `id=-1`, all floating fields zero, `timestamp=0`, and
`valid=0`. At initialization, valid detections are assigned in detector output
order to the first available slots. IDs begin at zero and increase by one.
IDs are not reused during the run.

## 3. Feature detection and initialization

At the first APS frame of a run:

1. Input is one grayscale uint8 APS image of the configured sensor resolution.
2. Run Shi–Tomasi corner detection separately in each of 8 × 6 rectangular
   image buckets, using `qualityLevel=0.01`, `minDistance=5 px`,
   `blockSize=3`, and Harris detection disabled.
3. Rank candidates in each bucket by descending minimum-eigenvalue response;
   ties are ordered by ascending y, then ascending x.
4. Select candidates round-robin over buckets in row-major bucket order,
   enforcing the 5 px minimum distance globally across bucket boundaries.
   Stop at 500 valid features or when no bucket can contribute another
   candidate. Therefore, fewer than 500 states may be initialized if fewer
   qualifying corners exist.
5. For every selected point initialize `vx=0`, `vy=0`, `confidence=1`,
   `timestamp=t_frame_common`, `valid=1`.

The 8 × 6 detector-selection buckets are distinct from the 16 px association
grid.

## 4. Event and timestamp representation

An input event is `(t, x, y, p)`:

- `t`: original signed integer DAVIS timestamp in microseconds;
- `x`, `y`: integer sensor pixel coordinates;
- `p`: original integer polarity value.

The parser MUST preserve `p` in event output. Polarity is not used by this
association or update algorithm.

Tonic may independently zero each data stream's timestamps. The tracker MUST
therefore receive Stage 1B synchronized windows and use the event timestamp
offset supplied by that synchronization layer. Define:

`t_event_common = t_event_original + event_timestamp_offset_us`

The offset is an integer number of microseconds. State timestamps and APS
timestamps used for prediction are in this common-axis microsecond domain.
Original event time is retained separately in output. The tracker MUST NOT
silently interpret timestamps as seconds or convert units.

For consecutive APS frames at common-axis times `t_k` and `t_(k+1)`, consume
events in the half-open interval `[t_k, t_(k+1))`. Process events in ascending
common timestamp; preserve source order for equal timestamps. State changes
from one event are visible to the next event, including events with equal time.

## 5. Spatial grid organization

For the reference 240 × 180 sensor, use 16 × 16 pixel cells, giving 15
columns × 12 rows. A feature at `(x, y)` is assigned to:

`cell_x = floor(x / 16)`  
`cell_y = floor(y / 16)`

Every active feature has one grid membership. Features may leave the sensor
coordinate range during prediction/update; their bin coordinates are still
computed by floor division and may be outside the nominal sensor grid. Event
coordinates are expected to have passed dataset coordinate validation.

The online implementation indexes stored feature positions. To conservatively
include any point that might move into the spatial gate during the temporal
gate, maintain:

`vmax_x = max_i(abs(vx_i))`  
`vmax_y = max_i(abs(vy_i))`

over all active states observed since the index was built or updated. These
bounds only increase during a frame interval. For an event at `(xe, ye)`,
compute:

`extent_x = R + vmax_x * T`  
`extent_y = R + vmax_y * T`

Visit every cell whose x/y cell coordinates intersect the closed rectangle
`[xe-extent_x, xe+extent_x] × [ye-extent_y, ye+extent_y]`, with one extra cell
on each side of each integer cell bound. This margin is part of the frozen
retrieval procedure. Retrieve candidate feature indices from those bins and
process them in ascending active-slot order. Each candidate is then tested by
the exact temporal and spatial gates below; the expanded rectangle itself is
not an association gate.

After a successful event update, move the updated feature's one grid
membership to the cell containing its new stored `(x, y)` by removing and
reinserting its membership, including when its cell did not change. Update
`vmax_x` and `vmax_y` if the new absolute velocity exceeds either bound. No
feature enters or leaves the index except through initialization or explicit
invalidation. If a calculated search bound is non-finite, retrieve every
indexed active feature and rely on the exact gates.

## 6. Association gates and conflict policy

For each retrieved active feature `i`, calculate:

`dt = t_event_common - timestamp_i`

The event is temporally eligible only if:

`abs(dt) < T`

The boundary is strict. Calculate the predicted point and squared residual:

`x_hat = x_i + vx_i * dt`  
`y_hat = y_i + vy_i * dt`  
`dx = x_event - x_hat`  
`dy = y_event - y_hat`  
`d2 = dx * dx + dy * dy`

The feature passes the spatial gate only if:

`d2 < R * R`

This boundary is also strict. No square root is used. The expanded grid
retrieval is conservative; the gates above alone determine matches.

Conflict policy:

1. No passing features: reject the event; do not change feature state.
2. One passing feature: select it.
3. Multiple passing features: count the event as ambiguous and select the
   feature with the smallest `d2`; exact distance ties go to the lower feature
   ID.
4. An event updates at most one feature. A feature may receive any number of
   chronological events.
5. Association is recomputed against current state for each event. Do not
   reuse an assignment made against an earlier state.

## 7. Prediction and alpha-beta update

For the selected feature, use the same `dt`, `x_hat`, `y_hat`, `dx`, and `dy`
as above. Since online events are chronologically ordered and state starts no
later than the event-window start, `dt` MUST be non-negative. A negative `dt`
is a synchronization/order error and MUST be surfaced; do not update.

Position update:

`x_new = x_hat + alpha * dx`  
`y_new = y_hat + alpha * dy`

Velocity update:

- If `dt == 0`, set `vx_new = vx_i`, `vy_new = vy_i`; position and confidence
  may still update.
- If `dt > 0`, define `dt_den = max(dt, 1 μs)` and calculate:

  `vx_new = vx_i + (beta / dt_den) * dx`  
  `vy_new = vy_i + (beta / dt_den) * dy`

The frozen reference coefficients are `alpha=0.5` and `beta=0.1`. Both are
dimensionless and constrained to `[0,1]`.

Confidence update, using the selected candidate's squared distance:

`q = min(d2 / (R * R), 1)`  
`confidence_new = confidence_i * (1 - 0.05 * q)`

Confidence remains in `[0,1]`. The update timestamp becomes
`t_event_common`. A successful update writes the new position, velocity,
confidence, and timestamp atomically; ID and `valid` remain unchanged.

Before committing, all proposed floating fields MUST be finite and representable
as float32. Otherwise reject the update atomically, report status
`numerical_rejection`, leave state and grid membership unchanged, and count it
as rejected. A numerical rejection does not invalidate a feature.

## 8. Invalidation and lifetime

The frozen runtime has **no automatic invalidation rule**: an unmatched event,
low confidence, predicted position outside the sensor, or numerical rejection
does not invalidate a feature. No slot is invalidated during a normal tracking
run and no new APS detections are allocated after initial setup. The memory
supports explicit slot invalidation outside the event loop; if a caller uses
that operation, it MUST rebuild the grid before resuming event processing.
Invalidating a slot sets the record to the inactive representation above. A
later independent run starts with a new initialization.

The hybrid Stage 7 evaluator's optional APS snapping is not part of this
frozen asynchronous tracker. Subsequent APS frames are used for endpoint
evaluation only; they do not correct or replace event-driven states.

## 9. Frame-time prediction and track output

After processing the event window, for each active feature compute a
non-mutating prediction at `t_(k+1)`:

`dt_end = t_(k+1) - timestamp_i`  
`x_end = x_i + vx_i * dt_end`  
`y_end = y_i + vy_i * dt_end`

`dt_end` MUST be non-negative. Do not change state timestamp or coordinates
for this prediction. The endpoint evaluator may match these predictions
one-to-one to next-frame Shi–Tomasi points within 15 px; that match is an
evaluation metric and does not alter event-driven states.

The canonical state output contains one record per slot using the FeatureState
field order and types in Section 2. The canonical prediction/track record is:

| Field | Type |
|---|---|
| `id` | signed int32 |
| `slot` | signed int32 |
| `x` | float64 prediction in pixels |
| `y` | float64 prediction in pixels |
| `vx` | float32 pixels/μs |
| `vy` | float32 pixels/μs |
| `confidence` | float32 in `[0,1]` |
| `timestamp` | signed int64 common-axis μs, equal to `t_(k+1)` |

The reference per-event evolution record uses these fields and types:

| Field | Type |
|---|---|
| `event_index` | signed int64 |
| `t_event` | signed int64, original event timestamp |
| `t_common_us` | signed int64 |
| `x_event`, `y_event`, `p` | signed int64 |
| `feature_id`, `feature_slot` | signed int32 |
| `candidate_count` | uint32, number passing both gates |
| `updated` | uint8 |
| `status` | NumPy Unicode string `U24` |
| `delta_t_us` | signed int64 |
| `predicted_x`, `predicted_y` | float64 |
| `residual_x`, `residual_y` | float64 |
| `updated_x`, `updated_y`, `updated_vx`, `updated_vy` | float64 |
| `position_update_magnitude`, `velocity_update_magnitude` | float64 diagnostics |
| `confidence_before`, `confidence_after` | float32 |
| `update_latency_us` | float64 software diagnostic |

Rows are emitted in stable chronological processing order. For an unmatched
event, `feature_id` and `feature_slot` are `-1`, `status="unmatched"`, and
floating diagnostics remain NaN except measured latency. For a numerical
rejection, the selected ID/slot, elapsed time, prediction, and residual are
retained, `updated=0`, `status="numerical_rejection"`, and uncommitted update
fields remain NaN. A successful row has `updated=1` and
`status="updated"`. Selected `d2` is used internally for the gate and
confidence update but is not a field in this evolution record; it can be
recomputed as `residual_x² + residual_y²`.

An experiment archive MUST preserve initial and final state arrays, ordered
per-event records, per-feature update counts, configuration, timestamp units,
and diagnostics. The reference NumPy archive contains `evolution`,
`events_per_feature`, `initial_states`, and `final_states`; CSV/JSON exports
may accompany it. Polarity is retained in all per-event records.

## 10. Data path

```text
Event (t, x, y, p)
  ↓
Event parser and common-axis timestamp offset
  ↓
Stable chronological ordering (source order for equal timestamps)
  ↓
Grid index lookup (conservative velocity-expanded cell rectangle)
  ↓
Candidate feature IDs / slots
  ↓
Temporal eligibility: abs(dt) < T
  ↓
Prediction: x_hat = x + vx*dt; y_hat = y + vy*dt
  ↓
Squared-distance gate: dx² + dy² < R²
  ↓
Deterministic association (nearest d², then lower feature ID)
  ↓
Confidence and alpha-beta state update
  ↓
Feature-state memory and grid re-bin
  ↓
Next-APS-time prediction and track output (state is not mutated)
```

## 11. Computational operation accounting

The counts below describe the algorithmic data path. They are parameterized
because the number of retrieved candidates varies by event. They exclude
Python interpreter, array allocation, sorting implementation, file I/O, and
plotting costs. Let:

- `N`: active features;
- `G`: cells visited by the conservative grid rectangle;
- `C`: feature IDs retrieved from those cells;
- `K`: retrieved features passing the temporal gate;
- `M`: features passing both gates (`0 <= M <= K`).

### Per event

| Stage | Arithmetic / comparisons | Memory reads | Memory writes |
|---|---|---|---|
| Parse/order | Read `t,x,y,p`; add timestamp offset once; chronological sort uses timestamp comparisons. | One event record. | One processing-order entry. |
| Grid bound | 2 velocity-bound × `T` multiplications, 2 radius additions, 4 event-coordinate bound additions/subtractions; cell-bound division/floor and integer comparisons. There are up to four integer ±1 margins. | `vmax_x`, `vmax_y`, `R`, `T`, grid dimensions. | None for lookup. |
| Grid lookup | Cell-coordinate comparisons; integer cell enumeration or comparison against occupied bin keys. | Bin directory/keys and member IDs in up to `G` cells. | None. |
| Temporal gate | Per retrieved feature: one timestamp subtraction, absolute-value/sign handling, one strict comparison `abs(dt)<T`. | Candidate state timestamp; candidate ID/slot. | None. |
| Prediction and spatial gate | Per temporally eligible feature: 2 velocity×`dt` multiplications, 2 position additions, 2 residual subtractions, 2 residual squarings, 1 sum, 1 strict comparison with `R²`. `R²` is computed once per configuration. | Candidate `x,y,vx,vy`; event `x,y`; configured `R²`. | None. |
| Conflict resolution | Up to `M-1` distance comparisons and, on exact ties, ID comparisons; candidate count increment/comparison. | Candidate `d2`, IDs. | Local winner/counter only. |
| State update | Recompute selected feature prediction/residual: 2 velocity×`dt` multiplications, 2 additions, 2 subtractions. Position: 2 coefficient multiplications + 2 additions. Positive-`dt` velocity: 1 `beta/dt_den` division + 2 multiplications + 2 additions. Confidence: 1 division `d2/R²`, min comparison, 1 penalty multiplication, 1 subtraction, 1 confidence multiplication. Finite/range checks compare proposed fields. | Selected state record; denominator/configuration; confidence; selected `d2`. | On success: `x,y,vx,vy,confidence,timestamp`; remove/reinsert grid membership; update counters and event output. On rejection: event output/counters only. |

`abs(dt)` may be implemented as a sign test plus conditional negation. The
comparison need not compute a square root. If `dt=0`, velocity divisions and
velocity corrections are skipped.

### Memory traffic model

For one event, association reads at most `C` candidate IDs and their state
records (up to `K` state records need prediction fields), plus the event and
grid directory data. A successful event writes one feature state, event
diagnostics, update counters, and one grid removal/insertion. A
rejected event writes no feature state. The final APS prediction reads each
active state once and writes one track record per active feature; it does not
write feature state.

The fixed feature memory is `500 × 33 = 16,500` bytes at the frozen operating
point, excluding grid and output buffers. A packed logical grid has 180 cells.
The Python dictionary/list grid is not a byte-accurate hardware memory model.
A hardware mapping must separately budget grid membership storage, per-cell
occupancy/indexing, event buffering, and output buffering.

## 12. Frozen-scope exclusions and limitations

This specification covers only the Python algorithmic reference and its
software data path. It does not define or implement FPGA, HLS, RTL, VIO, SLAM,
SNN, fixed-point arithmetic, clock-domain crossing, or hardware timing.
Stage 9 was one-factor-at-a-time on one five-interval DAVIS segment. EPE uses
next-frame Shi–Tomasi correspondences rather than ground-truth feature
identity. The parameters are a frozen experimental baseline, not a claim of
scientific optimum or universal performance.
