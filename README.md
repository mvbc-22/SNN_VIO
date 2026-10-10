
# Asynchronous Feature-State Front End for Hybrid Event-Frame-Inertial VIO

Research project for validating an asynchronous event-frame feature tracker
in Python. Hardware implementation is out of scope until the tracker has been
experimentally validated and the algorithm frozen.

## Roadmap

1A. DAVIS dataset loading and inspection
1B. DAVIS frame-to-frame temporal synchronization
2A. APS Shi-Tomasi feature initialization
2B. APS Shi-Tomasi vs FAST detector comparison
2. Compact feature-state representation
3. Event-to-feature association and asynchronous state updates
4. Tracking-quality comparison and algorithm validation

The existing DAVIS exploration and dense-feature experiments are in
[`notebooks/01_tonic_davis_test_modular.ipynb`](notebooks/01_tonic_davis_test_modular.ipynb)
and [`notebooks/02_dense_event_feature_extraction_all_windows.ipynb`](notebooks/02_dense_event_feature_extraction_all_windows.ipynb).
Stage 1A is implemented in [`src/davis_inspection.py`](src/davis_inspection.py)
and [`notebooks/03_davis_dataset_inspection.ipynb`](notebooks/03_davis_dataset_inspection.ipynb).

## Datasets

- **DAVIS 240C / `shapes_6dof`** — initial development sequence with events,
  APS frames, IMU, and OptiTrack ground truth. The local copy is expected at
  `data/DAVIS/DAVISDATA/shapes_6dof.bag`.
- **MVSEC** — evaluation dataset for later cross-dataset validation. Store
  scenes under `data/MVSEC/<scene>/`; available scene names are
  `indoor_flying`, `outdoor_day`, `outdoor_night`, and `motorcycle`.

Raw datasets are intentionally excluded from Git. Keep original bags unchanged;
the preparation script writes only a small, ignored metadata manifest.

## Environment and first run

The repository's Python environment is `.venv`. For a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell, create and use the same environment with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name snn-vio --display-name "SNN_VIO (.venv)"
```

In VS Code, select the `SNN_VIO (.venv)` kernel before running a notebook. After
installing or changing packages, restart the notebook kernel before rerunning
cells so the session loads the updated dependencies.

Validate the local DAVIS bag, check which MVSEC scenes are present, and create
`data/metadata/dataset_manifest.json`:

```bash
.venv/bin/python -m scripts.prepare_datasets
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_datasets
```

The `shapes_6dof` bag is already present in this workspace. If it is absent in a
fresh setup, Tonic can fetch it with:

```bash
.venv/bin/python -m scripts.prepare_datasets --download-davis
```

MVSEC downloads are opt-in because a scene comprises multiple large ROS bags.
For example, to fetch the indoor-flying scene:

```bash
.venv/bin/python -m scripts.prepare_datasets --download-mvsec indoor_flying
```

Repeat `--download-mvsec` to fetch additional scenes. You can place manually
downloaded MVSEC bag pairs in `data/MVSEC/<scene>/` and rerun the default
validation command without downloading. Override the manifest location with
`--manifest PATH` when needed.

## Stage 1A: Dataset inspection

Run Stage 1A with:

```bash
.venv/bin/python -m src.davis_inspection
```

This uses Tonic's DAVIS loader and writes a summary JSON and timestamp-range
plot under `results/dataset_inspection/`. Tonic returns integer timestamps in
microseconds, but independently resets each stream's first timestamp to zero.
The plot shows numeric stream-relative ranges on the same axis for inspection;
it does not synchronize or align streams. The loader reuses Tonic's arrays and
validates event fields, coordinates, polarity, frame dimensions, timestamp
ordering, and available IMU/ground-truth timestamps in bounded chunks.

Run the Stage 1A validation tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

The Stage 1A dataset inspection notebook is
[`notebooks/03_davis_dataset_inspection.ipynb`](notebooks/03_davis_dataset_inspection.ipynb).
Continue only after reviewing its outputs and deciding the next stage. The
existing dense event-image notebook is exploratory prior work; it is not an
APS feature tracker.

## Stage 1B: Frame-to-frame temporal synchronization

Run the Stage 1B windowing and report generation with:

```bash
.venv/bin/python -m src.frame_synchronization
```

This uses `load_davis` and the raw-source stream starts in
`data/metadata/dataset_manifest.json`. The manifest must describe the same bag
and agree with its size, stream counts, and timestamp durations; refresh a stale
manifest with `.venv/bin/python -m scripts.prepare_datasets`.

`make_frame_windows(recording, stream_start_timestamps_s)` creates a lazy
sequence of consecutive APS pairs. `windows.get_window(k)` returns the two APS
frames and their original Tonic timestamps, original-index ranges for event,
IMU, and ground-truth samples, and explicit common-axis frame times. Event
membership uses `[t_k, t_(k+1))`. Data samples remain views into the loaded
arrays. Tonic timestamps stay integer microseconds; raw manifest timestamps
are seconds; their origin differences are explicitly rounded to the
microsecond resolution to construct the common axis. No resampling,
interpolation, or clock correction occurs.

The report and five-interval timestamp plot are written to
`results/dataset_synchronization/`. Run tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

The runnable notebook is
[`notebooks/04_davis_frame_synchronization.ipynb`](notebooks/04_davis_frame_synchronization.ipynb).
Stage 1B does not include feature extraction, event tracking, VIO, SLAM, SNN,
FPGA, HLS, or RTL.

## Stage 2A: APS Shi-Tomasi feature initialization

Stage 2A detects Shi-Tomasi corners in APS images accessed via the Stage 1B
frame-window sequence. It spatially buckets candidates before selecting up to
50, 100, 200, 300, 500, and 1,000 features. The fixed-width feature record is
`[id:int32, x:float32, y:float32, quality:float32, valid:uint8]`; it contains
no patches or velocity. Detection-record IDs are unique within one experiment
run and do not claim cross-frame feature identity.

Run the comparison over five evenly spaced APS frames with:

```bash
.venv/bin/python -m src.aps_features
```

The defaults are quality level `0.01`, minimum distance `5` pixels, block size
`3`, and an `8 × 6` spatial bucket grid. These can be changed with command-line
options; `--frame-samples` controls how many APS frames are evaluated.
Measurements, a numeric feature archive, overlays, bucket maps, comparison
plots, and a short experiment report are written under
`results/aps_feature_extraction/`. Run the same experiment interactively in
[`notebooks/05_aps_shi_tomasi_features.ipynb`](notebooks/05_aps_shi_tomasi_features.ipynb).
The detector uses the `opencv-python` dependency.

On the current five-frame sample, mean detected counts were 50, 100, 200, 300,
500, and 633 for requested caps 50, 100, 200, 300, 500, and 1,000. Mean bucket
coverage was approximately 99% for all caps, while median corner quality
declined as lower-strength corners were included. The development heuristic
recommends **500** for the next evaluation: it was attained on all sampled
frames, while the 1,000 cap was not. This does not establish tracking quality.

Run the tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

No KLT, feature association, velocity, or event-driven tracking is included.

## Stage 2B: Shi-Tomasi vs FAST

Compare Shi-Tomasi and FAST on the same five evenly sampled consecutive APS
frame pairs:

```bash
.venv/bin/python -m src.detector_comparison --pair-samples 5 --timing-repeats 3 --fast-threshold 1
```

The default requested caps are 50, 100, 200, 300, 500, and 1,000 features.
Both detectors use the same `8 × 6` bucket grid and 5 px minimum spacing.
FAST uses TYPE_9_16 and non-maximum suppression. The measured run uses threshold
1 so FAST can supply enough candidates for the requested operating points;
this is a deliberately low, tunable threshold, not a claim that it is the
optimal FAST configuration. Change it with `--fast-threshold`. Runtime is
median wall-clock time over the configured repeats for candidate detection
plus per-cap spatial selection; dataset loading, decoding, synchronization, and
plotting are excluded.

The experiment measures count attainment, occupied-bucket coverage and
normalized entropy, detector-native response, raw-coordinate mutual-nearest
repeatability within 3 px on selected consecutive frames, and runtime. Response
values must not be compared numerically across detectors: Shi-Tomasi uses
minimum eigenvalue, while FAST uses OpenCV's keypoint response. Repeatability
is a detector-stability proxy only; no motion compensation, optical flow, or
KLT is used. Results and artifacts are written under
`results/detector_comparison/`. The runnable notebook is
[`notebooks/06_aps_detector_comparison.ipynb`](notebooks/06_aps_detector_comparison.ipynb).

On the five sampled frame pairs (10 frames), at the 500-feature operating
point, Shi-Tomasi detected 500.0/frame with 99.6% mean bucket coverage, 43.1%
raw-coordinate repeatability, and 9.374 ms mean measured runtime. FAST detected
499.8/frame with 100.0% coverage, 45.5% repeatability, and 3.834 ms runtime.
The baseline therefore recommends **FAST** at this operating point: it met the
count and coverage gates, had 2.4 percentage points higher repeatability, and
was about 2.4× faster in this measurement. At the 1,000 cap, neither method
reached the cap (Shi-Tomasi 632.7; FAST 550.3), so this comparison supports an
initial 500-feature detector choice, not a high-cap conclusion. Runtime is
machine/OpenCV dependent, and the small sample plus raw-coordinate metric are
limitations; validate thresholds and tracking quality in a later approved
stage. Run the full tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

No event association, event-driven tracking, VIO, SNN, FPGA, HLS, RTL, or SLAM
is implemented.

## Stage 3: Fixed-capacity FeatureState memory

Stage 3 adds a fixed-size NumPy structured array that models hardware-oriented
feature-state storage. Its record is
`[id:int32, x:float32, y:float32, vx:float32, vy:float32, confidence:float32, timestamp:int64, valid:uint8]`.
Coordinates use pixels, velocity uses pixels per microsecond, timestamps remain
integer microseconds from the existing Stage 1 loader, confidence is in `[0, 1]`,
and validity is 0/1. No image patches are stored.

Unused slots initialize to `id=-1`, zero position/velocity/confidence/timestamp,
and `valid=0`. `allocate()` chooses the first invalid slot and assigns a
monotonically increasing integer ID; a full memory raises `BufferError`.
`update()` writes position and timestamp and optionally replaces velocity or
confidence, preserving omitted fields and the slot ID. Backward timestamps,
non-finite values, and confidence outside `[0, 1]` are rejected.
`invalidate()` resets the entire slot to its invalid representation.
`predict(slot, timestamp)` computes `x + vx * Δt`, `y + vy * Δt` for an explicit
microsecond timestamp without mutating the stored state; prediction before the
state timestamp is rejected. `get_active()` returns a compact copy in slot
order. Velocity has no estimator yet and is initialized to zero. APS detections
start at confidence 1.0; detector response remains stored in the separate
Stage 2A feature records rather than being silently mapped to confidence.

Run the APS-frame-to-state initialization demonstration with:

```bash
.venv/bin/python -m src.feature_state_memory --feature-count 500 --slots 500
```

This loads the DAVIS synchronized APS sequence, detects Shi-Tomasi features in
the middle frame, allocates their positions into the configured fixed-size
memory, and saves the full memory, configuration, report, and overlay under
`results/feature_state_memory/`. The runnable notebook is
[`notebooks/07_feature_state_memory.ipynb`](notebooks/07_feature_state_memory.ipynb).
The Stage 3 unit tests cover initialization, allocation/full behavior, updates,
invalidation, prediction, active-state reads, validation, and end-to-end
Shi-Tomasi initialization. No event association or event processing is
implemented.

## Stage 4A: Brute-force event-to-feature association

Run the correctness-reference association on one APS-to-APS half-open event
interval:

```bash
.venv/bin/python -m src.event_association --interval-index 678 --feature-count 500 --slots 500 --radius-px 8 --time-window-us 50000
```

The runner detects Shi-Tomasi features on the interval's first APS frame,
initializes the existing fixed-capacity FeatureState memory, and tests every
interval event against every active state. The gates are strict
`d² < R²` and `abs(t_event - t_feature) < T`; the distance is squared without a
square root. `R` is configurable in pixels and `T` in microseconds. Events are
stably processed by timestamp, retaining source order at equal timestamps.
Original event timestamps and polarity are retained in output; the Stage 1
event-stream origin offset is added only to calculate a common-axis timestamp
against APS feature states. No silent unit conversion or state mutation occurs.

No match is rejected. If several features match, the event is counted as
ambiguous and assigned once to the smallest squared distance, with lower
feature ID as the exact-tie breaker. A feature may receive many events. Polarity
is preserved and visualized but does not affect association. Outputs include
event decisions, events-per-active-feature (including zero counts), diagnostics,
a report, and an APS/event/association overlay under
`results/event_association/`. The runnable notebook is
[`notebooks/08_event_association_brute_force.ipynb`](notebooks/08_event_association_brute_force.ipynb).

This is intentionally a brute-force reference, not an optimized implementation.
It does not update FeatureState, implement grid hashing, or claim validated
tracking. Run tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Stage 4B: Asynchronous FeatureState update

Run the full online alpha-beta update experiment on one APS interval:

```bash
.venv/bin/python -m src.event_state_update --interval-index 678 --feature-count 500 --slots 500 --radius-px 8 --time-window-us 50000 --alpha 0.5 --beta 0.1
```

Events are stably processed chronologically and associated with the Stage 4A
brute-force gates against the **current** state memory. Each accepted event
updates that state before the next event is associated. Position is updated
with `x_hat + alpha * rx`, `y_hat + alpha * ry`; velocity uses
`v + beta / dt * residual`. `alpha` and `beta` are configurable in `[0, 1]`.
For `dt=0`, the position update is applied while velocity is held unchanged.
For positive small `dt`, the denominator is `max(dt, --minimum-delta-t-us)`;
timestamps remain integer microseconds and velocity remains pixels per
microsecond.

Confidence uses the configurable penalty rule
`c' = c * (1 - penalty * min(d² / R², 1))`, with penalty in `[0, 1]`.
Only valid slots are associated; inactive slots remain invalid. A proposed
update is rejected atomically if it is non-finite or cannot be represented by
the float32 state fields. The experiment records per-event event/predicted/
updated positions, residual and update magnitudes, timestamp and confidence
evolution, successful updates, rejects, ambiguity, and event counts by feature.
Outputs and a two-frame trajectory visualization are saved under
`results/event_state_update/`. Cyan triangles mark predicted feature positions,
magenta squares mark updated positions, and polarity-colored markers show event
locations. The runnable notebook is
[`notebooks/09_asynchronous_feature_state_update.ipynb`](notebooks/09_asynchronous_feature_state_update.ipynb).

This remains the direct brute-force reference; no grid hashing, Kalman filter,
or image patches are implemented. It demonstrates algorithm behavior but is
not a tracking-quality validation.

On DAVIS interval 678 with 500 initialized states, `R=8 px`, `T=50,000 μs`,
`alpha=0.5`, `beta=0.1`, a 1 μs minimum positive `dt`, and confidence penalty
0.05, the run processed 10,879 events and made 1,555 successful online updates
(9,324 rejected; 576 of the decisions were ambiguous). All 500 slots remained
active; 132 features received updates, averaging 11.78 updates among those
features (maximum 44). Mean position correction magnitude was 2.030 px and
mean velocity correction magnitude was 0.01145 px/μs. The association ratio is
lower than Stage 4A's static-memory diagnostic because this stage re-associates
each event against the evolving memory; these values are experiment outputs,
not detector/tracker performance claims.

## Stage 5: Complete asynchronous feature tracker

Run the complete one-pair Python golden model:

```bash
.venv/bin/python -m src.asynchronous_feature_tracker --interval-index 678 --feature-count 500 --slots 500 --radius-px 8 --time-window-us 50000 --alpha 0.5 --beta 0.1 --reference-match-radius-px 15
```

The pipeline uses APS frame k for Shi-Tomasi and FeatureStateMemory
initialization; processes the half-open `[t_k, t_(k+1))` event view
chronologically using Stage 4A brute-force association and Stage 4B online
alpha-beta updates; predicts active states at the common-axis timestamp of frame
k+1; detects reference Shi-Tomasi features in frame k+1; then performs
deterministic greedy one-to-one nearest matching within the configured pixel
radius. Candidate pairs sort by squared distance, feature ID, predicted index,
then reference index. EPE is the mean Euclidean pixel error for successful
matches only; unmatched predictions count toward failure rate. The match radius
and all detector, association, update, and confidence parameters are recorded
with outputs.

The reported track lifetime is interval-censored: in a single pair, all
initially valid states are considered observable through the full APS interval,
since this baseline does not invalidate tracks. Events per successful update is
processed event count divided by successful state updates. Update latency is
measured per event inside the online association/update loop (including
rejections); processing time measures the entire in-memory pair pipeline from
initial frame detection through endpoint matching, excluding dataset loading
and output plotting.

Outputs include numeric records and diagnostics under
`results/asynchronous_feature_tracker/`, plus six plots: initial APS features,
events, event-driven trajectories, predicted positions, reference features,
and predicted/reference errors. Stage correctness checks report PASS/FAIL for
event processing, association, state update, prediction, and frame-to-frame
evaluation. The notebook is
[`notebooks/10_asynchronous_feature_tracker.ipynb`](notebooks/10_asynchronous_feature_tracker.ipynb).

This deliberately remains an unoptimized correctness baseline: brute-force
event-to-feature association is retained. No KLT, VIO, SNN, FPGA, HLS, RTL,
SLAM, grid hashing, Kalman filter, or image patches are included.

## Stage 6A: Conventional frame-only KLT baseline

Run the frame-only reference on the same APS pair used by Stage 5:

```bash
.venv/bin/python -m src.frame_only_baseline --interval-index 678 --feature-count 500 --reference-match-radius-px 15
```

This baseline detects Shi-Tomasi features in frame k using the same detector,
parameters, bucketing and requested feature count as Stage 5, then tracks those
same locations to frame k+1 using OpenCV pyramidal Lucas-Kanade optical flow
(`calcOpticalFlowPyrLK`). It detects next-frame Shi-Tomasi reference features
with the same configuration and uses Stage 5's one-to-one endpoint matching
procedure and match radius. The baseline does not access the event samples;
the event-stream panel in its six-plot set is explicitly labeled unused.

Metrics include matched-only EPE, interval-censored track lifetime, KLT
successful track count, failure rate, total in-memory pipeline time, KLT-call
latency, and amortized KLT time per initial feature. KLT failure includes
OpenCV status failure, non-finite results, and points outside the image. For
this one interval, successful tracks receive the full frame-pair duration and
failed tracks zero lifetime. Per-feature latency is a software amortization,
not hardware latency.

The `davis_frame_only_comparison_ready.json` output uses the same
`snn-vio-tracker-comparison-v1` schema as Stage 5's
`davis_asynchronous_comparison_ready.json`; both include the same
`common_metrics` keys for initial features, tracked features, successful
predictions/reference matches, EPE, lifetime, failure rate, processing time,
and operation latency. Detailed method-specific metrics and the latency
measurement boundary are also recorded, because Stage 5 event-loop latency and
the Stage 6A KLT-call latency are not identical operations. The baseline reports
KLT-only failure separately from the common endpoint evaluation failure rate.
Each JSON references its own NPZ arrays and records method, event-use flag,
configuration, and metric definitions. All six visualizations and a report are written under
`results/frame_only_baseline/`. The notebook is
[`notebooks/11_frame_only_klt_baseline.ipynb`](notebooks/11_frame_only_klt_baseline.ipynb).

OpenCV (already included in `requirements.txt`) supplies KLT. Run the complete
test suite with `.venv/bin/python -m unittest discover -s tests`. This is a
conventional frame-only reference only; no VIO is implemented.

## Stage 7: Three-way tracker comparison

Run the frame-only, event-driven, and hybrid systems over the same contiguous
APS segment:

```bash
.venv/bin/python -m src.three_way_comparison --interval-start 678 --interval-count 5 --feature-count 500 --slots 500 --reference-match-radius-px 15
```

All three receive the same initial Shi-Tomasi feature locations, APS frames,
and next-frame reference detections. System A carries those locations forward
with pyramidal KLT and does not access event data. System B uses event-driven
association and alpha-beta updates; next-frame APS features are used only by
the evaluator and do not modify its states. System C applies the same event
updates and then snaps each one-to-one matched state to its next-frame
Shi-Tomasi reference, advancing its timestamp while preserving velocity and
confidence. Unmatched hybrid states are not corrected or reinitialized.

The segment starts at interval 678 and spans five consecutive intervals by
default. Metrics share a one-to-one greedy endpoint match radius. EPE includes
successful predicted-to-reference matches only; unmatched tracks count toward
failure rate. Track survival/lifetime is measured by consecutive interval
matches until first failure and is an APS-reference proxy, not identity
ground truth. Events/update is total events divided by successful endpoint
feature-interval matches; frame-only is N/A. State updates count successful
KLT coordinate updates for A, alpha-beta updates for B, and alpha-beta plus
APS snaps for C.

Processing time includes identical shared Shi-Tomasi initialization/reference
detection costs plus method processing and endpoint matching, but excludes
dataset loading and plots. Latencies are not identical operations: frame-only
measures KLT-call latency per APS interval, while B/C measure per-event
association/update-loop latency including rejected events. Interpret these as
software measurements for this segment, not hardware latency or universal
performance.

Outputs under `results/three_way_comparison/` include comparison-ready JSON,
per-interval and per-feature lifetime CSVs, numeric NPZ arrays, the final
Markdown table, and plots for EPE distribution, track lifetime, feature
survival, processing time, update latency, and events/useful update. The
runnable notebook is
[`notebooks/12_three_way_tracker_comparison.ipynb`](notebooks/12_three_way_tracker_comparison.ipynb).
No VIO or optimization is introduced.

## Stage 8: Hardware-oriented spatial grid association

Compare the Stage 4A brute-force correctness reference with exact grid-based
candidate pruning on one DAVIS frame interval:

```bash
.venv/bin/python -m src.grid_event_association --interval-index 678 --feature-count 500 --slots 500 --radius-px 8 --time-window-us 50000 --cell-width 16 --cell-height 16
```

The sensor dimensions come from the APS frame; cell width and height are
configurable. A persistent grid indexes stored feature positions. Each event
queries a conservative neighborhood expanded by the maximum indexed absolute
velocity times the temporal gate, then applies the unchanged exact per-feature
prediction and strict squared-distance gate. This avoids missing a candidate
whose predicted position crosses multiple cells. During online updates, only
the successfully updated feature is re-binned; the velocity bound only grows,
so it remains conservative. Timestamps and stable chronological order are
preserved, including source order for equal timestamps. Ambiguous matches retain
the Stage 4 policy: minimum squared distance, then lower feature ID.

The comparison verifies all discrete assignment fields exactly and compares
selected squared distances with configurable absolute and relative tolerances
(both default to `1e-9`). A non-equivalent result exits with failure status.
The JSON, CSV performance table, and report are written to
`results/grid_event_association/`. They record candidate feature and distance
calculation counts, throughput, elapsed software runtime, rejected events,
ambiguity counts, and parity. The grid path is also available to
`run_asynchronous_updates` through its optional `spatial_grid` configuration;
the alpha-beta state update equations are unchanged. Grid indexing is Python
software validation, not an FPGA implementation or a hardware performance
estimate.

On the measured interval 678 (10,879 events, 500 initial features), the grid
returned identical assignments, reducing mean distance checks from 500 to
36.52 per event. Measured Python runtime was 9.98 s brute force versus 0.86 s
grid. This is a single-interval software result, not a general or hardware
performance claim.

Run the focused validation with:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Stage 9: Systematic parameter sweep

Run a reproducible one-factor-at-a-time sweep over feature count, square grid
cell size, spatial radius `R`, temporal threshold `T`, `alpha`, and `beta`:

```bash
.venv/bin/python -m src.parameter_sweep --interval-start 678 --interval-count 5
```

The default design uses the Stage 8 grid path and the same contiguous frame
segment for every configuration. Each variant changes one value from the
common baseline; this measures local sensitivity and does not measure
parameter interactions. Supply comma-separated level overrides with
`--feature-counts`, `--grid-cell-sizes`, `--radii-px`, `--time-windows-us`,
`--alphas`, and `--betas`. Configuration JSON records the exact tested
parameters. The long-form CSV includes per-system EPE, track lifetime,
survival, failure rate, events per update, candidate count, processing time,
latency, state-update count, and compact memory proxies. Tradeoff plots and a
multi-objective Pareto-candidate report are saved under
`results/parameter_sweep/`.

Feature storage bytes are calculated from the current fixed-width
`FeatureState` dtype. Logical grid cells/index bytes are compact-layout
estimates, not CPython memory usage or a synthesized hardware result. Accuracy
uses next-APS Shi-Tomasi matches; survival is a reference-match proxy. No
setting is frozen by this stage; broader sequences and parameter interactions
remain to be validated.

Run all tests with `.venv/bin/python -m unittest discover -s tests`.

## Stage 10: Frozen algorithm specification

The current research baseline is formally frozen in
[`ALGORITHM_SPEC.md`](ALGORITHM_SPEC.md). It defines the feature record,
initialization, timestamp domain, grid lookup, strict association gates,
alpha-beta/confidence updates, invalidation behavior, output records, and
computational/memory operation accounting. The selected parameters are a
reproducible baseline, not a proven optimum. No FPGA/HLS/RTL implementation is
included. Reopen the specification only if later experiments demonstrate a
failure requiring an algorithm change.
