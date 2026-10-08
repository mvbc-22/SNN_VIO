
# Asynchronous Feature-State Front End for Hybrid Event-Frame-Inertial VIO

Research project for a compact event-driven feature front end, visual-inertial
odometry, and eventual SNN/FPGA acceleration. The hardware implementation is
the primary research endpoint; the Python pipeline is its golden reference.

## Roadmap

0. Dataset setup and timestamp validation
1. Feature extraction from APS frames
2. Compact feature-state representation
3. Event-to-feature association
4. Asynchronous state updates
5. Tracking-quality comparison
6. Fixed-point model
7. HLS/RTL and FPGA prototype
8. VIO backend
9. SLAM

The existing DAVIS exploration and dense-feature experiments are in
[`notebooks/01_tonic_davis_test_modular.ipynb`](notebooks/01_tonic_davis_test_modular.ipynb)
and [`notebooks/02_dense_event_feature_extraction_all_windows.ipynb`](notebooks/02_dense_event_feature_extraction_all_windows.ipynb).

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

Validate the local DAVIS bag, check which MVSEC scenes are present, and create
`data/metadata/dataset_manifest.json`:

```bash
.venv/bin/python -m scripts.prepare_datasets
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

## Dataset synchronization and initial experiments

The preparation script reads the original ROS bag timestamps for `/dvs/events`,
`/dvs/image_raw`, `/dvs/imu`, and (when present) `/optitrack/davis`. It validates
that required streams are non-empty and ordered, then reports each stream on a
common microsecond timeline relative to the earliest stream start. This avoids
mistaking Tonic's independently zero-based stream timestamps for cross-sensor
alignment. The manifest records stream counts, time ranges, and timestamp
offsets; it does not interpolate samples or apply camera/IMU extrinsic or clock
calibration.

After dataset validation:

1. Run `notebooks/01_tonic_davis_test_modular.ipynb` to inspect event windows
   and event representations.
2. Run `notebooks/02_dense_event_feature_extraction_all_windows.ipynb` to
   establish the frame-based feature baseline and tracking inputs.
3. Use the validated shared timestamps as the basis for event-feature
   association and asynchronous feature-state updates.

The frame-based baseline is the Python golden model for subsequent
fixed-point, HLS/RTL, and FPGA work. Track feature coverage, event/IMU
throughput, update rate, latency, and odometry error as the stages are added.
