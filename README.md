
# Asynchronous Feature-State Front End for Hybrid Event-Frame-Inertial VIO

Research project for validating an asynchronous event-frame feature tracker
in Python. Hardware implementation is out of scope until the tracker has been
experimentally validated and the algorithm frozen.

## Roadmap

1A. DAVIS dataset loading and inspection
1B. DAVIS frame-to-frame temporal synchronization
1C. APS feature extraction
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
