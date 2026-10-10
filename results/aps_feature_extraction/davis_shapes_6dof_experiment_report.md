# Stage 2A — APS Shi–Tomasi feature extraction

- Algorithm: OpenCV Shi–Tomasi `goodFeaturesToTrack`; Harris mode disabled.
- Selected APS frame indices: `[0, 338, 677, 1016, 1355]`.
- Bucket grid: 8 × 6.
- Feature record: `id:int32, x:float32, y:float32, quality:float32, valid:uint8`.
- IDs are unique within this experiment output. They do not establish identity across frames.
- No event association, velocity, KLT, or tracking is performed.

| Requested cap | Mean detected | Mean bucket coverage | Mean normalized entropy | Mean median quality |
|---:|---:|---:|---:|---:|
| 50 | 50.0 | 99.2% | 0.993 | 0.000237805 |
| 100 | 100.0 | 99.2% | 0.994 | 0.000133479 |
| 200 | 200.0 | 99.2% | 0.991 | 7.07443e-05 |
| 300 | 300.0 | 99.2% | 0.985 | 4.91424e-05 |
| 500 | 500.0 | 99.2% | 0.969 | 2.84879e-05 |
| 1000 | 633.0 | 99.2% | 0.960 | 2.24664e-05 |

**Recommended count for the next experiment: 500.**
Selection rule: choose the highest requested cap with mean detected count at least 80% of the cap and mean occupancy of at least 50% of the spatial buckets. If no cap qualifies, choose the cap with the highest mean bucket coverage, then highest mean detected count. This is a development heuristic, not a tracking-performance result.

## Parameters

| Parameter | Value |
|---|---:|
| Quality level | 0.01 |
| Minimum feature distance | 5.0 px |
| Block/window size | 3 px |
| Bucket grid | 8 × 6 |

Feature quality is OpenCV's minimum-eigenvalue corner response, sampled at each detected point. It is an image-response score, not a calibrated confidence probability.

Plots: `davis_shapes_6dof_feature_overlays.png`, `davis_shapes_6dof_spatial_buckets.png`, and `davis_shapes_6dof_comparison.png`.
