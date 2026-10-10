# DAVIS frame-window synchronization report

- Recording: `shapes_6dof`
- Timestamp values in the loaded Tonic arrays: integer **microseconds**, preserved unchanged.
- Raw ROS-bag stream origins: **seconds** from `/home/administrator/Desktop/MVBC/Research_Work/06_Experiments/SNN/step_1_VIO/data/metadata/dataset_manifest.json`.
- Common comparison axis: microseconds relative to the earliest raw stream start
  (1468939993.0674160 s in the source clock); per-stream origin
  differences are explicitly rounded to the Tonic microsecond resolution.
- APS frame count: 1,356; consecutive half-open frame intervals:
  1,355.
- Origin offsets on the common axis (microseconds): {'events': 0, 'aps_frames': 19198, 'imu': 32647, 'ground_truth': 44530}.
- Events assigned within the first-to-last APS interval span:
  17,960,747.
- Empty event intervals: 0.
- Events before the first APS frame: 105.
- Events at or after the final APS frame: 1,625.
- IMU samples assigned inside the APS span:
  59,606.
- Ground-truth samples assigned inside the APS span:
  11,854.
- Event membership uses `[t_k, t_(k+1))`; samples remain index slices into the
  original arrays and are not copied per interval.
- IMU and ground truth are selected with the same half-open interval on the
  common timeline. Their returned sample timestamps remain stream-local Tonic
  microseconds; `common_timestamps_us` exposes mapped times separately.
- Validation passed: `True`.
- Synchronization method: add measured per-stream source-start offsets from the
  raw ROS-bag manifest. No interpolation, resampling, clock correction, or
  calibration is performed.
