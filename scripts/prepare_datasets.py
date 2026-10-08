#!/usr/bin/env python3
"""Validate local VIO datasets and record their shared timestamp ranges."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DAVIS_RECORDING = "shapes_6dof"
DAVIS_STREAMS = {
    "events": ("/dvs/events", True),
    "imu": ("/dvs/imu", True),
    "aps_frames": ("/dvs/image_raw", True),
    "ground_truth": ("/optitrack/davis", False),
}


def _download_datasets(
    data_dir: Path, download_davis: bool, mvsec_scenes: list[str]
) -> None:
    if not download_davis and not mvsec_scenes:
        return

    try:
        from tonic.datasets import DAVISDATA, MVSEC
    except ImportError as exc:
        raise RuntimeError(
            "Dataset downloads require Tonic. Install project requirements first."
        ) from exc

    if download_davis:
        DAVISDATA(
            save_to=str(data_dir / "DAVIS"),
            recording=DAVIS_RECORDING,
        )
    for scene in mvsec_scenes:
        MVSEC(save_to=str(data_dir), scene=scene)


def _stream_times(topic: dict[str, Any], name: str) -> np.ndarray:
    timestamps = np.asarray(topic.get("ts"))
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"{name} has no usable timestamps")
    timestamps = timestamps.astype(np.float64, copy=False)
    if not np.isfinite(timestamps).all():
        raise ValueError(f"{name} contains non-finite timestamps")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError(f"{name} timestamps are not monotonically ordered")
    return timestamps


def _inspect_davis(bag_path: Path, root: Path) -> dict[str, Any]:
    try:
        from importRosbag.importRosbag import importRosbag
    except ImportError as exc:
        raise RuntimeError(
            "Reading DAVIS bags requires the importRosbag package provided "
            "with Tonic. Install project requirements first."
        ) from exc

    topics = importRosbag(
        str(bag_path),
        importTopics=[topic for topic, _ in DAVIS_STREAMS.values()],
        disable_bar=True,
    )
    streams: dict[str, dict[str, Any]] = {}
    stream_arrays: dict[str, np.ndarray] = {}
    for name, (topic_name, required) in DAVIS_STREAMS.items():
        topic = topics.get(topic_name)
        if topic is None:
            if required:
                raise ValueError(f"DAVIS bag is missing required topic {topic_name}")
            continue

        timestamps = _stream_times(topic, name)
        stream_arrays[name] = timestamps
        streams[name] = {
            "topic": topic_name,
            "samples": int(timestamps.size),
            "timestamp_start_s": float(timestamps[0]),
            "timestamp_end_s": float(timestamps[-1]),
            "timestamps_monotonic": True,
        }

        if name == "events":
            width = int(topic["dimX"])
            height = int(topic["dimY"])
            x = np.asarray(topic["x"])
            y = np.asarray(topic["y"])
            if np.any(x >= width) or np.any(y >= height):
                raise ValueError("DAVIS event coordinates exceed the sensor dimensions")
            streams[name]["sensor_size_px"] = [width, height]

    origin_s = min(float(timestamps[0]) for timestamps in stream_arrays.values())
    for name, timestamps in stream_arrays.items():
        streams[name]["start_us"] = int(round((float(timestamps[0]) - origin_s) * 1e6))
        streams[name]["end_us"] = int(round((float(timestamps[-1]) - origin_s) * 1e6))
        streams[name]["duration_us"] = streams[name]["end_us"] - streams[name]["start_us"]

    return {
        "recording": DAVIS_RECORDING,
        "bag": str(bag_path.relative_to(root)),
        "size_bytes": bag_path.stat().st_size,
        "timestamp_origin_s": origin_s,
        "timestamp_units": "microseconds relative to earliest stream start",
        "streams": streams,
    }


def _inspect_mvsec(data_dir: Path, scene: str) -> dict[str, Any]:
    try:
        from tonic.datasets import MVSEC
    except ImportError as exc:
        raise RuntimeError(
            "MVSEC inventory requires Tonic. Install project requirements first."
        ) from exc

    scene_dir = data_dir / "MVSEC" / scene
    files = []
    for filename, _ in MVSEC.resources[scene]:
        path = scene_dir / filename
        files.append(
            {
                "path": str(path.relative_to(data_dir.parent)),
                "present": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {
        "scene": scene,
        "complete": all(file["present"] for file in files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DAVIS/MVSEC data and write a timestamp manifest."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (defaults to the repository containing this script).",
    )
    parser.add_argument(
        "--download-davis",
        action="store_true",
        help=f"Download the DAVIS {DAVIS_RECORDING} recording if it is absent.",
    )
    parser.add_argument(
        "--download-mvsec",
        action="append",
        choices=("indoor_flying", "outdoor_day", "outdoor_night", "motorcycle"),
        default=[],
        metavar="SCENE",
        help="Download an MVSEC scene (repeatable; each scene includes multiple bags).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/dataset_manifest.json"),
        help="Manifest output path, relative to --root unless absolute.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    data_dir = root / "data"
    _download_datasets(data_dir, args.download_davis, args.download_mvsec)

    bag_path = data_dir / "DAVIS" / "DAVISDATA" / f"{DAVIS_RECORDING}.bag"
    if not bag_path.is_file():
        parser.error(
            f"Required DAVIS recording is missing: {bag_path}. "
            "Place the bag there or rerun with --download-davis."
        )

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "davis": _inspect_davis(bag_path, root),
        "mvsec": {
            scene: _inspect_mvsec(data_dir, scene)
            for scene in ("indoor_flying", "outdoor_day", "outdoor_night", "motorcycle")
        },
    }

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    davis = manifest["davis"]
    print(f"DAVIS {davis['recording']}: {davis['bag']} ({davis['size_bytes']:,} bytes)")
    for name, stream in davis["streams"].items():
        print(
            f"  {name}: {stream['samples']:,} samples, "
            f"[{stream['start_us']:,}, {stream['end_us']:,}] us"
        )
    available_scenes = [
        scene for scene, details in manifest["mvsec"].items() if details["complete"]
    ]
    print(f"Complete MVSEC scenes: {', '.join(available_scenes) or 'none'}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
