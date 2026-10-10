"""Exact spatial candidate pruning for event-to-feature association."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialGridConfig:
    """Sensor and cell dimensions in pixels."""

    sensor_width: int
    sensor_height: int
    cell_width: int = 16
    cell_height: int = 16

    def __post_init__(self) -> None:
        for name in ("sensor_width", "sensor_height", "cell_width", "cell_height"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class SpatialGridIndex:
    """Persistent cell index over feature states with exact motion-aware lookup.

    The query extent includes the largest observed component-wise velocity
    multiplied by the temporal gate. This conservative bound ensures every
    feature whose predicted point can pass the spatial gate is retrieved.
    Velocity bounds only grow during incremental updates, keeping the index
    safe without rescanning all slots.
    """

    def __init__(
        self,
        active: np.ndarray,
        grid: SpatialGridConfig,
        radius_px: float,
        time_window_us: float,
    ) -> None:
        self.grid = grid
        self.radius_px = float(radius_px)
        self.time_window_us = float(time_window_us)
        self._bins: dict[tuple[int, int], list[int]] = {}
        self._cells: list[tuple[int, int]] = []
        self._max_abs_vx = 0.0
        self._max_abs_vy = 0.0
        for index, state in enumerate(active):
            self._insert(index, state)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor(float(x) / self.grid.cell_width),
            math.floor(float(y) / self.grid.cell_height),
        )

    def _insert(self, index: int, state: np.void) -> None:
        cell = self._cell(state["x"], state["y"])
        self._cells.append(cell)
        self._bins.setdefault(cell, []).append(index)
        self._max_abs_vx = max(self._max_abs_vx, abs(float(state["vx"])))
        self._max_abs_vy = max(self._max_abs_vy, abs(float(state["vy"])))

    def update(self, index: int, state: np.void) -> None:
        """Re-bin one state after an online state update."""
        old_cell = self._cells[index]
        cell_indices = self._bins[old_cell]
        cell_indices.remove(index)
        if not cell_indices:
            del self._bins[old_cell]
        new_cell = self._cell(state["x"], state["y"])
        self._cells[index] = new_cell
        self._bins.setdefault(new_cell, []).append(index)
        self._max_abs_vx = max(self._max_abs_vx, abs(float(state["vx"])))
        self._max_abs_vy = max(self._max_abs_vy, abs(float(state["vy"])))

    def _nearby_indices(self, event_x: int, event_y: int) -> list[int]:
        extent_x = self.radius_px + self._max_abs_vx * self.time_window_us
        extent_y = self.radius_px + self._max_abs_vy * self.time_window_us
        bounds = (
            event_x - extent_x,
            event_x + extent_x,
            event_y - extent_y,
            event_y + extent_y,
        )
        if not all(math.isfinite(value) for value in bounds):
            return list(range(len(self._cells)))
        min_x = math.floor(bounds[0] / self.grid.cell_width) - 1
        max_x = math.floor(bounds[1] / self.grid.cell_width) + 1
        min_y = math.floor(bounds[2] / self.grid.cell_height) - 1
        max_y = math.floor(bounds[3] / self.grid.cell_height) + 1
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if width * height <= max(1024, len(self._bins) * 4):
            indices: list[int] = []
            for cell_y in range(min_y, max_y + 1):
                for cell_x in range(min_x, max_x + 1):
                    indices.extend(self._bins.get((cell_x, cell_y), ()))
        else:
            indices = [
                index
                for (cell_x, cell_y), members in self._bins.items()
                if min_x <= cell_x <= max_x and min_y <= cell_y <= max_y
                for index in members
            ]
        indices.sort()
        return indices

    def select(
        self,
        active: np.ndarray,
        event_time: int,
        event_x: int,
        event_y: int,
        radius_squared: float,
        time_window: float,
    ) -> tuple[int, int, float | None, int, int]:
        """Return matches, best active index, and candidate/check counts."""
        candidate_indices = self._nearby_indices(event_x, event_y)
        candidate_count = 0
        distance_calculations = 0
        best_active_index = -1
        best_distance_squared: float | None = None
        best_feature_id: int | None = None
        for active_index in candidate_indices:
            state = active[active_index]
            delta_t = event_time - int(state["timestamp"])
            if abs(delta_t) >= time_window:
                continue
            distance_calculations += 1
            predicted_x = float(state["x"]) + float(state["vx"]) * delta_t
            predicted_y = float(state["y"]) + float(state["vy"]) * delta_t
            dx = event_x - predicted_x
            dy = event_y - predicted_y
            distance_squared = dx * dx + dy * dy
            if distance_squared < radius_squared:
                candidate_count += 1
                feature_id = int(state["id"])
                if (
                    best_distance_squared is None
                    or distance_squared < best_distance_squared
                    or (
                        distance_squared == best_distance_squared
                        and feature_id < best_feature_id
                    )
                ):
                    best_distance_squared = distance_squared
                    best_active_index = active_index
                    best_feature_id = feature_id
        return (
            candidate_count,
            best_active_index,
            best_distance_squared,
            len(candidate_indices),
            distance_calculations,
        )
