#!/usr/bin/env python3
"""Interactive, temporally constrained colour-marker tracker.

Compared with autocv.py this tracker does not assume a fixed number of slave
markers and does not sort every frame by nearest-neighbour alone.  It asks for
the marker count, then records a base click and ordered slave clicks.  Every
subsequent frame assigns colour candidates to the previous ordered chain with
an explicit motion gate, chain-length gate and one-to-one assignment.  A
candidate that cannot be explained by the previous frame is rejected instead
of silently changing marker order.

The setup wizard offers Auto and Manual sessions.  Both start paused at the
first frame; Auto detects during playback, while Manual detects only on an
explicit Detect command.  The analysis functions are usable without a GUI
and are kept small enough for unit tests with synthetic HSV/BGR frames.
"""

from __future__ import annotations

import argparse
import base64
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

if os.environ.get("DISPLAY"):
    os.environ["PYOPENGL_PLATFORM"] = "glx"

try:
    import glfw
    from imgui_compat import GlfwRenderer, imgui
except Exception as error:
    glfw = None  # type: ignore[assignment]
    imgui = None  # type: ignore[assignment]
    GlfwRenderer = None  # type: ignore[assignment]
    GUI_IMPORT_ERROR = error
else:
    GUI_IMPORT_ERROR = None


# ------------------------------- Settings --------------------------------

PROC_SCALE = 0.5
FRAME_SKIP = 0
ROI_X_PCT = 0.22
ROI_Y_PCT = 0.10
ROI_W_PCT = 0.45
ROI_H_PCT = 0.80
PREVIEW_MAX_WIDTH = 1200
PREVIEW_MAX_HEIGHT = 800

# Keep autocv.py's red-base/yellow-slave convention.
LOWER_RED_1 = np.array([0, 60, 60], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([170, 60, 60], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)
LOWER_YELLOW = np.array([15, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([40, 255, 255], dtype=np.uint8)

MIN_MARKER_AREA = 4.0
MAX_MARKER_AREA = 500.0
DEFAULT_MAX_MOTION_PX = 80.0
DEFAULT_GATING_SIGMA_PX = 3.0
MIN_CHAIN_LENGTH_RATIO = 0.55
MAX_CHAIN_LENGTH_RATIO = 1.80
MAX_LENGTH_CHANGE_RATIO = 0.35
MAX_ASSIGNMENT_COST = 1.0
MISSING_GRACE_FRAMES = 3
METRIC_EMA_ALPHA = 0.05
MAX_METRIC_CONDITION = 50.0


def roi_rect(width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(width * ROI_X_PCT),
        int(height * ROI_Y_PCT),
        int(width * ROI_W_PCT),
        int(height * ROI_H_PCT),
    )


def red_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(
        cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1),
        cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2),
    )


def yellow_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)


def marker_candidates(
    mask: np.ndarray,
    min_area: float = MIN_MARKER_AREA,
    max_area: Optional[float] = None,
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or (max_area is not None and area > max_area):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        points.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    return points


def nearest_marker(
    points: Iterable[tuple[float, float]],
    hint: tuple[float, float],
    max_distance: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    candidates = list(points)
    if not candidates:
        return None
    selected = min(candidates, key=lambda point: math.dist(point, hint))
    if max_distance is not None and math.dist(selected, hint) > max_distance:
        return None
    return selected


@dataclass
class TrackerConfig:
    slave_count: int
    max_motion_px: float = DEFAULT_MAX_MOTION_PX
    gating_sigma_px: float = DEFAULT_GATING_SIGMA_PX
    missing_grace_frames: int = MISSING_GRACE_FRAMES
    min_area: float = MIN_MARKER_AREA
    max_area: float = MAX_MARKER_AREA

    def __post_init__(self) -> None:
        if not 1 <= int(self.slave_count) <= 128:
            raise ValueError("slave_count must be in 1..128")
        if self.max_motion_px <= 0.0:
            raise ValueError("max_motion_px must be positive")
        if self.gating_sigma_px <= 0.0:
            raise ValueError("gating_sigma_px must be positive")


@dataclass
class TrackingResult:
    points: Optional[np.ndarray]
    accepted: bool
    reason: str
    cost: float = float("inf")
    missing_count: int = 0


def _greedy_assignment(costs: np.ndarray, max_cost: float) -> Optional[tuple[np.ndarray, float]]:
    """Minimum-cost one-to-one assignment without adding scipy."""
    rows, cols = costs.shape
    if rows == 0 or rows > cols:
        return None
    # Penalize out-of-gate edges heavily so a valid all-in-gate assignment is
    # preferred even when its total cost is not the absolute unconstrained
    # minimum.  Hungarian matching keeps contour order from swapping markers.
    work = np.where(costs <= max_cost, costs, max_cost + 1_000_000.0).astype(float)
    u = np.zeros(rows + 1, dtype=float)
    v = np.zeros(cols + 1, dtype=float)
    p = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0] = row
        column = 0
        minimum = np.full(cols + 1, np.inf, dtype=float)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column] = True
            current_row = p[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                value = work[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if value < minimum[candidate]:
                    minimum[candidate] = value
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    assignment = np.full(rows, -1, dtype=int)
    for column in range(1, cols + 1):
        if p[column] > 0:
            assignment[p[column] - 1] = column - 1
    selected = costs[np.arange(rows), assignment]
    if np.any(selected > max_cost):
        return None
    return assignment, float(np.mean(selected))


class OrderedMarkerTracker:
    """Track [base, slave1, ..., slaveN] in a fixed semantic order."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.previous: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.missing_frames = 0
        self.accepted_frames = 0

    @property
    def expected_count(self) -> int:
        return self.config.slave_count + 1

    def initialize(self, ordered_points: Sequence[Sequence[float]]) -> None:
        points = np.asarray(ordered_points, dtype=float)
        if points.shape != (self.expected_count, 2):
            raise ValueError(f"expected {self.expected_count} ordered points")
        ok, reason = validate_chain_geometry(points)
        if not ok:
            raise ValueError(f"cannot initialize tracker: {reason}")
        self.previous = points.copy()
        self.velocity = np.zeros_like(points)
        self.missing_frames = 0
        self.accepted_frames = 1

    def predict(self) -> Optional[np.ndarray]:
        if self.previous is None:
            return None
        velocity = self.velocity if self.velocity is not None else np.zeros_like(self.previous)
        return self.previous + velocity

    def update(self, candidates: Sequence[Sequence[float]]) -> TrackingResult:
        if self.previous is None:
            return TrackingResult(None, False, "tracker is not initialized")
        points = np.asarray(candidates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            return TrackingResult(None, False, "candidate points must have shape (N, 2)")
        if len(points) < self.expected_count:
            self.missing_frames += 1
            if self.missing_frames > self.config.missing_grace_frames:
                self.previous = None
                self.velocity = None
            return TrackingResult(
                None,
                False,
                f"candidate count {len(points)}/{self.expected_count} (too few)",
                missing_count=self.missing_frames,
            )

        predicted = self.predict()
        assert predicted is not None
        max_distance = self.config.max_motion_px + self.config.gating_sigma_px * max(
            1.0, float(np.median(np.linalg.norm(self.velocity, axis=1))) if self.velocity is not None else 0.0
        )
        base_distance = float(np.linalg.norm(predicted[0] - points[0]))
        if base_distance > max_distance:
            self.missing_frames += 1
            return TrackingResult(None, False, f"base motion gate rejected candidate (max {max_distance:.1f}px)", missing_count=self.missing_frames)

        # Candidate 0 is selected from the red mask and candidates 1..N from
        # the yellow mask.  Keep that colour/semantic boundary fixed: letting
        # a generic assignment swap the red base with a yellow slave would
        # recreate the ordering failure this tracker is meant to prevent.
        slave_candidates = points[1:]
        slave_distances = np.linalg.norm(
            predicted[1:, None, :] - slave_candidates[None, :, :], axis=2
        )
        slave_costs = slave_distances / max(1.0, max_distance)
        # Contour order is arbitrary.  A tiny chain-order tie-break keeps
        # equal-distance solutions in semantic order without overriding the
        # actual motion cost.
        axis = predicted[-1] - predicted[0]
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1.0 and len(predicted) > 1:
            axis = predicted[1] - predicted[0]
            axis_length = float(np.linalg.norm(axis))
        if axis_length > 1.0:
            projections = np.dot(points[1:] - predicted[0], axis / axis_length)
            candidate_ranks = np.empty(len(slave_candidates), dtype=float)
            candidate_ranks[np.argsort(projections)] = np.arange(len(slave_candidates))
            order_tie_break = np.abs(
                np.arange(self.config.slave_count, dtype=float)[:, None] - candidate_ranks[None, :]
            ) * 1.0e-6
            assignment_costs = slave_costs + order_tie_break
        else:
            assignment_costs = slave_costs
        result = _greedy_assignment(assignment_costs, MAX_ASSIGNMENT_COST)
        if result is None:
            self.missing_frames += 1
            return TrackingResult(None, False, f"slave motion gate rejected candidates (max {max_distance:.1f}px)", missing_count=self.missing_frames)
        slave_assignment, slave_cost = result
        ordered = np.empty((self.expected_count, 2), dtype=float)
        ordered[0] = points[0]
        ordered[1:] = slave_candidates[slave_assignment]
        if np.any(slave_distances[np.arange(self.config.slave_count), slave_assignment] > max_distance):
            self.missing_frames += 1
            return TrackingResult(None, False, "one or more slaves exceeded motion gate", missing_count=self.missing_frames)
        mean_cost = (base_distance / max(1.0, max_distance) + slave_cost * self.config.slave_count) / self.expected_count
        geometry_ok, geometry_reason = validate_chain_geometry(ordered)
        if not geometry_ok:
            self.missing_frames += 1
            return TrackingResult(None, False, f"geometry rejected: {geometry_reason}", mean_cost, self.missing_frames)
        if self.previous is not None:
            old_lengths = np.linalg.norm(np.diff(self.previous, axis=0), axis=1)
            new_lengths = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
            changes = np.abs(new_lengths - old_lengths) / np.maximum(old_lengths, 1.0)
            if float(np.max(changes)) > MAX_LENGTH_CHANGE_RATIO:
                self.missing_frames += 1
                return TrackingResult(None, False, f"segment length change {float(np.max(changes)):.2f} is implausible", mean_cost, self.missing_frames)

        old = self.previous.copy()
        self.previous = ordered.copy()
        self.velocity = ordered - old
        self.missing_frames = 0
        self.accepted_frames += 1
        return TrackingResult(ordered.copy(), True, "ok", mean_cost, 0)


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def cross(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
        first_vector = second - first
        second_vector = third - first
        return float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])
    first = cross(a, b, c)
    second = cross(a, b, d)
    third = cross(c, d, a)
    fourth = cross(c, d, b)
    return first * second < 0.0 and third * fourth < 0.0


def validate_chain_geometry(points: Sequence[Sequence[float]]) -> tuple[bool, str]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return False, "invalid chain shape"
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        return False, "zero length segment"
    # The base-to-P1 attachment may have a different physical length.  The
    # equal-link assumption applies to P1-P2, P2-P3, ... only.
    link_lengths = lengths[1:] if len(lengths) > 1 else lengths
    median = float(np.median(link_lengths))
    ratios = link_lengths / median
    if float(np.min(ratios)) < MIN_CHAIN_LENGTH_RATIO or float(np.max(ratios)) > MAX_CHAIN_LENGTH_RATIO:
        return False, f"segment ratio {float(np.min(ratios)):.2f}..{float(np.max(ratios)):.2f}"
    for first in range(len(points) - 1):
        for second in range(first + 2, len(points) - 1):
            if _segments_intersect(points[first], points[first + 1], points[second], points[second + 1]):
                return False, f"segments {first + 1} and {second + 1} cross"
    return True, "ok"


def estimate_equal_link_metric(points: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    """Estimate a camera metric from equal P-to-P link lengths.

    The base-P1 segment is deliberately excluded.  For a symmetric metric M,
    every moving-link vector v satisfies v.T @ M @ v = constant.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 5:
        return None
    segments = np.diff(array[1:], axis=0)
    if len(segments) < 4:
        return None
    matrix = np.column_stack(
        (
            segments[:, 0] ** 2,
            2.0 * segments[:, 0] * segments[:, 1],
            segments[:, 1] ** 2,
            -np.ones(len(segments)),
        )
    )
    try:
        _, _, vector_basis = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None
    a, b, c, common_length = vector_basis[-1]
    if a + c < 0.0:
        a, b, c, common_length = -a, -b, -c, -common_length
    if a <= 0.0 or a * c - b * b <= 1.0e-12 or common_length <= 0.0:
        return None
    metric = np.array([[a, b], [b, c]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues[0] <= 0.0 or eigenvalues[1] / eigenvalues[0] > MAX_METRIC_CONDITION:
        return None
    metric *= 2.0 / float(np.trace(metric))
    return metric


class MetricSmoother:
    """Smooth the fixed-camera metric used for perspective-corrected angles."""

    def __init__(self, alpha: float = METRIC_EMA_ALPHA) -> None:
        self.alpha = float(alpha)
        self.matrix: Optional[np.ndarray] = None

    def update(self, metric: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if metric is None:
            return self.matrix
        candidate = np.asarray(metric, dtype=float)
        if self.matrix is None:
            self.matrix = candidate.copy()
        else:
            self.matrix = (1.0 - self.alpha) * self.matrix + self.alpha * candidate
            self.matrix *= 2.0 / float(np.trace(self.matrix))
        return self.matrix


def signed_chain_angles(
    points: Sequence[Sequence[float]],
    metric: Optional[np.ndarray] = None,
) -> list[float]:
    """Return signed angles, optionally after metric/perspective correction."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        return []
    if metric is not None:
        try:
            lower = np.linalg.cholesky(np.asarray(metric, dtype=float))
            array = array @ lower
        except np.linalg.LinAlgError:
            pass
    angles: list[float] = []
    for index in range(len(array) - 2):
        first = array[index + 1] - array[index]
        second = array[index + 2] - array[index + 1]
        magnitude = float(np.linalg.norm(first) * np.linalg.norm(second))
        if magnitude <= 0.0:
            angles.append(float("nan"))
            continue
        cosine = max(-1.0, min(1.0, float(np.dot(first, second) / magnitude)))
        angle = math.degrees(math.acos(cosine))
        if first[0] * second[1] - first[1] * second[0] > 0.0:
            angle = -angle
        angles.append(round(angle, 3))
    return angles


def _find_base(points: Sequence[tuple[float, float]], hint: tuple[float, float]) -> Optional[tuple[float, float]]:
    return nearest_marker(points, hint)


def detect_candidates(
    frame: np.ndarray,
    base_hint: Optional[tuple[float, float]] = None,
    config: Optional[TrackerConfig] = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], np.ndarray]:
    """Return red and yellow candidates in full-frame coordinates."""
    cfg = config or TrackerConfig(1)
    height, width = frame.shape[:2]
    rx, ry, rw, rh = roi_rect(width, height)
    roi = frame[ry:ry + rh, rx:rx + rw]
    if roi.size == 0:
        return [], [], frame.copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    area_scale = width * height / float(3840 * 2160)
    min_area = max(1.0, cfg.min_area * area_scale)
    # Keep the lower bound resolution-aware, but do not shrink the upper
    # bound below the configured physical marker size.  Small preview videos
    # commonly contain the same-size blobs in pixels as test images.
    max_area = max(min_area + 1.0, cfg.max_area)
    # Preserve autocv.py's behaviour: with a clicked base hint, search the
    # complete frame for the red base so a base just outside the slave ROI is
    # still trackable.  Yellow slaves remain restricted to the configured ROI.
    if base_hint is not None:
        full_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = marker_candidates(red_mask(full_hsv), min_area)
    else:
        red = [(x + rx, y + ry) for x, y in marker_candidates(red_mask(hsv), min_area)]
    yellow = [(x + rx, y + ry) for x, y in marker_candidates(yellow_mask(hsv), min_area, max_area)]
    annotated = frame.copy()
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 100, 0), 2)
    cv2.putText(annotated, "ROI", (rx, max(20, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)
    if base_hint is not None:
        cv2.drawMarker(annotated, tuple(map(int, base_hint)), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    return red, yellow, annotated


def ordered_initial_points(
    red_candidates: Sequence[tuple[float, float]],
    yellow_candidates: Sequence[tuple[float, float]],
    base_hint: tuple[float, float],
    slave_hints: Sequence[tuple[float, float]],
    max_click_distance: float = 60.0,
) -> Optional[np.ndarray]:
    """Resolve clicked points to nearby same-colour blobs in the first frame."""
    base = _find_base(red_candidates, base_hint)
    if base is None or math.dist(base, base_hint) > max_click_distance:
        return None
    slaves: list[tuple[float, float]] = []
    remaining = list(yellow_candidates)
    for hint in slave_hints:
        selected = nearest_marker(remaining, hint, max_click_distance)
        if selected is None:
            return None
        slaves.append(selected)
        remaining.remove(selected)
    result = np.asarray([base, *slaves], dtype=float)
    ok, _reason = validate_chain_geometry(result)
    return result if ok else None


def draw_tracking(
    frame: np.ndarray,
    points: Optional[Sequence[Sequence[float]]],
    message: str = "",
    base_hint: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    draw = frame.copy()
    if base_hint is not None:
        cv2.drawMarker(draw, tuple(map(int, base_hint)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    if points is not None:
        array = np.asarray(points, dtype=float)
        if len(array):
            base = tuple(map(int, array[0]))
            cv2.circle(draw, base, 8, (0, 0, 255), -1)
            cv2.putText(draw, "BASE", (base[0] + 8, base[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for index, point in enumerate(array[1:], start=1):
                current = tuple(map(int, point))
                cv2.circle(draw, current, 6, (0, 255, 255), -1)
                cv2.putText(draw, f"P{index}", (current[0] + 8, current[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.line(draw, tuple(map(int, array[index - 1])), current, (0, 255, 0), 2)
    if message:
        cv2.putText(draw, message, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return draw


def process_tracking_frame(
    frame: np.ndarray,
    tracker: OrderedMarkerTracker,
    config: TrackerConfig,
    initial_points: np.ndarray,
) -> tuple[TrackingResult, np.ndarray, Optional[np.ndarray]]:
    """Run one automatic detection step and return its annotated preview."""
    base_hint = tuple(map(float, initial_points[0]))
    red, yellow, annotated = detect_candidates(frame, base_hint, config)
    predicted = tracker.predict()
    base = (
        nearest_marker(
            red,
            tuple(predicted[0]) if predicted is not None else base_hint,
            config.max_motion_px,
        )
        if red
        else None
    )
    candidates = ([base] if base is not None else []) + yellow
    result = tracker.update(candidates)
    return result, annotated, predicted


def tracking_fieldnames(config: TrackerConfig) -> list[str]:
    fields = ["frame", "video_time_s", "accepted", "reason", "assignment_cost"]
    fields += [
        field
        for index in range(1, config.slave_count + 2)
        for field in (f"p{index}_x", f"p{index}_y")
    ]
    fields += [f"angle_{index}_deg" for index in range(1, config.slave_count)]
    return fields


def tracking_row(
    frame_no: int,
    video_time_s: float,
    result: TrackingResult,
    config: TrackerConfig,
    metric: Optional[np.ndarray] = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame": frame_no,
        "video_time_s": round(float(video_time_s), 4),
        "accepted": int(result.accepted),
        "reason": result.reason,
        "assignment_cost": round(result.cost, 5) if math.isfinite(result.cost) else "",
    }
    if result.points is not None:
        for index, point in enumerate(result.points, start=1):
            row[f"p{index}_x"] = round(float(point[0]), 3)
            row[f"p{index}_y"] = round(float(point[1]), 3)
    else:
        for index in range(1, config.slave_count + 2):
            row[f"p{index}_x"] = ""
            row[f"p{index}_y"] = ""
    angles = signed_chain_angles(result.points, metric) if result.points is not None else []
    for index in range(1, config.slave_count):
        row[f"angle_{index}_deg"] = angles[index - 1] if index - 1 < len(angles) else ""
    return row


def _display_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, PREVIEW_MAX_WIDTH / max(1, width), PREVIEW_MAX_HEIGHT / max(1, height))
    if scale < 1.0:
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
    return frame, 1.0


def _tk_photo_from_bgr(tk_module: object, frame: np.ndarray) -> object:
    """Convert an OpenCV BGR frame to a Tk PhotoImage without Pillow."""
    # Tk 8.6 accepts PNG directly from an in-memory base64 payload.  Passing
    # a base64 PPM string together with ``format='PPM'`` is rejected by some
    # Tk builds with "couldn't recognize image data".  OpenCV's PNG encoder
    # already converts its BGR input to PNG's RGB channel order, so do not
    # call BGR2RGB here or the displayed colors will be swapped.
    ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("cannot encode preview frame as PNG for Tk")
    data = base64.b64encode(encoded.tobytes())
    try:
        return tk_module.PhotoImage(data=data)  # type: ignore[attr-defined]
    except Exception as error:
        raise RuntimeError(f"Tk could not decode the preview image: {error}") from error


class TkPointPicker:
    """Native Tk window used instead of OpenCV's Qt/xcb window."""

    def __init__(self, first_frame: np.ndarray, slave_count: int) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("CV setup - select markers")
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.root.bind("<Escape>", lambda _event: self.cancel())
        self.root.bind("<BackSpace>", lambda _event: self.undo())
        self.root.bind("<Return>", lambda _event: self.confirm())
        self.root.focus_force()
        self.slave_count = int(slave_count)
        self.clicks: list[tuple[float, float]] = []
        self.result: Optional[list[tuple[float, float]]] = None
        self.display, self.scale = _display_frame(first_frame)
        self.photo = _tk_photo_from_bgr(tk, self.display)

        self.instruction = tk.StringVar()
        tk.Label(self.root, textvariable=self.instruction, anchor="w").pack(fill="x", padx=10, pady=(10, 3))
        tk.Label(
            self.root,
            text="Click BASE, then SLAVE 1..N in order. Enter=confirm, Backspace=undo, Esc=cancel.",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 5))
        self.canvas = tk.Canvas(
            self.root,
            width=self.display.shape[1],
            height=self.display.shape[0],
            highlightthickness=0,
            background="black",
        )
        self.canvas.pack(padx=10, pady=5)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.image = self.photo
        self.canvas.bind("<Button-1>", self.on_click)

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=(5, 10))
        self.confirm_button = tk.Button(button_frame, text="Confirm", command=self.confirm, state="disabled")
        self.confirm_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        tk.Button(button_frame, text="Undo", command=self.undo).pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(button_frame, text="Cancel", command=self.cancel).pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.redraw()

    def redraw(self) -> None:
        if not self.clicks:
            self.instruction.set("Click BASE (red)")
        elif len(self.clicks) <= self.slave_count:
            self.instruction.set(f"Click SLAVE {len(self.clicks)} (yellow)")
        else:
            self.instruction.set("All markers selected. Press Enter or Confirm.")
        self.canvas.delete("marker")
        for index, point in enumerate(self.clicks):
            x = int(round(point[0] * self.scale))
            y = int(round(point[1] * self.scale))
            color = "#ff3030" if index == 0 else "#ffe000"
            size = 10
            self.canvas.create_line(x - size, y, x + size, y, fill=color, width=2, tags="marker")
            self.canvas.create_line(x, y - size, x, y + size, fill=color, width=2, tags="marker")
            label = "BASE" if index == 0 else f"P{index}"
            self.canvas.create_text(x + 12, y - 12, text=label, fill=color, anchor="sw", tags="marker")
        self.confirm_button.configure(state="normal" if len(self.clicks) == self.slave_count + 1 else "disabled")

    def on_click(self, event: object) -> None:
        if len(self.clicks) >= self.slave_count + 1:
            return
        x = max(0.0, min(float(self.display.shape[1] - 1) / self.scale, float(event.x) / self.scale))  # type: ignore[attr-defined]
        y = max(0.0, min(float(self.display.shape[0] - 1) / self.scale, float(event.y) / self.scale))  # type: ignore[attr-defined]
        self.clicks.append((x, y))
        self.redraw()

    def undo(self) -> None:
        if self.clicks:
            self.clicks.pop()
            self.redraw()

    def confirm(self) -> None:
        if len(self.clicks) == self.slave_count + 1:
            self.result = list(self.clicks)
            self.root.destroy()

    def cancel(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> Optional[list[tuple[float, float]]]:
        self.root.mainloop()
        return self.result


def _imgui_xy(value: object) -> tuple[float, float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)  # type: ignore[attr-defined]
    return float(value[0]), float(value[1])  # type: ignore[index]


class GlfwPointPicker:
    """GLFW/ImGui fallback for marker selection when Tk cannot open."""

    def __init__(self, first_frame: np.ndarray, slave_count: int) -> None:
        if glfw is None or imgui is None or GlfwRenderer is None:
            raise RuntimeError("GLFW/ImGui is not installed")
        from OpenGL import GL as gl

        self.gl = gl
        self.display, self.scale = _display_frame(first_frame)
        self.height, self.width = self.display.shape[:2]
        self.slave_count = int(slave_count)
        self.clicks: list[tuple[float, float]] = []
        self.result: Optional[list[tuple[float, float]]] = None
        self.done = False
        self.window = None
        self.renderer = None
        self.texture = 0

    def _create_texture(self) -> int:
        rgb = np.ascontiguousarray(cv2.cvtColor(self.display, cv2.COLOR_BGR2RGB))
        texture = int(self.gl.glGenTextures(1))
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, texture)
        self.gl.glPixelStorei(self.gl.GL_UNPACK_ALIGNMENT, 1)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            self.gl.GL_RGB,
            self.width,
            self.height,
            0,
            self.gl.GL_RGB,
            self.gl.GL_UNSIGNED_BYTE,
            rgb,
        )
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)
        return texture

    def _draw(self) -> None:
        assert imgui is not None
        flags = getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
        opened = imgui.begin("CV setup - select markers###cv-glfw-point-picker", True, flags=flags)
        if _imgui_open(opened):
            if not self.clicks:
                instruction = "Click BASE (red)"
            elif len(self.clicks) <= self.slave_count:
                instruction = f"Click SLAVE {len(self.clicks)} (yellow)"
            else:
                instruction = "All markers selected. Confirm or press Enter."
            imgui.text(instruction)
            imgui.text_disabled("Click BASE, then SLAVE 1..N in order. Backspace=undo, Esc=cancel.")
            imgui.separator()
            imgui.image(self.texture, float(self.width), float(self.height))

            if imgui.is_item_hovered() and imgui.is_mouse_clicked(0) and len(self.clicks) < self.slave_count + 1:
                image_x, image_y = _imgui_xy(imgui.get_item_rect_min())
                mouse_x, mouse_y = _imgui_xy(imgui.get_mouse_pos())
                x = max(0.0, min(float(self.width - 1), mouse_x - image_x)) / self.scale
                y = max(0.0, min(float(self.height - 1), mouse_y - image_y)) / self.scale
                self.clicks.append((x, y))

            draw_list = imgui.get_window_draw_list()
            image_x, image_y = _imgui_xy(imgui.get_item_rect_min())
            for index, (x, y) in enumerate(self.clicks):
                screen_x = image_x + x * self.scale
                screen_y = image_y + y * self.scale
                color = imgui.get_color_u32_rgba(1.0, 0.15, 0.15, 1.0) if index == 0 else imgui.get_color_u32_rgba(1.0, 0.90, 0.0, 1.0)
                draw_list.add_line(screen_x - 10, screen_y, screen_x + 10, screen_y, color, 2.0)
                draw_list.add_line(screen_x, screen_y - 10, screen_x, screen_y + 10, color, 2.0)
                draw_list.add_text(screen_x + 12, screen_y - 12, color, "BASE" if index == 0 else f"P{index}")

            if imgui.button("Confirm##glfw-point-confirm") and len(self.clicks) == self.slave_count + 1:
                self.result = list(self.clicks)
                self.done = True
            imgui.same_line()
            if imgui.button("Undo##glfw-point-undo") and self.clicks:
                self.clicks.pop()
            imgui.same_line()
            if imgui.button("Cancel##glfw-point-cancel"):
                self.done = True
        imgui.end()

    def run(self) -> Optional[list[tuple[float, float]]]:
        assert glfw is not None and imgui is not None and GlfwRenderer is not None
        if os.environ.get("DISPLAY") and hasattr(glfw, "init_hint") and hasattr(glfw, "PLATFORM_X11"):
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        try:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            if hasattr(glfw, "CONTEXT_CREATION_API") and hasattr(glfw, "NATIVE_CONTEXT_API"):
                glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            window_width = min(1400, max(900, self.width + 32))
            window_height = min(1000, max(650, self.height + 150))
            self.window = glfw.create_window(window_width, window_height, "CV setup - select markers", None, None)
            if not self.window:
                raise RuntimeError("glfw.create_window() failed")
            glfw.make_context_current(self.window)
            imgui.create_context()
            _install_imgui_theme()
            self.renderer = GlfwRenderer(self.window)
            self.texture = self._create_texture()
            while not glfw.window_should_close(self.window) and not self.done:
                glfw.poll_events()
                self.renderer.process_inputs()
                imgui.new_frame()
                self._draw()
                escape_key = getattr(imgui, "KEY_ESCAPE", None)
                backspace_key = getattr(imgui, "KEY_BACKSPACE", None)
                enter_key = getattr(imgui, "KEY_ENTER", None)
                if escape_key is not None and imgui.is_key_pressed(escape_key):
                    self.done = True
                elif backspace_key is not None and imgui.is_key_pressed(backspace_key) and self.clicks:
                    self.clicks.pop()
                elif enter_key is not None and imgui.is_key_pressed(enter_key) and len(self.clicks) == self.slave_count + 1:
                    self.result = list(self.clicks)
                    self.done = True
                imgui.render()
                self.renderer.render(imgui.get_draw_data())
                glfw.swap_buffers(self.window)
            return self.result
        finally:
            if self.texture:
                try:
                    self.gl.glDeleteTextures([self.texture])
                except Exception:
                    pass
            if self.renderer is not None:
                self.renderer.shutdown()
            if self.window is not None:
                glfw.destroy_window(self.window)
            glfw.terminate()


class TkVideoPreview:
    """Native Tk video window that avoids OpenCV Qt/xcb preview failures."""

    def __init__(self, title: str) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self._set_key("escape"))
        self.root.bind("<KeyPress-q>", lambda _event: self._set_key("escape"))
        self.root.bind("<KeyPress-Q>", lambda _event: self._set_key("escape"))
        self.root.focus_force()
        self.label = tk.Label(self.root, background="black")
        self.label.pack(fill="both", expand=True)
        self.photo = None
        self.closed = False
        self.pending_key: Optional[str] = None
        self.root.update_idletasks()

    def _set_key(self, key: str) -> None:
        self.pending_key = key

    def show(self, frame: np.ndarray) -> Optional[str]:
        if self.closed:
            return "escape"
        shown, _ = _display_frame(frame)
        self.photo = _tk_photo_from_bgr(self.tk, shown)
        self.label.configure(image=self.photo)
        self.label.image = self.photo
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True
            return "escape"
        key = self.pending_key
        self.pending_key = None
        return key

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass


class TkTrackingSession:
    """Frame-controlled tracking session for automatic and manual modes."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        first_frame: np.ndarray,
        initial_points: np.ndarray,
        config: TrackerConfig,
        mode: str,
    ) -> None:
        import tkinter as tk

        if mode not in {"auto", "manual"}:
            raise ValueError("mode must be 'auto' or 'manual'")
        self.tk = tk
        self.cap = cap
        self.current_frame = first_frame
        self.current_frame_no = 1
        if mode == "manual":
            # Manual mode may jump over many frames between Detect presses.
            # Keep colour/order/geometry validation, but do not reject a valid
            # marker solely because it moved farther than the auto-step gate.
            manual_motion_limit = math.hypot(first_frame.shape[1], first_frame.shape[0])
            self.config = TrackerConfig(
                config.slave_count,
                max_motion_px=max(config.max_motion_px, manual_motion_limit),
                gating_sigma_px=config.gating_sigma_px,
                missing_grace_frames=config.missing_grace_frames,
                min_area=config.min_area,
                max_area=config.max_area,
            )
        else:
            self.config = config
        self.initial_points = initial_points
        self.tracker = OrderedMarkerTracker(self.config)
        self.tracker.initialize(initial_points)
        self.metric = MetricSmoother()
        self.metric.update(estimate_equal_link_metric(initial_points))
        self.mode = mode
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            self.fps = 30.0
        self.frame_delay_ms = max(10, min(200, int(round(1000.0 / self.fps))))
        self.rows: list[dict[str, object]] = []
        self.playing = False
        self.finished = False
        self._after_id: Optional[object] = None
        self._last_annotated = None

        self.root = tk.Tk()
        self.root.title(f"CV tracking - {'AUTO' if mode == 'auto' else 'MANUAL'}")
        self.root.protocol("WM_DELETE_WINDOW", self.finish)
        self.root.bind("<Escape>", lambda _event: self.finish())
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<KeyPress-d>", lambda _event: self.detect_current())
        self.root.focus_force()

        self.info = tk.StringVar()
        tk.Label(self.root, textvariable=self.info, anchor="w").pack(
            fill="x", padx=10, pady=(10, 3)
        )
        self.video_label = tk.Label(self.root, background="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=5)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(5, 10))
        self.play_button = tk.Button(controls, text="Play", command=self.play)
        self.play_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.pause_button = tk.Button(controls, text="Pause", command=self.pause, state="disabled")
        self.pause_button.pack(side="left", expand=True, fill="x", padx=4)
        self.detect_button = tk.Button(
            controls,
            text="Detect",
            command=self.detect_current,
            state="normal" if mode == "manual" else "disabled",
        )
        self.detect_button.pack(side="left", expand=True, fill="x", padx=4)
        tk.Button(controls, text="Finish / Save", command=self.finish).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        hint = (
            "AUTO: Play/Pause controls automatic detection."
            if mode == "auto"
            else "MANUAL: Pause at a frame, then press Detect (D)."
        )
        tk.Label(self.root, text=hint, anchor="w").pack(fill="x", padx=10, pady=(0, 8))
        self._render(self.tracker.previous, "Paused at video start — press Play")

    def _cancel_scheduled(self) -> None:
        if self._after_id is None:
            return
        try:
            self.root.after_cancel(self._after_id)
        except self.tk.TclError:
            pass
        self._after_id = None

    def _set_controls(self) -> None:
        if self.finished:
            return
        self.play_button.configure(state="disabled" if self.playing else "normal")
        self.pause_button.configure(state="normal" if self.playing else "disabled")
        self.detect_button.configure(
            state="normal" if self.mode == "manual" and not self.playing else "disabled"
        )

    def _render(
        self,
        points: Optional[Sequence[Sequence[float]]],
        message: str,
        annotated: Optional[np.ndarray] = None,
    ) -> None:
        if annotated is None:
            _, _, annotated = detect_candidates(
                self.current_frame,
                tuple(map(float, self.initial_points[0])),
                self.config,
            )
        drawn = draw_tracking(
            annotated,
            points,
            message,
            tuple(map(float, self.initial_points[0])),
        )
        shown, _ = _display_frame(drawn)
        self.photo = _tk_photo_from_bgr(self.tk, shown)
        self.video_label.configure(image=self.photo)
        self.video_label.image = self.photo
        self.info.set(
            f"{self.mode.upper()}  |  frame {self.current_frame_no}  |  "
            f"{max(0.0, (self.current_frame_no - 1) / self.fps):.2f}s  |  {message}"
        )
        try:
            self.root.update_idletasks()
        except self.tk.TclError:
            self.finished = True

    def _schedule_next(self, delay: int = 0) -> None:
        self._cancel_scheduled()
        if self.playing and not self.finished:
            self._after_id = self.root.after(delay, self._advance)

    def play(self) -> None:
        if self.finished or self.playing:
            return
        self.playing = True
        self._set_controls()
        self._schedule_next(0)

    def pause(self) -> None:
        if self.finished:
            return
        self.playing = False
        self._cancel_scheduled()
        self._set_controls()
        self._render(self.tracker.previous, "Paused — press Detect" if self.mode == "manual" else "Paused")

    def toggle_play(self) -> None:
        self.pause() if self.playing else self.play()

    def _advance(self) -> None:
        self._after_id = None
        if not self.playing or self.finished:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self._set_controls()
            self._render(self.tracker.previous, "End of video")
            return
        if PROC_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=PROC_SCALE, fy=PROC_SCALE, interpolation=cv2.INTER_AREA)
        self.current_frame = frame
        self.current_frame_no += 1
        if self.mode == "auto":
            self.detect_current(auto_step=True)
        else:
            self._render(self.tracker.previous, "Playing — pause to detect")
        self._schedule_next(self.frame_delay_ms)

    def detect_current(self, auto_step: bool = False) -> None:
        if self.finished or (self.playing and not auto_step):
            return
        try:
            result, annotated, predicted = process_tracking_frame(
                self.current_frame, self.tracker, self.config, self.initial_points
            )
            if result.points is not None:
                self.metric.update(estimate_equal_link_metric(result.points))
            row = tracking_row(
                self.current_frame_no,
                max(0.0, (self.current_frame_no - 1) / self.fps),
                result,
                self.config,
                self.metric.matrix,
            )
            if self.mode == "manual" and self.rows and self.rows[-1]["frame"] == self.current_frame_no:
                self.rows[-1] = row
            else:
                self.rows.append(row)
            points = result.points if result.points is not None else predicted
            self._last_annotated = annotated
            self._render(points, result.reason, annotated)
            self._set_controls()
        except Exception as error:
            self._render(self.tracker.previous, f"Detection error: {error}")

    def finish(self) -> None:
        if self.finished:
            return
        self.playing = False
        self._cancel_scheduled()
        self.finished = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def run(self) -> list[dict[str, object]]:
        self.root.mainloop()
        return self.rows


def select_points_interactively(
    first_frame: np.ndarray,
    config: TrackerConfig,
) -> Optional[np.ndarray]:
    """Ask for base then slave clicks in a native desktop window."""
    errors: list[str] = []
    for picker_type in (TkPointPicker, GlfwPointPicker):
        picker = None
        try:
            picker = picker_type(first_frame, config.slave_count)
            clicks = picker.run()
            if clicks is None:
                return None
            break
        except Exception as error:
            if picker is not None and hasattr(picker, "root"):
                try:
                    picker.root.destroy()  # type: ignore[attr-defined]
                except Exception:
                    pass
            errors.append(f"{picker_type.__name__}: {type(error).__name__}: {error}")
    else:
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "unset"
        raise RuntimeError(
            "cannot open the marker selection window "
            f"(DISPLAY/WAYLAND={display}); "
            + " | ".join(errors)
        ) from None
    if clicks is None or len(clicks) != config.slave_count + 1:
        return None
    red, yellow, _ = detect_candidates(first_frame, clicks[0], config)
    return ordered_initial_points(red, yellow, clicks[0], clicks[1:])


def ask_slave_count(value: Optional[int] = None) -> int:
    if value is not None:
        return int(TrackerConfig(int(value)).slave_count)
    while True:
        raw = input("Slave point count (1-128): ").strip()
        try:
            count = int(raw)
            TrackerConfig(count)
            return count
        except (TypeError, ValueError) as error:
            print(f"Invalid count: {error}")


def choose_path_dialog(
    *,
    save: bool,
    title: str,
    initial: str = "",
    allow_terminal: bool = True,
) -> Optional[str]:
    """Open a native picker, with a terminal fallback for headless use.

    zenity, kdialog, and Tk's platform file dialog cover Linux/WSL/Windows
    desktops.  Only a genuinely headless machine reaches the terminal prompt.
    """
    available, selected = _native_path_dialog(save=save, title=title, initial=initial)
    if available:
        return selected
    if not allow_terminal:
        return None

    prompt = "CSV output path" if save else "Video path"
    try:
        value = input(f"{prompt} (GUI picker unavailable): ").strip()
    except EOFError:
        return None
    return value or None


def _native_path_dialog(
    *,
    save: bool,
    title: str,
    initial: str = "",
) -> tuple[bool, Optional[str]]:
    """Return ``(available, selection)`` without ever prompting in a terminal."""
    if shutil.which("zenity"):
        command = ["zenity", "--file-selection", f"--title={title}"]
        if initial:
            command.append(f"--filename={initial}")
        if save:
            command.extend(["--save", "--confirm-overwrite"])
        else:
            command.extend([
                "--file-filter=Video files | *.mp4 *.avi *.mov *.mkv *.webm",
                "--file-filter=All files | *",
            ])
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        selected = result.stdout.strip()
        return True, selected or None

    if shutil.which("kdialog"):
        command = [
            "kdialog",
            "--getsavefilename" if save else "--getopenfilename",
            initial or ".",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm)" if not save else "CSV files (*.csv)",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        selected = result.stdout.strip()
        return True, selected or None

    # Tk's filedialog is a real desktop file chooser on Windows and the
    # normal GTK/Windows-backed chooser available to Python on Linux/WSLg.
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        root.update_idletasks()

        initial_path = Path(initial).expanduser() if initial else Path.cwd()
        if initial_path.is_dir():
            initial_dir = str(initial_path)
            initial_file = ""
        else:
            initial_dir = str(initial_path.parent if initial_path.parent.is_dir() else Path.cwd())
            initial_file = initial_path.name

        if save:
            selected = filedialog.asksaveasfilename(
                parent=root,
                title=title,
                initialdir=initial_dir,
                initialfile=initial_file,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
        else:
            selected = filedialog.askopenfilename(
                parent=root,
                title=title,
                initialdir=initial_dir,
                initialfile=initial_file,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"),
                    ("All files", "*.*"),
                ],
            )
        return True, selected or None
    except Exception:
        return False, None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def _set_imgui_color(style: object, name: str, rgba: tuple[float, float, float, float]) -> None:
    if imgui is None:
        return
    color_id = getattr(imgui, name, None)
    if color_id is None:
        return
    try:
        style.colors[color_id] = rgba
    except Exception:
        pass


def _install_imgui_theme() -> None:
    """Use the same light Elesim palette for the setup wizard."""
    if imgui is None:
        return
    style = imgui.get_style()
    for attr, value in (
        ("window_rounding", 6.0),
        ("child_rounding", 5.0),
        ("frame_rounding", 4.0),
        ("grab_rounding", 4.0),
        ("popup_rounding", 5.0),
        ("scrollbar_rounding", 6.0),
        ("tab_rounding", 5.0),
        ("window_border_size", 1.0),
        ("child_border_size", 1.0),
        ("frame_border_size", 1.0),
    ):
        if hasattr(style, attr):
            setattr(style, attr, value)
    for attr, value in (
        ("item_spacing", (8.0, 10.5)),
        ("frame_padding", (8.0, 6.0)),
        ("window_padding", (11.0, 13.5)),
        ("cell_padding", (7.0, 6.0)),
    ):
        if hasattr(style, attr):
            setattr(style, attr, value)

    colors = {
        "COLOR_TEXT": (0.10, 0.11, 0.13, 1.00),
        "COLOR_TEXT_DISABLED": (0.48, 0.50, 0.54, 1.00),
        "COLOR_WINDOW_BACKGROUND": (0.94, 0.95, 0.96, 1.00),
        "COLOR_CHILD_BACKGROUND": (0.985, 0.985, 0.99, 1.00),
        "COLOR_POPUP_BACKGROUND": (1.00, 1.00, 1.00, 0.98),
        "COLOR_BORDER": (0.74, 0.76, 0.80, 1.00),
        "COLOR_BORDER_SHADOW": (1.00, 1.00, 1.00, 0.00),
        "COLOR_FRAME_BACKGROUND": (1.00, 1.00, 1.00, 1.00),
        "COLOR_FRAME_BACKGROUND_HOVERED": (0.91, 0.95, 1.00, 1.00),
        "COLOR_FRAME_BACKGROUND_ACTIVE": (0.84, 0.90, 1.00, 1.00),
        "COLOR_TITLE_BACKGROUND": (0.88, 0.89, 0.91, 1.00),
        "COLOR_TITLE_BACKGROUND_ACTIVE": (0.82, 0.86, 0.92, 1.00),
        "COLOR_TITLE_BACKGROUND_COLLAPSED": (0.90, 0.91, 0.93, 1.00),
        "COLOR_MENU_BAR_BACKGROUND": (0.91, 0.92, 0.94, 1.00),
        "COLOR_SCROLLBAR_BACKGROUND": (0.93, 0.94, 0.95, 1.00),
        "COLOR_SCROLLBAR_GRAB": (0.70, 0.72, 0.76, 1.00),
        "COLOR_SCROLLBAR_GRAB_HOVERED": (0.62, 0.65, 0.70, 1.00),
        "COLOR_SCROLLBAR_GRAB_ACTIVE": (0.52, 0.56, 0.62, 1.00),
        "COLOR_CHECK_MARK": (0.00, 0.45, 0.95, 1.00),
        "COLOR_SLIDER_GRAB": (0.00, 0.48, 1.00, 1.00),
        "COLOR_SLIDER_GRAB_ACTIVE": (0.00, 0.36, 0.86, 1.00),
        "COLOR_BUTTON": (0.90, 0.91, 0.93, 1.00),
        "COLOR_BUTTON_HOVERED": (0.82, 0.89, 0.98, 1.00),
        "COLOR_BUTTON_ACTIVE": (0.70, 0.82, 0.98, 1.00),
        "COLOR_HEADER": (0.86, 0.88, 0.91, 1.00),
        "COLOR_HEADER_HOVERED": (0.78, 0.86, 0.98, 1.00),
        "COLOR_HEADER_ACTIVE": (0.66, 0.78, 0.96, 1.00),
        "COLOR_SEPARATOR": (0.78, 0.80, 0.84, 1.00),
        "COLOR_SEPARATOR_HOVERED": (0.52, 0.66, 0.86, 1.00),
        "COLOR_SEPARATOR_ACTIVE": (0.34, 0.54, 0.82, 1.00),
        "COLOR_TAB": (0.86, 0.88, 0.91, 1.00),
        "COLOR_TAB_HOVERED": (0.75, 0.84, 0.98, 1.00),
        "COLOR_TAB_ACTIVE": (0.94, 0.97, 1.00, 1.00),
        "COLOR_NAV_HIGHLIGHT": (0.00, 0.45, 0.95, 0.80),
    }
    for name, rgba in colors.items():
        _set_imgui_color(style, name, rgba)

    io = imgui.get_io()
    fonts = getattr(io, "fonts", None)
    if fonts is not None and hasattr(fonts, "add_font_from_file_ttf"):
        for candidate in (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf"),
            Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ):
            if not candidate.exists():
                continue
            try:
                ranges = fonts.get_glyph_ranges_korean() if hasattr(fonts, "get_glyph_ranges_korean") else None
                font = fonts.add_font_from_file_ttf(str(candidate), 18.0, glyph_ranges=ranges)
                if hasattr(io, "font_default") and font is not None:
                    io.font_default = font
                break
            except Exception:
                continue


def _imgui_open(result: object) -> bool:
    if hasattr(result, "opened"):
        return bool(result.opened)
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


@dataclass
class WizardSelection:
    video_path: str
    csv_path: str
    slave_count: int
    mode: str = "auto"


_TK_WIZARD_STARTED = False


class CvInputWizard:
    """A small GLFW/ImGui file browser used by ``python cv.py``.

    The directory list is the Browse view, and the final page-like controls
    also collect the CSV destination, slave count, and session mode before
    tracking starts.  A native Tk wizard is used when GLFW cannot start.
    """

    def __init__(self, initial_dir: Optional[Path] = None, slave_count: int = 6) -> None:
        base_dir = (initial_dir or Path.cwd()).expanduser().resolve()
        # When launched as ``cd new && python3 cv.py``, start in the project
        # directory so the existing 400_*.mp4 files are visible immediately.
        if initial_dir is None and base_dir.name == "new" and base_dir.parent.is_dir():
            base_dir = base_dir.parent
        self.current_dir = base_dir
        if not self.current_dir.is_dir():
            self.current_dir = Path.cwd().resolve()
        self.video_path = ""
        self.csv_path = ""
        self.slave_count = int(slave_count)
        self.mode = "auto"
        self.error = ""
        self.browser_notice = ""
        self.browser_target = "video"
        self.browser_open = False
        self.finished = False
        self.cancelled = False
        self._entries: list[Path] = []

    def refresh(self) -> None:
        try:
            entries = list(self.current_dir.iterdir())
        except OSError as error:
            self._entries = []
            self.error = f"Cannot read directory: {error}"
            return
        directories = sorted((entry for entry in entries if entry.is_dir()), key=lambda item: item.name.lower())
        suffixes = VIDEO_SUFFIXES if self.browser_target == "video" else {".csv"}
        files = sorted(
            (entry for entry in entries if entry.is_file() and entry.suffix.lower() in suffixes),
            key=lambda item: item.name.lower(),
        )
        self._entries = [*directories, *files]

    def choose_video(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        self.video_path = str(resolved)
        self.csv_path = str(resolved.with_name(resolved.stem + "_tracking.csv"))
        self.error = ""
        self.browser_notice = ""
        self.browser_target = "video"

    def choose_csv(self, path_text: str) -> None:
        if not path_text.strip():
            return
        path = Path(path_text).expanduser()
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        self.csv_path = str(path.resolve())
        self.error = ""
        self.browser_notice = ""

    def _set_browser_directory_from_path(self, path_text: str) -> None:
        if not path_text.strip():
            return
        candidate = Path(path_text).expanduser()
        directory = candidate.parent if candidate.suffix else candidate
        if directory.is_dir():
            self.current_dir = directory.resolve()

    def browse_video(self) -> None:
        available, selected = _native_path_dialog(
            save=False,
            title="Select input video",
            initial=self.video_path or str(self.current_dir),
        )
        if available:
            if selected:
                self.choose_video(Path(selected))
            return

        # Last-resort fallback for a desktop without zenity, kdialog, or Tk.
        self.browser_target = "video"
        self._set_browser_directory_from_path(self.video_path)
        self.browser_notice = ""
        self.error = ""
        self.browser_open = True

    def browse_csv(self) -> None:
        available, selected = _native_path_dialog(
            save=True,
            title="Select tracking CSV output",
            initial=self.csv_path or str(self.current_dir / "tracking.csv"),
        )
        if available:
            if selected:
                self.choose_csv(selected)
            return

        # Last-resort fallback for a desktop without zenity, kdialog, or Tk.
        self.browser_target = "csv"
        self._set_browser_directory_from_path(self.csv_path)
        self.browser_notice = ""
        self.error = ""
        self.browser_open = True

    def _draw_file_browser_window(self) -> None:
        if imgui is None or not self.browser_open:
            return
        target_label = "input video" if self.browser_target == "video" else "output CSV"
        title = f"File browser - {target_label}###cv-file-browser-window"
        once = getattr(imgui, "ONCE", 0)
        io = imgui.get_io()
        imgui.set_next_window_position(
            max(20.0, float(io.display_size.x) * 0.14),
            max(20.0, float(io.display_size.y) * 0.12),
            once,
        )
        imgui.set_next_window_size(760.0, 560.0, once)
        opened = imgui.begin(
            title,
            self.browser_open,
            flags=getattr(imgui, "WINDOW_NO_COLLAPSE", 0),
        )
        if _imgui_open(opened):
            imgui.text_colored(f"BROWSE {target_label.upper()}", 0.05, 0.32, 0.72, 1.0)
            imgui.text_wrapped(f"Current folder: {self.current_dir}")
            if imgui.button("Up##file-browser-up", 100.0, 0.0) and self.current_dir.parent != self.current_dir:
                self.current_dir = self.current_dir.parent
                self.refresh()
            imgui.same_line()
            if imgui.button("Refresh##file-browser-refresh", 100.0, 0.0):
                self.refresh()

            if self.browser_target == "csv" and self.video_path:
                imgui.same_line()
                if imgui.button("Use automatic path##file-browser-auto-csv", 180.0, 0.0):
                    video = Path(self.video_path)
                    self.choose_csv(str(video.with_name(video.stem + "_tracking.csv")))
                    self.browser_open = False

            imgui.separator()
            list_result = imgui.begin_child("cv-file-browser-list-window", 0.0, -52.0, True)
            if _imgui_open(list_result):
                if not self._entries:
                    if self.browser_target == "video":
                        imgui.text_disabled("No video files or folders found here.")
                    else:
                        imgui.text_disabled("No CSV files found here.")
                for entry in self._entries[:300]:
                    if entry.is_dir():
                        if imgui.button(f"[DIR]    {entry.name}##file-browser-dir-{entry}"):
                            self.current_dir = entry
                            self.refresh()
                            break
                    elif self.browser_target == "video":
                        if imgui.button(f"[VIDEO]  {entry.name}##file-browser-video-{entry}"):
                            self.choose_video(entry)
                            self.browser_open = False
                            break
                    elif imgui.button(f"[CSV]    {entry.name}##file-browser-csv-{entry}"):
                        self.choose_csv(str(entry))
                        self.browser_open = False
                        break
            imgui.end_child()
            if imgui.button("Cancel##file-browser-cancel", 100.0, 0.0):
                self.browser_open = False
        imgui.end()
        if isinstance(opened, tuple) and not opened[0]:
            self.browser_open = False

    def draw(self) -> None:
        self.refresh()
        io = imgui.get_io()
        always = getattr(imgui, "ALWAYS", 0)
        imgui.set_next_window_position(0.0, 0.0, always)
        imgui.set_next_window_size(float(io.display_size.x), float(io.display_size.y), always)
        flags = (
            getattr(imgui, "WINDOW_NO_TITLE_BAR", 0)
            | getattr(imgui, "WINDOW_NO_MOVE", 0)
            | getattr(imgui, "WINDOW_NO_RESIZE", 0)
            | getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
        )
        opened = imgui.begin("CV SETUP WIZARD###cv-wizard", True, flags=flags)
        visible = opened[0] if isinstance(opened, tuple) else bool(opened)
        if visible:
            imgui.text_colored("CV TRACKER / SETUP", 0.05, 0.32, 0.72, 1.0)
            imgui.same_line()
            imgui.text_disabled("Ordered color markers with temporal gating")
            imgui.separator()

            available_width = max(1.0, float(imgui.get_content_region_available_width()))
            controls_width = available_width
            controls_open = imgui.begin_child("wizard-controls", controls_width, 0.0, True)
            if controls_open[0] if isinstance(controls_open, tuple) else controls_open:
                imgui.text_colored("01  INPUT VIDEO", 0.05, 0.32, 0.72, 1.0)
                imgui.text_disabled("Select the input video with Browse.")
                if imgui.button("Browse video...##wizard-browse-video", 180.0, 0.0):
                    self.browse_video()
                if self.video_path:
                    imgui.text_wrapped(str(Path(self.video_path).expanduser()))
                else:
                    imgui.text_disabled("No input video selected")

                imgui.separator()
                imgui.text_colored("02  OUTPUT & MARKERS", 0.05, 0.32, 0.72, 1.0)
                imgui.text_disabled("Choose an output destination with Browse.")
                if imgui.button("Browse output CSV...##wizard-browse-csv", 180.0, 0.0):
                    self.browse_csv()
                if self.csv_path:
                    imgui.text_wrapped(str(Path(self.csv_path).expanduser()))
                else:
                    imgui.text_disabled("Output path will be generated beside the video")
                changed, count = imgui.input_int("Slave count##wizard-slaves", self.slave_count)
                if changed:
                    self.slave_count = int(count)
                imgui.text_disabled("Allowed marker count: 1..128")

                imgui.separator()
                imgui.text_colored("03  SESSION MODE", 0.05, 0.32, 0.72, 1.0)
                mode_index = 0 if self.mode == "auto" else 1
                changed, mode_index = imgui.combo("Mode##wizard-mode", mode_index, ["Auto", "Manual"])
                if changed:
                    self.mode = "auto" if int(mode_index) == 0 else "manual"
                if self.mode == "auto":
                    imgui.text_disabled("Play/Pause controls continuous automatic detection.")
                else:
                    imgui.text_disabled("Pause at a frame, then press Detect to process that frame.")

                valid_video = bool(self.video_path) and Path(self.video_path).expanduser().is_file()
                valid_count = 1 <= int(self.slave_count) <= 128
                if self.error:
                    imgui.text_colored(self.error, 0.82, 0.12, 0.10, 1.0)
                elif self.browser_notice:
                    imgui.text_disabled(self.browser_notice)
                elif valid_video and valid_count:
                    imgui.text_colored("READY  Input and marker count are valid", 0.10, 0.48, 0.22, 1.0)
                else:
                    imgui.text_disabled("Select a valid video and marker count to start.")

                imgui.separator()
                if imgui.button("Start tracking##wizard-start"):
                    if not valid_video:
                        self.error = "Select a valid input video first"
                    elif not valid_count:
                        self.error = "Slave count must be 1..128"
                    else:
                        self.choose_csv(
                            self.csv_path
                            or str(Path(self.video_path).with_name(Path(self.video_path).stem + "_tracking.csv"))
                        )
                        self.finished = True
                imgui.same_line()
                if imgui.button("Cancel##wizard-cancel"):
                    self.cancelled = True
            imgui.end_child()
        imgui.end()
        self._draw_file_browser_window()


def run_tk_input_wizard(default_slaves: Optional[int] = None) -> Optional[WizardSelection]:
    """Run the native Tk setup wizard."""
    global _TK_WIZARD_STARTED
    _TK_WIZARD_STARTED = False
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
    except Exception as error:
        print(f"Tkinter setup wizard unavailable: {error}", file=sys.stderr)
        return None
    _TK_WIZARD_STARTED = True

    root.title("CV setup wizard")
    root.geometry("620x300")
    video_var = tk.StringVar()
    csv_var = tk.StringVar()
    count_var = tk.StringVar(value=str(default_slaves or 6))
    mode_var = tk.StringVar(value="auto")
    result: list[Optional[WizardSelection]] = [None]

    def browse_video() -> None:
        selected = filedialog.askopenfilename(
            parent=root,
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"), ("All files", "*.*")],
        )
        if selected:
            video_var.set(selected)
            if not csv_var.get().strip():
                path = Path(selected)
                csv_var.set(str(path.with_name(path.stem + "_tracking.csv")))

    def browse_csv() -> None:
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Select tracking CSV output",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            csv_var.set(selected)

    def start() -> None:
        video = Path(video_var.get()).expanduser()
        if not video.is_file():
            error_var.set("Select a valid video file.")
            return
        try:
            count = int(count_var.get())
            TrackerConfig(count)
        except (TypeError, ValueError) as error:
            error_var.set(f"Invalid slave count: {error}")
            return
        output = Path(csv_var.get()).expanduser() if csv_var.get().strip() else video.with_name(video.stem + "_tracking.csv")
        if output.suffix.lower() != ".csv":
            output = output.with_suffix(".csv")
        result[0] = WizardSelection(str(video), str(output), count, mode_var.get())
        root.destroy()

    def cancel() -> None:
        root.destroy()

    error_var = tk.StringVar()
    tk.Label(root, text="CV TRACKER / SETUP", font=("TkDefaultFont", 14, "bold"), anchor="w").pack(
        fill="x", padx=12, pady=(12, 8)
    )
    form = tk.Frame(root)
    form.pack(fill="x", padx=12)
    form.columnconfigure(1, weight=1)
    tk.Label(form, text="Input video").grid(row=0, column=0, sticky="w", pady=4)
    tk.Entry(form, textvariable=video_var).grid(row=0, column=1, sticky="ew", pady=4)
    tk.Button(form, text="Browse...", command=browse_video).grid(row=0, column=2, padx=(6, 0), pady=4)
    tk.Label(form, text="Output CSV").grid(row=1, column=0, sticky="w", pady=4)
    tk.Entry(form, textvariable=csv_var).grid(row=1, column=1, sticky="ew", pady=4)
    tk.Button(form, text="Browse...", command=browse_csv).grid(row=1, column=2, padx=(6, 0), pady=4)
    tk.Label(form, text="Slave count").grid(row=2, column=0, sticky="w", pady=4)
    tk.Entry(form, textvariable=count_var, width=10).grid(row=2, column=1, sticky="w", pady=4)
    tk.Label(form, text="Session mode").grid(row=3, column=0, sticky="w", pady=4)
    mode_frame = tk.Frame(form)
    mode_frame.grid(row=3, column=1, sticky="w", pady=4)
    tk.Radiobutton(mode_frame, text="Auto", variable=mode_var, value="auto").pack(side="left")
    tk.Radiobutton(mode_frame, text="Manual", variable=mode_var, value="manual").pack(side="left", padx=12)
    tk.Label(root, textvariable=error_var, foreground="#b00020", anchor="w").pack(fill="x", padx=12, pady=(6, 0))
    tk.Label(
        root,
        text="After setup, the video opens paused at frame 1. Auto detects while playing; Manual detects only when you press Detect while paused.",
        anchor="w",
        justify="left",
        wraplength=590,
    ).pack(fill="x", padx=12, pady=(8, 4))
    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=(4, 12))
    tk.Button(buttons, text="Start session", command=start).pack(side="left", expand=True, fill="x", padx=(0, 5))
    tk.Button(buttons, text="Cancel", command=cancel).pack(side="left", expand=True, fill="x", padx=(5, 0))
    root.bind("<Return>", lambda _event: start())
    root.bind("<Escape>", lambda _event: cancel())
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return result[0]


def _run_glfw_input_wizard(default_slaves: Optional[int] = None) -> Optional[WizardSelection]:
    """Run the GLFW/ImGui wizard after Tkinter is unavailable."""
    if glfw is None or imgui is None or GlfwRenderer is None:
        if GUI_IMPORT_ERROR:
            print(f"CV GUI dependencies are missing: {GUI_IMPORT_ERROR}", file=sys.stderr)
        return None
    if os.environ.get("DISPLAY") and hasattr(glfw, "init_hint") and hasattr(glfw, "PLATFORM_X11"):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    try:
        initialized = glfw.init()
    except Exception as error:
        print(f"CV GLFW startup failed: {error}", file=sys.stderr)
        return None
    if not initialized:
        return None
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    if hasattr(glfw, "CONTEXT_CREATION_API") and hasattr(glfw, "NATIVE_CONTEXT_API"):
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    if hasattr(glfw, "OPENGL_PROFILE") and hasattr(glfw, "OPENGL_CORE_PROFILE"):
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    try:
        window = glfw.create_window(1100, 800, "CV setup wizard", None, None)
    except Exception as error:
        print(f"CV GLFW window creation failed: {error}", file=sys.stderr)
        glfw.terminate()
        return None
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    if hasattr(glfw, "get_current_context") and glfw.get_current_context() is None:
        glfw.destroy_window(window)
        glfw.terminate()
        return None
    imgui.create_context()
    _install_imgui_theme()
    renderer = None
    wizard = CvInputWizard(slave_count=default_slaves or 6)
    gui_error: Optional[Exception] = None
    try:
        renderer = GlfwRenderer(window)
        while not glfw.window_should_close(window) and not wizard.finished and not wizard.cancelled:
            glfw.poll_events()
            renderer.process_inputs()
            imgui.new_frame()
            wizard.draw()
            imgui.render()
            renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    except Exception as error:
        print(f"CV GUI wizard failed: {error}", file=sys.stderr)
        gui_error = error
    finally:
        if renderer is not None:
            renderer.shutdown()
        glfw.destroy_window(window)
        glfw.terminate()
    if gui_error is not None:
        return None
    if wizard.cancelled or not wizard.finished:
        return None
    return WizardSelection(wizard.video_path, wizard.csv_path, wizard.slave_count, wizard.mode)


def run_input_wizard(default_slaves: Optional[int] = None) -> Optional[WizardSelection]:
    """Try Tkinter first, then use GLFW/ImGui if Tk cannot start."""
    global _TK_WIZARD_STARTED
    try:
        selection = run_tk_input_wizard(default_slaves)
    except Exception as error:
        print(f"Tkinter setup wizard failed: {error}", file=sys.stderr)
        _TK_WIZARD_STARTED = False
        selection = None
    if _TK_WIZARD_STARTED:
        return selection
    return _run_glfw_input_wizard(default_slaves)


def choose_input_and_output_paths(
    default_slaves: Optional[int] = None,
) -> Optional[tuple[str, str, int, str]]:
    """Autocv-style GUI selection for a no-argument launch."""
    wizard_selection = run_input_wizard(default_slaves)
    if wizard_selection is not None:
        return (
            wizard_selection.video_path,
            wizard_selection.csv_path,
            wizard_selection.slave_count,
            wizard_selection.mode,
        )

    # If a desktop session is present, do not silently downgrade to a
    # terminal prompt after a GUI/OpenGL startup error.  That was the confusing
    # behaviour of the previous implementation.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        print(
            "CV input wizard could not start. Check the GLFW/OpenGL display "
            "and run with PYOPENGL_PLATFORM=glx on WSLg.",
            file=sys.stderr,
        )
        return None

    # Keep a terminal fallback for machines with no display server at all.
    video = choose_path_dialog(save=False, title="Select input video")
    if not video:
        return None
    default_csv = str(Path(video).with_name(Path(video).stem + "_tracking.csv"))
    csv_path = choose_path_dialog(save=True, title="Select tracking CSV output", initial=default_csv)
    if not csv_path:
        return None
    csv_file = Path(csv_path).expanduser()
    if csv_file.suffix.lower() != ".csv":
        csv_file = csv_file.with_suffix(".csv")
    count = default_slaves if default_slaves is not None else ask_slave_count()
    return str(Path(video).expanduser()), str(csv_file), int(count), "auto"


def track_video(
    video_path: str,
    csv_path: str,
    config: TrackerConfig,
    preview: bool = True,
    base_hint: Optional[tuple[float, float]] = None,
    mode: str = "auto",
) -> int:
    if mode not in {"auto", "manual"}:
        raise ValueError("mode must be 'auto' or 'manual'")
    if mode == "manual" and not preview:
        raise RuntimeError("manual mode requires the video session window; remove --headless")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"cannot read first frame: {video_path}")
    if PROC_SCALE != 1.0:
        first_frame = cv2.resize(first_frame, None, fx=PROC_SCALE, fy=PROC_SCALE, interpolation=cv2.INTER_AREA)
    if base_hint is None:
        initial = select_points_interactively(first_frame, config)
    else:
        red, yellow, _ = detect_candidates(first_frame, base_hint, config)
        # A base hint alone is sufficient for setup only when the caller also
        # supplies ordered slave hints through the helper API; interactive
        # setup is safer for normal CLI use.
        initial = select_points_interactively(first_frame, config)
    if initial is None:
        cap.release()
        return 0

    output = Path(csv_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    try:
        if preview:
            session = TkTrackingSession(cap, first_frame, initial, config, mode)
            rows = session.run()
        else:
            tracker = OrderedMarkerTracker(config)
            tracker.initialize(initial)
            metric = MetricSmoother()
            metric.update(estimate_equal_link_metric(initial))
            frame_no = 1
            while True:
                for _ in range(FRAME_SKIP):
                    cap.grab()
                ok, frame = cap.read()
                if not ok:
                    break
                if PROC_SCALE != 1.0:
                    frame = cv2.resize(frame, None, fx=PROC_SCALE, fy=PROC_SCALE, interpolation=cv2.INTER_AREA)
                frame_no += 1
                result, _annotated, _predicted = process_tracking_frame(frame, tracker, config, initial)
                if result.points is not None:
                    metric.update(estimate_equal_link_metric(result.points))
                rows.append(
                    tracking_row(
                        frame_no,
                        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        result,
                        config,
                        metric.matrix,
                    )
                )
    finally:
        cap.release()

    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tracking_fieldnames(config))
        writer.writeheader()
        writer.writerows(rows)
    return sum(int(row["accepted"]) for row in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ordered colour marker tracker with temporal gating")
    parser.add_argument("video", nargs="?", help="video/image sequence path; omitted opens a file picker")
    parser.add_argument("--csv", default="", help="tracking CSV path; omitted opens a save picker")
    parser.add_argument("--slaves", type=int, default=None, help="slave point count; omitted asks interactively")
    parser.add_argument("--mode", choices=("auto", "manual"), default="auto", help="tracking session mode")
    parser.add_argument("--headless", action="store_true", help="disable preview after interactive setup")
    parser.add_argument("--max-motion-px", type=float, default=DEFAULT_MAX_MOTION_PX)
    parser.add_argument("--missing-grace", type=int, default=MISSING_GRACE_FRAMES)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    mode = args.mode
    if args.video:
        video_path = str(Path(args.video).expanduser())
        if args.csv:
            csv_path = str(Path(args.csv).expanduser())
        else:
            csv_path = str(Path(video_path).with_name(Path(video_path).stem + "_tracking.csv"))
        count = ask_slave_count(args.slaves)
    else:
        selected = choose_input_and_output_paths(args.slaves)
        if selected is None:
            print("No input/output path selected.")
            return 0
        video_path, csv_path, wizard_count, wizard_mode = selected
        count = int(args.slaves) if args.slaves is not None else wizard_count
        mode = wizard_mode
    config = TrackerConfig(count, max_motion_px=args.max_motion_px, missing_grace_frames=args.missing_grace)
    try:
        accepted = track_video(video_path, csv_path, config, preview=not args.headless, mode=mode)
    except Exception as error:
        print(f"tracking failed: {error}", file=sys.stderr)
        return 2
    print(f"accepted frames: {accepted}; CSV: {Path(csv_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
