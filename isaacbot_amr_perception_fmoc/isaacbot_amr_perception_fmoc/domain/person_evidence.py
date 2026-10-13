# Copyright 2026 Mechatronics Academy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Which depth points an image detector says are a person.

ADBSCAN finds anything person-sized: a box, a chair, a pole, the isaacbot's own front. A 2D detector
(YOLO "person") knows what a person looks like but not where it is in 3D. Here the cloud's points
are projected into the detector's image, and the points inside a person box are split out of their
clusters into a person part of their own:

* A person standing next to a box or a wall is often one ADBSCAN cluster with it, too wide to be a
  person. The box in the image only holds the person, so the part inside it is the person again.
* A person box also covers whatever stands behind the person. Of the clusters inside a box, the
  nearest ones (within depth_band of the nearest) are the person, and of those the one with most
  points, so a wall or pole behind them, or a small thing in front of their legs, is not.
* Of that cluster, only the points within body_depth of its median depth are the person: a few
  points of something just behind them made a person 1.2 m deep, too wide to be followed.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PersonBox:
    """A person detection in image pixels: top-left x, y and width, height."""

    x: float
    y: float
    width: float
    height: float
    score: float


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera of the detector's image: 3x3 intrinsics K (z forward optical frame)."""

    k: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class PersonEvidenceConfig:
    # Boxes are grown by this many pixels on every side (box edges are loose, depth is offset).
    margin_px: float = 10.0
    # Fewer points of a cluster inside a box than this are not a person part.
    min_points: int = 10
    # m: clusters inside a box this much further away than the nearest one are behind the person.
    depth_band: float = 0.5
    # m: points of the person's cluster further than this from its median depth are not them.
    body_depth: float = 0.3

    def __post_init__(self):
        if self.margin_px < 0.0:
            raise ValueError('margin_px must not be negative')
        if self.min_points < 1:
            raise ValueError('min_points must be >= 1')
        if self.depth_band <= 0.0 or self.body_depth <= 0.0:
            raise ValueError('depth_band and body_depth must be positive')


def to_camera(points: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    """(N, 3) base-frame points -> (N, 3) in the camera's optical frame (z = depth)."""
    return points @ base_to_camera[:3, :3].T + base_to_camera[:3, 3]


def project(points: np.ndarray, base_to_camera: np.ndarray, camera: CameraModel) -> np.ndarray:
    """
    (N, 3) base-frame points -> (N, 2) pixel coordinates; NaN for points behind the camera.

    base_to_camera: 4x4 transform taking base-frame points into the camera's optical frame.
    """
    cam = to_camera(points, base_to_camera)
    z = cam[:, 2]
    uv = np.full((len(points), 2), np.nan)
    front = z > 1e-6
    if front.any():
        hom = cam[front] @ camera.k.T
        uv[front] = hom[:, :2] / hom[:, 2:3]
    return uv


def split_person_parts(points: np.ndarray, labels: np.ndarray, base_to_camera: np.ndarray,
                       camera: CameraModel, boxes: Sequence[PersonBox],
                       config: PersonEvidenceConfig) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Split each person box's points out of their cluster into a new cluster label.

    points: (N, 3) in the base frame; labels: (N,) ADBSCAN labels, -1 for noise.
    Returns the new labels (the rest of a split cluster keeps its label) and {label: box score}
    for the person parts. One part per box, boxes taken by score.
    """
    labels = labels.copy()
    parts: Dict[int, float] = {}
    if not boxes or not (labels >= 1).any():
        return labels, parts
    uv = project(points, base_to_camera, camera)
    depth = to_camera(points, base_to_camera)[:, 2]
    next_label = int(labels.max()) + 1
    m = config.margin_px
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        with np.errstate(invalid='ignore'):
            inside = ((uv[:, 0] >= box.x - m) & (uv[:, 0] <= box.x + box.width + m)
                      & (uv[:, 1] >= box.y - m) & (uv[:, 1] <= box.y + box.height + m))
        inside &= (labels >= 1) & ~np.isin(labels, list(parts))
        found = []   # (median depth, points, label)
        for c in np.unique(labels[inside]):
            member = inside & (labels == c)
            n = int(np.count_nonzero(member))
            if n >= config.min_points:
                found.append((float(np.median(depth[member])), n, int(c)))
        if not found:
            continue
        nearest = min(d for d, _, _ in found)
        _, _, c = max((f for f in found if f[0] <= nearest + config.depth_band),
                      key=lambda f: f[1])
        part = inside & (labels == c)
        part &= np.abs(depth - np.median(depth[part])) <= config.body_depth
        labels[part] = next_label
        parts[next_label] = box.score
        next_label += 1
    return labels, parts
