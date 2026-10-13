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

import numpy as np
import pytest

from isaacbot_amr_perception_fmoc.domain.person_evidence import (
    PersonBox, PersonEvidenceConfig, project, split_person_parts)
from synthetic_scene import base_to_color_optical, color_camera, person_box

CAMERA = color_camera()
BASE_TO_CAM = base_to_color_optical()
CFG = PersonEvidenceConfig(margin_px=0.0)


def column(x, y, n=50, z0=0.4, z1=1.6):
    """A vertical line of n points at (x, y), z0 to z1 m up, in base_link."""
    return np.stack([np.full(n, x), np.full(n, y), np.linspace(z0, z1, n)], axis=1)


def split(points, labels, boxes, config=CFG):
    return split_person_parts(points, np.asarray(labels), BASE_TO_CAM, CAMERA, boxes, config)


def test_a_point_straight_ahead_projects_to_the_image_centre():
    uv = project(np.array([[2.0, 0.0, 0.5]]), BASE_TO_CAM, CAMERA)
    assert uv[0] == pytest.approx([CAMERA.k[0, 2], CAMERA.k[1, 2]])


def test_points_behind_the_camera_do_not_project():
    uv = project(np.array([[-1.0, 0.0, 0.5]]), BASE_TO_CAM, CAMERA)
    assert np.isnan(uv).all()


def test_a_cluster_inside_a_person_box_becomes_a_person_part():
    points = np.vstack([column(2.0, 0.0), column(2.0, 1.0)])
    labels, parts = split(points, [1] * 50 + [2] * 50, [person_box(2.0, 0.0, score=0.8)])
    assert parts == {3: 0.8}
    assert (labels[:50] == 3).all() and (labels[50:] == 2).all()


def test_no_boxes_change_nothing_and_noise_stays_noise():
    points = np.vstack([column(2.0, 0.0), column(2.0, 0.0, n=5)])
    labels = [1] * 50 + [-1] * 5
    new, parts = split(points, labels, [])
    assert parts == {} and (new == labels).all()
    new, parts = split(points, labels, [person_box(2.0, 0.0)])
    assert parts == {2: 0.9} and (new[50:] == -1).all()


def test_a_person_merged_with_a_box_beside_them_is_split_out():
    # ADBSCAN made one cluster of the person and the box next to them (too wide for a person).
    person, box = column(1.5, 0.0), np.vstack([column(1.5, -y, z1=0.9) for y in (0.4, 0.6, 0.8)])
    labels, parts = split(np.vstack([person, box]), [1] * 200, [person_box(1.5, 0.0)])
    assert parts == {2: 0.9}
    assert (labels[:50] == 2).all() and (labels[50:] == 1).all()


def test_what_stands_behind_the_person_is_not_the_person():
    person = column(2.0, 0.0)
    pole = column(3.5, 0.05, n=80)   # more points, but further away inside the same box
    labels, parts = split(np.vstack([person, pole]), [1] * 50 + [2] * 80, [person_box(2.0, 0.0)])
    assert parts == {3: 0.9} and (labels[:50] == 3).all() and (labels[50:] == 2).all()


def test_of_clusters_at_the_persons_depth_the_largest_is_the_person():
    # Something small just in front of the person's legs, inside the box.
    person, thing = column(2.0, 0.0), column(1.8, 0.0, n=15, z0=0.35, z1=0.6)
    labels, parts = split(np.vstack([person, thing]), [1] * 50 + [2] * 15, [person_box(2.0, 0.0)])
    assert parts == {3: 0.9} and (labels[:50] == 3).all() and (labels[50:] == 2).all()


def test_points_far_from_the_persons_depth_are_not_the_person():
    # One cluster: the person, and a few points of something 0.8 m behind them, in the box.
    person, behind = column(1.5, 0.0), column(2.3, 0.05, n=6, z1=0.6)
    labels, parts = split(np.vstack([person, behind]), [1] * 56, [person_box(1.5, 0.0)])
    assert parts == {2: 0.9}
    assert (labels[:50] == 2).all() and (labels[50:] == 1).all()


def test_too_few_points_inside_are_not_a_person_part():
    points = column(2.0, 0.0, n=8)
    assert split(points, [1] * 8, [person_box(2.0, 0.0)])[1] == {}
    loose = PersonEvidenceConfig(margin_px=0.0, min_points=5)
    assert split(points, [1] * 8, [person_box(2.0, 0.0)], loose)[1] == {2: 0.9}


def test_one_part_per_box_best_score_first():
    points = np.vstack([column(2.0, 0.0), column(2.0, 1.0)])
    boxes = [person_box(2.0, 0.0, score=0.6), person_box(2.0, 1.0, score=0.95)]
    labels, parts = split(points, [1] * 50 + [2] * 50, boxes)
    assert parts == {3: 0.95, 4: 0.6}
    assert (labels[50:] == 3).all() and (labels[:50] == 4).all()


def test_margin_grows_the_box():
    points = column(2.0, 0.0)
    full = person_box(2.0, 0.0)
    beside = PersonBox(full.x + full.width + 5.0, full.y, full.width, full.height, 0.7)
    assert split(points, [1] * 50, [beside])[1] == {}
    grown = PersonEvidenceConfig(margin_px=full.width)
    assert split(points, [1] * 50, [beside], grown)[1] == {2: 0.7}


def test_config_is_validated():
    with pytest.raises(ValueError):
        PersonEvidenceConfig(min_points=0)
    with pytest.raises(ValueError):
        PersonEvidenceConfig(margin_px=-1.0)
    with pytest.raises(ValueError):
        PersonEvidenceConfig(depth_band=0.0)
    with pytest.raises(ValueError):
        PersonEvidenceConfig(body_depth=0.0)
