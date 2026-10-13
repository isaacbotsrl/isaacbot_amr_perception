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

from isaacbot_amr_perception_fmoc.application.track_person import PersonDetections, TrackPerson
from isaacbot_amr_perception_fmoc.domain.adbscan import AdbscanConfig
from isaacbot_amr_perception_fmoc.domain.cloud_filter import CloudFilterConfig
from isaacbot_amr_perception_fmoc.domain.target_tracker import RobotMotion, RobotPose, TrackerConfig, TrackState
from synthetic_scene import (
    base_to_color_optical, camera_to_base, color_camera, person_box, render)


def test_follows_a_person_walking_away_past_a_bystander():
    use_case = TrackPerson(CloudFilterConfig(), AdbscanConfig(), TrackerConfig())
    robot = RobotPose(0.0, 0.0, 0.0)
    result = use_case.process(render(people=[(1.6, 0.0), (2.0, -1.2)], wall_x=4.5),
                              camera_to_base(), robot, RobotMotion(0.0, 0.0), 0.0)
    assert result.track.state == TrackState.TRACKING
    assert abs(result.track.y) < 0.1
    xs = []
    for i in range(1, 12):
        px = 1.6 + 0.15 * i
        result = use_case.process(render(people=[(px, 0.0), (2.0, -1.2)], wall_x=4.5),
                                  camera_to_base(), robot, RobotMotion(0.5, 0.0), 0.2 * i)
        assert result.track.state == TrackState.TRACKING
        xs.append(result.track.x)
        assert abs(result.track.y) < 0.15  # never jumps to the bystander
    assert np.all(np.diff(xs) > 0)


def test_max_points_caps_the_cloud():
    use_case = TrackPerson(CloudFilterConfig(row_stride=1, col_stride=1), AdbscanConfig(),
                           TrackerConfig(), max_points=500)
    result = use_case.process(render(people=[(1.6, 0.0)]), camera_to_base(),
                              RobotPose(0.0, 0.0, 0.0), RobotMotion(0.0, 0.0), 0.0)
    assert result.points_used == 500


def _detections(*boxes):
    return PersonDetections(list(boxes), color_camera(), base_to_color_optical())


def test_with_a_detector_only_a_person_box_is_acquired():
    still = RobotMotion(0.0, 0.0)
    robot = RobotPose(0.0, 0.0, 0.0)
    scene = render(people=[(1.5, 0.0), (2.0, -1.2)], wall_x=4.5)
    # Person-sized in the zone, but the detector found no person there (a pole, a box).
    use_case = TrackPerson(CloudFilterConfig(), AdbscanConfig(), TrackerConfig())
    result = use_case.process(scene, camera_to_base(), robot, still, 0.0,
                              _detections(person_box(2.0, -1.2)))
    assert result.track.state == TrackState.NONE
    assert sum(result.person) == 1  # the bystander is a person, the zone object is not
    result = use_case.process(scene, camera_to_base(), robot, still, 0.2, _detections())
    assert result.track.state == TrackState.NONE and not any(result.person)
    # The detector confirms the one in the zone.
    result = use_case.process(scene, camera_to_base(), robot, still, 0.4,
                              _detections(person_box(1.5, 0.0), person_box(2.0, -1.2)))
    assert result.track.state == TrackState.TRACKING
    assert abs(result.track.x - 1.5) < 0.3 and abs(result.track.y) < 0.1


def test_a_person_standing_next_to_a_box_is_acquired():
    # A wide eps (as on the noisy real cloud) makes the person and the box beside them one
    # cluster, 1.1 m wide: too wide for a person. The person box splits the person back out.
    scene = render(people=[(1.5, 0.0), (1.6, -0.5), (1.6, -0.8)], wall_x=4.5)
    merging = AdbscanConfig(min_eps=0.3, min_eps_per_m=0.0)
    still, robot = RobotMotion(0.0, 0.0), RobotPose(0.0, 0.0, 0.0)
    plain = TrackPerson(CloudFilterConfig(), merging, TrackerConfig())
    result = plain.process(scene, camera_to_base(), robot, still, 0.0)
    assert result.track.state == TrackState.NONE and len(result.clusters) == 1
    use_case = TrackPerson(CloudFilterConfig(), merging, TrackerConfig())
    result = use_case.process(scene, camera_to_base(), robot, still, 0.0,
                              _detections(person_box(1.5, 0.0)))
    assert result.track.state == TrackState.TRACKING
    assert abs(result.track.y) < 0.1
    assert sorted(result.person) == [False, True]


def test_without_a_detector_every_cluster_may_be_a_person():
    use_case = TrackPerson(CloudFilterConfig(), AdbscanConfig(), TrackerConfig())
    result = use_case.process(render(people=[(1.5, 0.0)]), camera_to_base(),
                              RobotPose(0.0, 0.0, 0.0), RobotMotion(0.0, 0.0), 0.0)
    assert result.track.state == TrackState.TRACKING
    assert result.person is None
