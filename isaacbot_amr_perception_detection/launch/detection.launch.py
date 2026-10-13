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
YOLO detection on the colour stream -> detections (vision_msgs/Detection2DArray).

  ros2 launch isaacbot_amr_perception_detection detection.launch.py model_path:=/path/yolo11n.onnx labels:=person

labels: comma-separated class names to publish (e.g. "person" for follow-me); empty = all classes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    # [''] = every class (the node's convention for "no filter").
    labels = [s.strip() for s in arg('labels').split(',') if s.strip()] or ['']
    return [Node(
        package='isaacbot_amr_perception_detection',
        executable='detection_node',
        namespace=arg('namespace'),
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('isaacbot_amr_perception_detection'), 'config', 'detection.yaml']),
            {'model_path': LaunchConfiguration('model_path'),
             'use_gpu': LaunchConfiguration('use_gpu'),
             'max_rate_hz': LaunchConfiguration('max_rate_hz'),
             'allowed_labels': labels,
             'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[('image_raw', LaunchConfiguration('image_topic'))],
        output='screen',
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value=EnvironmentVariable('ISAACBOT_AMR_SYSTEM_NAMESPACE', default_value='')),
        DeclareLaunchArgument('image_topic', default_value='camera/color/image_raw'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('use_gpu', default_value='false'),
        DeclareLaunchArgument('max_rate_hz', default_value='3.0'),
        DeclareLaunchArgument('labels', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_launch_setup),
    ])
