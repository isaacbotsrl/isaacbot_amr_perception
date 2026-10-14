# Copyright 2026 IsaacBot
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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value=EnvironmentVariable('ISAACBOT_AMR_SYSTEM_NAMESPACE', default_value='')),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='isaacbot_amr_perception_terrain',
            executable='terrain_node',
            namespace=namespace,
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare('isaacbot_amr_perception_terrain'), 'config', 'terrain.yaml']),
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            output='screen',
        ),
    ])
