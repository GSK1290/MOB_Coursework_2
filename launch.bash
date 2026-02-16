#!/bin/bash

colcon build
source install/setup.bash

ros2 launch pf_localisation example_pf.launch.py