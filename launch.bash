#!/bin/bash

cleanup() {
    echo "Cleaning up processes..."
    
   
    if [ ! -z "$RECORD_PID" ]; then
        echo "Stopping bag recording gracefully..."
        kill -SIGINT $RECORD_PID 2>/dev/null
        sleep 3 
        
        kill -9 $RECORD_PID 2>/dev/null
        pkill -9 -P $RECORD_PID 2>/dev/null
    fi
    
    if [ ! -z "$LAUNCH_PID" ]; then
        kill -9 $LAUNCH_PID 2>/dev/null
        pkill -9 -P $LAUNCH_PID 2>/dev/null
    fi
    
    pkill -9 -f "ros2 bag record"
    pkill -9 -f "pf_localisation"
    pkill -9 -f "rviz2"
    pkill -9 -f "ros2 launch"
    
    sleep 1
    echo "Cleanup complete!"
}

trap cleanup EXIT INT TERM

colcon build
source install/setup.bash

ros2 launch pf_localisation example_pf.launch.py &
LAUNCH_PID=$!

sleep 3

ros2 bag record -o simpath2_pf /base_pose_ground_truth /estimatedpose &
RECORD_PID=$!

sleep 2

ros2 bag play ./src/pf_localisation/data/sim_data/simpath2 --clock

echo "Playback finished. Waiting for bag to finalize..."
sleep 2 
