#!/bin/bash

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    
    # Stop bag recording gracefully first (SIGINT allows it to close properly)
    if [ ! -z "$RECORD_PID" ]; then
        echo "Stopping bag recording gracefully..."
        kill -SIGINT $RECORD_PID 2>/dev/null
        sleep 3  # Give it time to write metadata
        
        # Force kill if still running
        kill -9 $RECORD_PID 2>/dev/null
        pkill -9 -P $RECORD_PID 2>/dev/null
    fi
    
    # Kill launch process and its children
    if [ ! -z "$LAUNCH_PID" ]; then
        kill -9 $LAUNCH_PID 2>/dev/null
        pkill -9 -P $LAUNCH_PID 2>/dev/null
    fi
    
    # Nuclear option - kill all related processes
    pkill -9 -f "ros2 bag record"
    pkill -9 -f "pf_localisation"
    pkill -9 -f "rviz2"
    pkill -9 -f "ros2 launch"
    
    sleep 1
    echo "Cleanup complete!"
}

# Trap exit signals
trap cleanup EXIT INT TERM

# Build and source
colcon build
source install/setup.bash

# Launch localization in background
ros2 launch pf_localisation example_pf.launch.py &
LAUNCH_PID=$!

# Give launch time to start up
sleep 3

# Start bag recording in background
ros2 bag record -o simpath2_amcl /base_pose_ground_truth /amcl_pose &
RECORD_PID=$!

# Give recording time to initialize
sleep 2

# Play the bag file (this will block until done)
ros2 bag play ./src/pf_localisation/data/sim_data/simpath2 --clock

echo "Playback finished. Waiting for bag to finalize..."
sleep 2  # Give bag recorder time to finish writing

# Cleanup will be called automatically