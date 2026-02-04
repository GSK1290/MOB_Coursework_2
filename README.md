# Introduction to Mobile Robotic - Coursework 2
The objective of this exercise is for you to implement, analyze, and understand the working of a particle filter localisation algorithm. You will be working in simulation, using the `Stage` environment and `RViz 2`.

## 1. Setup your workspace


Clone the Github repository 
```
git clone https://github.com/LCCZK/MOB_Coursework_2.git
```

Go the the root directory of the repo and update all submodules.
```
cd MOB_Coursework_2
git submodule update --init --recursive
```
Source youe ros2 installation and set the ROS_LOCALHOST_ONLY environment variable
```
echo "source /opt/ros/jazzy/setup.sh" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
```
## 2. Working with ROS
Build your ROS workspace with `colcon`, **\*run form the root directory of the repo  --symlink-install is not supported for this workspace**.
```
colcon build
```
Source your work space with 
```
source install/setup.bash 
```

## 3. Your tasks
You only have to make changes to `pf.py` under `src/pf_localisation/pf_localisation/pf.py`, implement the logic to initiate a particle cloud, resample the particle cloud and estimate a pose from your particle cloud.

## 4. Test your implementation with simulation
A example launch file is provided under `src/pf_localisation/launch/example_pf.launch.py`, you can try it with:
```
ros2 launch pf_localisation example_pf.launch.py
```
You may also write your own launch script if you wish so.