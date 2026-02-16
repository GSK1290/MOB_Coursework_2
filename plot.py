import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy as np


typestore = get_typestore(Stores.ROS2_JAZZY)
bag_path = './test_estsim25' 

gt_x, gt_y = [], []
est_x, est_y = [], []

with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
    connections = [c for c in reader.connections if c.topic in ['/base_pose_ground_truth', '/estimatedpose']]
    
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        
        try:
            pos = msg.pose.pose.position
        except AttributeError:
            pos = msg.pose.position

        if connection.topic == '/base_pose_ground_truth':
            gt_x.append(pos.x)
            gt_y.append(pos.y)
        else:
            est_x.append(pos.x)
            est_y.append(pos.y)


plt.figure(figsize=(10, 6))
plt.plot(gt_x, gt_y, label='Ground Truth', color='blue', linestyle='--')
plt.plot(est_x, est_y, label='Estimated Pose', color='red', linestyle='-.')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()
