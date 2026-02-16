import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# ---------- Settings ----------
bag_path = './simpath2_estpose3'   # your bag folder or .mcap file
GT_TOPIC = '/base_pose_ground_truth'
EST_TOPIC = '/estimatedpose'
# ------------------------------

typestore = get_typestore(Stores.ROS2_JAZZY)

gt_x, gt_y, gt_t = [], [], []
est_x, est_y, est_t = [], [], []

t0 = None

with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:

    connections = [
        c for c in reader.connections
        if c.topic in [GT_TOPIC, EST_TOPIC]
    ]

    for connection, timestamp, rawdata in reader.messages(connections=connections):

        if t0 is None:
            t0 = timestamp

        t = (timestamp - t0) * 1e-9  # convert ns → seconds

        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        # Handle both Odometry and PoseStamped formats
        try:
            pos = msg.pose.pose.position
        except AttributeError:
            pos = msg.pose.position

        if connection.topic == GT_TOPIC:
            gt_x.append(pos.x)
            gt_y.append(pos.y)
            gt_t.append(t)

        elif connection.topic == EST_TOPIC:
            est_x.append(pos.x-15.1)
            est_y.append(pos.y-15.1)
            est_t.append(t)


# ---------------- Plotting ----------------

plt.figure(figsize=(12, 6))

# X vs Time
plt.subplot(1, 2, 1)
plt.plot(gt_t, gt_x, label='Ground Truth X')
plt.plot(est_t, est_x, label='Estimated X')
plt.xlabel('Time (s)')
plt.ylabel('X Position (m)')
plt.title('X Position vs Time')
plt.legend()
plt.grid(True)

# Y vs Time
plt.subplot(1, 2, 2)
plt.plot(gt_t, gt_y, label='Ground Truth Y')
plt.plot(est_t, est_y, label='Estimated Y')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
plt.title('Y Position vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("x_y_vs_time.png", dpi=300)
plt.show()
