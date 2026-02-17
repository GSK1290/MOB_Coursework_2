import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy as np

BAG_SELF = "./simpath2_self_amcl"
TOPIC_SELF = "/estimatedpose"
TOPIC_GT = "/base_pose_ground_truth"

BAG_AMCL = "./simpath2_amcl"
TOPIC_AMCL = "/amcl_pose"

SELF_OFFSET_X = -15.1 
SELF_OFFSET_Y = -15.1


def apply_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def extract_xy_time(bag_path: str, topic: str, typestore):
    t0 = None
    t_list, x_list, y_list = [], [], []

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise RuntimeError(f"Topic '{topic}' not found in {bag_path}")

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            if t0 is None:
                t0 = timestamp

            t = (timestamp - t0) * 1e-9
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

            try:
                pos = msg.pose.pose.position
            except AttributeError:
                pos = msg.pose.position

            t_list.append(t)
            x_list.append(pos.x)
            y_list.append(pos.y)

    return np.array(t_list), np.array(x_list), np.array(y_list)


def build_global_common_time(t_gt, t_self, t_amcl):
    """
    Create a single time vector that is inside the overlap of all 3 streams.
    We use GT timestamps within the global overlap window as t_common.
    """
    start = max(t_gt[0], t_self[0], t_amcl[0])
    end = min(t_gt[-1], t_self[-1], t_amcl[-1])

    if start >= end:
        raise RuntimeError("No overlapping time range between GT, Self, and AMCL.")

    mask = (t_gt >= start) & (t_gt <= end)
    t_common = t_gt[mask]

    if len(t_common) < 2:
        raise RuntimeError("Not enough overlapping samples to plot (t_common too small).")

    return t_common


def interp_xy(t_common, t_src, x_src, y_src):
    x_i = np.interp(t_common, t_src, x_src)
    y_i = np.interp(t_common, t_src, y_src)
    return x_i, y_i


def main():
    apply_plot_style()
    typestore = get_typestore(Stores.ROS2_JAZZY)

    t_self, x_self, y_self = extract_xy_time(BAG_SELF, TOPIC_SELF, typestore)
    t_gt, x_gt, y_gt = extract_xy_time(BAG_SELF, TOPIC_GT, typestore)
    t_amcl, x_amcl, y_amcl = extract_xy_time(BAG_AMCL, TOPIC_AMCL, typestore)

    x_self = x_self + SELF_OFFSET_X
    y_self = y_self + SELF_OFFSET_Y
    x_amcl = x_amcl + SELF_OFFSET_X
    y_amcl = y_amcl + SELF_OFFSET_Y

    t_common = build_global_common_time(t_gt, t_self, t_amcl)

    x_gt_i, y_gt_i = interp_xy(t_common, t_gt, x_gt, y_gt)
    x_self_i, y_self_i = interp_xy(t_common, t_self, x_self, y_self)
    x_amcl_i, y_amcl_i = interp_xy(t_common, t_amcl, x_amcl, y_amcl)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t_common, x_gt_i, linewidth=2.2, label="Ground Truth")
    ax1.plot(t_common, x_self_i, linewidth=2.2, linestyle="--", label="Self PF")
    ax1.plot(t_common, x_amcl_i, linewidth=2.2, linestyle=":", label="AMCL")
    ax1.set_ylabel("X Position (m)")
    ax1.set_title("Position vs Time (simpath2)")
    ax1.grid(True)

    ax2.plot(t_common, y_gt_i, linewidth=2.2, label="Ground Truth")
    ax2.plot(t_common, y_self_i, linewidth=2.2, linestyle="--", label="Self PF")
    ax2.plot(t_common, y_amcl_i, linewidth=2.2, linestyle=":", label="AMCL")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y Position (m)")
    ax2.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("simpath2_comparison_xy_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig2, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(x_gt_i, y_gt_i, linewidth=2.2, label="Ground Truth")
    ax.plot(x_self_i, y_self_i, linewidth=2.2, linestyle="--", label="Self PF")
    ax.plot(x_amcl_i, y_amcl_i, linewidth=2.2, linestyle=":", label="AMCL")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Comparison (simpath2)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig("simpath2_comparison_trajectory.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:")
    print(" - simpath2_comparison_xy_time.png")
    print(" - simpath2_comparison_trajectory.png")


if __name__ == "__main__":
    main()
