import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy as np

BAG_GT = "./simpath2_self_amcl"          
TOPIC_GT = "/base_pose_ground_truth"

BAG_SELF_AMCL = "./simpath2_self_amcl"  
TOPIC_SELF_AMCL = "/estimatedpose"

BAG_NAV_AMCL = "./simpath2_amcl"        
TOPIC_NAV_AMCL = "/amcl_pose"

BAG_PF = "./simpath2_pf"                
TOPIC_PF = "/estimatedpose"

OFFSET_SELF_AMCL_X, OFFSET_SELF_AMCL_Y = -15.1, -15.1
OFFSET_NAV_AMCL_X,  OFFSET_NAV_AMCL_Y  = -15.1, -15.1
OFFSET_PF_X,        OFFSET_PF_Y        = -15.1, -15.1


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
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"Topic '{topic}' not found in {bag_path}")

        for conn, timestamp, rawdata in reader.messages(connections=conns):
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


def build_global_common_time(t_ref, streams):
    start = max([t_ref[0]] + [t[0] for t in streams])
    end   = min([t_ref[-1]] + [t[-1] for t in streams])

    if start >= end:
        raise RuntimeError("No overlapping time range across all bags/topics.")

    mask = (t_ref >= start) & (t_ref <= end)
    t_common = t_ref[mask]

    if len(t_common) < 2:
        raise RuntimeError("Overlap exists but too few samples to plot.")

    return t_common


def interp_xy(t_common, t_src, x_src, y_src):
    x_i = np.interp(t_common, t_src, x_src)
    y_i = np.interp(t_common, t_src, y_src)
    return x_i, y_i


def main():
    apply_plot_style()
    typestore = get_typestore(Stores.ROS2_JAZZY)

    t_gt, x_gt, y_gt = extract_xy_time(BAG_GT, TOPIC_GT, typestore)

    t_self, x_self, y_self = extract_xy_time(BAG_SELF_AMCL, TOPIC_SELF_AMCL, typestore)
    t_nav,  x_nav,  y_nav  = extract_xy_time(BAG_NAV_AMCL, TOPIC_NAV_AMCL, typestore)
    t_pf,   x_pf,   y_pf   = extract_xy_time(BAG_PF, TOPIC_PF, typestore)

    x_self += OFFSET_SELF_AMCL_X; y_self += OFFSET_SELF_AMCL_Y
    x_nav  += OFFSET_NAV_AMCL_X;  y_nav  += OFFSET_NAV_AMCL_Y
    x_pf   += OFFSET_PF_X;        y_pf   += OFFSET_PF_Y

    t_common = build_global_common_time(t_gt, [t_self, t_nav, t_pf])

    x_gt_i, y_gt_i     = interp_xy(t_common, t_gt,   x_gt,   y_gt)
    x_self_i, y_self_i = interp_xy(t_common, t_self, x_self, y_self)
    x_nav_i, y_nav_i   = interp_xy(t_common, t_nav,  x_nav,  y_nav)
    x_pf_i, y_pf_i     = interp_xy(t_common, t_pf,   x_pf,   y_pf)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # X
    ax1.plot(t_common, x_gt_i, linewidth=2.2, label="Ground Truth")
    ax1.plot(t_common, x_self_i, linewidth=2.2, linestyle="--", label="Self-AMCL PF (/estimatedpose)")
    ax1.plot(t_common, x_nav_i, linewidth=2.2, linestyle=":",  label="Nav AMCL (/amcl_pose)")
    ax1.plot(t_common, x_pf_i, linewidth=2.2, linestyle="-.", label="PF baseline (/estimatedpose)")
    ax1.set_ylabel("X Position (m)")
    ax1.set_title("simpath2: Ground Truth vs Self-AMCL PF vs Nav AMCL vs PF baseline")
    ax1.grid(True)

    # Y
    ax2.plot(t_common, y_gt_i, linewidth=2.2, label="Ground Truth")
    ax2.plot(t_common, y_self_i, linewidth=2.2, linestyle="--", label="Self-AMCL PF (/estimatedpose)")
    ax2.plot(t_common, y_nav_i,  linewidth=2.2, linestyle=":",  label="Nav AMCL (/amcl_pose)")
    ax2.plot(t_common, y_pf_i,   linewidth=2.2, linestyle="-.", label="PF baseline (/estimatedpose)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y Position (m)")
    ax2.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("simpath2_all_comparison_xy_time.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
