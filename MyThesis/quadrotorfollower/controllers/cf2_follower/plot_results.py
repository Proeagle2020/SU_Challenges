# -*- coding: utf-8 -*-

import os
import csv

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

CF1_FILE = os.path.join(
    DATA_DIR,
    "leader_trajectory.csv"
)

CF2_FILE = os.path.join(
    DATA_DIR,
    "follower_trajectory.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    data = {}

    if not rows:
        return data

    for key in rows[0].keys():

        try:

            data[key] = np.array(
                [
                    float(r[key])
                    for r in rows
                ],
                dtype=float
            )

        except Exception:

            data[key] = np.array(
                [
                    r[key]
                    for r in rows
                ]
            )

    return data


# ============================================================
# LOAD
# ============================================================

if not os.path.exists(CF1_FILE):

    raise FileNotFoundError(
        "leader_trajectory.csv not found."
    )

if not os.path.exists(CF2_FILE):

    raise FileNotFoundError(
        "follower_trajectory.csv not found."
    )


cf1 = load_csv(
    CF1_FILE
)

cf2 = load_csv(
    CF2_FILE
)


# ============================================================
# FIGURE 1 - 3D PATH
# ============================================================

fig = plt.figure(
    figsize=(10, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    cf1["x"],
    cf1["y"],
    cf1["z"],
    label="CF1 Leader",
    linewidth=2
)

ax.plot(
    cf2["x"],
    cf2["y"],
    cf2["z"],
    label="CF2 Follower",
    linewidth=2
)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_title(
    "3D Trajectory Comparison"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_3D_path.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 2 - SPEED
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    cf1["time"],
    cf1["speed"],
    label="CF1"
)

plt.plot(
    cf2["time"],
    cf2["speed"],
    label="CF2"
)

plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")

plt.title(
    "Speed Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_speed.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 3 - ANGULAR VELOCITY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for key, label in [
    ("wx", "Wx"),
    ("wy", "Wy"),
    ("wz", "Wz")
]:

    plt.plot(
        cf1["time"],
        cf1[key],
        label=f"CF1 {label}"
    )

    plt.plot(
        cf2["time"],
        cf2[key],
        "--",
        label=f"CF2 {label}"
    )


plt.xlabel("Time [s]")

plt.ylabel(
    "Angular Velocity [rad/s]"
)

plt.title(
    "Angular Velocity Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_angular_velocity.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 4 - ROLL
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    cf1["time"],
    cf1["roll"],
    label="CF1 Roll"
)

plt.plot(
    cf2["time"],
    cf2["roll"],
    label="CF2 Roll"
)

plt.xlabel("Time [s]")
plt.ylabel("Roll [rad]")

plt.title(
    "Roll Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_roll.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 5 - PITCH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    cf1["time"],
    cf1["pitch"],
    label="CF1 Pitch"
)

plt.plot(
    cf2["time"],
    cf2["pitch"],
    label="CF2 Pitch"
)

plt.xlabel("Time [s]")
plt.ylabel("Pitch [rad]")

plt.title(
    "Pitch Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_pitch.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 6 - YAW
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    cf1["time"],
    cf1["yaw"],
    label="CF1 Yaw"
)

plt.plot(
    cf2["time"],
    cf2["yaw"],
    label="CF2 Yaw"
)

plt.xlabel("Time [s]")
plt.ylabel("Yaw [rad]")

plt.title(
    "Yaw Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_yaw.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 7 - LINEAR ACCELERATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    cf1["time"],
    cf1["linear_acceleration"],
    label="CF1"
)

plt.plot(
    cf2["time"],
    cf2["linear_acceleration"],
    label="CF2"
)

plt.xlabel("Time [s]")

plt.ylabel(
    "Linear Acceleration [m/s²]"
)

plt.title(
    "Linear Acceleration Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_linear_acceleration.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 8 - ANGULAR ACCELERATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    cf1["time"],
    cf1["angular_acceleration"],
    label="CF1"
)

plt.plot(
    cf2["time"],
    cf2["angular_acceleration"],
    label="CF2"
)

plt.xlabel("Time [s]")

plt.ylabel(
    "Angular Acceleration [rad/s²]"
)

plt.title(
    "Angular Acceleration Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_angular_acceleration.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PATH ALIGNMENT
# ============================================================

n1 = len(cf1["x"])
n2 = len(cf2["x"])

n = min(
    n1,
    n2
)

cf1x = cf1["x"][:n]
cf1y = cf1["y"][:n]
cf1z = cf1["z"][:n]

cf2x = cf2["x"][:n]
cf2y = cf2["y"][:n]
cf2z = cf2["z"][:n]


# ============================================================
# TRANSLATION ALIGNMENT
# ============================================================

cf2x_aligned = (
    cf2x -
    cf2x[0]
    +
    cf1x[0]
)

cf2y_aligned = (
    cf2y -
    cf2y[0]
    +
    cf1y[0]
)

cf2z_aligned = cf2z


# ============================================================
# POSITION ERROR
# ============================================================

error = np.sqrt(
    (cf1x - cf2x_aligned) ** 2
    +
    (cf1y - cf2y_aligned) ** 2
    +
    (cf1z - cf2z_aligned) ** 2
)


time_error = cf1["time"][:n]


# ============================================================
# FIGURE 9 - POSITION ERROR
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time_error,
    error,
    linewidth=2
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "3D Position Error [m]"
)

plt.title(
    "Leader-Follower Position Error"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_position_error.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 10 - XY
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    cf1x,
    cf1y,
    label="CF1 Leader",
    linewidth=2
)

plt.plot(
    cf2x_aligned,
    cf2y_aligned,
    label="CF2 Follower",
    linewidth=2
)

plt.xlabel(
    "X [m]"
)

plt.ylabel(
    "Y [m]"
)

plt.title(
    "Horizontal Trajectory Comparison"
)

plt.axis("equal")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "10_XY_path.png"
    ),
    dpi=200
)

plt.close()


print("")
print("==============================================")
print("        PLOTS GENERATED SUCCESSFULLY")
print("==============================================")

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print(
    "10 plots generated."
)

print("==============================================")