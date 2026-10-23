# -*- coding: utf-8 -*-

"""
Crazyflie 2 - FOLLOWER
Webots R2025a

Architecture:

WAIT
   |
   v
TAKEOFF
   |
   v
ALIGN_XY
   |
   v
ALIGN_Z_YAW
   |
   v
START_STABLE
   |
   v
REPLAY_ABSOLUTE
   |
   v
FINAL_APPROACH
   |
   v
FINAL_HOVER

Main idea:
1. Read CF1 trajectory from leader_trajectory.csv
2. Find the stable beginning of CF1 flight
3. Move CF2 to EXACTLY the same XYZ/Yaw start point
4. Replay the ABSOLUTE CF1 trajectory
5. Use:
       reference velocity
       +
       position feedback
6. Finish above CF1 final ground position
7. Automatically generate plots
"""

from controller import Robot

import os
import csv
import math


# ============================================================
# PID
# ============================================================

from pid_controller import (
    pid_velocity_fixed_height_controller
)


# ============================================================
# CONFIGURATION
# ============================================================

TAKEOFF_HEIGHT = 1.00

ALIGN_XY_SPEED = 0.35

ALIGN_XY_KP = 0.80

ALIGN_Z_KP = 0.80

ALIGN_YAW_KP = 0.80

ALIGN_YAW_MAX_RATE = 0.30

REPLAY_POSITION_KP = 1.00

REPLAY_FEED_FORWARD = 1.00

REPLAY_MAX_SPEED = 0.80

REPLAY_ALTITUDE_KP = 0.45

REPLAY_MIN_HEIGHT = 0.75

REPLAY_MAX_HEIGHT = 1.80

FINAL_HOVER_HEIGHT = 1.50

FINAL_POSITION_KP = 0.70

FINAL_MAX_SPEED = 0.30

TAKEOFF_ALTITUDE_RATE = 0.45

FINAL_ALTITUDE_RATE = 0.30

ALTITUDE_MIN = 0.20

ALTITUDE_MAX = 2.50

START_POSITION_TOLERANCE = 0.035

START_Z_TOLERANCE = 0.035

START_YAW_TOLERANCE = 0.04

START_VELOCITY_TOLERANCE = 0.08

START_STABLE_TIME = 1.0

GROUND_Z = 0.005

LOW_ALTITUDE_LIMIT = 0.55

LOW_ALTITUDE_RECOVERY = 0.95


# ============================================================
# PATHS
# ============================================================

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

LEADER_CSV = os.path.join(
    DATA_DIR,
    "leader_trajectory.csv"
)

FOLLOWER_CSV = os.path.join(
    DATA_DIR,
    "follower_trajectory.csv"
)

START_FLAG = os.path.join(
    DATA_DIR,
    "START_REPLAY.flag"
)

PLOTS_DIR = os.path.join(
    DATA_DIR,
    "plots"
)


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def finite(value):

    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def clamp(value, low, high):

    return max(
        low,
        min(high, value)
    )


def move_toward(
    current,
    target,
    max_delta
):

    if target > current:

        return min(
            current + max_delta,
            target
        )

    return max(
        current - max_delta,
        target
    )


def limit_vector(
    vx,
    vy,
    maximum
):

    magnitude = math.sqrt(
        vx * vx +
        vy * vy
    )

    if magnitude <= maximum:

        return vx, vy

    scale = maximum / magnitude

    return (
        vx * scale,
        vy * scale
    )


def normalize_angle(angle):

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


# ============================================================
# TRAJECTORY CLASS
# ============================================================

class Trajectory:

    def __init__(self, rows):

        self.rows = rows

        self.t0 = rows[0]["time"]

        self.t1 = rows[-1]["time"]

        self.duration = (
            self.t1 -
            self.t0
        )

    def sample(
        self,
        replay_time
    ):

        target_time = (
            self.t0 +
            replay_time
        )

        if target_time <= self.t0:

            return self.rows[0]

        if target_time >= self.t1:

            return self.rows[-1]

        # Binary-style manual search
        low = 0
        high = len(self.rows) - 1

        while low <= high:

            middle = (
                low + high
            ) // 2

            if (
                self.rows[middle]["time"]
                <
                target_time
            ):

                low = middle + 1

            else:

                high = middle - 1

        i = max(
            0,
            min(
                high,
                len(self.rows) - 2
            )
        )

        a = self.rows[i]
        b = self.rows[i + 1]

        dt = (
            b["time"] -
            a["time"]
        )

        alpha = (
            target_time -
            a["time"]
        ) / max(
            dt,
            1e-9
        )

        result = {}

        keys = [
            "x",
            "y",
            "z",

            "vx_global",
            "vy_global",
            "vz_global",

            "roll",
            "pitch",
            "yaw",

            "forward_desired",
            "sideways_desired",
            "yaw_desired",
            "height_desired"
        ]

        for key in keys:

            result[key] = (
                a[key]
                +
                alpha *
                (
                    b[key] -
                    a[key]
                )
            )

        result["time"] = target_time

        return result


# ============================================================
# LOAD LEADER CSV
# ============================================================

def load_leader_csv():

    if not os.path.exists(
        LEADER_CSV
    ):

        raise RuntimeError(
            "\nLeader CSV not found:\n"
            +
            LEADER_CSV
        )

    rows = []

    with open(
        LEADER_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for raw in reader:

            try:

                row = {}

                row["time"] = float(
                    raw["time"]
                )

                row["x"] = float(
                    raw["x"]
                )

                row["y"] = float(
                    raw["y"]
                )

                row["z"] = float(
                    raw["z"]
                )

                row["vx_global"] = float(
                    raw.get(
                        "vx_global",
                        0.0
                    ) or 0.0
                )

                row["vy_global"] = float(
                    raw.get(
                        "vy_global",
                        0.0
                    ) or 0.0
                )

                row["vz_global"] = float(
                    raw.get(
                        "vz_global",
                        0.0
                    ) or 0.0
                )

                row["roll"] = float(
                    raw.get(
                        "roll",
                        0.0
                    ) or 0.0
                )

                row["pitch"] = float(
                    raw.get(
                        "pitch",
                        0.0
                    ) or 0.0
                )

                row["yaw"] = float(
                    raw.get(
                        "yaw",
                        0.0
                    ) or 0.0
                )

                row["forward_desired"] = float(
                    raw.get(
                        "forward_desired",
                        0.0
                    ) or 0.0
                )

                row["sideways_desired"] = float(
                    raw.get(
                        "sideways_desired",
                        0.0
                    ) or 0.0
                )

                row["yaw_desired"] = float(
                    raw.get(
                        "yaw_desired",
                        0.0
                    ) or 0.0
                )

                row["height_desired"] = float(
                    raw.get(
                        "height_desired",
                        1.0
                    ) or 1.0
                )

            except Exception:

                continue

            if not all(
                finite(row[k])
                for k in [
                    "time",
                    "x",
                    "y",
                    "z"
                ]
            ):

                continue

            rows.append(row)

    if len(rows) < 20:

        raise RuntimeError(
            "Leader trajectory has too few valid samples."
        )

    return rows


# ============================================================
# FIND STABLE REPLAY WINDOW
# ============================================================

def find_replay_window(rows):

    # --------------------------------------------------------
    # Find start of stable flight
    # --------------------------------------------------------

    stable_counter = 0

    start_index = 0

    for i, row in enumerate(rows):

        horizontal_speed = math.sqrt(
            row["vx_global"] ** 2
            +
            row["vy_global"] ** 2
        )

        airborne = (
            row["z"]
            >
            GROUND_Z + 0.70
        )

        vertical_stable = (
            abs(
                row["vz_global"]
            )
            <
            0.25
        )

        horizontal_stable = (
            horizontal_speed
            <
            0.25
        )

        if (
            airborne
            and
            vertical_stable
            and
            horizontal_stable
        ):

            stable_counter += 1

            if stable_counter >= 8:

                start_index = (
                    i -
                    7
                )

                break

        else:

            stable_counter = 0

    # --------------------------------------------------------
    # Find landing
    # --------------------------------------------------------

    end_index = len(rows) - 1

    low_counter = 0

    for i in range(
        start_index + 20,
        len(rows)
    ):

        if rows[i]["z"] < 0.55:

            low_counter += 1

            if low_counter >= 6:

                end_index = (
                    i -
                    5
                )

                break

        else:

            low_counter = 0

    if end_index <= start_index:

        end_index = len(rows) - 1

    return (
        start_index,
        end_index
    )


# ============================================================
# AUTOMATIC PLOT GENERATION
# ============================================================

def generate_plots():

    print("")
    print("==============================================")
    print("          GENERATING PLOTS")
    print("==============================================")

    try:

        import numpy as np
        import matplotlib.pyplot as plt

    except Exception as error:

        print(
            "WARNING: matplotlib/numpy import failed."
        )

        print(error)

        print(
            "Install matplotlib with:"
        )

        print(
            "python -m pip install matplotlib"
        )

        return

    if not os.path.exists(
        LEADER_CSV
    ):

        print(
            "Leader CSV not found."
        )

        return

    if not os.path.exists(
        FOLLOWER_CSV
    ):

        print(
            "Follower CSV not found."
        )

        return

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    def read_csv(path):

        with open(
            path,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            data = []

            for row in reader:

                converted = {}

                for key, value in row.items():

                    try:

                        converted[key] = float(
                            value
                        )

                    except Exception:

                        converted[key] = value

                data.append(
                    converted
                )

        return data

    try:

        leader = read_csv(
            LEADER_CSV
        )

        follower = read_csv(
            FOLLOWER_CSV
        )

    except Exception as error:

        print(
            "ERROR reading CSV files:"
        )

        print(error)

        return

    # --------------------------------------------------------
    # Extract replay data
    # --------------------------------------------------------

    follower_replay = [
        r for r in follower
        if r.get("state") == "REPLAY"
    ]

    if len(follower_replay) < 5:

        print(
            "Not enough follower replay data."
        )

        return

    leader_rows = [
        r for r in leader
        if all(
            key in r
            for key in [
                "x",
                "y",
                "z"
            ]
        )
    ]

    # --------------------------------------------------------
    # Determine leader replay window again
    # --------------------------------------------------------

    try:

        start_idx, end_idx = (
            find_replay_window(
                leader
            )
        )

        leader_replay = leader[
            start_idx:
            end_idx + 1
        ]

    except Exception:

        leader_replay = leader

    # --------------------------------------------------------
    # Leader replay time
    # --------------------------------------------------------

    leader_t0 = float(
        leader_replay[0]["time"]
    )

    t1 = np.array(
        [
            float(r["time"]) -
            leader_t0
            for r in leader_replay
        ]
    )

    x1 = np.array(
        [float(r["x"]) for r in leader_replay]
    )

    y1 = np.array(
        [float(r["y"]) for r in leader_replay]
    )

    z1 = np.array(
        [float(r["z"]) for r in leader_replay]
    )

    # --------------------------------------------------------
    # Follower replay time
    # --------------------------------------------------------

    follower_t0 = float(
        follower_replay[0]["time"]
    )

    t2 = np.array(
        [
            float(r["time"]) -
            follower_t0
            for r in follower_replay
        ]
    )

    x2 = np.array(
        [float(r["x"]) for r in follower_replay]
    )

    y2 = np.array(
        [float(r["y"]) for r in follower_replay]
    )

    z2 = np.array(
        [float(r["z"]) for r in follower_replay]
    )

    # --------------------------------------------------------
    # Interpolate leader at follower times
    # --------------------------------------------------------

    x1i = np.interp(
        t2,
        t1,
        x1
    )

    y1i = np.interp(
        t2,
        t1,
        y1
    )

    z1i = np.interp(
        t2,
        t1,
        z1
    )

    position_error = np.sqrt(
        (x1i - x2) ** 2
        +
        (y1i - y2) ** 2
        +
        (z1i - z2) ** 2
    )

    # ========================================================
    # 1 - 3D
    # ========================================================

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot(
        x1,
        y1,
        z1,
        label="CF1 Leader",
        linewidth=2
    )

    ax.plot(
        x2,
        y2,
        z2,
        label="CF2 Follower",
        linewidth=2
    )

    ax.scatter(
        [x1[0]],
        [y1[0]],
        [z1[0]],
        s=50,
        label="Start"
    )

    ax.scatter(
        [x1[-1]],
        [y1[-1]],
        [z1[-1]],
        s=50,
        label="End"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        "Absolute 3D Trajectory"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "01_3D_trajectory.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 2 - XY
    # ========================================================

    plt.figure(
        figsize=(9, 7)
    )

    plt.plot(
        x1,
        y1,
        label="CF1 Leader",
        linewidth=2
    )

    plt.plot(
        x2,
        y2,
        label="CF2 Follower",
        linewidth=2
    )

    plt.scatter(
        [x1[0]],
        [y1[0]],
        s=50,
        label="Common Start"
    )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    plt.title(
        "Absolute XY Trajectory"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "02_XY_trajectory.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 3 - ALTITUDE
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        t1,
        z1,
        label="CF1"
    )

    plt.plot(
        t2,
        z2,
        label="CF2"
    )

    plt.xlabel("Replay Time [s]")
    plt.ylabel("Z [m]")

    plt.title(
        "Altitude Comparison"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "03_altitude.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 4 - SPEED
    # ========================================================

    speed1 = np.array(
        [
            math.sqrt(
                float(r["vx_global"]) ** 2
                +
                float(r["vy_global"]) ** 2
                +
                float(r["vz_global"]) ** 2
            )
            for r in leader_replay
        ]
    )

    speed2 = np.array(
        [
            math.sqrt(
                float(r["vx_global"]) ** 2
                +
                float(r["vy_global"]) ** 2
                +
                float(r["vz_global"]) ** 2
            )
            for r in follower_replay
        ]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        t1,
        speed1,
        label="CF1"
    )

    plt.plot(
        t2,
        speed2,
        label="CF2"
    )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Speed [m/s]"
    )

    plt.title(
        "Speed Comparison"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "04_speed.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 5 - ROLL
    # ========================================================

    roll1 = np.array(
        [
            float(r["roll"])
            for r in leader_replay
        ]
    )

    roll2 = np.array(
        [
            float(r["roll"])
            for r in follower_replay
        ]
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t1,
        roll1,
        label="CF1"
    )

    plt.plot(
        t2,
        roll2,
        label="CF2"
    )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Roll [rad]"
    )

    plt.title(
        "Roll Comparison"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "05_roll.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 6 - PITCH
    # ========================================================

    pitch1 = np.array(
        [
            float(r["pitch"])
            for r in leader_replay
        ]
    )

    pitch2 = np.array(
        [
            float(r["pitch"])
            for r in follower_replay
        ]
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t1,
        pitch1,
        label="CF1"
    )

    plt.plot(
        t2,
        pitch2,
        label="CF2"
    )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Pitch [rad]"
    )

    plt.title(
        "Pitch Comparison"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "06_pitch.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 7 - YAW
    # ========================================================

    yaw1 = np.array(
        [
            float(r["yaw"])
            for r in leader_replay
        ]
    )

    yaw2 = np.array(
        [
            float(r["yaw"])
            for r in follower_replay
        ]
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        t1,
        yaw1,
        label="CF1"
    )

    plt.plot(
        t2,
        yaw2,
        label="CF2"
    )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Yaw [rad]"
    )

    plt.title(
        "Yaw Comparison"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "07_yaw.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 8 - ANGULAR VELOCITY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    for key, label in [
        ("wx", "Wx"),
        ("wy", "Wy"),
        ("wz", "Wz")
    ]:

        leader_values = np.array(
            [
                float(r[key])
                for r in leader_replay
            ]
        )

        follower_values = np.array(
            [
                float(r[key])
                for r in follower_replay
            ]
        )

        plt.plot(
            t1,
            leader_values,
            label=f"CF1 {label}"
        )

        plt.plot(
            t2,
            follower_values,
            "--",
            label=f"CF2 {label}"
        )

    plt.xlabel(
        "Replay Time [s]"
    )

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
            PLOTS_DIR,
            "08_angular_velocity.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 9 - POSITION ERROR
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        t2,
        position_error,
        linewidth=2
    )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "3D Position Error [m]"
    )

    plt.title(
        "CF1-CF2 Position Tracking Error"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "09_position_error.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 10 - ERROR XYZ
    # ========================================================

    ex = x1i - x2
    ey = y1i - y2
    ez = z1i - z2

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        t2,
        ex,
        label="Ex"
    )

    plt.plot(
        t2,
        ey,
        label="Ey"
    )

    plt.plot(
        t2,
        ez,
        label="Ez"
    )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Error [m]"
    )

    plt.title(
        "Position Tracking Errors"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "10_position_xyz_error.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 11 - COMMANDS
    # ========================================================

    command_names = [
        (
            "forward_desired",
            "Forward"
        ),
        (
            "sideways_desired",
            "Sideways"
        ),
        (
            "yaw_desired",
            "Yaw"
        )
    ]

    plt.figure(
        figsize=(10, 7)
    )

    for key, label in command_names:

        values = np.array(
            [
                float(r[key])
                for r in leader_replay
            ]
        )

        plt.plot(
            t1,
            values,
            label=label
        )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Command"
    )

    plt.title(
        "Leader Flight Commands"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "11_leader_commands.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # 12 - MOTOR COMMANDS
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    for key in [
        "motor1",
        "motor2",
        "motor3",
        "motor4"
    ]:

        values = np.array(
            [
                float(r[key])
                for r in leader_replay
            ]
        )

        plt.plot(
            t1,
            values,
            label=f"CF1 {key}"
        )

    plt.xlabel(
        "Replay Time [s]"
    )

    plt.ylabel(
        "Motor Command"
    )

    plt.title(
        "CF1 Motor Commands"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "12_motor_commands.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # METRICS
    # ========================================================

    rmse = math.sqrt(
        float(
            np.mean(
                position_error ** 2
            )
        )
    )

    maximum_error = float(
        np.max(
            position_error
        )
    )

    final_error = float(
        position_error[-1]
    )

    metrics_file = os.path.join(
        PLOTS_DIR,
        "tracking_metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CRAZYFLIE LEADER-FOLLOWER METRICS\n"
        )

        file.write(
            "===================================\n\n"
        )

        file.write(
            f"Position RMSE = {rmse:.6f} m\n"
        )

        file.write(
            f"Maximum Position Error = "
            f"{maximum_error:.6f} m\n"
        )

        file.write(
            f"Final Position Error = "
            f"{final_error:.6f} m\n"
        )

    print(
        "Plots generated:"
    )

    print(
        PLOTS_DIR
    )

    print(
        f"Position RMSE = {rmse:.4f} m"
    )

    print(
        f"Maximum Error = {maximum_error:.4f} m"
    )

    print(
        f"Final Error = {final_error:.4f} m"
    )

    print(
        "=============================================="
    )


# ============================================================
# ROBOT
# ============================================================

robot = Robot()

timestep = int(
    robot.getBasicTimeStep()
)


# ============================================================
# MOTORS
# ============================================================

m1_motor = robot.getDevice(
    "m1_motor"
)

m2_motor = robot.getDevice(
    "m2_motor"
)

m3_motor = robot.getDevice(
    "m3_motor"
)

m4_motor = robot.getDevice(
    "m4_motor"
)

for motor in [
    m1_motor,
    m2_motor,
    m3_motor,
    m4_motor
]:

    motor.setPosition(
        float("inf")
    )

    motor.setVelocity(
        0.0
    )


# ============================================================
# SENSORS
# ============================================================

imu = robot.getDevice(
    "inertial_unit"
)

gps = robot.getDevice(
    "gps"
)

gyro = robot.getDevice(
    "gyro"
)

imu.enable(timestep)
gps.enable(timestep)
gyro.enable(timestep)


# ============================================================
# REMOVE OLD FLAG
# ============================================================

try:

    if os.path.exists(
        START_FLAG
    ):

        os.remove(
            START_FLAG
        )

except Exception:

    pass


# ============================================================
# CREATE DATA DIRECTORY
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# FOLLOWER CSV
# ============================================================

log = open(
    FOLLOWER_CSV,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(log)

writer.writerow([
    "time",

    "replay_time",

    "x",
    "y",
    "z",

    "vx_global",
    "vy_global",
    "vz_global",

    "speed",

    "roll",
    "pitch",
    "yaw",

    "wx",
    "wy",
    "wz",

    "angular_speed",

    "linear_acceleration",
    "angular_acceleration",

    "x_ref",
    "y_ref",
    "z_ref",

    "vx_ref",
    "vy_ref",
    "vz_ref",

    "forward_ref",
    "sideways_ref",
    "yaw_ref",

    "desired_forward",
    "desired_sideways",
    "desired_yaw",
    "desired_height",

    "position_error",

    "motor1",
    "motor2",
    "motor3",
    "motor4",

    "state"
])


# ============================================================
# PID
# ============================================================

pid = pid_velocity_fixed_height_controller()


# ============================================================
# INITIAL VARIABLES
# ============================================================

state = "WAIT"

home_x = None
home_y = None
home_z = None

trajectory = None

leader_rows = None

start_index = None
end_index = None

leader_start_x = None
leader_start_y = None
leader_start_z = None
leader_start_yaw = None

leader_final_x = None
leader_final_y = None

final_target_x = None
final_target_y = None

replay_time = 0.0

height_command = GROUND_Z

stable_timer = 0.0

plots_generated = False

last_time = None

last_x = None
last_y = None
last_z = None

last_vx = 0.0
last_vy = 0.0
last_vz = 0.0

last_wx = 0.0
last_wy = 0.0
last_wz = 0.0

last_print_time = -1.0


# ============================================================
# MAIN LOOP
# ============================================================

print("")
print("==============================================")
print("       CRAZYFLIE 2 - FOLLOWER")
print("==============================================")
print("ABSOLUTE TRAJECTORY TRACKING")
print("CF2 aligns to CF1 START point")
print("Feed-forward + Position Feedback")
print("Automatic Plot Generation")
print("==============================================")


while robot.step(timestep) != -1:

    now = robot.getTime()

    # ========================================================
    # DT
    # ========================================================

    if last_time is None:

        dt = timestep / 1000.0

    else:

        dt = now - last_time

    dt = max(
        dt,
        0.001
    )

    # ========================================================
    # SENSORS
    # ========================================================

    gps_values = gps.getValues()

    imu_values = imu.getRollPitchYaw()

    gyro_values = gyro.getValues()

    x = float(
        gps_values[0]
    )

    y = float(
        gps_values[1]
    )

    z = float(
        gps_values[2]
    )

    roll = float(
        imu_values[0]
    )

    pitch = float(
        imu_values[1]
    )

    yaw = float(
        imu_values[2]
    )

    wx = float(
        gyro_values[0]
    )

    wy = float(
        gyro_values[1]
    )

    wz = float(
        gyro_values[2]
    )

    # ========================================================
    # VALID SENSOR DATA
    # ========================================================

    valid = all(
        finite(v)
        for v in [
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            wx,
            wy,
            wz
        ]
    )

    if not valid:

        continue

    # ========================================================
    # VELOCITY
    # ========================================================

    if last_time is None:

        vx = 0.0
        vy = 0.0
        vz = 0.0

    else:

        vx = (
            x -
            last_x
        ) / dt

        vy = (
            y -
            last_y
        ) / dt

        vz = (
            z -
            last_z
        ) / dt

    vx = clamp(
        vx,
        -4.0,
        4.0
    )

    vy = clamp(
        vy,
        -4.0,
        4.0
    )

    vz = clamp(
        vz,
        -4.0,
        4.0
    )

    speed = math.sqrt(
        vx * vx +
        vy * vy +
        vz * vz
    )

    angular_speed = math.sqrt(
        wx * wx +
        wy * wy +
        wz * wz
    )

    # ========================================================
    # ACCELERATION
    # ========================================================

    if last_time is None:

        linear_acceleration = 0.0

        angular_acceleration = 0.0

    else:

        ax_est = (
            vx -
            last_vx
        ) / dt

        ay_est = (
            vy -
            last_vy
        ) / dt

        az_est = (
            vz -
            last_vz
        ) / dt

        linear_acceleration = math.sqrt(
            ax_est ** 2 +
            ay_est ** 2 +
            az_est ** 2
        )

        alpha_x = (
            wx -
            last_wx
        ) / dt

        alpha_y = (
            wy -
            last_wy
        ) / dt

        alpha_z = (
            wz -
            last_wz
        ) / dt

        angular_acceleration = math.sqrt(
            alpha_x ** 2 +
            alpha_y ** 2 +
            alpha_z ** 2
        )

    # ========================================================
    # HOME
    # ========================================================

    if home_x is None:

        home_x = x
        home_y = y
        home_z = z

        print("")
        print(
            ">>> CF2 HOME POSITION"
        )

        print(
            f"X={home_x:.4f} "
            f"Y={home_y:.4f} "
            f"Z={home_z:.4f}"
        )

    # ========================================================
    # DEFAULT COMMANDS
    # ========================================================

    desired_forward = 0.0
    desired_sideways = 0.0
    desired_yaw_rate = 0.0

    desired_altitude = height_command

    x_ref = float("nan")
    y_ref = float("nan")
    z_ref = float("nan")

    vx_ref = 0.0
    vy_ref = 0.0
    vz_ref = 0.0

    forward_ref = 0.0
    sideways_ref = 0.0
    yaw_ref = 0.0

    # ========================================================
    # STATE: WAIT
    # ========================================================

    if state == "WAIT":

        height_command = (
            GROUND_Z
        )

        desired_altitude = (
            height_command
        )

        desired_forward = 0.0
        desired_sideways = 0.0
        desired_yaw_rate = 0.0

        # ----------------------------------------------------
        # Wait for CF1 to finish
        # ----------------------------------------------------

        if os.path.exists(
            START_FLAG
        ):

            try:

                leader_rows = (
                    load_leader_csv()
                )

                (
                    start_index,
                    end_index
                ) = find_replay_window(
                    leader_rows
                )

                leader_start = (
                    leader_rows[
                        start_index
                    ]
                )

                leader_start_x = (
                    leader_start["x"]
                )

                leader_start_y = (
                    leader_start["y"]
                )

                leader_start_z = (
                    leader_start["z"]
                )

                leader_start_yaw = (
                    leader_start["yaw"]
                )

                final_row = (
                    leader_rows[-1]
                )

                leader_final_x = (
                    final_row["x"]
                )

                leader_final_y = (
                    final_row["y"]
                )

                final_target_x = (
                    leader_final_x
                )

                final_target_y = (
                    leader_final_y
                )

                replay_rows = (
                    leader_rows[
                        start_index:
                        end_index + 1
                    ]
                )

                trajectory = Trajectory(
                    replay_rows
                )

                print("")
                print(
                    "=============================================="
                )

                print(
                    "          LEADER DATA LOADED"
                )

                print(
                    "=============================================="
                )

                print(
                    "CF1 REPLAY START:"
                )

                print(
                    f"X={leader_start_x:.4f}"
                )

                print(
                    f"Y={leader_start_y:.4f}"
                )

                print(
                    f"Z={leader_start_z:.4f}"
                )

                print(
                    f"Yaw={leader_start_yaw:.4f}"
                )

                print("")

                print(
                    "CF1 FINAL POSITION:"
                )

                print(
                    f"X={leader_final_x:.4f}"
                )

                print(
                    f"Y={leader_final_y:.4f}"
                )

                print("")

                print(
                    "CF2 HOME:"
                )

                print(
                    f"X={home_x:.4f}"
                )

                print(
                    f"Y={home_y:.4f}"
                )

                print("")

                print(
                    f"Replay duration = "
                    f"{trajectory.duration:.3f} s"
                )

                # Start at current altitude
                height_command = z

                pid.reset()

                stable_timer = 0.0

                state = "TAKEOFF"

                print(
                    ">>> CF2 STATE = TAKEOFF"
                )

            except Exception as error:

                print(
                    "ERROR:"
                )

                print(error)

    # ========================================================
    # STATE: TAKEOFF
    # ========================================================

    elif state == "TAKEOFF":

        desired_forward = 0.0
        desired_sideways = 0.0
        desired_yaw_rate = 0.0

        height_command = move_toward(
            height_command,
            TAKEOFF_HEIGHT,
            TAKEOFF_ALTITUDE_RATE * dt
        )

        desired_altitude = (
            height_command
        )

        stable_condition = (
            abs(
                z -
                TAKEOFF_HEIGHT
            )
            <
            0.08
            and
            abs(vz)
            <
            0.12
            and
            abs(roll)
            <
            0.08
            and
            abs(pitch)
            <
            0.08
        )

        if stable_condition:

            stable_timer += dt

        else:

            stable_timer = 0.0

        if (
            now -
            last_print_time
            >
            0.50
        ):

            print(
                f"TAKEOFF: "
                f"Z={z:.3f} "
                f"Target={height_command:.3f} "
                f"Vz={vz:.3f}"
            )

            last_print_time = now

        if stable_timer >= 0.8:

            stable_timer = 0.0

            state = "ALIGN_XY"

            pid.reset()

            print("")
            print(
                ">>> TAKEOFF STABLE"
            )

            print(
                ">>> CF2 STATE = ALIGN_XY"
            )

            print(
                f"Target X={leader_start_x:.3f} "
                f"Y={leader_start_y:.3f}"
            )

    # ========================================================
    # STATE: ALIGN_XY
    # ========================================================

    elif state == "ALIGN_XY":

        error_x = (
            leader_start_x -
            x
        )

        error_y = (
            leader_start_y -
            y
        )

        desired_vx_global = (
            ALIGN_XY_KP *
            error_x
        )

        desired_vy_global = (
            ALIGN_XY_KP *
            error_y
        )

        desired_vx_global, desired_vy_global = (
            limit_vector(
                desired_vx_global,
                desired_vy_global,
                ALIGN_XY_SPEED
            )
        )

        # world -> body

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        desired_forward = (
            desired_vx_global *
            cos_yaw
            +
            desired_vy_global *
            sin_yaw
        )

        desired_sideways = (
            -desired_vx_global *
            sin_yaw
            +
            desired_vy_global *
            cos_yaw
        )

        desired_yaw_rate = 0.0

        # Keep safe altitude

        height_command = move_toward(
            height_command,
            TAKEOFF_HEIGHT,
            0.30 * dt
        )

        desired_altitude = (
            height_command
        )

        position_error_xy = math.sqrt(
            error_x ** 2 +
            error_y ** 2
        )

        velocity_xy = math.sqrt(
            vx ** 2 +
            vy ** 2
        )

        if (
            position_error_xy
            <
            START_POSITION_TOLERANCE
            and
            velocity_xy
            <
            START_VELOCITY_TOLERANCE
        ):

            stable_timer += dt

        else:

            stable_timer = 0.0

        if (
            now -
            last_print_time
            >
            0.50
        ):

            print(
                f"ALIGN_XY: "
                f"X={x:.3f}/"
                f"{leader_start_x:.3f} "
                f"Y={y:.3f}/"
                f"{leader_start_y:.3f} "
                f"Err={position_error_xy:.3f}"
            )

            last_print_time = now

        if stable_timer >= 0.50:

            state = "ALIGN_Z_YAW"

            stable_timer = 0.0

            pid.reset()

            print("")
            print(
                ">>> XY ALIGNMENT COMPLETE"
            )

            print(
                ">>> CF2 STATE = ALIGN_Z_YAW"
            )

    # ========================================================
    # STATE: ALIGN_Z_YAW
    # ========================================================

    elif state == "ALIGN_Z_YAW":

        # ----------------------------------------------------
        # Keep exact XY
        # ----------------------------------------------------

        error_x = (
            leader_start_x -
            x
        )

        error_y = (
            leader_start_y -
            y
        )

        desired_vx_global = (
            0.60 *
            error_x
        )

        desired_vy_global = (
            0.60 *
            error_y
        )

        desired_vx_global, desired_vy_global = (
            limit_vector(
                desired_vx_global,
                desired_vy_global,
                0.18
            )
        )

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        desired_forward = (
            desired_vx_global * cy
            +
            desired_vy_global * sy
        )

        desired_sideways = (
            -desired_vx_global * sy
            +
            desired_vy_global * cy
        )

        # ----------------------------------------------------
        # Z alignment
        # ----------------------------------------------------

        z_error = (
            leader_start_z -
            z
        )

        height_target = (
            leader_start_z
            +
            ALIGN_Z_KP *
            z_error
        )

        height_target = clamp(
            height_target,
            0.75,
            1.90
        )

        height_command = move_toward(
            height_command,
            height_target,
            0.35 * dt
        )

        desired_altitude = (
            height_command
        )

        # ----------------------------------------------------
        # Yaw alignment
        # ----------------------------------------------------

        yaw_error = normalize_angle(
            leader_start_yaw -
            yaw
        )

        desired_yaw_rate = clamp(
            ALIGN_YAW_KP *
            yaw_error,
            -ALIGN_YAW_MAX_RATE,
            ALIGN_YAW_MAX_RATE
        )

        position_error_xy = math.sqrt(
            error_x ** 2 +
            error_y ** 2
        )

        if (
            position_error_xy
            <
            START_POSITION_TOLERANCE
            and
            abs(z_error)
            <
            START_Z_TOLERANCE
            and
            abs(yaw_error)
            <
            START_YAW_TOLERANCE
            and
            abs(vx)
            <
            START_VELOCITY_TOLERANCE
            and
            abs(vy)
            <
            START_VELOCITY_TOLERANCE
            and
            abs(vz)
            <
            START_VELOCITY_TOLERANCE
        ):

            stable_timer += dt

        else:

            stable_timer = 0.0

        if (
            now -
            last_print_time
            >
            0.50
        ):

            print(
                f"ALIGN START: "
                f"XYerr={position_error_xy:.3f} "
                f"Zerr={z_error:.3f} "
                f"YawErr={yaw_error:.3f}"
            )

            last_print_time = now

        if stable_timer >= 0.70:

            state = "START_STABLE"

            stable_timer = 0.0

            pid.reset()

            print("")
            print(
                "=============================================="
            )

            print(
                "       CF2 AT EXACT CF1 START"
            )

            print(
                f"X = {x:.4f}"
            )

            print(
                f"Y = {y:.4f}"
            )

            print(
                f"Z = {z:.4f}"
            )

            print(
                f"Yaw = {yaw:.4f}"
            )

            print(
                "=============================================="
            )

            print(
                ">>> CF2 STATE = START_STABLE"
            )

    # ========================================================
    # STATE: START_STABLE
    # ========================================================

    elif state == "START_STABLE":

        # ----------------------------------------------------
        # Hold the exact start position
        # ----------------------------------------------------

        error_x = (
            leader_start_x -
            x
        )

        error_y = (
            leader_start_y -
            y
        )

        vx_global_cmd = (
            0.70 *
            error_x
        )

        vy_global_cmd = (
            0.70 *
            error_y
        )

        vx_global_cmd, vy_global_cmd = (
            limit_vector(
                vx_global_cmd,
                vy_global_cmd,
                0.12
            )
        )

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        desired_forward = (
            vx_global_cmd * cy
            +
            vy_global_cmd * sy
        )

        desired_sideways = (
            -vx_global_cmd * sy
            +
            vy_global_cmd * cy
        )

        yaw_error = normalize_angle(
            leader_start_yaw -
            yaw
        )

        desired_yaw_rate = clamp(
            0.60 *
            yaw_error,
            -0.15,
            0.15
        )

        height_command = move_toward(
            height_command,
            leader_start_z,
            0.25 * dt
        )

        desired_altitude = (
            height_command
        )

        stable_condition = (
            abs(
                leader_start_x -
                x
            )
            <
            START_POSITION_TOLERANCE
            and
            abs(
                leader_start_y -
                y
            )
            <
            START_POSITION_TOLERANCE
            and
            abs(
                leader_start_z -
                z
            )
            <
            START_Z_TOLERANCE
            and
            abs(
                yaw_error
            )
            <
            START_YAW_TOLERANCE
            and
            math.sqrt(
                vx * vx +
                vy * vy +
                vz * vz
            )
            <
            START_VELOCITY_TOLERANCE
        )

        if stable_condition:

            stable_timer += dt

        else:

            stable_timer = 0.0

        if stable_timer >= START_STABLE_TIME:

            replay_time = 0.0

            state = "REPLAY"

            pid.reset()

            print("")
            print(
                "=============================================="
            )

            print(
                "        REPLAY STARTED"
            )

            print(
                "        ABSOLUTE TRAJECTORY"
            )

            print(
                "=============================================="
            )

            print(
                ">>> CF2 STATE = REPLAY"
            )

    # ========================================================
    # STATE: REPLAY
    # ========================================================

    elif state == "REPLAY":

        replay_time += dt

        ref = trajectory.sample(
            replay_time
        )

        # ----------------------------------------------------
        # Reference data
        # ----------------------------------------------------

        x_ref = ref["x"]
        y_ref = ref["y"]
        z_ref = ref["z"]

        vx_ref = ref["vx_global"]
        vy_ref = ref["vy_global"]
        vz_ref = ref["vz_global"]

        forward_ref = (
            ref["forward_desired"]
        )

        sideways_ref = (
            ref["sideways_desired"]
        )

        yaw_ref = (
            ref["yaw"]
        )

        # ----------------------------------------------------
        # Position feedback
        # ----------------------------------------------------

        error_x = (
            x_ref -
            x
        )

        error_y = (
            y_ref -
            y
        )

        # ----------------------------------------------------
        # Feedforward + Feedback
        # ----------------------------------------------------

        vx_global_cmd = (
            REPLAY_FEED_FORWARD *
            vx_ref
            +
            REPLAY_POSITION_KP *
            error_x
        )

        vy_global_cmd = (
            REPLAY_FEED_FORWARD *
            vy_ref
            +
            REPLAY_POSITION_KP *
            error_y
        )

        vx_global_cmd, vy_global_cmd = (
            limit_vector(
                vx_global_cmd,
                vy_global_cmd,
                REPLAY_MAX_SPEED
            )
        )

        # ----------------------------------------------------
        # Global -> Body
        # ----------------------------------------------------

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        desired_forward = (
            vx_global_cmd * cy
            +
            vy_global_cmd * sy
        )

        desired_sideways = (
            -vx_global_cmd * sy
            +
            vy_global_cmd * cy
        )

        desired_forward = clamp(
            desired_forward,
            -REPLAY_MAX_SPEED,
            REPLAY_MAX_SPEED
        )

        desired_sideways = clamp(
            desired_sideways,
            -REPLAY_MAX_SPEED,
            REPLAY_MAX_SPEED
        )

        # ----------------------------------------------------
        # Yaw control
        # ----------------------------------------------------

        yaw_error = normalize_angle(
            yaw_ref -
            yaw
        )

        desired_yaw_rate = clamp(
            0.60 *
            yaw_error
            +
            0.15 *
            ref["yaw_desired"],
            -0.35,
            0.35
        )

        # ----------------------------------------------------
        # Altitude control
        # ----------------------------------------------------

        z_error = (
            z_ref -
            z
        )

        altitude_target = (
            z_ref
            +
            REPLAY_ALTITUDE_KP *
            z_error
            +
            0.10 *
            vz_ref
        )

        altitude_target = clamp(
            altitude_target,
            REPLAY_MIN_HEIGHT,
            REPLAY_MAX_HEIGHT
        )

        height_command = move_toward(
            height_command,
            altitude_target,
            0.60 * dt
        )

        desired_altitude = (
            height_command
        )

        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        position_error = math.sqrt(
            error_x ** 2 +
            error_y ** 2 +
            z_error ** 2
        )

        # ----------------------------------------------------
        # Low altitude safety
        # ----------------------------------------------------

        if z < LOW_ALTITUDE_LIMIT:

            desired_forward = 0.0
            desired_sideways = 0.0
            desired_yaw_rate = 0.0

            height_command = max(
                height_command,
                LOW_ALTITUDE_RECOVERY
            )

            desired_altitude = (
                height_command
            )

            if (
                now -
                last_print_time
                >
                1.0
            ):

                print(
                    "WARNING: CF2 LOW ALTITUDE"
                )

                last_print_time = now

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            now -
            last_print_time
            >
            2.0
        ):

            print(
                f"REPLAY "
                f"t={replay_time:.1f}/"
                f"{trajectory.duration:.1f} "
                f"X={x:.2f}/"
                f"{x_ref:.2f} "
                f"Y={y:.2f}/"
                f"{y_ref:.2f} "
                f"Z={z:.2f}/"
                f"{z_ref:.2f} "
                f"Err={position_error:.3f}"
            )

            last_print_time = now

        # ----------------------------------------------------
        # End of trajectory
        # ----------------------------------------------------

        if (
            replay_time
            >=
            trajectory.duration
        ):

            state = "FINAL_APPROACH"

            stable_timer = 0.0

            pid.reset()

            print("")
            print(
                ">>> REPLAY FINISHED"
            )

            print(
                ">>> CF2 STATE = FINAL_APPROACH"
            )

    # ========================================================
    # STATE: FINAL_APPROACH
    # ========================================================

    elif state == "FINAL_APPROACH":

        error_x = (
            final_target_x -
            x
        )

        error_y = (
            final_target_y -
            y
        )

        desired_vx_global = (
            FINAL_POSITION_KP *
            error_x
        )

        desired_vy_global = (
            FINAL_POSITION_KP *
            error_y
        )

        desired_vx_global, desired_vy_global = (
            limit_vector(
                desired_vx_global,
                desired_vy_global,
                FINAL_MAX_SPEED
            )
        )

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        desired_forward = (
            desired_vx_global * cy
            +
            desired_vy_global * sy
        )

        desired_sideways = (
            -desired_vx_global * sy
            +
            desired_vy_global * cy
        )

        desired_yaw_rate = 0.0

        height_command = move_toward(
            height_command,
            FINAL_HOVER_HEIGHT,
            FINAL_ALTITUDE_RATE * dt
        )

        desired_altitude = (
            height_command
        )

        horizontal_error = math.sqrt(
            error_x ** 2 +
            error_y ** 2
        )

        stable_condition = (
            horizontal_error
            <
            0.08
            and
            abs(vx)
            <
            0.08
            and
            abs(vy)
            <
            0.08
            and
            abs(
                z -
                FINAL_HOVER_HEIGHT
            )
            <
            0.08
            and
            abs(vz)
            <
            0.08
        )

        if stable_condition:

            stable_timer += dt

        else:

            stable_timer = 0.0

        if (
            stable_timer
            >=
            1.50
        ):

            state = "FINAL_HOVER"

            stable_timer = 0.0

            height_command = (
                FINAL_HOVER_HEIGHT
            )

            desired_altitude = (
                FINAL_HOVER_HEIGHT
            )

            print("")
            print(
                "=============================================="
            )

            print(
                "          MISSION COMPLETE"
            )

            print(
                "          FINAL HOVER"
            )

            print(
                f"Final target X="
                f"{final_target_x:.4f}"
            )

            print(
                f"Final target Y="
                f"{final_target_y:.4f}"
            )

            print(
                f"Hover Z="
                f"{FINAL_HOVER_HEIGHT:.4f}"
            )

            print(
                "=============================================="
            )

    # ========================================================
    # STATE: FINAL_HOVER
    # ========================================================

    elif state == "FINAL_HOVER":

        error_x = (
            final_target_x -
            x
        )

        error_y = (
            final_target_y -
            y
        )

        desired_vx_global = (
            0.50 *
            error_x
        )

        desired_vy_global = (
            0.50 *
            error_y
        )

        desired_vx_global, desired_vy_global = (
            limit_vector(
                desired_vx_global,
                desired_vy_global,
                0.15
            )
        )

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        desired_forward = (
            desired_vx_global * cy
            +
            desired_vy_global * sy
        )

        desired_sideways = (
            -desired_vx_global * sy
            +
            desired_vy_global * cy
        )

        desired_yaw_rate = 0.0

        height_command = (
            FINAL_HOVER_HEIGHT
        )

        desired_altitude = (
            FINAL_HOVER_HEIGHT
        )

        # ----------------------------------------------------
        # Generate plots once
        # ----------------------------------------------------

        if not plots_generated:

            log.flush()

            generate_plots()

            plots_generated = True

    # ========================================================
    # PID
    # ========================================================

    motor_power = pid.pid(
        dt,

        desired_forward,
        desired_sideways,
        desired_yaw_rate,

        desired_altitude,

        roll,
        pitch,
        wz,

        z,

        # Body velocity
        vx * math.cos(yaw)
        +
        vy * math.sin(yaw),

        -vx * math.sin(yaw)
        +
        vy * math.cos(yaw)
    )

    # ========================================================
    # MOTOR SIGNS
    # ========================================================

    m1 = motor_power[0]
    m2 = motor_power[1]
    m3 = motor_power[2]
    m4 = motor_power[3]

    # Motors off while waiting
    if state == "WAIT":

        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0

    m1_motor.setVelocity(
        -m1
    )

    m2_motor.setVelocity(
        m2
    )

    m3_motor.setVelocity(
        -m3
    )

    m4_motor.setVelocity(
        m4
    )

    # ========================================================
    # LOG
    # ========================================================

    writer.writerow([
        now,

        replay_time,

        x,
        y,
        z,

        vx,
        vy,
        vz,

        speed,

        roll,
        pitch,
        yaw,

        wx,
        wy,
        wz,

        angular_speed,

        linear_acceleration,
        angular_acceleration,

        x_ref,
        y_ref,
        z_ref,

        vx_ref,
        vy_ref,
        vz_ref,

        forward_ref,
        sideways_ref,
        yaw_ref,

        desired_forward,
        desired_sideways,
        desired_yaw_rate,
        desired_altitude,

        (
            math.sqrt(
                (
                    x_ref - x
                ) ** 2
                +
                (
                    y_ref - y
                ) ** 2
                +
                (
                    z_ref - z
                ) ** 2
            )
            if
            finite(x_ref)
            and
            finite(y_ref)
            and
            finite(z_ref)
            else
            float("nan")
        ),

        m1,
        m2,
        m3,
        m4,

        state
    ])

    # ========================================================
    # UPDATE MEMORY
    # ========================================================

    last_time = now

    last_x = x
    last_y = y
    last_z = z

    last_vx = vx
    last_vy = vy
    last_vz = vz

    last_wx = wx
    last_wy = wy
    last_wz = wz

    # ========================================================
    # FLUSH
    # ========================================================

    if int(now * 10) % 5 == 0:

        try:
            log.flush()
        except Exception:
            pass


# ============================================================
# CLOSE
# ============================================================

try:
    log.flush()
    log.close()
except Exception:
    pass

print(
    "CF2 controller stopped."
)