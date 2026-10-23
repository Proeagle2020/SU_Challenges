# -*- coding: utf-8 -*-
"""
CCN.py
Four-Crazyflie sequential obstacle-navigation controller for Webots R2025a.

Coordinate convention:
    +X = forward
    +Y = left
    +Z = up

Initial formation:
    D1 = (0,0)
    D2 = (0,1)
    D3 = (1,0)
    D4 = (1,1)

Motion sequence:
    D3 -> D4 -> D1 -> D2

Mission geometry:
    Column 1 = (2,0.5), diameter 0.1 m, height 2 m
    Column 2 = (3,0.5), diameter 0.1 m, height 2 m

    Wall 1 center = (5,0.2), length 1 m, thickness 0.05 m
    Wall 2 center = (5,0.7), length 1 m, thickness 0.05 m

The controller uses:
    - GPS for global position/velocity feedback
    - IMU + gyro for attitude/yaw
    - Camera + OpenCV for obstacle confirmation
    - MultiRanger for close-range safety and wall-centering
    - File flags ONLY for one-time mission synchronization.
      No os.replace()/temporary status files are used, so the
      previous WinError 5 race is avoided.
"""

from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller

from math import cos, sin, sqrt, atan2, isfinite
from pathlib import Path
import csv
import os
import re
import shutil
import time as wallclock

import numpy as np
import cv2


# ============================================================
# SETTINGS
# ============================================================

TIME_STEP = 32

FLYING_ALTITUDE = 1.0
LAND_HEIGHT = 0.05

# Faster than the previous 0.20 m/s version,
# but still conservative enough for the obstacle course.
MAX_HORIZONTAL_SPEED = 0.45

KP_POS = 1.15
KP_YAW = 1.0

SETTLE_TIME = 1.0
ALTITUDE_THRESHOLD = 0.75

# Final formation = same square, shifted +6 m in X.
FINAL_X_SHIFT = 6.0


# ============================================================
# SEQUENCE
# ============================================================

START_ORDER = [3, 4, 1, 2]

PREVIOUS_DRONE = {
    3: None,
    4: 3,
    1: 4,
    2: 1,
}


# ============================================================
# INITIAL POSITIONS
# ============================================================

START_POSITIONS = {
    1: (0.0, 0.0),
    2: (0.0, 1.0),
    3: (1.0, 0.0),
    4: (1.0, 1.0),
}


# ============================================================
# OBSTACLE GEOMETRY
# ============================================================

COLUMN_1_X = 2.0
COLUMN_2_X = 3.0
COLUMN_Y = 0.5

COLUMN_GAP_CENTER_X = 2.5

# Exit after passing the two columns.
COLUMN_EXIT_X = 3.55

# Walls are centered at X=5.0 and extend 0.5 m on each side
# because their length is 1 m.
WALL_CENTER_X = 5.0

WALL_1_Y = 0.2
WALL_2_Y = 0.7

WALL_PASS_Y = 0.45

WALL_ENTRY_X = 4.35
WALL_EXIT_X = 5.65


# ============================================================
# WAYPOINTS
# ============================================================

def build_targets(start_x, start_y, final_x, final_y):
    """
    The mission is deliberately explicit:

        1) move to the space between the two columns
        2) cross the longitudinal gap between them
        3) move to the wall corridor
        4) pass between the two walls
        5) form the final square
    """

    return [
        # Approach the region in front of the two columns.
        (COLUMN_GAP_CENTER_X, start_y),

        # Move into the exact space between the two columns.
        (COLUMN_GAP_CENTER_X, COLUMN_Y),

        # Exit from between the two columns.
        (COLUMN_EXIT_X, COLUMN_Y),

        # Enter the corridor between the two walls.
        (WALL_ENTRY_X, WALL_PASS_Y),

        # Pass through the wall corridor.
        (WALL_EXIT_X, WALL_PASS_Y),

        # Final square position.
        (final_x, final_y),
    ]


# ============================================================
# ROBOT / ID
# ============================================================

robot = Robot()

timestep = int(robot.getBasicTimeStep())
robot_name = robot.getName()

match = re.search(r"(\d+)$", robot_name)

if match:
    drone_id = int(match.group(1))
else:
    drone_id = 1

if drone_id not in START_POSITIONS:
    print("ERROR: invalid drone ID")
    raise SystemExit


start_x, start_y = START_POSITIONS[drone_id]

final_x = start_x + FINAL_X_SHIFT
final_y = start_y

targets = build_targets(
    start_x,
    start_y,
    final_x,
    final_y,
)

print()
print("================================================")
print(f"Robot       : {robot_name}")
print(f"Drone ID    : {drone_id}")
print(f"Start       : ({start_x:.2f}, {start_y:.2f})")
print(f"Final       : ({final_x:.2f}, {final_y:.2f})")
print(f"Order       : {START_ORDER}")
print("================================================")
print()


# ============================================================
# SHARED MISSION SYNCHRONIZATION
# ============================================================

controller_dir = Path(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

controllers_dir = controller_dir.parent

sync_root = controllers_dir / "_mission_sync"

sync_root.mkdir(
    parents=True,
    exist_ok=True
)


def remove_old_missions():
    """
    Drone 3 starts a new Webots run.

    Delete old mission directories so a controller that starts
    early cannot accidentally attach to a previous simulation.
    """

    try:

        for path in sync_root.glob("mission_*"):

            if not path.is_dir():
                continue

            try:

                shutil.rmtree(
                    path,
                    ignore_errors=True
                )

            except OSError:
                pass

    except OSError:
        pass


def create_new_mission_for_drone3():

    # Remove old runs FIRST.
    remove_old_missions()

    mission_name = (
        f"mission_{wallclock.time_ns()}"
    )

    mission_dir_local = (
        sync_root / mission_name
    )

    mission_dir_local.mkdir(
        parents=True,
        exist_ok=False
    )

    ready_file = (
        mission_dir_local
        /
        "MISSION_READY.flag"
    )

    with open(
        ready_file,
        "x",
        encoding="utf-8"
    ) as f:

        f.write(
            f"mission={mission_name}\n"
        )

    return mission_dir_local


def find_current_mission():

    """
    Find the newest mission created by D3.

    IMPORTANT:
    We intentionally do NOT compare folder timestamps with the
    startup time of the current controller. That was the bug
    that caused Drone 4 to miss the new mission when D3 created
    it slightly earlier.

    If D4/D1/D2 initially attach to an old folder just before
    D3 deletes it, refresh_mission_if_deleted() below will make
    them rescan and attach to the new run.
    """

    deadline = wallclock.time() + 60.0

    while wallclock.time() < deadline:

        candidates = []

        try:

            for path in sync_root.glob(
                "mission_*"
            ):

                if not path.is_dir():
                    continue

                if not (
                    path /
                    "MISSION_READY.flag"
                ).exists():
                    continue

                candidates.append(path)

        except OSError:
            candidates = []

        if candidates:

            # Most recently created mission first.
            candidates.sort(
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True
            )

            return candidates[0]

        wallclock.sleep(0.02)

    print(
        f"Drone {drone_id}: ERROR - "
        "mission directory was not found."
    )

    raise SystemExit


if drone_id == 3:

    mission_dir = (
        create_new_mission_for_drone3()
    )

else:

    mission_dir = find_current_mission()


print(
    f"Drone {drone_id}: "
    f"Mission directory = "
    f"{mission_dir.name}"
)


def refresh_mission_if_deleted():

    """
    If D3 removed an old mission after this controller had
    already discovered it, automatically find the new mission.
    """

    global mission_dir

    try:

        if (
            mission_dir.is_dir()
            and
            (
                mission_dir
                /
                "MISSION_READY.flag"
            ).exists()
        ):

            return

    except OSError:
        pass

    if drone_id != 3:

        mission_dir = find_current_mission()

        print(
            f"Drone {drone_id}: "
            f"Attached to mission "
            f"{mission_dir.name}"
        )


def flag_path(name):

    refresh_mission_if_deleted()

    return (
        mission_dir / name
    )


def create_flag(name):

    global mission_dir

    path = flag_path(name)

    try:

        with open(
            path,
            "x",
            encoding="utf-8"
        ) as f:

            f.write(
                f"drone={drone_id}\n"
            )

    except FileExistsError:

        pass

    except OSError as error:

        # If the mission directory disappeared between the
        # existence check and file creation, reconnect once.
        if drone_id != 3:

            mission_dir = find_current_mission()

            path = (
                mission_dir / name
            )

            try:

                with open(
                    path,
                    "x",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        f"drone={drone_id}\n"
                    )

            except FileExistsError:

                pass

            except OSError as retry_error:

                print(
                    f"Drone {drone_id}: "
                    f"Mission sync error: "
                    f"{retry_error}"
                )

        else:

            print(
                f"Drone {drone_id}: "
                f"Mission sync error: "
                f"{error}"
            )


def flag_exists(name):

    refresh_mission_if_deleted()

    try:

        return (
            mission_dir / name
        ).exists()

    except OSError:

        return False


def wait_for_previous_drone():

    previous = (
        PREVIOUS_DRONE[drone_id]
    )

    if previous is None:

        return

    required_flag = (
        f"D{previous}_FINAL.flag"
    )

    print(
        f"Drone {drone_id}: "
        f"Waiting for Drone {previous}..."
    )

    while (
        robot.step(timestep) != -1
    ):

        if flag_exists(
            required_flag
        ):

            break

    print(
        f"Drone {drone_id}: "
        f"Drone {previous} finished."
    )


# ============================================================
# FILE LOG
# ============================================================

csv_file = controller_dir / (
    f"mission_log_D{drone_id}.csv"
)


mission_start_time = None
first_obstacle_time = None
columns_clear_time = None
walls_clear_time = None
final_time = None
landing_start_time = None
landing_complete_time = None


def elapsed(event_time):
    if (
        mission_start_time is None
        or event_time is None
    ):
        return None

    return (
        event_time
        -
        mission_start_time
    )


def save_results():

    obstacle_time = None
    landing_time = None
    total_time = None

    if (
        mission_start_time is not None
        and walls_clear_time is not None
    ):
        obstacle_time = (
            walls_clear_time
            -
            mission_start_time
        )

    if (
        landing_start_time is not None
        and landing_complete_time is not None
    ):
        landing_time = (
            landing_complete_time
            -
            landing_start_time
        )

    if (
        mission_start_time is not None
        and landing_complete_time is not None
    ):
        total_time = (
            landing_complete_time
            -
            mission_start_time
        )

    new_file = not csv_file.exists()

    with open(
        csv_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "Drone",
                "Takeoff_s",
                "MissionStart_s",
                "FirstObstacle_s",
                "ColumnsClear_s",
                "WallsClear_s",
                "FinalFormation_s",
                "LandingStart_s",
                "LandingComplete_s",
                "ObstacleTraversal_s",
                "LandingDuration_s",
                "TotalMission_s",
            ])

        writer.writerow([
            drone_id,
            None if takeoff_time is None
            else round(takeoff_time, 3),
            None if mission_start_time is None
            else round(mission_start_time, 3),
            None if first_obstacle_time is None
            else round(first_obstacle_time, 3),
            None if columns_clear_time is None
            else round(columns_clear_time, 3),
            None if walls_clear_time is None
            else round(walls_clear_time, 3),
            None if final_time is None
            else round(final_time, 3),
            None if landing_start_time is None
            else round(landing_start_time, 3),
            None if landing_complete_time is None
            else round(landing_complete_time, 3),
            None if obstacle_time is None
            else round(obstacle_time, 3),
            None if landing_time is None
            else round(landing_time, 3),
            None if total_time is None
            else round(total_time, 3),
        ])


# ============================================================
# MOTORS
# ============================================================

m1_motor = robot.getDevice("m1_motor")
m2_motor = robot.getDevice("m2_motor")
m3_motor = robot.getDevice("m3_motor")
m4_motor = robot.getDevice("m4_motor")

for motor in [
    m1_motor,
    m2_motor,
    m3_motor,
    m4_motor
]:
    motor.setPosition(float("inf"))

m1_motor.setVelocity(-1)
m2_motor.setVelocity(1)
m3_motor.setVelocity(-1)
m4_motor.setVelocity(1)


# ============================================================
# IMU / GPS / GYRO
# ============================================================

imu = robot.getDevice("inertial_unit")
gps = robot.getDevice("gps")
gyro = robot.getDevice("gyro")

if imu is None or gps is None or gyro is None:
    print("ERROR: Required attitude/GPS device missing.")
    raise SystemExit

imu.enable(timestep)
gps.enable(timestep)
gyro.enable(timestep)


# ============================================================
# RANGE SENSORS
# ============================================================

range_front = robot.getDevice("range_front")
range_back = robot.getDevice("range_back")
range_left = robot.getDevice("range_left")
range_right = robot.getDevice("range_right")

for sensor in [
    range_front,
    range_back,
    range_left,
    range_right
]:
    if sensor is not None:
        sensor.enable(timestep)


print("Range sensors:")
print("  front:", "OK" if range_front else "MISSING")
print("  back :", "OK" if range_back else "MISSING")
print("  left :", "OK" if range_left else "MISSING")
print("  right:", "OK" if range_right else "MISSING")


# ============================================================
# CAMERA
# ============================================================

camera = robot.getDevice("camera")

if camera is None:
    print(
        "ERROR: Camera 'camera' was not found."
    )
    raise SystemExit

CAMERA_PERIOD = 64

camera.enable(CAMERA_PERIOD)

camera_width = camera.getWidth()
camera_height = camera.getHeight()

print(
    f"Camera: "
    f"{camera_width}x{camera_height}, "
    f"FOV={camera.getFov():.3f}"
)


# ============================================================
# PID
# ============================================================

PID_controller = (
    pid_velocity_fixed_height_controller()
)


def reset_pid():

    for attr in [
        "past_vx_error",
        "past_vy_error",
        "past_alt_error",
        "past_pitch_error",
        "past_roll_error",
        "altitude_integrator",
    ]:

        if hasattr(PID_controller, attr):

            setattr(
                PID_controller,
                attr,
                0.0
            )


def safe_number(
    value,
    default=0.0
):

    try:

        x = float(value)

        if isfinite(x):
            return x

    except Exception:
        pass

    return default


# ============================================================
# RANGE
# ============================================================

def read_range(sensor):

    if sensor is None:
        return 2.0

    raw = safe_number(
        sensor.getValue(),
        2000.0
    )

    # Crazyflie lookup table:
    # 0 -> 0 m
    # 2000 -> 2 m

    distance = raw / 1000.0

    return max(
        0.0,
        min(2.0, distance)
    )


# ============================================================
# CAMERA IMAGE
# ============================================================

def get_camera_image():

    raw = camera.getImage()

    if raw is None:
        return None

    expected = (
        camera_width
        *
        camera_height
        *
        4
    )

    if len(raw) != expected:
        return None

    image = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    image = image.reshape(
        (
            camera_height,
            camera_width,
            4
        )
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2BGR
    )


# ============================================================
# ROBUST VISION
# ============================================================

def vision_obstacle(image):

    """
    Returns:
        detected,
        direction,
        left_score,
        center_score,
        right_score

    direction:
        +1 = left
        -1 = right
         0 = unknown

    Important:
        The previous implementation treated tiny edge
        densities such as C=0.03 as obstacles. That caused
        a persistent false-positive state and very slow motion.

        This version requires:
            1) substantially higher central edge density, AND
            2) a meaningful contour with meaningful height.

        Camera is therefore active, but normal background
        texture is not allowed to dominate the controller.
    """

    if image is None:

        return (
            False,
            0,
            0.0,
            0.0,
            0.0
        )

    h, w = image.shape[:2]

    # Ignore extreme top and bottom regions.
    y1 = int(0.12 * h)
    y2 = int(0.78 * h)

    x1 = int(0.08 * w)
    x2 = int(0.92 * w)

    roi = image[
        y1:y2,
        x1:x2
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        80,
        160
    )

    # Close small gaps.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    rw = edges.shape[1]

    left_region = edges[
        :,
        :int(0.33 * rw)
    ]

    center_region = edges[
        :,
        int(0.33 * rw):
        int(0.67 * rw)
    ]

    right_region = edges[
        :,
        int(0.67 * rw):
    ]

    left_score = (
        np.count_nonzero(left_region)
        /
        left_region.size
    )

    center_score = (
        np.count_nonzero(center_region)
        /
        center_region.size
    )

    right_score = (
        np.count_nonzero(right_region)
        /
        right_region.size
    )

    # --------------------------------------------------------
    # Contour filtering
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = float(
        edges.shape[0]
        *
        edges.shape[1]
    )

    valid_center_contour = False

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 40:
            continue

        bx, by, bw, bh = cv2.boundingRect(
            contour
        )

        center_x = (
            bx
            +
            bw / 2.0
        )

        center_norm = center_x / rw

        height_norm = (
            bh
            /
            edges.shape[0]
        )

        box_norm = (
            bw * bh
            /
            roi_area
        )

        # Meaningful object in the center.
        if (
            0.25 <= center_norm <= 0.75
            and
            height_norm >= 0.18
            and
            box_norm >= 0.008
        ):

            valid_center_contour = True

            break

    # The old 0.03-0.05 values are normal background.
    strong_center_edges = (
        center_score >= 0.10
    )

    detected = (
        strong_center_edges
        and
        valid_center_contour
    )

    direction = 0

    if detected:

        # Lower edge density = more free visual space.
        if left_score < right_score:

            direction = +1

        elif right_score < left_score:

            direction = -1

        else:

            direction = 0

    return (
        detected,
        direction,
        left_score,
        center_score,
        right_score
    )


# ============================================================
# VISION FILTER
# ============================================================

vision_hits = 0

vision_misses = 0

vision_active = False

vision_direction = 0

vision_left = 0.0
vision_center = 0.0
vision_right = 0.0

last_vision_time = -100.0

VISION_PERIOD = 0.12

VISION_CONFIRM = 2


def update_vision(current_time):

    global vision_hits
    global vision_misses
    global vision_active
    global vision_direction

    global vision_left
    global vision_center
    global vision_right

    global last_vision_time

    if (
        current_time
        -
        last_vision_time
        <
        VISION_PERIOD
    ):

        return

    image = get_camera_image()

    (
        detected,
        direction,
        left_score,
        center_score,
        right_score
    ) = vision_obstacle(image)

    vision_left = left_score
    vision_center = center_score
    vision_right = right_score

    last_vision_time = current_time

    if detected:

        vision_hits += 1
        vision_misses = 0

    else:

        vision_hits = max(
            0,
            vision_hits - 1
        )

        vision_misses += 1

    if vision_hits >= VISION_CONFIRM:

        vision_active = True
        vision_direction = direction

    elif vision_misses >= 2:

        vision_active = False
        vision_direction = 0


# ============================================================
# STATES
# ============================================================

TAKEOFF = 0
WAIT_SEQUENCE = 1
MOVE = 2
FINAL_WAIT = 3
LAND = 4
DONE = 5

state = TAKEOFF


# ============================================================
# INITIAL GPS SAMPLE
# ============================================================

for _ in range(10):

    if robot.step(timestep) == -1:
        raise SystemExit


gps_data = gps.getValues()

past_x_global = safe_number(
    gps_data[0],
    start_x
)

past_y_global = safe_number(
    gps_data[1],
    start_y
)

past_time = robot.getTime()


# ============================================================
# STATUS
# ============================================================

create_flag(
    f"D{drone_id}_READY.flag"
)

print(
    f"Drone {drone_id}: READY"
)


settle_start_time = None

mission_start_time = None

waypoint_index = 0


# ============================================================
# MAIN LOOP
# ============================================================

while robot.step(timestep) != -1:

    current_time = robot.getTime()

    dt = current_time - past_time

    if not isfinite(dt):
        dt = timestep / 1000.0

    dt = max(
        dt,
        timestep / 1000.0
    )


    # ========================================================
    # SENSOR READ
    # ========================================================

    imu_data = imu.getRollPitchYaw()

    roll = safe_number(
        imu_data[0],
        0.0
    )

    pitch = safe_number(
        imu_data[1],
        0.0
    )

    yaw = safe_number(
        imu_data[2],
        0.0
    )

    gyro_data = gyro.getValues()

    yaw_rate = safe_number(
        gyro_data[2],
        0.0
    )

    gps_data = gps.getValues()

    x_global = safe_number(
        gps_data[0],
        past_x_global
    )

    y_global = safe_number(
        gps_data[1],
        past_y_global
    )

    altitude = safe_number(
        gps_data[2],
        FLYING_ALTITUDE
    )


    # ========================================================
    # VELOCITY
    # ========================================================

    v_x_global = (
        x_global - past_x_global
    ) / dt

    v_y_global = (
        y_global - past_y_global
    ) / dt

    v_x_global = safe_number(
        v_x_global,
        0.0
    )

    v_y_global = safe_number(
        v_y_global,
        0.0
    )


    # ========================================================
    # WORLD -> BODY
    #
    # +X = forward
    # +Y = left
    # ========================================================

    cy = cos(yaw)
    sy = sin(yaw)

    v_x = (
        v_x_global * cy
        +
        v_y_global * sy
    )

    v_y = (
        -v_x_global * sy
        +
        v_y_global * cy
    )

    v_x = safe_number(
        v_x,
        0.0
    )

    v_y = safe_number(
        v_y,
        0.0
    )


    # ========================================================
    # RANGE
    # ========================================================

    front = read_range(range_front)
    back = read_range(range_back)
    left = read_range(range_left)
    right = read_range(range_right)


    # ========================================================
    # CAMERA
    # ========================================================

    if state == MOVE:

        update_vision(
            current_time
        )


    # ========================================================
    # DEFAULT COMMAND
    # ========================================================

    forward_desired = 0.0

    sideways_desired = 0.0

    yaw_desired = 0.0

    height_desired = FLYING_ALTITUDE


    # ========================================================
    # TAKEOFF
    # ========================================================

    if state == TAKEOFF:

        height_desired = FLYING_ALTITUDE

        forward_desired = 0.0
        sideways_desired = 0.0


        if abs(
            altitude
            -
            FLYING_ALTITUDE
        ) < 0.10:

            if settle_start_time is None:

                settle_start_time = (
                    current_time
                )

            elif (
                current_time
                -
                settle_start_time
                >=
                SETTLE_TIME
            ):

                print(
                    f"Drone {drone_id}: "
                    "TAKEOFF COMPLETE"
                )

                reset_pid()

                create_flag(
                    f"D{drone_id}_AIRBORNE.flag"
                )

                state = WAIT_SEQUENCE

        else:

            settle_start_time = None


    # ========================================================
    # WAIT FOR TURN
    # ========================================================

    elif state == WAIT_SEQUENCE:

        height_desired = FLYING_ALTITUDE

        forward_desired = 0.0
        sideways_desired = 0.0

        previous = PREVIOUS_DRONE[
            drone_id
        ]

        can_start = False

        if previous is None:

            can_start = True

        else:

            can_start = flag_exists(
                f"D{previous}_FINAL.flag"
            )

        if can_start:

            mission_start_time = (
                current_time
            )

            first_obstacle_time = None
            columns_clear_time = None
            walls_clear_time = None
            final_time = None

            print()
            print(
                "============================================"
            )
            print(
                f"Drone {drone_id}: "
                "MISSION START"
            )
            print(
                f"Position: "
                f"({x_global:.2f}, "
                f"{y_global:.2f}, "
                f"{altitude:.2f})"
            )
            print(
                "============================================"
            )

            reset_pid()

            state = MOVE

            create_flag(
                f"D{drone_id}_RUNNING.flag"
            )


    # ========================================================
    # MOVE
    # ========================================================

    elif state == MOVE:

        target_x, target_y = targets[
            waypoint_index
        ]

        err_x = (
            target_x
            -
            x_global
        )

        err_y = (
            target_y
            -
            y_global
        )

        distance = sqrt(
            err_x * err_x
            +
            err_y * err_y
        )


        # ----------------------------------------------------
        # Base position control
        # ----------------------------------------------------

        speed_factor = min(
            1.0,
            distance / 0.50
        )

        vel_x = (
            KP_POS
            *
            err_x
            *
            speed_factor
        )

        vel_y = (
            KP_POS
            *
            err_y
            *
            speed_factor
        )


        # ====================================================
        # CAMERA ASSIST
        # ====================================================
        #
        # Vision is an avoidance/safety input, not the
        # primary navigator. The known geometry is handled by
        # the mission targets.
        #

        if vision_active:

            # Only add a lateral avoidance bias when the
            # camera has a real confirmed central obstacle.

            if vision_direction > 0:

                # +Y = left
                vel_y += 0.16

            elif vision_direction < 0:

                # -Y = right
                vel_y -= 0.16

            else:

                # Unknown visual direction:
                # use side ranges.
                if left > right:

                    vel_y += 0.12

                elif right > left:

                    vel_y -= 0.12


            # If the central object is strong, slow X.
            if vision_center >= 0.18:

                vel_x *= 0.65


        # ====================================================
        # CLOSE RANGE FRONT SAFETY
        # ====================================================

        if front < 0.18:

            vel_x = 0.0

            # Escape to the side with more free space.

            if left > right:

                vel_y += 0.22

            elif right > left:

                vel_y -= 0.22


        elif front < 0.55:

            scale = (
                front - 0.18
            ) / (
                0.55 - 0.18
            )

            scale = max(
                0.0,
                min(1.0, scale)
            )

            if vel_x > 0:

                vel_x *= scale


        # ====================================================
        # COLUMN GAP CENTERING
        # ====================================================
        #
        # When crossing between the two columns we want:
        #
        #       Y ~= 0.5
        #
        # This is the actual requested passage point for the
        # given column coordinates.
        #

        if (
            2.15 <= x_global <= 3.20
        ):

            gap_y_error = (
                COLUMN_Y
                -
                y_global
            )

            vel_y += (
                0.75
                *
                gap_y_error
            )


        # ====================================================
        # WALL CORRIDOR CENTERING
        # ====================================================
        #
        # Walls:
        #   y=0.2
        #   y=0.7
        #
        # Center:
        #   y=0.45
        #

        if (
            4.25 <= x_global <= 5.75
        ):

            wall_y_error = (
                WALL_PASS_Y
                -
                y_global
            )

            vel_y += (
                0.90
                *
                wall_y_error
            )

            # Additional sensor-based centering.
            # +Y = left, -Y = right.
            #
            # If left range is larger than right range,
            # the vehicle is closer to the right wall,
            # so move left.

            vel_y += (
                0.30
                *
                (left - right)
            )


        # ====================================================
        # SIDE SAFETY
        # ====================================================

        if (
            left < 0.12
            and
            vel_y > 0
        ):

            vel_y = 0.0


        if (
            right < 0.12
            and
            vel_y < 0
        ):

            vel_y = 0.0


        # ====================================================
        # SPEED LIMIT
        # ====================================================

        speed = sqrt(
            vel_x * vel_x
            +
            vel_y * vel_y
        )

        if speed > MAX_HORIZONTAL_SPEED:

            vel_x = (
                vel_x
                /
                speed
            ) * MAX_HORIZONTAL_SPEED

            vel_y = (
                vel_y
                /
                speed
            ) * MAX_HORIZONTAL_SPEED


        # ====================================================
        # WORLD -> BODY
        # ========================================================

        forward_desired = (
            vel_x * cy
            +
            vel_y * sy
        )

        sideways_desired = (
            -vel_x * sy
            +
            vel_y * cy
        )


        # ====================================================
        # YAW HOLD
        # ========================================================

        yaw_error = atan2(
            sin(-yaw),
            cos(-yaw)
        )

        yaw_desired = (
            KP_YAW
            *
            yaw_error
        )

        yaw_desired = max(
            -0.5,
            min(
                0.5,
                yaw_desired
            )
        )


        # ====================================================
        # TIMING: FIRST OBSTACLE
        # ========================================================

        obstacle_now = (
            vision_active
            or
            front < 0.55
        )

        if (
            first_obstacle_time is None
            and
            obstacle_now
        ):

            first_obstacle_time = (
                current_time
            )

            print(
                f"Drone {drone_id}: "
                f"Obstacle detected at "
                f"{elapsed(current_time):.2f}s"
            )


        # ====================================================
        # TIMING: COLUMNS CLEARED
        # ========================================================

        if (
            columns_clear_time is None
            and
            x_global >= COLUMN_EXIT_X
        ):

            columns_clear_time = (
                current_time
            )

            print(
                f"Drone {drone_id}: "
                f"Columns cleared in "
                f"{elapsed(current_time):.2f}s"
            )


        # ====================================================
        # TIMING: WALLS CLEARED
        # ========================================================

        if (
            walls_clear_time is None
            and
            x_global >= WALL_EXIT_X
        ):

            walls_clear_time = (
                current_time
            )

            print(
                f"Drone {drone_id}: "
                f"Walls cleared in "
                f"{elapsed(current_time):.2f}s"
            )


        # ====================================================
        # TARGET REACHED
        # ========================================================

        horizontal_speed = sqrt(
            v_x_global * v_x_global
            +
            v_y_global * v_y_global
        )

        if (
            distance < 0.10
            and
            horizontal_speed < 0.12
        ):

            print(
                f"Drone {drone_id}: "
                f"Reached waypoint "
                f"{waypoint_index + 1}/"
                f"{len(targets)}"
            )

            waypoint_index += 1

            reset_pid()

            if (
                waypoint_index
                >=
                len(targets)
            ):

                final_time = (
                    current_time
                )

                print()
                print(
                    "********************************************"
                )
                print(
                    f"Drone {drone_id}: "
                    "FINAL FORMATION REACHED"
                )
                print(
                    f"Position: "
                    f"({x_global:.2f}, "
                    f"{y_global:.2f}, "
                    f"{altitude:.2f})"
                )
                print(
                    f"Mission time: "
                    f"{elapsed(final_time):.2f}s"
                )

                if (
                    walls_clear_time
                    is not None
                ):

                    print(
                        f"Obstacle traversal time: "
                        f"{walls_clear_time - mission_start_time:.2f}s"
                    )

                print(
                    "********************************************"
                )
                print()

                create_flag(
                    f"D{drone_id}_FINAL.flag"
                )

                reset_pid()

                state = FINAL_WAIT


    # ========================================================
    # WAIT FOR ALL FOUR FINAL FLAGS
    # ========================================================

    elif state == FINAL_WAIT:

        height_desired = FLYING_ALTITUDE

        forward_desired = 0.0
        sideways_desired = 0.0
        yaw_desired = 0.0

        all_final = True

        for d in [1, 2, 3, 4]:

            if not flag_exists(
                f"D{d}_FINAL.flag"
            ):

                all_final = False

                break

        if all_final:

            landing_start_time = (
                current_time
            )

            print(
                f"Drone {drone_id}: "
                "ALL FOUR DRONES FINAL"
            )

            print(
                f"Drone {drone_id}: "
                "SYNCHRONIZED LANDING START"
            )

            create_flag(
                f"D{drone_id}_LANDING.flag"
            )

            reset_pid()

            state = LAND


    # ========================================================
    # LAND
    # ========================================================

    elif state == LAND:

        height_desired = LAND_HEIGHT

        forward_desired = 0.0
        sideways_desired = 0.0
        yaw_desired = 0.0

        if altitude < 0.10:

            m1_motor.setVelocity(0.0)
            m2_motor.setVelocity(0.0)
            m3_motor.setVelocity(0.0)
            m4_motor.setVelocity(0.0)

            landing_complete_time = (
                current_time
            )

            landing_duration = (
                landing_complete_time
                -
                landing_start_time
            )

            total_duration = (
                landing_complete_time
                -
                mission_start_time
            )

            obstacle_duration = None

            if (
                walls_clear_time
                is not None
            ):

                obstacle_duration = (
                    walls_clear_time
                    -
                    mission_start_time
                )


            print()
            print(
                "############################################"
            )

            print(
                f"Drone {drone_id}: "
                "LANDED SUCCESSFULLY"
            )

            if obstacle_duration is not None:

                print(
                    f"Obstacle traversal: "
                    f"{obstacle_duration:.2f}s"
                )

            print(
                f"Landing duration: "
                f"{landing_duration:.2f}s"
            )

            print(
                f"TOTAL mission time: "
                f"{total_duration:.2f}s"
            )

            print(
                "############################################"
            )
            print()

            create_flag(
                f"D{drone_id}_LANDED.flag"
            )

            save_results()

            state = DONE


    # ========================================================
    # DONE
    # ========================================================

    elif state == DONE:

        m1_motor.setVelocity(0.0)
        m2_motor.setVelocity(0.0)
        m3_motor.setVelocity(0.0)
        m4_motor.setVelocity(0.0)

        continue


    # ========================================================
    # ALTITUDE SAFETY
    # ========================================================

    if altitude < ALTITUDE_THRESHOLD:

        forward_desired = 0.0
        sideways_desired = 0.0
        height_desired = FLYING_ALTITUDE


    # ========================================================
    # PID
    # ========================================================

    try:

        motor_power = PID_controller.pid(
            dt,
            forward_desired,
            sideways_desired,
            yaw_desired,
            height_desired,
            roll,
            pitch,
            yaw_rate,
            altitude,
            v_x,
            v_y
        )

    except Exception as error:

        print(
            f"Drone {drone_id}: "
            f"PID ERROR = {error}"
        )

        motor_power = [
            48.0,
            48.0,
            48.0,
            48.0
        ]


    # ========================================================
    # NaN / INF PROTECTION
    # ========================================================

    safe_power = []

    for p in motor_power:

        p = safe_number(
            p,
            48.0
        )

        p = max(
            0.0,
            min(
                600.0,
                p
            )
        )

        safe_power.append(p)


    # ========================================================
    # MOTOR OUTPUT
    # ========================================================

    m1_motor.setVelocity(
        -safe_power[0]
    )

    m2_motor.setVelocity(
        safe_power[1]
    )

    m3_motor.setVelocity(
        -safe_power[2]
    )

    m4_motor.setVelocity(
        safe_power[3]
    )


    # ========================================================
    # UPDATE
    # ========================================================

    past_time = current_time

    past_x_global = x_global

    past_y_global = y_global
