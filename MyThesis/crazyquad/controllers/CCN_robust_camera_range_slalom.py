# -*- coding: utf-8 -*-

from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller

from math import cos, sin, sqrt, atan2, isfinite
import csv
import os
import re

import numpy as np
import cv2


# ============================================================
# FOUR CRAZYFLIE - ROBUST OBSTACLE COURSE
# ============================================================
#
# Coordinate system:
#   +X = forward
#   +Y = left
#   +Z = up
#
# Initial formation:
#   D1 = (0,0)
#   D2 = (0,1)
#   D3 = (1,0)
#   D4 = (1,1)
#
# Order:
#   D3 -> D4 -> D1 -> D2
#
# Mission:
#   1) all drones take off to 1 m
#   2) front row (D3,D4) moves first
#   3) each drone follows a safe slalom around the two columns
#   4) each drone enters Y=0.45
#   5) each drone passes through the gap between the two walls
#   6) each drone reaches its final square position
#   7) all drones land at the common landing time
#
# IMPORTANT DESIGN CHANGE:
#
# Camera + OpenCV are ACTIVE for obstacle detection and timing,
# but are NOT allowed to inject large lateral commands into the
# flight controller.
#
# Range Finder = close-range collision safety.
# GPS         = route tracking.
# IMU/Gyro   = attitude.
# PID         = low-level control.
#
# This avoids the previous failure in which the camera generated
# false/unstable obstacle directions and D3 was driven away from
# its safe planned path.
# ============================================================


# ============================================================
# BASIC SETTINGS
# ============================================================

TIME_STEP = 32

FLYING_ALTITUDE = 1.0
LAND_HEIGHT = 0.05

# Faster than previous versions, but not aggressive.
MAX_HORIZONTAL_SPEED = 0.40

KP_POS = 1.15
KP_YAW = 1.0

SETTLE_TIME = 1.0

ALTITUDE_THRESHOLD = 0.75


# ============================================================
# SEQUENCE
# ============================================================
#
# Because +X is forward, D3/D4 are physically in front.
#

START_ORDER = [3, 4, 1, 2]

# Absolute simulation-time delays AFTER the drone itself
# reaches 1 m.
START_DELAY = {
    3: 0.0,
    4: 28.0,
    1: 56.0,
    2: 84.0,
}

# All nominal routes should finish before this time.
LANDING_START_TIME = 125.0


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
# FINAL FORMATION
# ============================================================

FINAL_SHIFT_X = 6.0


# ============================================================
# OBSTACLES
# ============================================================

COLUMN_1_X = 2.0
COLUMN_2_X = 3.0
COLUMN_Y = 0.5
COLUMN_RADIUS = 0.05

WALL_X = 5.0
WALL_1_Y = 0.2
WALL_2_Y = 0.7

WALL_CENTER_Y = 0.45

WALL_ENTRY_X = 4.20
WALL_EXIT_X = 5.80


# ============================================================
# SLALOM CLEARANCE
# ============================================================

# Lower and upper passage lines.
#
# The free path is deliberately away from the 0.1 m diameter
# columns.

LOW_Y = 0.15
HIGH_Y = 0.85


# ============================================================
# RANGE FINDER
# ============================================================

MAX_RANGE = 2.0

FRONT_STOP_DISTANCE = 0.20
FRONT_SLOW_DISTANCE = 0.50

SIDE_CRITICAL_DISTANCE = 0.12


# ============================================================
# CAMERA
# ============================================================

CAMERA_NAME = "camera"

CAMERA_PERIOD = 64

VISION_PERIOD = 0.12

VISION_CONFIRM_FRAMES = 2


# ============================================================
# BLUE VISION
# ============================================================
#
# Obstacles in the supplied scene are blue.
#
# Vision is used for:
#   - detecting a blue obstacle
#   - timestamping first visual detection
#   - logging
#
# Vision is NOT used to create a large lateral command.
#

BLUE_LOW = np.array(
    [90, 55, 40],
    dtype=np.uint8
)

BLUE_HIGH = np.array(
    [135, 255, 255],
    dtype=np.uint8
)

MIN_CONTOUR_AREA = 70


# ============================================================
# ROBOT
# ============================================================

robot = Robot()

timestep = int(robot.getBasicTimeStep())

robot_name = robot.getName()

match = re.search(
    r"(\d+)$",
    robot_name
)

drone_id = (
    int(match.group(1))
    if match
    else 1
)


if drone_id not in START_POSITIONS:

    print("ERROR: Unknown drone ID.")

    raise SystemExit


start_x, start_y = START_POSITIONS[
    drone_id
]


final_x = (
    start_x
    +
    FINAL_SHIFT_X
)

final_y = start_y


print()
print("================================================")
print(f"Robot       : {robot_name}")
print(f"Drone ID    : {drone_id}")
print(f"Start       : ({start_x:.2f}, {start_y:.2f})")
print(f"Final       : ({final_x:.2f}, {final_y:.2f})")
print(f"Sequence    : {START_ORDER}")
print(
    f"Start delay : "
    f"{START_DELAY[drone_id]:.1f} s"
)
print("================================================")
print()


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

    motor.setPosition(
        float("inf")
    )


m1_motor.setVelocity(-1)
m2_motor.setVelocity(1)
m3_motor.setVelocity(-1)
m4_motor.setVelocity(1)


# ============================================================
# IMU / GPS / GYRO
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


if (
    imu is None
    or
    gps is None
    or
    gyro is None
):

    print(
        "ERROR: IMU/GPS/Gyro missing."
    )

    raise SystemExit


imu.enable(timestep)
gps.enable(timestep)
gyro.enable(timestep)


# ============================================================
# RANGE FINDERS
# ============================================================

range_front = robot.getDevice(
    "range_front"
)

range_back = robot.getDevice(
    "range_back"
)

range_left = robot.getDevice(
    "range_left"
)

range_right = robot.getDevice(
    "range_right"
)


for sensor in [
    range_front,
    range_back,
    range_left,
    range_right
]:

    if sensor is not None:

        sensor.enable(timestep)


print("Range sensors:")

print(
    "  front:",
    "OK" if range_front else "MISSING"
)

print(
    "  back :",
    "OK" if range_back else "MISSING"
)

print(
    "  left :",
    "OK" if range_left else "MISSING"
)

print(
    "  right:",
    "OK" if range_right else "MISSING"
)


# ============================================================
# CAMERA
# ============================================================

camera = robot.getDevice(
    CAMERA_NAME
)


if camera is None:

    print(
        "ERROR: Camera not found."
    )

    raise SystemExit


camera.enable(
    CAMERA_PERIOD
)


camera_width = camera.getWidth()
camera_height = camera.getHeight()


print(
    f"Camera: "
    f"{camera_width}x"
    f"{camera_height}, "
    f"FOV={camera.getFov():.3f}"
)


# ============================================================
# PID
# ============================================================

PID = (
    pid_velocity_fixed_height_controller()
)


def reset_pid():

    for name in [

        "past_vx_error",
        "past_vy_error",
        "past_alt_error",
        "past_pitch_error",
        "past_roll_error",
        "altitude_integrator"

    ]:

        if hasattr(PID, name):

            setattr(
                PID,
                name,
                0.0
            )


def finite_value(
    value,
    default=0.0
):

    try:

        value = float(value)

        if isfinite(value):

            return value

    except Exception:

        pass

    return default


# ============================================================
# RANGE
# ============================================================

def get_range(sensor):

    if sensor is None:

        return MAX_RANGE


    raw = finite_value(
        sensor.getValue(),
        2000.0
    )


    distance = (
        raw
        /
        1000.0
    )


    return max(
        0.0,
        min(
            MAX_RANGE,
            distance
        )
    )


# ============================================================
# CAMERA IMAGE
# ============================================================

def get_image():

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
# BLUE OBSTACLE DETECTOR
# ============================================================

def detect_blue_obstacle(
    image
):

    if image is None:

        return (
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            None
        )


    h, w = image.shape[:2]


    # The lower-middle part of the forward image is
    # more useful for obstacle detection than the extreme
    # top/bottom edges.

    y1 = int(
        0.10 * h
    )

    y2 = int(
        0.82 * h
    )

    x1 = int(
        0.08 * w
    )

    x2 = int(
        0.92 * w
    )


    roi = image[
        y1:y2,
        x1:x2
    ]


    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )


    mask = cv2.inRange(
        hsv,
        BLUE_LOW,
        BLUE_HIGH
    )


    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    best = None
    best_area = 0.0


    left_score = (
        np.mean(
            mask[
                :,
                :int(mask.shape[1] * 0.33)
            ]
            >
            0
        )
    )


    center_score = (
        np.mean(
            mask[
                :,
                int(mask.shape[1] * 0.33):
                int(mask.shape[1] * 0.67)
            ]
            >
            0
        )
    )


    right_score = (
        np.mean(
            mask[
                :,
                int(mask.shape[1] * 0.67):
            ]
            >
            0
        )
    )


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < MIN_CONTOUR_AREA:

            continue


        x, y, cw, ch = (
            cv2.boundingRect(
                contour
            )
        )


        height_ratio = (
            ch
            /
            mask.shape[0]
        )


        width_ratio = (
            cw
            /
            mask.shape[1]
        )


        # Reject tiny blue artifacts.

        if (
            height_ratio < 0.08
            and
            width_ratio < 0.04
        ):

            continue


        if area > best_area:

            best_area = area

            best = (
                x,
                y,
                cw,
                ch
            )


    detected = (
        best is not None
        and
        (
            center_score > 0.025
            or
            best_area > 150
        )
    )


    center_x = None


    if best is not None:

        x, y, cw, ch = best

        center_x = (
            x
            +
            cw / 2.0
        )


    return (
        detected,
        left_score,
        center_score,
        right_score,
        best_area,
        center_x
    )


# ============================================================
# VISION FILTER STATE
# ============================================================

vision_hits = 0
vision_misses = 0

vision_active = False

vision_left = 0.0
vision_center = 0.0
vision_right = 0.0

vision_area = 0.0
vision_center_x = None

last_vision_time = -100.0


def update_vision(
    now
):

    global vision_hits
    global vision_misses
    global vision_active

    global vision_left
    global vision_center
    global vision_right
    global vision_area
    global vision_center_x

    global last_vision_time


    if (
        now
        -
        last_vision_time
        <
        VISION_PERIOD
    ):

        return


    image = get_image()


    (
        detected,
        left_score,
        center_score,
        right_score,
        area_score,
        center_x
    ) = detect_blue_obstacle(
        image
    )


    vision_left = left_score
    vision_center = center_score
    vision_right = right_score

    vision_area = area_score

    vision_center_x = center_x


    last_vision_time = now


    if detected:

        vision_hits += 1
        vision_misses = 0

    else:

        vision_hits = max(
            0,
            vision_hits - 1
        )

        vision_misses += 1


    if vision_hits >= VISION_CONFIRM_FRAMES:

        vision_active = True


    if vision_misses >= VISION_CONFIRM_FRAMES:

        vision_active = False


# ============================================================
# BUILD THE ACTUAL SAFE SLALOM PATH
# ============================================================
#
# IMPORTANT:
#
# D1 and D3 start at y=0:
#
#   below C1
#       -> through longitudinal gap
#       -> above C2
#
# D2 and D4 start at y=1:
#
#   above C1
#       -> through longitudinal gap
#       -> below C2
#
# This exactly reflects the supplied obstacle coordinates:
#
#   C1=(2,0.5)
#   C2=(3,0.5)
#
# ============================================================


if start_y < 0.5:

    path = [

        # 1. Approach first obstacle
        (1.55, start_y),

        # 2. Move below column 1
        (1.70, LOW_Y),
        (2.25, LOW_Y),

        # 3. Enter the longitudinal gap
        (2.50, LOW_Y),

        # 4. Cross the gap vertically
        (2.50, HIGH_Y),

        # 5. Pass ABOVE column 2
        (3.35, HIGH_Y),

        # 6. Return to wall corridor center
        (4.10, WALL_CENTER_Y),

        # 7. Enter between walls
        (WALL_ENTRY_X, WALL_CENTER_Y),

        # 8. Leave between walls
        (WALL_EXIT_X, WALL_CENTER_Y),

        # 9. Final formation
        (final_x, final_y)
    ]

else:

    path = [

        # 1. Approach first obstacle
        (1.55, start_y),

        # 2. Move above column 1
        (1.70, HIGH_Y),
        (2.25, HIGH_Y),

        # 3. Enter the longitudinal gap
        (2.50, HIGH_Y),

        # 4. Cross the gap vertically
        (2.50, LOW_Y),

        # 5. Pass BELOW column 2
        (3.35, LOW_Y),

        # 6. Return to wall corridor
        (4.10, WALL_CENTER_Y),

        # 7. Enter between walls
        (WALL_ENTRY_X, WALL_CENTER_Y),

        # 8. Leave between walls
        (WALL_EXIT_X, WALL_CENTER_Y),

        # 9. Final formation
        (final_x, final_y)
    ]


print(
    f"Drone {drone_id}: "
    f"Path has {len(path)} waypoints."
)


for i, point in enumerate(path, 1):

    print(
        f"  WP{i}: "
        f"X={point[0]:.2f}, "
        f"Y={point[1]:.2f}"
    )


# ============================================================
# STATES
# ============================================================

STATE_TAKEOFF = 0
STATE_WAIT_START = 1
STATE_MOVE = 2
STATE_FINAL_HOLD = 3
STATE_LAND = 4
STATE_DONE = 5

state = STATE_TAKEOFF


# ============================================================
# MISSION TIMING
# ============================================================

takeoff_time = None

mission_start_time = None

first_visual_obstacle_time = None

first_range_obstacle_time = None

column1_clear_time = None

column2_clear_time = None

wall_entry_time = None

wall_exit_time = None

final_time = None

landing_start_time = None

landing_complete_time = None


# ============================================================
# CSV
# ============================================================

controller_dir = os.path.dirname(
    os.path.abspath(__file__)
)


csv_path = os.path.join(
    controller_dir,
    f"mission_log_D{drone_id}.csv"
)


def save_results():

    obstacle_duration = None

    landing_duration = None

    total_duration = None


    if (
        mission_start_time is not None
        and
        wall_exit_time is not None
    ):

        obstacle_duration = (
            wall_exit_time
            -
            mission_start_time
        )


    if (
        landing_start_time is not None
        and
        landing_complete_time is not None
    ):

        landing_duration = (
            landing_complete_time
            -
            landing_start_time
        )


    if (
        mission_start_time is not None
        and
        landing_complete_time is not None
    ):

        total_duration = (
            landing_complete_time
            -
            mission_start_time
        )


    file_exists = os.path.exists(
        csv_path
    )


    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)


        if not file_exists:

            writer.writerow([

                "Drone",

                "Takeoff_s",

                "MissionStart_s",

                "FirstVisualObstacle_s",

                "FirstRangeObstacle_s",

                "Column1Clear_s",

                "Column2Clear_s",

                "WallEntry_s",

                "WallExit_s",

                "FinalFormation_s",

                "LandingStart_s",

                "LandingComplete_s",

                "ObstacleTraversal_s",

                "LandingDuration_s",

                "TotalMission_s"

            ])


        writer.writerow([

            drone_id,

            None
            if takeoff_time is None
            else round(
                takeoff_time,
                3
            ),

            None
            if mission_start_time is None
            else round(
                mission_start_time,
                3
            ),

            None
            if first_visual_obstacle_time is None
            else round(
                first_visual_obstacle_time
                -
                mission_start_time,
                3
            ),

            None
            if first_range_obstacle_time is None
            else round(
                first_range_obstacle_time
                -
                mission_start_time,
                3
            ),

            None
            if column1_clear_time is None
            else round(
                column1_clear_time
                -
                mission_start_time,
                3
            ),

            None
            if column2_clear_time is None
            else round(
                column2_clear_time
                -
                mission_start_time,
                3
            ),

            None
            if wall_entry_time is None
            else round(
                wall_entry_time
                -
                mission_start_time,
                3
            ),

            None
            if wall_exit_time is None
            else round(
                wall_exit_time
                -
                mission_start_time,
                3
            ),

            None
            if final_time is None
            else round(
                final_time
                -
                mission_start_time,
                3
            ),

            None
            if landing_start_time is None
            else round(
                landing_start_time
                -
                mission_start_time,
                3
            ),

            None
            if landing_complete_time is None
            else round(
                landing_complete_time
                -
                mission_start_time,
                3
            ),

            None
            if obstacle_duration is None
            else round(
                obstacle_duration,
                3
            ),

            None
            if landing_duration is None
            else round(
                landing_duration,
                3
            ),

            None
            if total_duration is None
            else round(
                total_duration,
                3
            ),

        ])


# ============================================================
# SENSOR WARM-UP
# ============================================================

for _ in range(10):

    if robot.step(timestep) == -1:

        raise SystemExit


gps_data = gps.getValues()


past_x = finite_value(
    gps_data[0],
    start_x
)


past_y = finite_value(
    gps_data[1],
    start_y
)


past_time = robot.getTime()


print(
    f"Drone {drone_id}: "
    "Sensors initialized."
)


# ============================================================
# MAIN LOOP
# ============================================================

waypoint_index = 0

settle_time_start = None


while robot.step(timestep) != -1:


    now = robot.getTime()


    # ========================================================
    # DT
    # ========================================================

    dt = now - past_time


    if not isfinite(dt):

        dt = (
            timestep
            /
            1000.0
        )


    dt = max(
        dt,
        timestep / 1000.0
    )


    # ========================================================
    # IMU
    # ========================================================

    imu_data = imu.getRollPitchYaw()


    roll = finite_value(
        imu_data[0],
        0.0
    )


    pitch = finite_value(
        imu_data[1],
        0.0
    )


    yaw = finite_value(
        imu_data[2],
        0.0
    )


    # ========================================================
    # GYRO
    # ========================================================

    gyro_data = gyro.getValues()


    yaw_rate = finite_value(
        gyro_data[2],
        0.0
    )


    # ========================================================
    # GPS
    # ========================================================

    gps_data = gps.getValues()


    x = finite_value(
        gps_data[0],
        past_x
    )


    y = finite_value(
        gps_data[1],
        past_y
    )


    z = finite_value(
        gps_data[2],
        FLYING_ALTITUDE
    )


    # ========================================================
    # VELOCITY
    # ========================================================

    vx_world = (
        x - past_x
    ) / dt


    vy_world = (
        y - past_y
    ) / dt


    vx_world = finite_value(
        vx_world,
        0.0
    )


    vy_world = finite_value(
        vy_world,
        0.0
    )


    # ========================================================
    # WORLD -> BODY
    # ========================================================

    cy = cos(yaw)
    sy = sin(yaw)


    vx_body = (
        vx_world * cy
        +
        vy_world * sy
    )


    vy_body = (
        -vx_world * sy
        +
        vy_world * cy
    )


    vx_body = finite_value(
        vx_body,
        0.0
    )


    vy_body = finite_value(
        vy_body,
        0.0
    )


    # ========================================================
    # RANGE
    # ========================================================

    front = get_range(
        range_front
    )

    back = get_range(
        range_back
    )

    left = get_range(
        range_left
    )

    right = get_range(
        range_right
    )


    # ========================================================
    # CAMERA VISION
    # ========================================================

    if state == STATE_MOVE:

        update_vision(
            now
        )


    # ========================================================
    # DEFAULT COMMANDS
    # ========================================================

    forward_desired = 0.0

    sideways_desired = 0.0

    yaw_desired = 0.0

    height_desired = (
        FLYING_ALTITUDE
    )


    # ========================================================
    # TAKEOFF
    # ========================================================

    if state == STATE_TAKEOFF:


        height_desired = (
            FLYING_ALTITUDE
        )


        forward_desired = 0.0

        sideways_desired = 0.0


        if abs(
            z - FLYING_ALTITUDE
        ) < 0.10:


            if settle_time_start is None:

                settle_time_start = now


            elif (
                now
                -
                settle_time_start
                >=
                SETTLE_TIME
            ):


                takeoff_time = now


                print(
                    f"Drone {drone_id}: "
                    "TAKEOFF COMPLETE"
                )


                reset_pid()


                state = (
                    STATE_WAIT_START
                )


    # ========================================================
    # WAIT FOR SEQUENCE SLOT
    # ========================================================

    elif state == STATE_WAIT_START:


        height_desired = (
            FLYING_ALTITUDE
        )


        forward_desired = 0.0

        sideways_desired = 0.0


        elapsed_takeoff = (
            now
            -
            takeoff_time
        )


        if (

            elapsed_takeoff
            >=
            START_DELAY[drone_id]

        ):


            mission_start_time = now


            print()
            print(
                "============================================"
            )

            print(
                f"Drone {drone_id}: "
                "MISSION START"
            )

            print(
                f"Start = "
                f"({x:.2f}, "
                f"{y:.2f}, "
                f"{z:.2f})"
            )

            print(
                "============================================"
            )


            reset_pid()


            waypoint_index = 0


            state = STATE_MOVE


    # ========================================================
    # MOVE
    # ========================================================

    elif state == STATE_MOVE:


        target_x, target_y = path[
            waypoint_index
        ]


        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        err_x = (
            target_x
            -
            x
        )


        err_y = (
            target_y
            -
            y
        )


        distance = sqrt(

            err_x * err_x
            +
            err_y * err_y
        )


        # ----------------------------------------------------
        # Nominal velocity controller
        # ----------------------------------------------------

        velocity_scale = min(

            1.0,

            distance / 0.50
        )


        vel_x = (

            KP_POS
            *
            err_x
            *
            velocity_scale
        )


        vel_y = (

            KP_POS
            *
            err_y
            *
            velocity_scale
        )


        # ====================================================
        # CAMERA = DETECTION / LOGGING ONLY
        # ====================================================
        #
        # IMPORTANT:
        # No lateral camera steering.
        #
        # This is intentional.
        #

        if vision_active:

            if (
                first_visual_obstacle_time
                is None
            ):

                first_visual_obstacle_time = (
                    now
                )


                print(
                    f"Drone {drone_id}: "
                    f"VISION obstacle at "
                    f"{now - mission_start_time:.2f}s"
                )


        # ====================================================
        # RANGE FINDER = REAL SAFETY
        # ====================================================

        if (
            first_range_obstacle_time
            is None
            and
            front < FRONT_SLOW_DISTANCE
        ):

            first_range_obstacle_time = (
                now
            )


            print(
                f"Drone {drone_id}: "
                f"RANGE obstacle at "
                f"{now - mission_start_time:.2f}s "
                f"(front={front:.2f}m)"
            )


        # ----------------------------------------------------
        # Front obstacle
        # ----------------------------------------------------

        if front < FRONT_STOP_DISTANCE:


            # Stop forward motion.

            vel_x = 0.0


            # Follow the already planned safe side.
            #
            # For lower start:
            #   before the second obstacle -> lower side
            #   after it                -> upper side
            #
            # For upper start it is reversed.

            if start_y < 0.5:

                if x < 2.6:

                    vel_y = min(
                        0.15,
                        max(
                            vel_y,
                            0.08
                        )
                    )

                else:

                    vel_y = max(
                        -0.15,
                        min(
                            vel_y,
                            -0.08
                        )
                    )

            else:

                if x < 2.6:

                    vel_y = max(
                        -0.15,
                        min(
                            vel_y,
                            -0.08
                        )
                    )

                else:

                    vel_y = min(
                        0.15,
                        max(
                            vel_y,
                            0.08
                        )
                    )


        elif front < FRONT_SLOW_DISTANCE:


            # Smoothly slow down.

            factor = (

                front
                -
                FRONT_STOP_DISTANCE
            ) / (

                FRONT_SLOW_DISTANCE
                -
                FRONT_STOP_DISTANCE
            )


            factor = max(
                0.15,
                min(
                    1.0,
                    factor
                )
            )


            if vel_x > 0:

                vel_x *= factor


        # ====================================================
        # SIDE SAFETY
        # ====================================================

        if (
            left < SIDE_CRITICAL_DISTANCE
            and
            vel_y > 0
        ):

            vel_y = 0.0


        if (
            right < SIDE_CRITICAL_DISTANCE
            and
            vel_y < 0
        ):

            vel_y = 0.0


        # ====================================================
        # WALL CENTERING
        # ====================================================
        #
        # Between x=4.2 and 5.8 the geometric target is Y=0.45.
        #
        # Range correction is intentionally small.
        #

        if (
            WALL_ENTRY_X
            <=
            x
            <=
            WALL_EXIT_X
        ):


            center_error = (
                WALL_CENTER_Y
                -
                y
            )


            vel_y += (
                0.40
                *
                center_error
            )


            sensor_error = (
                left
                -
                right
            )


            vel_y += (
                0.10
                *
                sensor_error
            )


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
        # ====================================================

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
        # ====================================================

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
        # TIMING MARKERS
        # ====================================================

        if (
            column1_clear_time is None
            and
            x >= 2.25
        ):

            column1_clear_time = now


            print(
                f"Drone {drone_id}: "
                f"Column 1 cleared = "
                f"{now - mission_start_time:.2f}s"
            )


        if (
            column2_clear_time is None
            and
            x >= 3.35
        ):

            column2_clear_time = now


            print(
                f"Drone {drone_id}: "
                f"Column 2 cleared = "
                f"{now - mission_start_time:.2f}s"
            )


        if (
            wall_entry_time is None
            and
            x >= WALL_ENTRY_X
        ):

            wall_entry_time = now


            print(
                f"Drone {drone_id}: "
                f"Wall corridor entered = "
                f"{now - mission_start_time:.2f}s"
            )


        if (
            wall_exit_time is None
            and
            x >= WALL_EXIT_X
        ):

            wall_exit_time = now


            print(
                f"Drone {drone_id}: "
                f"Walls cleared = "
                f"{now - mission_start_time:.2f}s"
            )


        # ====================================================
        # WAYPOINT REACHED
        # ====================================================

        horizontal_speed = sqrt(

            vx_world * vx_world
            +
            vy_world * vy_world
        )


        if (

            distance < 0.10

            and

            horizontal_speed < 0.18

        ):


            print(
                f"Drone {drone_id}: "
                f"WP{waypoint_index + 1}/"
                f"{len(path)} reached"
            )


            waypoint_index += 1


            reset_pid()


            if (
                waypoint_index
                >=
                len(path)
            ):


                final_time = now


                print()
                print(
                    "********************************************"
                )


                print(
                    f"Drone {drone_id}: "
                    "FINAL FORMATION REACHED"
                )


                print(
                    f"Position = "
                    f"({x:.2f}, "
                    f"{y:.2f}, "
                    f"{z:.2f})"
                )


                print(
                    f"Mission time = "
                    f"{now - mission_start_time:.2f}s"
                )


                if wall_exit_time is not None:

                    print(
                        f"Obstacle traversal = "
                        f"{wall_exit_time - mission_start_time:.2f}s"
                    )


                print(
                    "********************************************"
                )


                print()


                reset_pid()


                state = STATE_FINAL_HOLD


    # ========================================================
    # FINAL FORMATION HOLD
    # ========================================================

    elif state == STATE_FINAL_HOLD:


        height_desired = (
            FLYING_ALTITUDE
        )


        forward_desired = 0.0

        sideways_desired = 0.0

        yaw_desired = 0.0


        # Keep checking camera passively,
        # but it does not steer the drone.

        if (
            now
            >=
            LANDING_START_TIME
        ):


            landing_start_time = now


            print(
                f"Drone {drone_id}: "
                "LANDING START"
            )


            reset_pid()


            state = STATE_LAND


    # ========================================================
    # LAND
    # ========================================================

    elif state == STATE_LAND:


        height_desired = LAND_HEIGHT

        forward_desired = 0.0

        sideways_desired = 0.0

        yaw_desired = 0.0


        if z < 0.10:


            m1_motor.setVelocity(0.0)
            m2_motor.setVelocity(0.0)
            m3_motor.setVelocity(0.0)
            m4_motor.setVelocity(0.0)


            landing_complete_time = now


            obstacle_duration = None

            landing_duration = None

            total_duration = None


            if (
                wall_exit_time is not None
                and
                mission_start_time is not None
            ):

                obstacle_duration = (

                    wall_exit_time
                    -
                    mission_start_time
                )


            if (
                landing_start_time is not None
            ):

                landing_duration = (

                    landing_complete_time
                    -
                    landing_start_time
                )


            if (
                mission_start_time is not None
            ):

                total_duration = (

                    landing_complete_time
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
                    f"Obstacle traversal = "
                    f"{obstacle_duration:.2f}s"
                )


            if landing_duration is not None:

                print(
                    f"Landing duration = "
                    f"{landing_duration:.2f}s"
                )


            if total_duration is not None:

                print(
                    f"TOTAL mission time = "
                    f"{total_duration:.2f}s"
                )


            print(
                "############################################"
            )


            print()


            save_results()


            state = STATE_DONE


    # ========================================================
    # DONE
    # ========================================================

    elif state == STATE_DONE:


        m1_motor.setVelocity(0.0)
        m2_motor.setVelocity(0.0)
        m3_motor.setVelocity(0.0)
        m4_motor.setVelocity(0.0)


        continue


    # ========================================================
    # ALTITUDE SAFETY
    # ========================================================

    if z < ALTITUDE_THRESHOLD:


        forward_desired = 0.0

        sideways_desired = 0.0

        height_desired = (
            FLYING_ALTITUDE
        )


    # ========================================================
    # PID
    # ========================================================

    try:

        motor_power = PID.pid(

            dt,

            forward_desired,

            sideways_desired,

            yaw_desired,

            height_desired,

            roll,

            pitch,

            yaw_rate,

            z,

            vx_body,

            vy_body
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
    # NAN / INF PROTECTION
    # ========================================================

    safe_power = []


    for p in motor_power:


        p = finite_value(
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
    # MOTOR COMMAND
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
    # VISION STATUS REPORT
    # ========================================================

    if (
        vision_active
        and
        now - last_vision_time
        < 0.05
    ):

        # This block is deliberately silent most cycles.
        # The first detection is reported above.
        pass


    # ========================================================
    # UPDATE
    # ========================================================

    past_time = now

    past_x = x

    past_y = y
