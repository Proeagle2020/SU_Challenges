# -*- coding: utf-8 -*-

from controller import Robot
from controller import Keyboard
from math import cos, sin, sqrt

import os
import csv

from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing


# ============================================================
# CONFIGURATION
# ============================================================

FLYING_ATTITUDE = 1.0

ALTITUDE_STEP = 0.10

MIN_ALTITUDE = 0.15
MAX_ALTITUDE = 2.50

GROUND_Z = 0.005

# Repeat interval when W/S is held
ALTITUDE_REPEAT_INTERVAL = 0.40


# ============================================================
# PATH
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

TRAJECTORY_FILE = os.path.join(
    DATA_DIR,
    "leader_trajectory.csv"
)

START_FLAG = os.path.join(
    DATA_DIR,
    "START_REPLAY.flag"
)


# ============================================================
# ROBOT
# ============================================================

robot = Robot()

timestep = int(
    robot.getBasicTimeStep()
)


# ============================================================
# DATA DIRECTORY
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


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
# REMOVE OLD LEADER CSV
# ============================================================

try:

    if os.path.exists(
        TRAJECTORY_FILE
    ):
        os.remove(
            TRAJECTORY_FILE
        )

except Exception:
    pass


# ============================================================
# MOTORS
# ============================================================

m1_motor = robot.getDevice(
    "m1_motor"
)

m1_motor.setPosition(
    float("inf")
)

m1_motor.setVelocity(
    -1
)


m2_motor = robot.getDevice(
    "m2_motor"
)

m2_motor.setPosition(
    float("inf")
)

m2_motor.setVelocity(
    1
)


m3_motor = robot.getDevice(
    "m3_motor"
)

m3_motor.setPosition(
    float("inf")
)

m3_motor.setVelocity(
    -1
)


m4_motor = robot.getDevice(
    "m4_motor"
)

m4_motor.setPosition(
    float("inf")
)

m4_motor.setVelocity(
    1
)


# ============================================================
# SENSORS
# ============================================================

imu = robot.getDevice(
    "inertial_unit"
)

imu.enable(
    timestep
)


gps = robot.getDevice(
    "gps"
)

gps.enable(
    timestep
)


gyro = robot.getDevice(
    "gyro"
)

gyro.enable(
    timestep
)


accelerometer = robot.getDevice(
    "accelerometer"
)

accelerometer.enable(
    timestep
)


camera = robot.getDevice(
    "camera"
)

camera.enable(
    timestep
)


range_front = robot.getDevice(
    "range_front"
)

range_front.enable(
    timestep
)


range_left = robot.getDevice(
    "range_left"
)

range_left.enable(
    timestep
)


range_back = robot.getDevice(
    "range_back"
)

range_back.enable(
    timestep
)


range_right = robot.getDevice(
    "range_right"
)

range_right.enable(
    timestep
)


# ============================================================
# KEYBOARD
# ============================================================

keyboard = Keyboard()

keyboard.enable(
    timestep
)


# ============================================================
# ORIGINAL CF1 PID
# ============================================================

PID_crazyflie = pid_velocity_fixed_height_controller()


# ============================================================
# WALL FOLLOWING
# ============================================================

wall_following = WallFollowing(
    angle_value_buffer=0.01,
    reference_distance_from_wall=0.5,
    max_forward_speed=0.3,
    init_state=(
        WallFollowing
        .StateWallFollowing
        .FORWARD
    )
)


# ============================================================
# CONTROL
# ============================================================

height_desired = FLYING_ATTITUDE

autonomous_mode = False

last_altitude_command_time = -999.0


# ============================================================
# SENSOR MEMORY
# ============================================================

past_x_global = 0.0
past_y_global = 0.0
past_z_global = 0.0

past_vx_global = 0.0
past_vy_global = 0.0
past_vz_global = 0.0

past_wx = 0.0
past_wy = 0.0
past_wz = 0.0

past_time = robot.getTime()

first_time = True


# ============================================================
# FLIGHT DETECTION
# ============================================================

takeoff_detected = False

recording_started = False

landing_detected = False

landing_timer = 0.0

LANDING_STABLE_TIME = 1.0


# ============================================================
# CSV
# ============================================================

csv_file = open(
    TRAJECTORY_FILE,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(
    csv_file
)


writer.writerow(
    [
        "time",

        "x",
        "y",
        "z",

        "vx_global",
        "vy_global",
        "vz_global",

        "vx_body",
        "vy_body",

        "speed",

        "roll",
        "pitch",
        "yaw",

        "wx",
        "wy",
        "wz",

        "angular_speed",

        "acc_body_x",
        "acc_body_y",
        "acc_body_z",

        "acc_world_x",
        "acc_world_y",
        "acc_world_z",

        "ax",
        "ay",
        "az",

        "linear_acceleration",

        "alpha_x",
        "alpha_y",
        "alpha_z",

        "angular_acceleration",

        "forward_desired",
        "sideways_desired",

        "yaw_desired",

        "height_desired",

        "motor1",
        "motor2",
        "motor3",
        "motor4",

        "autonomous_mode"
    ]
)


# ============================================================
# HEADER
# ============================================================

print("")
print("==============================================")
print("       CRAZYFLIE 1 - LEADER")
print("==============================================")

print("Controls:")

print(
    "UP ARROW       -> Forward"
)

print(
    "DOWN ARROW     -> Backward"
)

print(
    "LEFT ARROW     -> Left"
)

print(
    "RIGHT ARROW    -> Right"
)

print(
    "Q / q          -> Yaw Left"
)

print(
    "E / e          -> Yaw Right"
)

print(
    "W / w          -> Increase Altitude"
)

print(
    "S / s          -> Decrease Altitude"
)

print(
    "PAGE UP        -> Increase Altitude"
)

print(
    "PAGE DOWN      -> Decrease Altitude"
)

print(
    "A / a          -> Autonomous ON"
)

print(
    "D / d          -> Autonomous OFF"
)

print("")

print(
    f"Altitude step = {ALTITUDE_STEP:.2f} m"
)

print(
    f"Altitude range = "
    f"{MIN_ALTITUDE:.2f} - "
    f"{MAX_ALTITUDE:.2f} m"
)

print(
    f"Trajectory = "
    f"{TRAJECTORY_FILE}"
)

print(
    f"Ground Z = {GROUND_Z:.4f}"
)

print("==============================================")


# ============================================================
# MAIN LOOP
# ============================================================

while robot.step(timestep) != -1:

    current_time = robot.getTime()


    # ========================================================
    # DT
    # ========================================================

    dt = (
        current_time -
        past_time
    )

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

    accelerometer_values = (
        accelerometer.getValues()
    )


    # ========================================================
    # GPS
    # ========================================================

    roll = imu_values[0]
    pitch = imu_values[1]
    yaw = imu_values[2]


    # ========================================================
    # FIRST SAMPLE
    # ========================================================

    if first_time:

        past_x_global = gps_values[0]

        past_y_global = gps_values[1]

        past_z_global = gps_values[2]

        past_time = current_time

        first_time = False

        continue


    # ========================================================
    # POSITION
    # ========================================================

    x_global = gps_values[0]

    y_global = gps_values[1]

    z_global = gps_values[2]


    # ========================================================
    # VELOCITY
    # ========================================================

    vx_global = (
        x_global -
        past_x_global
    ) / dt


    vy_global = (
        y_global -
        past_y_global
    ) / dt


    vz_global = (
        z_global -
        past_z_global
    ) / dt


    # ========================================================
    # LIMIT VELOCITY SPIKES
    # ========================================================

    vx_global = max(
        -5.0,
        min(
            5.0,
            vx_global
        )
    )


    vy_global = max(
        -5.0,
        min(
            5.0,
            vy_global
        )
    )


    vz_global = max(
        -5.0,
        min(
            5.0,
            vz_global
        )
    )


    # ========================================================
    # BODY VELOCITY
    # ========================================================

    cos_yaw = cos(yaw)

    sin_yaw = sin(yaw)


    vx_body = (
        vx_global *
        cos_yaw
        +
        vy_global *
        sin_yaw
    )


    vy_body = (
        -vx_global *
        sin_yaw
        +
        vy_global *
        cos_yaw
    )


    # ========================================================
    # SPEED
    # ========================================================

    speed = sqrt(
        vx_global ** 2
        +
        vy_global ** 2
        +
        vz_global ** 2
    )


    # ========================================================
    # GYRO
    # ========================================================

    wx = gyro_values[0]

    wy = gyro_values[1]

    wz = gyro_values[2]


    angular_speed = sqrt(
        wx ** 2
        +
        wy ** 2
        +
        wz ** 2
    )


    # ========================================================
    # ACCELEROMETER
    # ========================================================

    acc_body_x = accelerometer_values[0]

    acc_body_y = accelerometer_values[1]

    acc_body_z = accelerometer_values[2]


    # ========================================================
    # LINEAR ACCELERATION
    #
    # Derived from GPS velocity.
    # ========================================================

    ax = (
        vx_global -
        past_vx_global
    ) / dt


    ay = (
        vy_global -
        past_vy_global
    ) / dt


    az = (
        vz_global -
        past_vz_global
    ) / dt


    linear_acceleration = sqrt(
        ax ** 2
        +
        ay ** 2
        +
        az ** 2
    )


    # ========================================================
    # ANGULAR ACCELERATION
    # ========================================================

    alpha_x = (
        wx -
        past_wx
    ) / dt


    alpha_y = (
        wy -
        past_wy
    ) / dt


    alpha_z = (
        wz -
        past_wz
    ) / dt


    angular_acceleration = sqrt(
        alpha_x ** 2
        +
        alpha_y ** 2
        +
        alpha_z ** 2
    )


    # ========================================================
    # COMMANDS
    # ========================================================

    forward_desired = 0.0

    sideways_desired = 0.0

    yaw_desired = 0.0


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = keyboard.getKey()


    while key > 0:

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        if key == Keyboard.UP:

            forward_desired += 0.5


        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        elif key == Keyboard.DOWN:

            forward_desired -= 0.5


        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        elif key == Keyboard.LEFT:

            sideways_desired += 0.5


        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        elif key == Keyboard.RIGHT:

            sideways_desired -= 0.5


        # ----------------------------------------------------
        # YAW LEFT
        # ----------------------------------------------------

        elif (
            key == ord("Q")
            or
            key == ord("q")
        ):

            yaw_desired = 1.0


        # ----------------------------------------------------
        # YAW RIGHT
        # ----------------------------------------------------

        elif (
            key == ord("E")
            or
            key == ord("e")
        ):

            yaw_desired = -1.0


        # ====================================================
        # ALTITUDE UP
        # ====================================================

        elif (
            key == ord("W")
            or
            key == ord("w")
            or
            key == Keyboard.PAGEUP
        ):

            if (
                current_time -
                last_altitude_command_time
                >=
                ALTITUDE_REPEAT_INTERVAL
            ):

                height_desired += (
                    ALTITUDE_STEP
                )

                height_desired = max(
                    MIN_ALTITUDE,
                    min(
                        MAX_ALTITUDE,
                        height_desired
                    )
                )

                print(
                    f"ALTITUDE UP -> "
                    f"{height_desired:.2f} m"
                )

                last_altitude_command_time = (
                    current_time
                )


        # ====================================================
        # ALTITUDE DOWN
        # ====================================================

        elif (
            key == ord("S")
            or
            key == ord("s")
            or
            key == Keyboard.PAGEDOWN
        ):

            if (
                current_time -
                last_altitude_command_time
                >=
                ALTITUDE_REPEAT_INTERVAL
            ):

                height_desired -= (
                    ALTITUDE_STEP
                )

                height_desired = max(
                    MIN_ALTITUDE,
                    min(
                        MAX_ALTITUDE,
                        height_desired
                    )
                )

                print(
                    f"ALTITUDE DOWN -> "
                    f"{height_desired:.2f} m"
                )

                last_altitude_command_time = (
                    current_time
                )


        # ====================================================
        # AUTONOMOUS ON
        # ====================================================

        elif (
            key == ord("A")
            or
            key == ord("a")
        ):

            autonomous_mode = True

            print(
                "Autonomous mode: ON"
            )


        # ====================================================
        # AUTONOMOUS OFF
        # ====================================================

        elif (
            key == ord("D")
            or
            key == ord("d")
        ):

            autonomous_mode = False

            print(
                "Autonomous mode: OFF"
            )


        key = keyboard.getKey()


    # ========================================================
    # RANGE SENSOR
    # ========================================================

    range_front_value = (
        range_front.getValue()
        /
        1000.0
    )


    range_right_value = (
        range_right.getValue()
        /
        1000.0
    )


    direction = (
        WallFollowing
        .WallFollowingDirection
        .LEFT
    )


    cmd_vel_x, cmd_vel_y, cmd_ang_w, state_wf = (
        wall_following.wall_follower(
            range_front_value,
            range_right_value,
            yaw,
            direction,
            current_time
        )
    )


    # ========================================================
    # AUTONOMOUS MODE
    # ========================================================

    if autonomous_mode:

        forward_desired = (
            cmd_vel_x
        )

        sideways_desired = (
            cmd_vel_y
        )

        yaw_desired = (
            cmd_ang_w
        )


    # ========================================================
    # ALTITUDE LIMIT
    # ========================================================

    height_desired = max(
        MIN_ALTITUDE,
        min(
            MAX_ALTITUDE,
            height_desired
        )
    )


    # ========================================================
    # PID
    # ========================================================

    motor_power = (
        PID_crazyflie.pid(
            dt,

            forward_desired,
            sideways_desired,
            yaw_desired,

            height_desired,

            roll,
            pitch,
            wz,

            z_global,

            vx_body,
            vy_body
        )
    )


    # ========================================================
    # MOTORS
    # ========================================================

    motor1 = motor_power[0]

    motor2 = motor_power[1]

    motor3 = motor_power[2]

    motor4 = motor_power[3]


    m1_motor.setVelocity(
        -motor1
    )

    m2_motor.setVelocity(
        motor2
    )

    m3_motor.setVelocity(
        -motor3
    )

    m4_motor.setVelocity(
        motor4
    )


    # ========================================================
    # TAKEOFF DETECTION
    # ========================================================

    if (
        not takeoff_detected
        and
        z_global > 0.20
    ):

        takeoff_detected = True

        recording_started = True

        print("")
        print(
            ">>> TAKEOFF DETECTED"
        )

        print(
            ">>> RECORDING STARTED"
        )


    # ========================================================
    # RECORD
    # ========================================================

    if recording_started:

        writer.writerow(
            [
                current_time,

                x_global,
                y_global,
                z_global,

                vx_global,
                vy_global,
                vz_global,

                vx_body,
                vy_body,

                speed,

                roll,
                pitch,
                yaw,

                wx,
                wy,
                wz,

                angular_speed,

                acc_body_x,
                acc_body_y,
                acc_body_z,

                ax,
                ay,
                az,

                ax,
                ay,
                az,

                linear_acceleration,

                alpha_x,
                alpha_y,
                alpha_z,

                angular_acceleration,

                forward_desired,
                sideways_desired,

                yaw_desired,

                height_desired,

                motor1,
                motor2,
                motor3,
                motor4,

                int(
                    autonomous_mode
                )
            ]
        )

        csv_file.flush()


    # ========================================================
    # LANDING DETECTION
    # ========================================================

    if recording_started:

        close_to_ground = (
            z_global <= 0.12
        )

        slow_motion = (
            abs(vx_global) < 0.15
            and
            abs(vy_global) < 0.15
            and
            abs(vz_global) < 0.15
        )

        level = (
            abs(roll) < 0.15
            and
            abs(pitch) < 0.15
        )


        if (
            close_to_ground
            and
            slow_motion
            and
            level
        ):

            landing_timer += dt

        else:

            landing_timer = 0.0


        if (
            landing_timer
            >=
            LANDING_STABLE_TIME
        ):

            landing_detected = True

            print("")
            print(
                "=============================================="
            )

            print(
                "             LEADER LANDED"
            )

            print(
                "=============================================="
            )

            print(
                f"Final X = {x_global:.4f}"
            )

            print(
                f"Final Y = {y_global:.4f}"
            )

            print(
                f"Final Z = {z_global:.4f}"
            )


            # ------------------------------------------------
            # STOP MOTORS
            # ------------------------------------------------

            m1_motor.setVelocity(
                0.0
            )

            m2_motor.setVelocity(
                0.0
            )

            m3_motor.setVelocity(
                0.0
            )

            m4_motor.setVelocity(
                0.0
            )


            # ------------------------------------------------
            # CLOSE CSV
            # ------------------------------------------------

            csv_file.flush()

            csv_file.close()


            # ------------------------------------------------
            # CREATE REPLAY FLAG
            # ------------------------------------------------

            try:

                with open(
                    START_FLAG,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        "START_REPLAY\n"
                    )

                    f.write(
                        f"x={x_global}\n"
                    )

                    f.write(
                        f"y={y_global}\n"
                    )

                    f.write(
                        f"z={z_global}\n"
                    )


                print(
                    ">>> START_REPLAY FLAG CREATED"
                )

                print(
                    START_FLAG
                )

            except Exception as error:

                print(
                    "ERROR creating START_REPLAY.flag:"
                )

                print(error)


            break


    # ========================================================
    # MEMORY
    # ========================================================

    past_time = current_time

    past_x_global = x_global

    past_y_global = y_global

    past_z_global = z_global

    past_vx_global = vx_global

    past_vy_global = vy_global

    past_vz_global = vz_global

    past_wx = wx

    past_wy = wy

    past_wz = wz


# ============================================================
# CLOSE
# ============================================================

try:

    if not csv_file.closed:

        csv_file.flush()
        csv_file.close()

except Exception:

    pass


print(
    "CF1 controller stopped."
)