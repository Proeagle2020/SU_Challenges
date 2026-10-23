"""
Spot robot - stable forward walking with minimal lateral deviation.
Cameras enabled for visualization (view via Scene Tree).
"""

from controller import Robot
import math
import sys

NUMBER_OF_JOINTS = 12

MOTOR_NAMES = [
    "front left shoulder abduction motor",
    "front left shoulder rotation motor",
    "front left elbow motor",
    "front right shoulder abduction motor",
    "front right shoulder rotation motor",
    "front right elbow motor",
    "rear left shoulder abduction motor",
    "rear left shoulder rotation motor",
    "rear left elbow motor",
    "rear right shoulder abduction motor",
    "rear right shoulder rotation motor",
    "rear right elbow motor"
]

LED_NAMES = [
    "left top led", "left middle up led", "left middle down led", "left bottom led",
    "right top led", "right middle up led", "right middle down led", "right bottom led"
]

CAMERA_NAMES = [
    "left head camera", "right head camera", "left flank camera",
    "right flank camera", "rear camera"
]

robot = Robot()
time_step = int(robot.getBasicTimeStep())

motors = []
leds = []
cameras = []

# Initialize motors
for name in MOTOR_NAMES:
    motors.append(robot.getDevice(name))

# LEDs on
for name in LED_NAMES:
    led = robot.getDevice(name)
    leds.append(led)
    led.set(1)

# Cameras enabled
for name in CAMERA_NAMES:
    cam = robot.getDevice(name)
    cameras.append(cam)
    cam.enable(2 * time_step)
    print(f"Camera '{name}' enabled.")

def step():
    if robot.step(time_step) == -1:
        sys.exit(0)

def set_joint_positions(targets):
    for i in range(NUMBER_OF_JOINTS):
        motors[i].setPosition(targets[i])

def movement_decomposition(target, duration):
    n_steps = int(duration * 1000 / time_step)
    current_pos = [motor.getTargetPosition() for motor in motors]
    step_diff = [(target[i] - current_pos[i]) / n_steps for i in range(NUMBER_OF_JOINTS)]
    for _ in range(n_steps):
        for i in range(NUMBER_OF_JOINTS):
            current_pos[i] += step_diff[i]
            motors[i].setPosition(current_pos[i])
        step()

def walk_forward(t, speed=5):
    """
    Trot gait: diagonal pairs (FL+RR) and (FR+RL) move together.
    Minimal abduction to avoid lateral tilt.
    """
    abd = 0.02  # almost zero abduction

    freq = speed * 2.5
    phase = 2 * math.pi * freq * t

    hip_amp = 0.10
    knee_amp = 0.12

    # FL and RR (diagonal pair 1)
    fl_hip = hip_amp * math.sin(phase)
    fl_knee = knee_amp * math.sin(phase + math.pi/2) + 0.05

    # FR and RL (diagonal pair 2, opposite phase)
    fr_hip = hip_amp * math.sin(phase + math.pi)
    fr_knee = knee_amp * math.sin(phase + math.pi + math.pi/2) + 0.05

    rl_hip = fr_hip
    rl_knee = fr_knee
    rr_hip = fl_hip
    rr_knee = fl_knee

    fl_abd = -abd
    fr_abd = abd
    rl_abd = -abd
    rr_abd = abd

    target = [
        fl_abd, fl_hip, fl_knee,
        fr_abd, fr_hip, fr_knee,
        rl_abd, rl_hip, rl_knee,
        rr_abd, rr_hip, rr_knee
    ]

    # Clamp to safe ranges
    for i in [2, 5, 8, 11]:
        if target[i] < -0.35:
            target[i] = -0.35
        elif target[i] > 1.2:
            target[i] = 1.2
    return target

def main():
    # Initial standing with slight knee bend
    stand = [
        -0.02, 0.0, 0.05,
         0.02, 0.0, 0.05,
        -0.02, 0.0, 0.05,
         0.02, 0.0, 0.05
    ]
    movement_decomposition(stand, 1.5)

    # Continuous walking
    while True:
        t = robot.getTime()
        target = walk_forward(t, speed=0.15)
        set_joint_positions(target)
        step()

if __name__ == "__main__":
    main()