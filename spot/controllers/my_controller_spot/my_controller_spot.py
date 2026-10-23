"""
Copyright 1996-2024 Cyberbotics Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Description: Simple controller to present the Spot robot (Python version).
"""

from controller import Robot, Motor, LED, Camera
import math
import sys

NUMBER_OF_LEDS = 8
NUMBER_OF_JOINTS = 12
NUMBER_OF_CAMERAS = 5

# Motor names
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

# Camera names
CAMERA_NAMES = [
    "left head camera",
    "right head camera",
    "left flank camera",
    "right flank camera",
    "rear camera"
]

# LED names
LED_NAMES = [
    "left top led",
    "left middle up led",
    "left middle down led",
    "left bottom led",
    "right top led",
    "right middle up led",
    "right middle down led",
    "right bottom led"
]

# Global robot instance and devices
robot = Robot()
time_step = int(robot.getBasicTimeStep())

motors = []
cameras = []
leds = []

def step():
    """Perform one simulation step and exit if the simulation is over."""
    if robot.step(time_step) == -1:
        sys.exit(0)

def movement_decomposition(target, duration):
    """
    Smoothly move all joints from their current target positions to the given target
    positions over the specified duration (in seconds).
    """
    n_steps = int(duration * 1000 / time_step)
    current_pos = [motor.getTargetPosition() for motor in motors]
    step_diff = [(target[i] - current_pos[i]) / n_steps for i in range(NUMBER_OF_JOINTS)]

    for _ in range(n_steps):
        for i in range(NUMBER_OF_JOINTS):
            current_pos[i] += step_diff[i]
            motors[i].setPosition(current_pos[i])
        step()

def lie_down(duration):
    target = [
        -0.40, -0.99, 1.59,   # Front left leg
         0.40, -0.99, 1.59,   # Front right leg
        -0.40, -0.99, 1.59,   # Rear left leg
         0.40, -0.99, 1.59    # Rear right leg
    ]
    movement_decomposition(target, duration)

def stand_up(duration):
    target = [
        -0.1, 0.0, 0.0,   # Front left leg
         0.1, 0.0, 0.0,   # Front right leg
        -0.1, 0.0, 0.0,   # Rear left leg
         0.1, 0.0, 0.0    # Rear right leg
    ]
    movement_decomposition(target, duration)

def sit_down(duration):
    target = [
        -0.20, -0.40, -0.19,  # Front left leg
         0.20, -0.40, -0.19,  # Front right leg
        -0.40, -0.90,  1.18,  # Rear left leg
         0.40, -0.90,  1.18   # Rear right leg
    ]
    movement_decomposition(target, duration)

def give_paw():
    # Stabilize posture
    target1 = [
        -0.20, -0.30,  0.05,  # Front left leg
         0.20, -0.40, -0.19,  # Front right leg
        -0.40, -0.90,  1.18,  # Rear left leg
         0.49, -0.90,  0.80   # Rear right leg
    ]
    movement_decomposition(target1, 4)

    # Wave the front right leg
    start_time = robot.getTime()
    while robot.getTime() - start_time < 8:
        motors[4].setPosition(0.2 * math.sin(2 * robot.getTime()) + 0.6)  # shoulder rotation
        motors[5].setPosition(0.4 * math.sin(2 * robot.getTime()))        # elbow
        step()

    # Return to sitting posture
    target2 = [
        -0.20, -0.40, -0.19,  # Front left leg
         0.20, -0.40, -0.19,  # Front right leg
        -0.40, -0.90,  1.18,  # Rear left leg
         0.40, -0.90,  1.18   # Rear right leg
    ]
    movement_decomposition(target2, 4)

def main():
    # Get and enable cameras (only the two front cameras are enabled)
    for name in CAMERA_NAMES:
        cam = robot.getDevice(name)
        cameras.append(cam)
    cameras[0].enable(2 * time_step)
    cameras[1].enable(2 * time_step)

    # Get LEDs and turn them on
    for name in LED_NAMES:
        led = robot.getDevice(name)
        leds.append(led)
        led.set(1)

    # Get motors
    for name in MOTOR_NAMES:
        motors.append(robot.getDevice(name))

    # Main loop: perform a sequence of movements
    while True:
        lie_down(4.0)
        stand_up(4.0)
        sit_down(4.0)
        give_paw()
        stand_up(4.0)
        lie_down(3.0)
        stand_up(3.0)
        lie_down(2.0)
        stand_up(2.0)
        lie_down(1.0)
        stand_up(1.0)
        lie_down(0.75)
        stand_up(0.75)
        lie_down(0.5)
        stand_up(0.5)

if __name__ == "__main__":
    main()