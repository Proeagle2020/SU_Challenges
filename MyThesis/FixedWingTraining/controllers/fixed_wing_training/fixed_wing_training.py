"""Simple educational stabilization example for Webots R2025a.

This controller is a classroom demonstration, not a validated aircraft model.
It only demonstrates conservative damping and altitude-related force feedback.
It does not implement waypoint navigation, target tracking, or mission guidance.
"""
from controller import Supervisor
import math

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
node = robot.getSelf()

MASS = 8.0
G = 9.81
ALTITUDE_REF = 2.0
ALTITUDE_KP = 12.0
ALTITUDE_KD = 7.0
MAX_VERTICAL_CORRECTION = 35.0

print("FixedWingTraining: educational controller started.")
print("Simplified forces only; not a validated aerodynamic simulation.")

while robot.step(timestep) != -1:
    p = node.getPosition()
    v = node.getVelocity()
    y = p[1]
    vy = v[1]

    # Apply a bounded vertical correction around the reference height.
    correction = ALTITUDE_KP * (ALTITUDE_REF - y) - ALTITUDE_KD * vy
    correction = max(-MAX_VERTICAL_CORRECTION,
                     min(MAX_VERTICAL_CORRECTION, correction))

    # Apply force in world coordinates; compensate gravity approximately.
    node.addForce([0.0, MASS * 9.81 + correction, 0.0], False)

    # Gentle angular-rate damping only, for an educational test.
    wx, wy, wz = v[3], v[4], v[5]
    node.addTorque([-2.0 * wx, -2.0 * wy, -2.0 * wz], False)

    if int(robot.getTime() * 2) != int((robot.getTime() - timestep / 1000.0) * 2):
        print(f"t={robot.getTime():5.1f}s  altitude={y:5.2f} m  vertical_speed={vy:5.2f} m/s")
