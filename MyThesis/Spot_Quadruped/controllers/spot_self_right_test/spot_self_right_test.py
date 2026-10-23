
from controller import Robot
import math
import numpy as np

TIME_STEP = 32
MOTOR_SPEED = 2.5
STAND = np.array([-0.10,0,0, 0.10,0,0, -0.10,0,0, 0.10,0,0], dtype=float)

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
    "rear right elbow motor",
]

class StabilityTest:
    def __init__(self):
        self.robot = Robot()
        self.dt = int(self.robot.getBasicTimeStep()) or TIME_STEP
        self.imu = self.robot.getDevice("inertial unit")
        self.gps = self.robot.getDevice("gps")
        self.imu.enable(self.dt)
        self.gps.enable(self.dt)

        self.motors = []
        for name in MOTOR_NAMES:
            m = self.robot.getDevice(name)
            self.motors.append(m)
            if m:
                m.setVelocity(min(m.getMaxVelocity(), MOTOR_SPEED))

        print("=" * 72)
        print("SPOT STABILITY TEST - NO GAIT / NO SUPERVISOR")
        print("Phase 1: move slowly to verified standing pose.")
        print("Phase 2: hold 10 s and report roll/pitch/z.")
        print("No exploration is performed in this test.")
        print("=" * 72)

    def set_targets(self, q):
        for i,m in enumerate(self.motors):
            if not m:
                continue
            v = float(q[i])
            try:
                lo, hi = m.getMinPosition(), m.getMaxPosition()
                if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
                    v = max(lo+0.03, min(hi-0.03, v))
            except Exception:
                pass
            m.setVelocity(min(m.getMaxVelocity(), MOTOR_SPEED))
            m.setPosition(v)

    def pose(self):
        p = self.gps.getValues()
        r = self.imu.getRollPitchYaw()
        return float(p[2]), math.degrees(float(r[0])), math.degrees(float(r[1]))

    def run(self):
        # Ramp from current targets to the safe standing pose.
        start = np.zeros(12, dtype=float)
        for i,m in enumerate(self.motors):
            if m:
                try:
                    start[i] = m.getTargetPosition()
                except Exception:
                    start[i] = 0.0

        duration = 4.0
        steps = max(1, int(duration*1000/self.dt))
        for k in range(steps):
            a = (k+1)/steps
            a = 0.5 - 0.5*math.cos(math.pi*a)
            self.set_targets((1-a)*start + a*STAND)
            if self.robot.step(self.dt) == -1:
                return

        print("[TEST] Standing pose reached. Holding for 10 seconds.")
        t0 = self.robot.getTime()
        last = -1.0
        while self.robot.step(self.dt) != -1:
            now = self.robot.getTime()
            z, roll, pitch = self.pose()
            if now-last > 1.0:
                last = now
                print(f"[STATUS] t={now-t0:5.1f}s z={z:.3f} roll={roll:+.1f} pitch={pitch:+.1f}")
            if now-t0 >= 10.0:
                break

        print("[TEST] Stability test finished.")
        print("[TEST] If roll/pitch stay near zero and z stays near 0.61, proceed to gait test.")

if __name__ == "__main__":
    StabilityTest().run()
