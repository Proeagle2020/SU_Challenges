"""
Crazyflie Controller – Takeoff, Hover, Controlled Landing
Uses altitude ramp for smooth descent with limited vertical speed.
"""

from controller import Robot
from controller import Camera

from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing
import math

class PID:
    def __init__(self, kp, ki, kd, output_limit=(-0.6, 0.6), integral_limit=0.08):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.output_limit = output_limit
        self.integral_limit = integral_limit

    def update(self, measurement, setpoint, dt):
        error = setpoint - measurement
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(self.output_limit[0], min(self.output_limit[1], output))
        
class Camera (Device):
    def enable(self, samplingPeriod):

class CrazyflieController:
    def __init__(self):
        print("\n=== CONTROLLED LANDING CONTROLLER ===\n")
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Motors (official signs: m1-, m2+, m3-, m4+)
        motor_names = ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]
        self.motors = []
        self.MAX_VEL = 600.0
        self.signs = [-1, 1, -1, 1]
        for name in motor_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
                self.motors.append(motor)
        print(f"✅ {len(self.motors)} motors initialized.")

        # Sensors
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.timestep)
        self.imu = self.robot.getDevice("inertial_unit")
        if self.imu:
            self.imu.enable(self.timestep)

        # State variables
        self.altitude = 0.0
        self.prev_altitude = 0.0
        self.vertical_velocity = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.time = 0.0

        # PID controllers
        self.pid_z = PID(1.2, 0.15, 0.5, output_limit=(-0.5, 0.5), integral_limit=0.08)
        self.pid_roll = PID(1.0, 0.0, 0.2, output_limit=(-0.15, 0.15))
        self.pid_pitch = PID(1.0, 0.0, 0.2, output_limit=(-0.15, 0.15))
        self.pid_yaw = PID(0.8, 0.0, 0.1, output_limit=(-0.1, 0.1))

        # Base thrust – start low and increase if needed
        self.base_thrust = 0.30   # adjust so drone hovers at ~1m with small error

        # Flight state
        self.state = "TAKEOFF"
        self.target_alt = 1.0
        self.hover_duration = 5.0
        self.hover_timer = 0.0

        # Landing ramp
        self.landing_target_alt = 0.0
        self.descent_rate = 0.25   # m/s (gentle)
        self.landing_started = False

        print("=== Ready. Press 'A' to reset ===\n")

    def update_sensors(self):
        if self.gps:
            vals = self.gps.getValues()
            self.altitude = vals[2]
        if self.imu:
            rpy = self.imu.getRollPitchYaw()
            self.roll = rpy[0]
            self.pitch = rpy[1]
            self.yaw = rpy[2]

    def set_motor_velocities(self, thrusts):
        for i, thrust in enumerate(thrusts):
            thrust = max(0.0, min(1.0, thrust))
            speed = thrust * self.MAX_VEL * self.signs[i]
            self.motors[i].setVelocity(speed)

    def compute_mixer(self, thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd):
        total = max(0.05, min(0.95, thrust_cmd))
        m1 = total - roll_cmd - pitch_cmd + yaw_cmd
        m2 = total + roll_cmd - pitch_cmd - yaw_cmd
        m3 = total - roll_cmd + pitch_cmd - yaw_cmd
        m4 = total + roll_cmd + pitch_cmd + yaw_cmd
        return [max(0.0, min(1.0, m)) for m in [m1, m2, m3, m4]]

    def run(self):
        prev_time = self.robot.getTime()
        while self.robot.step(self.timestep) != -1:
            now = self.robot.getTime()
            dt = now - prev_time
            prev_time = now
            self.time = now

            self.update_sensors()

            # Compute vertical velocity (simple differentiation)
            self.vertical_velocity = (self.altitude - self.prev_altitude) / dt if dt > 0 else 0.0
            self.prev_altitude = self.altitude

            # Determine altitude setpoint
            if self.state == "TAKEOFF" or self.state == "HOVER":
                alt_sp = self.target_alt
            elif self.state == "LANDING":
                # Ramp down from current altitude to 0
                if not self.landing_started:
                    self.landing_target_alt = self.altitude
                    self.landing_started = True
                    print("🛬 Landing... (descent rate: {:.2f} m/s)".format(self.descent_rate))
                # Decrease target altitude at a fixed rate
                self.landing_target_alt -= self.descent_rate * dt
                if self.landing_target_alt < 0.0:
                    self.landing_target_alt = 0.0
                alt_sp = self.landing_target_alt
            else:
                alt_sp = self.altitude

            # Altitude PID
            z_out = self.pid_z.update(self.altitude, alt_sp, dt)
            thrust_cmd = self.base_thrust + z_out
            thrust_cmd = max(0.05, min(0.95, thrust_cmd))

            # Attitude stabilisation
            roll_cmd = self.pid_roll.update(self.roll, 0.0, dt)
            pitch_cmd = self.pid_pitch.update(self.pitch, 0.0, dt)
            yaw_cmd = self.pid_yaw.update(self.yaw, 0.0, dt)

            # Mix and apply
            motor_thrusts = self.compute_mixer(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd)
            self.set_motor_velocities(motor_thrusts)

            # Debug (every 0.5s)
            if int(self.time * 2) % 1 == 0 and int(self.time * 2) != int((self.time - dt) * 2):
                print(f"T:{self.time:.1f}  Alt:{self.altitude:.2f}  sp:{alt_sp:.2f}  vz:{self.vertical_velocity:.2f}  thrust:{thrust_cmd:.3f}")

            # State machine
            if self.state == "TAKEOFF":
                if self.altitude >= self.target_alt - 0.05:
                    print(f"✅ Takeoff complete at {self.altitude:.2f} m")
                    self.state = "HOVER"
                    self.hover_timer = 0.0

            elif self.state == "HOVER":
                self.hover_timer += dt
                if self.hover_timer >= self.hover_duration:
                    print(f"⏱️ Hover finished. Starting landing ramp...")
                    self.state = "LANDING"
                    self.landing_started = False   # will re‑init in the landing state

            elif self.state == "LANDING":
                # When close to ground and slow, cut motors
                if self.altitude < 0.05 and abs(self.vertical_velocity) < 0.05:
                    self.set_motor_velocities([0.0, 0.0, 0.0, 0.0])
                    print("✅ Landed safely!")
                    self.state = "DONE"

            elif self.state == "DONE":
                self.set_motor_velocities([0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    ctrl = CrazyflieController()
    ctrl.run()