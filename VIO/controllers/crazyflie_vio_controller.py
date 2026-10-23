"""
کنترلر CrazyfliePlus برای شبیه‌سازی VIO
- مدل موتور واقعی (thrust = MOTOR_CONSTANT * omega^2)
- PID وضعیت + ارتفاع + کنترل دستی
- ذخیره‌سازی فریم‌ها و لاگ IMU
"""

from controller import Robot, Keyboard
import math
import os
import csv
import numpy as np
import cv2

# ----- ثابت‌های فیزیکی Crazyflie (از نمونه Webots) -----
GRAVITY = 9.81
MASS = 0.027           # kg
ARM_LENGTH = 0.046     # m (فاصله موتور تا مرکز)
MOTOR_CONSTANT = 2.16e-8  # thrust = k * omega^2
MAX_RPM = 22000
MAX_THRUST_PER_MOTOR = MOTOR_CONSTANT * (MAX_RPM * 2 * math.pi / 60) ** 2
HOVER_THRUST_PER_MOTOR = MASS * GRAVITY / 4   # 0.066 N

# ----- ضرایب PID -----
KP_ROLL_PITCH = 250.0   # افزایش یافته برای پایداری بهتر
KD_ROLL_PITCH = 100.0
KP_ALT = 40.0
KI_ALT = 10.0
KD_ALT = 20.0
KP_YAW_RATE = 2.0

# ----- پارامترهای شبیه‌سازی -----
TIME_STEP = 32  # میلی‌ثانیه
TARGET_ALTITUDE = 0.5  # متر
OUTPUT_DIR = "vio_data"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)
CSV_FILE = os.path.join(OUTPUT_DIR, "imu_log.csv")

# ----- کلاس کنترلر -----
class CrazyflieController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.keyboard = Keyboard()
        self.keyboard.enable(self.time_step)

        # حسگرها
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.tof = self.robot.getDevice("tof")
        self.tof.enable(self.time_step)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)
        self.cam_width = self.camera.getWidth()
        self.cam_height = self.camera.getHeight()

        # موتورها
        motor_names = ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]
        self.motors = []
        for name in motor_names:
            motor = self.robot.getDevice(name)
            motor.setPosition(float('inf'))  # حالت سرعت
            motor.setVelocity(0.0)
            self.motors.append(motor)

        # متغیرهای PID
        self.alt_integral = 0.0
        self.prev_alt_error = 0.0
        self.frame_count = 0
        self.rpy = [0,0,0]  # رول، پیچ، یاو

        # فایل CSV
        self.csv_file = open(CSV_FILE, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp', 'roll', 'pitch', 'yaw', 'altitude'])

    def thrust_to_omega(self, thrust):
        """تبدیل نیروی تراست (N) به سرعت زاویه‌ای (rad/s) با در نظر گرفتن علامت"""
        if abs(thrust) < 1e-10:
            return 0.0
        omega = math.sqrt(abs(thrust) / MOTOR_CONSTANT)
        return math.copysign(omega, thrust)

    def read_sensors(self):
        """خواندن مقادیر حسگرها"""
        # IMU: زوایای اویلر (رادیان)
        r, p, y = self.imu.getRollPitchYaw()
        # ارتفاع از سنسور ToF (فاصله تا زمین به متر)
        alt = self.tof.getValue()
        # سرعت‌های زاویه‌ای (رادیان بر ثانیه)
        gx, gy, gz = self.gyro.getValues()
        # تصویر دوربین
        img = self.camera.getImage()
        if img:
            frame = np.frombuffer(img, np.uint8).reshape((self.cam_height, self.cam_width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            frame = None
        return r, p, y, alt, gx, gy, gz, frame

    def control_loop(self):
        while self.robot.step(self.time_step) != -1:
            r, p, y, alt, gx, gy, gz, frame = self.read_sensors()
            t = self.robot.getTime()

            # ذخیره‌سازی داده‌ها (هر فریم)
            self.writer.writerow([t, r, p, y, alt])
            if frame is not None:
                fname = os.path.join(FRAME_DIR, f"frame_{self.frame_count:06d}.png")
                cv2.imwrite(fname, frame)
                self.frame_count += 1

            # ----- دریافت فرمان از کیبورد -----
            vx, vy, yaw_rate = 0.0, 0.0, 0.0
            key = self.keyboard.getKey()
            while key != -1:
                if key == ord('W'): vx = 0.3
                elif key == ord('S'): vx = -0.3
                elif key == ord('A'): vy = 0.3
                elif key == ord('D'): vy = -0.3
                elif key == ord('Q'): yaw_rate = 0.5
                elif key == ord('E'): yaw_rate = -0.5
                elif key == ord('R'): 
                    global TARGET_ALTITUDE
                    TARGET_ALTITUDE += 0.05
                elif key == ord('F'): 
                    TARGET_ALTITUDE -= 0.05
                    if TARGET_ALTITUDE < 0.1: TARGET_ALTITUDE = 0.1
                key = self.keyboard.getKey()

            # ----- کنترل ارتفاع (PID) -----
            alt_error = TARGET_ALTITUDE - alt
            self.alt_integral += alt_error * (self.time_step / 1000.0)
            alt_deriv = (alt_error - self.prev_alt_error) / (self.time_step / 1000.0)
            self.prev_alt_error = alt_error
            thrust_total = MASS * GRAVITY + KP_ALT * alt_error + KI_ALT * self.alt_integral + KD_ALT * alt_deriv
            thrust_total = max(0.0, min(thrust_total, 4 * MAX_THRUST_PER_MOTOR))

            # ----- کنترل وضعیت (حلقه PD) -----
            # زوایای مطلوب از ورودی سرعت افقی (تقریبی)
            des_roll = -0.15 * vy      # vy مثبت -> رول منفی (شیب به چپ)
            des_pitch = 0.15 * vx      # vx مثبت -> پیچ مثبت (شیب به جلو)
            # محدود کردن زاویه‌ها
            des_roll = max(-0.4, min(des_roll, 0.4))
            des_pitch = max(-0.4, min(des_pitch, 0.4))

            roll_error = des_roll - r
            pitch_error = des_pitch - p
            # مشتق با استفاده از ژیروسکوپ
            roll_rate_error = - gx         # gx سرعت رول است
            pitch_rate_error = - gy        # gy سرعت پیچ

            # گشتاورهای کنترلی
            tau_x = KP_ROLL_PITCH * roll_error + KD_ROLL_PITCH * roll_rate_error
            tau_y = KP_ROLL_PITCH * pitch_error + KD_ROLL_PITCH * pitch_rate_error
            tau_z = KP_YAW_RATE * yaw_rate  # کنترل نرخ یاو

            # ----- تخصیص تراست به موتورها (پیکربندی X) -----
            # ترتیب موتورها: m1 (جلو-راست)، m2 (عقب-چپ)، m3 (جلو-چپ)، m4 (عقب-راست)
            # محاسبه نیروی هر موتور با مخلوط‌کن استاندارد
            m1 = thrust_total/4 + tau_x/(4*ARM_LENGTH) + tau_y/(4*ARM_LENGTH) - tau_z/4
            m2 = thrust_total/4 - tau_x/(4*ARM_LENGTH) - tau_y/(4*ARM_LENGTH) - tau_z/4
            m3 = thrust_total/4 - tau_x/(4*ARM_LENGTH) + tau_y/(4*ARM_LENGTH) + tau_z/4
            m4 = thrust_total/4 + tau_x/(4*ARM_LENGTH) - tau_y/(4*ARM_LENGTH) + tau_z/4

            # محدود کردن نیروها
            forces = [max(0.0, min(f, MAX_THRUST_PER_MOTOR)) for f in (m1, m2, m3, m4)]

            # تبدیل نیرو به سرعت موتور و اعمال
            for i, f in enumerate(forces):
                omega = self.thrust_to_omega(f)
                # موتور در Webots سرعت را به rad/s نیاز دارد
                self.motors[i].setVelocity(omega)

        self.csv_file.close()
        print(f"پایان شبیه‌سازی. {self.frame_count} تصویر ذخیره شد.")

# اجرا
controller = CrazyflieController()
controller.control_loop()