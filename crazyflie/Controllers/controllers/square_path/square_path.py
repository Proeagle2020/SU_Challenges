"""
کنترلر Supervisor برای Crazyflie 2.1 در Webots:
بلند شدن، حرکت در مسیر مربعی، بازگشت به نقطه شروع و فرود.
"""

from controller import Supervisor
import math
import sys

# پارامترهای حرکت
TAKEOFF_HEIGHT = 0.5        # ارتفاع پرواز (متر)
SQUARE_SIDE = 1.0           # طول ضلع مربع (متر)
SPEED = 0.3                 # سرعت حرکت (متر بر ثانیه)
TIME_STEP = 32              # میلی‌ثانیه

# زوایا (ثابت نگه داشتن ربات افقی)
FLAT_ROTATION = [0, 0, 1, 0]   # بدون چرخش (محور افقی)

class CrazyflieSupervisor:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        if self.timestep <= 0:
            self.timestep = TIME_STEP
        
        # گره ربات در صحنه
        self.node = self.robot.getFromDef("CRAZYFLIE")  # نام DEF ربات را تنظیم کنید
        if self.node is None:
            # اگر DEF مشخص نیست، از اولین ربات استفاده کن (احتیاط)
            self.node = self.robot.getSelf()
        if self.node is None:
            print("Error: Robot node not found. Set its DEF name to 'CRAZYFLIE' or adjust code.")
            sys.exit(1)
        
        self.translation_field = self.node.getField("translation")
        self.rotation_field = self.node.getField("rotation")
        
        # موقعیت اولیه
        self.start_pos = self.translation_field.getSFVec3f()
        self.x0, self.y0, self.z0 = self.start_pos
        
        # حالت‌های پرواز
        self.state = "TAKEOFF"
        self.target = list(self.start_pos)  # هدف فعلی
        self.step_counter = 0

    def set_target_position(self, x, y, z):
        self.target = [x, y, z]
    
    def move_to_target(self):
        """حرکت نرم به سمت هدف با درون‌یابی خطی"""
        pos = self.translation_field.getSFVec3f()
        dx = self.target[0] - pos[0]
        dy = self.target[1] - pos[1]
        dz = self.target[2] - pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist < 0.001:  # به هدف رسیدیم
            self.translation_field.setSFVec3f(self.target)
            return True
        
        step_dist = SPEED * (self.timestep / 1000.0)
        if step_dist >= dist:
            self.translation_field.setSFVec3f(self.target)
            return True
        
        # حرکت گام‌به‌گام
        frac = step_dist / dist
        new_x = pos[0] + dx * frac
        new_y = pos[1] + dy * frac
        new_z = pos[2] + dz * frac
        self.translation_field.setSFVec3f([new_x, new_y, new_z])
        return False

    def run(self):
        print("Crazyflie square path controller started.")
        # نقاط مسیر (ترتیب: بلند شدن → جلو → راست → عقب → چپ → برگشت → فرود)
        # حرکت مربع در صفحه XY در ارتفاع ثابت
        takeoff_point = [self.x0, self.y0, self.z0 + TAKEOFF_HEIGHT]
        square_points = [
            [self.x0 + SQUARE_SIDE, self.y0, self.z0 + TAKEOFF_HEIGHT],  # جلو
            [self.x0 + SQUARE_SIDE, self.y0 + SQUARE_SIDE, self.z0 + TAKEOFF_HEIGHT],  # راست
            [self.x0, self.y0 + SQUARE_SIDE, self.z0 + TAKEOFF_HEIGHT],  # عقب
            [self.x0, self.y0, self.z0 + TAKEOFF_HEIGHT]                 # چپ (بازگشت به بالای نقطه شروع)
        ]
        land_point = [self.x0, self.y0, self.z0]  # فرود
        
        # صف مسیر
        path = [takeoff_point] + square_points + [land_point]
        current_target_idx = 0
        
        while self.robot.step(self.timestep) != -1:
            # تنظیم جهت افقی
            self.rotation_field.setSFRotation(FLAT_ROTATION)
            
            # بررسی رسیدن به هدف
            at_target = self.move_to_target()
            
            if at_target:
                # رفتن به هدف بعدی
                if current_target_idx < len(path) - 1:
                    current_target_idx += 1
                    self.set_target_position(*path[current_target_idx])
                    print(f"Moving to {path[current_target_idx]}")
                else:
                    # مأموریت تمام شد
                    print("Mission complete. Landing done.")
                    self.robot.simulationSetMode(self.robot.SIMULATION_MODE_PAUSE)
                    break

# اجرا
controller = CrazyflieSupervisor()
controller.run()