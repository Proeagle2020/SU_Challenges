"""
کنترلر سینماتیک معکوس برای بازوی دو لینکه در Webots
با GPS برای خواندن موقعیت End-Effector
"""

from controller import Robot
import math

# =====================================================
# 1. تنظیمات اولیه
# =====================================================
robot = Robot()
TIME_STEP = 16

# نام موتورها
MOTOR1_NAME = 'motor_1'
MOTOR2_NAME = 'motor_2'

# طول لینک‌ها
L1 = 0.35
L2 = 0.35

# =====================================================
# 2. دریافت و فعال‌سازی GPS
# =====================================================
gps = robot.getDevice("End_gps")  # نام دقیق را از Scene Tree بررسی کنید
if gps is None:
    print("خطا: GPS پیدا نشد!")
    exit(1)
gps.enable(TIME_STEP)

# =====================================================
# 3. دریافت موتورها
# =====================================================
motor1 = robot.getDevice(MOTOR1_NAME)
motor2 = robot.getDevice(MOTOR2_NAME)

if motor1 is None or motor2 is None:
    print("یکی از موتورها پیدا نشد!")
    print("لیست تمام دستگاه‌ها:")
    for device in robot.getDevices():
        print(f"  - {device.getName()}")
    exit(1)

# تنظیم موتورها روی حالت موقعیت (آماده‌سازی)
motor1.setPosition(0)
motor2.setPosition(0)

# =====================================================
# 4. سنسورهای موقعیت (اختیاری - در صورت وجود)
# =====================================================
# اگر سنسور PS_1 و PS_2 را در Scene Tree تعریف کرده‌اید، فعال کنید
# در غیر این صورت، این بخش را کامنت یا حذف کنید
sensor1 = robot.getDevice('PS_1')
sensor2 = robot.getDevice('PS_2')
if sensor1 is not None:
    sensor1.enable(TIME_STEP)
if sensor2 is not None:
    sensor2.enable(TIME_STEP)

# =====================================================
# 5. تعیین مختصات هدف
# =====================================================
x_target =  -0.15
y_target =  0.35

print("شبیه سازی ربات دولینکی")
print("=========================================================")
print(f"هدف: x = {x_target}, y = {y_target}")

# =====================================================
# 6. محاسبه سینماتیک معکوس
# =====================================================
r = math.sqrt(x_target**2 + y_target**2)

if r > L1 + L2 or r < abs(L1 - L2):
    print(f"نقطه ({x_target}, {y_target}) خارج از فضای کاری است.")
    print(f"حداکثر فاصله: {L1 + L2:.2f} متر")
    print(f"حداقل فاصله: {abs(L1 - L2):.2f} متر")
    exit(1)

cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
cos_theta2 = max(-1.0, min(1.0, cos_theta2))

# حالت آرنج بالا
theta2 = math.acos(cos_theta2)
# اگر می‌خواهید آرنج پایین، خط زیر را فعال کنید:
# theta2 = -math.acos(cos_theta2)

phi = math.atan2(y_target, x_target)
beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
theta1 = phi - beta
print("=========================================================")
print(f"زاویه مفصل اول (θ1) = درجه{math.degrees(theta1):.2f} ")
print(f"زاویه مفصل دوم (θ2) = درجه{math.degrees(theta2):.2f} ")
print("=========================================================")
motor1.setPosition(theta1)
motor2.setPosition(theta2)

# =====================================================
# 7. حلقه اصلی شبیه‌سازی
# =====================================================
print("در حال حرکت به نقطه هدف...")

while robot.step(TIME_STEP) != -1:
    # خواندن موقعیت GPS
    pos = gps.getValues()
    x = pos[0]
    y = pos[1]
    speed = gps.getSpeed()
    
    # خواندن سنسورهای موقعیت (در صورت وجود)
    if sensor1 is not None:
        current_pos1 = (sensor1.getValue()*360)/(2 * 3.14)
        current_pos2 = (sensor2.getValue()*360)/(2 * 3.14)
        print(f"موقعیت موتورها: θ1={current_pos1:.3f}, θ2={current_pos2:.3f}")
    
    # نمایش اطلاعات
    print(f"End-Effector: x={x:.3f}, y={y:.3f}, speed={speed:.3f} m/s")

print("به نقطه هدف رسید.")