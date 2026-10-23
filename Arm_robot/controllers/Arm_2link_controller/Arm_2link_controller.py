from controller import Robot, Motor
import math

# 1. ایجاد یک نمونه از ربات
robot = Robot()

# 2. دریافت زمان گام (Timestep) شبیه‌سازی (برای حلقه اصلی مورد نیاز است)
timestep = int(robot.getBasicTimeStep())

# 3. دریافت موتورها (مفاصل)
# ⚠️ نکته بسیار مهم: نام‌های داخل پرانتز (مثلاً 'joint1' و 'joint2')
# باید دقیقاً با نام موتورهایی که در درخت صحنه (Scene Tree) وباتس وارد کرده‌اید، یکی باشد.
motor1 = robot.getDevice('motor_1') # مفصل اول (روی پایه آبی)
motor2 = robot.getDevice('motor_2') # مفصل دوم (روی میله قرمز)

# 4. حلقه اصلی اجرای برنامه
while robot.step(timestep) != -1:
    
    # دریافت زمان جاری شبیه‌سازی برای محاسبه حرکت سینوسی
    current_time = robot.getTime()
    
    # محاسبه زاویه‌ها بر اساس زمان (ایجاد حرکت نرم و پیوسته)
    # زاویه موتور اول بین +1.5 و -1.5 رادیان در نوسان است
    angle1 = 1.5 * math.sin(current_time * 0.5) 
    
    # زاویه موتور دوم بین +1.0 و -1.0 رادیان در نوسان است
    angle2 = 1.0 * math.cos(current_time * 0.3) 
    
    # 5. اعمال زاویه‌های محاسبه شده به موتورها
    motor1.setPosition(angle1)
    motor2.setPosition(angle2)