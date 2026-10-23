# -*- coding: utf-8 -*-
"""
file: crazyflie_py_wallfollowing.py
Fixed: High-speed emergency sliding evasion, Motor cut on landing.
"""

from controller import Robot, Keyboard
from math import cos, sin, sqrt, atan2, pi
from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

FLYING_ATTITUDE = 1

def plot_and_save_map(trajectory, obstacles, start, end, filename='mission_map.png'):
    if not trajectory: return
    plt.clf()
    traj = np.array(trajectory)
    plt.plot(traj[:,0], traj[:,1], 'b-', label='Trajectory')
    if obstacles: plt.scatter(np.array(obstacles)[:,0], np.array(obstacles)[:,1], c='red', marker='^', label='Obstacles')
    if start: plt.plot(start[0], start[1], 'go', markersize=12)
    if end: plt.plot(end[0], end[1], 'y*', markersize=15)
    plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig(filename, dpi=150); print(f"Map saved: {filename}")
    plt.draw(); plt.pause(0.5); plt.show(block=False)

def save_data_to_csv(trajectory, obstacles, start, end, filename='mission_data.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Type','X','Y'])
        if start: writer.writerow(['Start', start[0], start[1]])
        for x,y in trajectory: writer.writerow(['Trajectory', x, y])
        for x,y in obstacles: writer.writerow(['Obstacle', x, y])
        if end: writer.writerow(['End', end[0], end[1]])
    print(f"Data saved: {filename}")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
m1_motor = robot.getDevice("m1_motor"); m1_motor.setPosition(float('inf')); m1_motor.setVelocity(-1)
m2_motor = robot.getDevice("m2_motor"); m2_motor.setPosition(float('inf')); m2_motor.setVelocity(1)
m3_motor = robot.getDevice("m3_motor"); m3_motor.setPosition(float('inf')); m3_motor.setVelocity(-1)
m4_motor = robot.getDevice("m4_motor"); m4_motor.setPosition(float('inf')); m4_motor.setVelocity(1)

# Sensors
imu = gyro = gps = camera = range_front = range_left = range_right = range_back = None
has_imu=has_gyro=has_gps=has_camera=has_range_front=has_range_left=has_range_right=has_range_back=False
try:
    imu = robot.getDevice("inertial_unit"); imu.enable(timestep); has_imu = True
except: pass
try:
    gps = robot.getDevice("gps"); gps.enable(timestep); has_gps = True
except: pass
try:
    gyro = robot.getDevice("gyro"); gyro.enable(timestep); has_gyro = True
except: pass
try:
    camera = robot.getDevice("camera"); camera.enable(timestep); has_camera = True
except: pass
try:
    range_front = robot.getDevice("range_front"); range_front.enable(timestep); has_range_front = True
except: pass
try:
    range_left = robot.getDevice("range_left"); range_left.enable(timestep); has_range_left = True
except: pass
try:
    range_right = robot.getDevice("range_right"); range_right.enable(timestep); has_range_right = True
except: pass
try:
    range_back = robot.getDevice("range_back"); range_back.enable(timestep); has_range_back = True
except: pass

keyboard = Keyboard(); keyboard.enable(timestep)

past_x = past_y = past_z = 0.0; past_time = 0.0; first_time = True
POSITION_THRESHOLD = 0.001; last_pos = None

PID = pid_velocity_fixed_height_controller()
height_desired = FLYING_ATTITUDE
wall_follower = WallFollowing(angle_value_buffer=0.01, reference_distance_from_wall=0.5,
                              max_forward_speed=0.3, init_state=WallFollowing.StateWallFollowing.FORWARD)
autonomous_mode = False

mission_active = False; mission_state = "IDLE"
takeoff_start_time = 0.0; avoid_start_time = 0.0; last_M_press = 0.0
avoid_direction = 1
AVOID_TIMEOUT = 10.0; WALL_FOLLOW_DIST = 0.5; WALL_FOLLOW_SPEED = 0.12
RETURN_SPEED = 0.2
start_x0=0.0; start_y0=0.0; target_x=0.0; target_y=10.0

# ========= پارامترهای کنترلی =========
Kp_line = 0.25
Kd_line = 0.08
Kp_yaw  = 0.50
# ====================================

prev_cross_error = 0.0; prev_along_error = 0.0

# ========= افزایش فاصله تشخیص برای جلوگیری از برخورد =========
OBSTACLE_DETECTION_DIST = 1.5
EMERGENCY_STOP_DIST = 0.25
SAFE_PASSAGE_WIDTH = 0.5
# ============================================================

previous_sideways = 0.0; SIDEWAYS_SLEW = 0.4 # افزایش نرخ واکنش جانبی
previous_yaw_des = 0.0; YAW_SLEW = 0.4

trajectory_points=[]; obstacle_points=[]; start_pos=None; end_pos=None; map_plotted=False
last_obs_pos=None; OBS_MIN_DIST=0.3

WIDE_SIDE_THRESHOLD = 0.4
WIDE_TIME_THRESHOLD = 1.0
wide_timer = 0.0
STUCK_TIME_THRESHOLD = 1.5
stuck_timer = 0.0

# متغیرهای حالت SLIDING
slide_start_time = 0.0
slide_direction = 1

CLIMB_RATE = 0.3; CLIMB_FWD = 0.1; CLIMB_TIMEOUT = 10.0; CLIMB_CLEARANCE = 2.0; MAX_CLIMB = 3.0
CROSS_ERROR_RETURN_THRESHOLD = 0.05

# ========= فرود نرم و قطع موتور =========
LANDING_DESCENT_RATE = 0.15
LANDING_SLOW_RATE = 0.04
landing_start_altitude = 0.0
landing_start_time = 0.0
# =========================================

print("=== Fixed: 1.5m Early Detection & Emergency Slide ===")

while robot.step(timestep) != -1:
    dt = robot.getTime() - past_time
    if dt <= 0: dt = 0.001

    if first_time:
        if has_gps:
            past_x, past_y, past_z = gps.getValues()
        past_time = robot.getTime()
        first_time = False

    roll=pitch=yaw=yaw_rate=0.0
    x_global=y_global=altitude=0.0
    vx_global=vy_global=vz_global=0.0
    if has_imu:
        roll, pitch, yaw = imu.getRollPitchYaw()
    if has_gyro:
        yaw_rate = gyro.getValues()[2]
    if has_gps:
        x_global, y_global, altitude = gps.getValues()
        vx_global = (x_global - past_x) / dt
        vy_global = (y_global - past_y) / dt
        vz_global = (altitude - past_z) / dt

    rf = 10.0; rl = 10.0; rr = 10.0
    if has_range_front: rf = range_front.getValue() / 1000.0
    if has_range_left:  rl = range_left.getValue() / 1000.0
    if has_range_right: rr = range_right.getValue() / 1000.0

    if has_gps:
        cur = (x_global, y_global, altitude)
        if last_pos is None or any(abs(a-b)>POSITION_THRESHOLD for a,b in zip(last_pos, cur)):
            print(f"Pos:({x_global:.2f},{y_global:.2f}) Alt:{altitude:.2f} Fr:{rf:.2f} L:{rl:.2f} R:{rr:.2f} State:{mission_state}", end='\r')
            last_pos = cur

    cos_yaw = cos(yaw); sin_yaw = sin(yaw)
    vx_body = vx_global * cos_yaw + vy_global * sin_yaw
    vy_body = -vx_global * sin_yaw + vy_global * cos_yaw

    forward_des = sideways_des = yaw_des = 0.0; height_diff = 0.0

    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP: forward_des += 0.5
        elif key == Keyboard.DOWN: forward_des -= 0.5
        elif key == Keyboard.RIGHT: sideways_des -= 0.5
        elif key == Keyboard.LEFT: sideways_des += 0.5
        elif key == ord('Q'): yaw_des = 1
        elif key == ord('E'): yaw_des = -1
        elif key == ord('W'): height_diff = 0.1
        elif key == ord('S'): height_diff = -0.1
        elif key == ord('A'): autonomous_mode = True; print("\nAuto ON")
        elif key == ord('D'): autonomous_mode = False; print("\nAuto OFF")
        elif key == ord('M'):
            t = robot.getTime()
            if t - last_M_press > 0.3:
                last_M_press = t
                if not mission_active:
                    if mission_state == "IDLE":
                        mission_active = True; mission_state = "TAKEOFF"
                        takeoff_start_time = t
                        if has_gps: start_x0, start_y0 = x_global, y_global
                        start_pos = (start_x0, start_y0)
                        trajectory_points.clear(); obstacle_points.clear()
                        last_obs_pos = None; map_plotted = False
                        print(f"\nMission started. Target:({target_x},{target_y})")
                else:
                    if mission_state not in ("DONE","IDLE"):
                        mission_active = False; mission_state = "IDLE"
                        end_pos = (x_global, y_global)
                        print("\nMission aborted.")
                        if not map_plotted:
                            plot_and_save_map(trajectory_points, obstacle_points, start_pos, end_pos)
                            save_data_to_csv(trajectory_points, obstacle_points, start_pos, end_pos)
                            map_plotted = True
                    elif mission_state == "DONE":
                        mission_active = False; mission_state = "IDLE"
                        print("\nReset to IDLE")
        key = keyboard.getKey()

    height_desired += height_diff * dt

    if autonomous_mode:
        cmd_x, cmd_y, cmd_w, _ = wall_follower.wall_follower(rf, rr, yaw, WallFollowing.WallFollowingDirection.LEFT, robot.getTime())
        forward_des = cmd_x; sideways_des = cmd_y; yaw_des = cmd_w

    roll_deg = abs(roll)*180/pi; pitch_deg = abs(pitch)*180/pi
    critical = (roll_deg>60 or pitch_deg>60)

    if mission_active and has_gps:
        if not critical:
            trajectory_points.append((x_global, y_global))
        if 0.01 < rf < 0.5:
            ox = x_global + rf*cos_yaw; oy = y_global + rf*sin_yaw
            if last_obs_pos is None or sqrt((ox-last_obs_pos[0])**2+(oy-last_obs_pos[1])**2) > OBS_MIN_DIST:
                obstacle_points.append((ox, oy)); last_obs_pos = (ox, oy)

        dx = target_x - start_x0; dy = target_y - start_y0
        L = sqrt(dx*dx+dy*dy)
        cross_error = (dy*(x_global-start_x0) - dx*(y_global-start_y0))/L if L>0.001 else 0.0
        cross_deriv = (cross_error - prev_cross_error)/dt; prev_cross_error = cross_error
        dist_target = sqrt((x_global-target_x)**2 + (y_global-target_y)**2)

        angle_to_target = atan2(target_y - y_global, target_x - x_global)
        yaw_err = (angle_to_target - yaw + pi)%(2*pi) - pi
        
        target_yaw_des = max(-0.5, min(0.5, Kp_yaw * yaw_err))
        yaw_des = previous_yaw_des + (target_yaw_des - previous_yaw_des) * min(1.0, YAW_SLEW * dt)
        previous_yaw_des = yaw_des

        obstacle_in_front = (0.01 < rf < OBSTACLE_DETECTION_DIST)

        if obstacle_in_front and mission_state not in ("DONE","SLIDING","CLIMB_WIDE","CLIMB_DESCEND","WALL_FOLLOW","RETURN"):
            if (rl < WIDE_SIDE_THRESHOLD) and (rr < WIDE_SIDE_THRESHOLD):
                wide_timer += dt
            else:
                wide_timer = 0.0
        else:
            wide_timer = 0.0

        speed_global = sqrt(vx_global**2 + vy_global**2)
        if mission_state == "FORWARD":
            if obstacle_in_front and speed_global < 0.05:
                stuck_timer += dt
            else:
                stuck_timer = 0.0
        else:
            stuck_timer = 0.0

        if mission_state not in ("CLIMB_WIDE","CLIMB_DESCEND","WALL_FOLLOW","RETURN","DONE","SLIDING"):
            if wide_timer >= WIDE_TIME_THRESHOLD:
                print("\n>>> Wide wall confirmed! CLIMB_WIDE.")
                mission_state = "CLIMB_WIDE"
                climb_start = robot.getTime()
                height_desired = altitude + 0.5
                wide_timer = 0.0
            elif stuck_timer >= STUCK_TIME_THRESHOLD:
                print("\n>>> Stuck confirmed! Entering WALL_FOLLOW.")
                mission_state = "WALL_FOLLOW"
                avoid_start_time = robot.getTime()
                avoid_direction = 1 if rl > rr else -1
                stuck_timer = 0.0

        # ===== ماشین حالت =====
        if mission_state == "TAKEOFF":
            height_desired = 1.0; forward_des = 0; sideways_des = 0; yaw_des = 0
            if altitude >= 0.95:
                mission_state = "FORWARD"; print("\nTakeoff done.")

        elif mission_state == "FORWARD":
            height_desired = 1.0
            desired_side = -Kp_line * cross_error - Kd_line * cross_deriv

            # ========= حالت جدید: عبور از ستون با سر خوردن (SLIDING) =========
            if obstacle_in_front:
                # اگر سنسورهای چپ و راست دور هستند و مانع فقط در جلوست (ستون)
                if rl > 0.5 and rr > 0.5:
                    print(f"\n>>> Pillar at {rf:.2f}m! AGGRESSIVE SLIDE.")
                    mission_state = "SLIDING"
                    slide_start_time = robot.getTime()
                    slide_direction = 1.0 if rl > rr else -1.0
                    # ترمز کامل و حرکت جانبی سریع
                    forward_des = 0.0
                    sideways_des = 0.0
                    yaw_des = 0.0
                    continue
                else:
                    # موانع باریک یا دیوار (با سرعت کم از کنار برو)
                    forward_des = 0.10
                    gap_side = 1.0 if rl > rr else -1.0
                    desired_side += gap_side * 0.25
            else:
                # مسیر آزاد
                forward_des = 0.35
            
            # ========= ترمز تدریجی در نزدیکی مقصد =========
            if dist_target < 0.5 and mission_state == "FORWARD":
                forward_des = min(0.25, max(0.0, dist_target * 0.6))
            # ==============================================
            
            target_sideways = max(-0.25, min(0.25, desired_side))
            delta_side = target_sideways - previous_sideways
            if abs(delta_side) > SIDEWAYS_SLEW*dt: 
                delta_side = SIDEWAYS_SLEW*dt * (1 if delta_side>0 else -1)
            sideways_des = previous_sideways + delta_side
            previous_sideways = sideways_des

            if dist_target < 0.15:
                forward_des = 0.0; sideways_des = 0.0; yaw_des = 0.0
                previous_yaw_des = 0.0
                mission_state = "DONE"
                landing_start_altitude = altitude
                landing_start_time = robot.getTime()
                print("\nTarget reached! Starting soft landing...")
                if not map_plotted:
                    plot_and_save_map(trajectory_points, obstacle_points, start_pos, end_pos)
                    save_data_to_csv(trajectory_points, obstacle_points, start_pos, end_pos)
                    map_plotted = True

        # ========= حالت سر خوردن جانبی (ترکیب ترمز شدید + رول) =========
        elif mission_state == "SLIDING":
            elapsed = robot.getTime() - slide_start_time
            height_desired = 1.0
            
            # مرحله 1: ترمز کامل و سر خوردن جانبی (برای 0.5 ثانیه)
            if elapsed < 0.6:
                forward_des = 0.0
                sideways_des = slide_direction * 0.3  # اعمال زاویه رول برای عبور نرم
                yaw_des = slide_direction * 0.4       # چرخش همزمان
            # مرحله 2: توقف سر خوردن و بررسی مسیر
            else:
                forward_des = 0.0
                sideways_des = 0.0
                yaw_des = 0.0
                
                if rf > 1.2:
                    print("\n>>> Slide successful. Resuming FORWARD.")
                    mission_state = "FORWARD"
                elif elapsed > 2.5:
                    print("\n>>> Slide timeout. Switching to WALL_FOLLOW.")
                    mission_state = "WALL_FOLLOW"
                    avoid_start_time = robot.getTime()
                    avoid_direction = 1 if rl > rr else -1
        # ===========================================================

        elif mission_state == "CLIMB_WIDE":
            height_desired = min(altitude + CLIMB_RATE*dt, MAX_CLIMB)
            forward_des = CLIMB_FWD; sideways_des = 0; yaw_des = max(-0.3, min(0.3, Kp_yaw*yaw_err))
            if rf > CLIMB_CLEARANCE:
                print("\nCleared wide obstacle."); mission_state = "FORWARD"; height_desired = 1.0
            elif altitude >= MAX_CLIMB - 0.1 and rf < CLIMB_CLEARANCE:
                print("\nWall too high – switching to WALL_FOLLOW.")
                mission_state = "WALL_FOLLOW"; avoid_start_time = robot.getTime()
                height_desired = 1.0; avoid_direction = 1 if rl > rr else -1
            elif robot.getTime() - climb_start > CLIMB_TIMEOUT:
                mission_state = "FORWARD"; height_desired = 1.0

        elif mission_state == "WALL_FOLLOW":
            height_desired = 1.0; forward_des = WALL_FOLLOW_SPEED
            side = rr if avoid_direction==1 else rl
            if side>10 or side<0.01: side = 0.5
            err_side = side - WALL_FOLLOW_DIST
            raw_side = -avoid_direction * 0.4 * err_side
            raw_side = max(-0.2, min(0.2, raw_side))
            
            delta = raw_side - previous_sideways
            if abs(delta) > SIDEWAYS_SLEW*dt: delta = SIDEWAYS_SLEW*dt * (1 if delta>0 else -1)
            sideways_des = previous_sideways + delta; previous_sideways = sideways_des
            
            yaw_des = max(-0.2, min(0.2, 0.2*Kp_yaw*yaw_err))

            if rf < 0.3:
                print("\n>>> Blocked in WALL_FOLLOW. Direct SLIDE.")
                mission_state = "SLIDING"
                slide_start_time = robot.getTime()
                slide_direction = 1 if rl > rr else -1
                continue

            if (rl < WIDE_SIDE_THRESHOLD) and (rr < WIDE_SIDE_THRESHOLD):
                wide_timer += dt
                if wide_timer >= WIDE_TIME_THRESHOLD:
                    print("\nWall detected during follow – CLIMB_WIDE.")
                    mission_state = "CLIMB_WIDE"; climb_start = robot.getTime()
                    height_desired = altitude + 0.5; wide_timer = 0.0
            else:
                wide_timer = 0.0

            if (rf > 1.0 or rf < 0.01) and side > 0.8:
                mission_state = "RETURN"; avoid_start_time = robot.getTime()
                print("\nPassed obstacle, returning to line.")
            elif robot.getTime() - avoid_start_time > AVOID_TIMEOUT:
                mission_state = "FORWARD"; print("\nWALL_FOLLOW timeout.")

        elif mission_state == "RETURN":
            height_desired = 1.0
            forward_des = 0.25
            
            cross_clamped = max(-0.2, min(0.2, cross_error))
            desired_sideways = -0.4 * cross_clamped - 0.1 * cross_deriv
            desired_sideways = max(-0.2, min(0.2, desired_sideways))

            delta = desired_sideways - previous_sideways
            max_delta = SIDEWAYS_SLEW * dt
            if abs(delta) > max_delta:
                delta = max_delta * (1 if delta > 0 else -1)
            sideways_des = previous_sideways + delta
            previous_sideways = sideways_des

            yaw_des = max(-0.3, min(0.3, Kp_yaw * yaw_err))
            
            if obstacle_in_front:
                print("\nObstacle in RETURN! SLIDING.")
                mission_state = "SLIDING"
                slide_start_time = robot.getTime()
                slide_direction = 1 if rl > rr else -1
                continue

            elif abs(cross_error) < CROSS_ERROR_RETURN_THRESHOLD and robot.getTime() - avoid_start_time > 1.0:
                mission_state = "FORWARD"
                print("\nReturned to line.")
            elif robot.getTime() - avoid_start_time > 3.0:
                mission_state = "FORWARD"
                print("\nReturn timeout.")

        # ========= فرود نرم و قطع موتور (رفع کامل یویو) =========
        elif mission_state == "DONE":
            if altitude > 0.2:
                rate = LANDING_DESCENT_RATE
            else:
                rate = LANDING_SLOW_RATE
                
            height_desired = max(0.0, altitude - (rate * dt))
            forward_des = 0.0; sideways_des = 0.0; yaw_des = 0.0
            
            # اگر ارتفاع به زیر 3 سانتی‌متر رسید، موتورها را قطع کن تا پرش نکند
            if altitude < 0.03:
                m1_motor.setVelocity(0.0)
                m2_motor.setVelocity(0.0)
                m3_motor.setVelocity(0.0)
                m4_motor.setVelocity(0.0)
                print("\nLanded perfectly (Motor Cut).")
                mission_active = False
                mission_state = "IDLE"
                # ادامه نده تا دوباره فرمان ندهد
                continue
        # ========================================================

    # ترمز اضطراری نهایی (در صورت خطا)
    if 0.01 < rf < EMERGENCY_STOP_DIST and mission_state != "DONE":
        forward_des = 0.0
        sideways_des = 0.0
        yaw_des = 0.0
        if mission_state not in ("EMERGENCY_STOP"):
            mission_state = "EMERGENCY_STOP"

    elif mission_state == "EMERGENCY_STOP":
        forward_des = 0.0; sideways_des = 0.0; yaw_des = 0.0
        if speed_global < 0.02:
            mission_state = "FORWARD"

    forward_des = max(0.0, min(0.35, forward_des))
    sideways_des = max(-0.3, min(0.3, sideways_des))

    try:
        motor_power = PID.pid(dt, forward_des, sideways_des, yaw_des, height_desired,
                              roll, pitch, yaw_rate, altitude, vx_body, vy_body, vz_global)
    except TypeError:
        motor_power = PID.pid(dt, forward_des, sideways_des, yaw_des, height_desired,
                              roll, pitch, yaw_rate, altitude, vx_body, vy_body)

    m1_motor.setVelocity(-motor_power[0]); m2_motor.setVelocity(motor_power[1])
    m3_motor.setVelocity(-motor_power[2]); m4_motor.setVelocity(motor_power[3])

    past_time = robot.getTime()
    if has_gps: past_x, past_y, past_z = x_global, y_global, altitude