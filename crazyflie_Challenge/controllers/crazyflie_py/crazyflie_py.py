# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2023 Bitcraze


"""
file: crazyflie_py_wallfollowing.py

Controls the crazyflie and implements a wall following method in webots in Python

Author:   Kimberly McGuire (Bitcraze AB)
"""


from controller import Robot
from controller import Keyboard

from math import cos, sin

from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing
import cv2
import numpy as np


FLYING_ATTITUDE = 1

if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    
    camera2 = robot.getDevice("camera_d1")
    camera2.enable(timestep)
    camera2.recognitionEnable(timestep)
    
    camera3 = robot.getDevice("camera_d2")
    camera3.enable(timestep)
    camera3.recognitionEnable(timestep)
    
    display1=robot.getDevice("display")
    display1.attachCamera(camera)
    
    
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    if camera2 is None:
      print("Camera_d1 is not found")
    else:
        print("Camera_d1 is Active")
    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)
    
    print(robot.getBasicTimeStep())
    
    y_l=np.array([10,100,100])
    y_h=np.array([40,255,255])
    
    def detect_yellow_ball(image):
    
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
        mask=cv2.inRange(hsv,y_l,y_h)
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours,key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(largest)
        if radius >1 :
            return(int(x),int(y),int(radius))
        return None
        
    # Initialize variables

    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE

    wall_following = WallFollowing(angle_value_buffer=0.01, reference_distance_from_wall=0.5,
                                   max_forward_speed=0.3, init_state=WallFollowing.StateWallFollowing.FORWARD)

    autonomous_mode = False

    print("\n")

    print("====== Controls =======\n\n")

    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n ")
    print("- Use W and S to go up and down\n ")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n")

    # Main loop:
    while robot.step(timestep) != -1:

        dt = robot.getTime() - past_time
        actual_state = {}
            
        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT:
                sideways_desired -= 0.5
            elif key == Keyboard.LEFT:
                sideways_desired += 0.5
            elif key == ord('Q'):
                yaw_desired = + 1
            elif key == ord('E'):
                yaw_desired = - 1
            elif key == ord('W'):
                height_diff_desired = 0.1
            elif key == ord('S'):
                height_diff_desired = - 0.1
            elif key == ord('A'):
                if autonomous_mode is False:
                    autonomous_mode = True
                    print("Autonomous mode: ON")
            elif key == ord('D'):
                if autonomous_mode is True:
                    autonomous_mode = False
                    print("Autonomous mode: OFF")
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt
        
        
        raw_image = camera.getImage()
        width,height = camera.getWidth(),camera.getHeight()
        image = np.frombuffer(raw_image,np.uint8).reshape((height,width,4))
        image_bgr = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        
        result = detect_yellow_ball(image_bgr)
        
        if result:
            x,y,radius=result
            fov=camera.getFov()
            aspect_ratio=width/height
            
            angle_x=(x/width-0.5)*fov
            angle_y=(0.5-y/height)*(fov/aspect_ratio)
            print(f"Yellow ball found! Center:({x},{y}),Radius:{radius} pixels")
            print(f"Horizontal angle:{angle_x:.2f}rad,Vertical angle:{angle_y:.2f} rad")
        else:
            print("Yellow Ball not found")
        
        
        gps_position=gps.getValues()
        roll,pitch,yaw=imu.getRollPitchYaw()
        objects=camera2.getRecognitionObjects()
        ball_found=False
        for obj in objects:
            if obj.get_colors() and list(obj.get_colors()[0])==[1,1,0]:
                rel_pos=obj.et_position()
                cos_yaw=math.cos(yaw)
                sin_yaw=math.sin(yaw)
                world_x=gps_position[0]+(rel_pos[0]*cos_yaw-rel_pos[2]*sin_yaw)
                world_y=gps_position[1]+(rel_pos[0]*sin_yaw-rel_pos[2]*cos_yaw)
                world_z=gps_position[2]+rel_pos[1]
                print(f"Yellow ball:X={world_x:.2f},Y={world_y:.2f},Z={world_z:.2f}")
                ball_found=True
                break
        # get range in meters
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000

        # Choose a wall following direction
        # if you choose direction left, use the right range value
        # if you choose direction right, use the left range value
        direction = WallFollowing.WallFollowingDirection.LEFT
        range_side_value = range_right_value

        # Get the velocity commands from the wall following state machine
        cmd_vel_x, cmd_vel_y, cmd_ang_w, state_wf = wall_following.wall_follower(
            range_front_value, range_side_value, yaw, direction, robot.getTime())

        if autonomous_mode:
            sideways_desired = cmd_vel_y
            forward_desired = cmd_vel_x
            yaw_desired = cmd_ang_w

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
