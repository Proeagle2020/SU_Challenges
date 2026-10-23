# -*- coding: utf-8 -*-

import numpy as np


class pid_velocity_fixed_height_controller():

    def __init__(self):

        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0

        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0

        self.altitude_integrator = 0.0

        self.last_time = 0.0


    def pid(
        self,
        dt,
        desired_vx,
        desired_vy,
        desired_yaw_rate,
        desired_altitude,

        actual_roll,
        actual_pitch,
        actual_yaw_rate,

        actual_altitude,
        actual_vx,
        actual_vy
    ):

        # ====================================================
        # ORIGINAL CF1 GAINS
        # DO NOT CHANGE THESE FOR NOW
        # ====================================================

        gains = {

            "kp_att_y": 1.0,

            "kd_att_y": 0.5,

            "kp_att_rp": 0.5,

            "kd_att_rp": 0.1,

            "kp_vel_xy": 2.0,

            "kd_vel_xy": 0.5,

            "kp_z": 10.0,

            "ki_z": 5.0,

            "kd_z": 5.0
        }


        dt = max(
            dt,
            0.001
        )


        # ====================================================
        # VX
        # ====================================================

        vx_error = (
            desired_vx -
            actual_vx
        )


        vx_deriv = (
            vx_error -
            self.past_vx_error
        ) / dt


        # ====================================================
        # VY
        # ====================================================

        vy_error = (
            desired_vy -
            actual_vy
        )


        vy_deriv = (
            vy_error -
            self.past_vy_error
        ) / dt


        # ====================================================
        # DESIRED PITCH
        # ====================================================

        desired_pitch = (
            gains["kp_vel_xy"]
            *
            np.clip(
                vx_error,
                -1,
                1
            )

            +

            gains["kd_vel_xy"]
            *
            vx_deriv
        )


        # ====================================================
        # DESIRED ROLL
        # ====================================================

        desired_roll = -(
            gains["kp_vel_xy"]
            *
            np.clip(
                vy_error,
                -1,
                1
            )

            +

            gains["kd_vel_xy"]
            *
            vy_deriv
        )


        # Do NOT add the new Follower limitations here.
        # Keep CF1 behavior unchanged.


        self.past_vx_error = (
            vx_error
        )

        self.past_vy_error = (
            vy_error
        )


        # ====================================================
        # ALTITUDE
        # ====================================================

        alt_error = (
            desired_altitude -
            actual_altitude
        )


        alt_deriv = (
            alt_error -
            self.past_alt_error
        ) / dt


        self.altitude_integrator += (
            alt_error *
            dt
        )


        self.altitude_integrator = (
            np.clip(
                self.altitude_integrator,
                -2,
                2
            )
        )


        alt_command = (
            gains["kp_z"]
            *
            alt_error

            +

            gains["kd_z"]
            *
            alt_deriv

            +

            gains["ki_z"]
            *
            self.altitude_integrator

            +

            48
        )


        self.past_alt_error = (
            alt_error
        )


        # ====================================================
        # PITCH ATTITUDE
        # ====================================================

        pitch_error = (
            desired_pitch -
            actual_pitch
        )


        pitch_deriv = (
            pitch_error -
            self.past_pitch_error
        ) / dt


        # ====================================================
        # ROLL ATTITUDE
        # ====================================================

        roll_error = (
            desired_roll -
            actual_roll
        )


        roll_deriv = (
            roll_error -
            self.past_roll_error
        ) / dt


        # ====================================================
        # YAW
        # ====================================================

        yaw_rate_error = (
            desired_yaw_rate -
            actual_yaw_rate
        )


        # ====================================================
        # ATTITUDE COMMANDS
        # ====================================================

        roll_command = (
            gains["kp_att_rp"]
            *
            np.clip(
                roll_error,
                -1,
                1
            )

            +

            gains["kd_att_rp"]
            *
            roll_deriv
        )


        pitch_command = -(
            gains["kp_att_rp"]
            *
            np.clip(
                pitch_error,
                -1,
                1
            )

            +

            gains["kd_att_rp"]
            *
            pitch_deriv
        )


        yaw_command = (
            gains["kp_att_y"]
            *
            np.clip(
                yaw_rate_error,
                -1,
                1
            )
        )


        self.past_pitch_error = (
            pitch_error
        )

        self.past_roll_error = (
            roll_error
        )


        # ====================================================
        # MOTOR MIXER
        # ====================================================

        m1 = (
            alt_command
            -
            roll_command
            +
            pitch_command
            +
            yaw_command
        )


        m2 = (
            alt_command
            -
            roll_command
            -
            pitch_command
            -
            yaw_command
        )


        m3 = (
            alt_command
            +
            roll_command
            -
            pitch_command
            +
            yaw_command
        )


        m4 = (
            alt_command
            +
            roll_command
            +
            pitch_command
            -
            yaw_command
        )


        # ====================================================
        # LIMIT
        # ====================================================

        m1 = np.clip(
            m1,
            0,
            600
        )

        m2 = np.clip(
            m2,
            0,
            600
        )

        m3 = np.clip(
            m3,
            0,
            600
        )

        m4 = np.clip(
            m4,
            0,
            600
        )


        return [
            m1,
            m2,
            m3,
            m4
        ]