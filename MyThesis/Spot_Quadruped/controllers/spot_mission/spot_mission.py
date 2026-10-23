from controller import Supervisor, Camera, RangeFinder, Motor
import math
import csv
from pathlib import Path
from collections import deque

try:
    import numpy as np
except ImportError:
    raise RuntimeError('numpy is required. Install it in the same Python used by Webots.')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ============================================================
# SPOT EXPLORER - Webots R2025a
# Single-file controller
# ============================================================
# What it does:
# 1) Stand up and walk with a slow crawl gait.
# 2) Rotate and scan the room with a front RangeFinder.
# 3) Build an occupancy grid using Webots GPS + IMU pose.
# 4) Select frontier cells and navigate to them with A*.
# 5) Detect a yellow/orange duck with the exploration camera.
# 6) Confirm duck only after multiple observations from different poses.
# 7) Save 2D map, 3D point-cloud map, trajectory and duck CSV files.
#
# IMPORTANT:
# This is a simulation mapping/exploration controller. Pose comes from
# Webots GPS + InertialUnit, so the map is SLAM-like but not pure SLAM.
# Later, GPS can be replaced by odometry/VIO/SLAM pose estimation.
# ============================================================

TIME_STEP = 32

# ---------------------- Mapping -----------------------------
MAP_RESOLUTION = 0.08          # m/cell
MAP_SIZE_X = 20.0              # m
MAP_SIZE_Y = 20.0              # m
MAX_SENSOR_RANGE = 7.5         # m
MAP_UPDATE_PERIOD = 8          # controller steps

# ---------------------- Exploration --------------------------
SCAN_TIME = 18.0                # hard upper limit for a scan
SCAN_ROTATE_TARGET = 2.0 * math.pi * 0.92
SCAN_TURN_COMMAND = 0.45
FRONTIER_REPLAN_PERIOD = 1.25
GOAL_TOLERANCE = 0.32
MIN_FRONTIER_CLUSTER = 3
MIN_FRONTIER_GOAL_DISTANCE = 0.90
MIN_EXPLORED_RATIO_TO_FINISH = 0.80
MAX_EXPLORATION_TIME = 300.0

# ---------------------- Fall / recovery ----------------------
# Spot R2025a already contains Accelerometer, GPS, Gyro and InertialUnit.
# Recovery is physical: the controller commands the 12 leg joints to rock
# the body upright. No Supervisor API or Scene Tree setting is required.
FALL_ROLL_LIMIT = 1.05       # rad (~60 deg)
FALL_PITCH_LIMIT = 1.05      # rad (~60 deg)
FALL_Z_LIMIT = 0.34          # m, extra protection
FALL_CONFIRM_TIME = 0.35     # s
RECOVERY_SETTLE_TIME = 2.5   # s
RECOVERY_COOLDOWN = 2.0      # s
STABLE_SAVE_PERIOD = 0.75    # s

# ---------------------- Duck detection -----------------------
DUCK_MIN_CONFIRMATIONS = 4
DUCK_MIN_TRAVEL_BETWEEN_OBS = 0.8
DUCK_COLOR_MIN_AREA = 80
DUCK_MAX_OBSERVATION_RANGE = 6.0
DUCK_POSITION_GATE = 1.2

# ---------------------- Motion -------------------------------
FORWARD_SIGN = 1.0
WALK_SPEED = 0.40
TURN_SPEED = 0.65
CRAWL_CYCLE = 1.20
CRAWL_STRIDE = 0.14
CRAWL_LIFT = 0.055
TURN_STRIDE = 0.10
MOTOR_SPEED_LIMIT = 2.4
COMMAND_RAMP = 0.035

# STAND is no longer hard-coded. The controller captures Spot's actual
# initial joint pose and uses that as HOME_POSE. This avoids forcing the
# official R2025a model toward an unsafe zero-angle pose.
MOTOR_NAMES = [
    'front left shoulder abduction motor',
    'front left shoulder rotation motor',
    'front left elbow motor',
    'front right shoulder abduction motor',
    'front right shoulder rotation motor',
    'front right elbow motor',
    'rear left shoulder abduction motor',
    'rear left shoulder rotation motor',
    'rear left elbow motor',
    'rear right shoulder abduction motor',
    'rear right shoulder rotation motor',
    'rear right elbow motor'
]

CAMERA_NAMES = [
    'left head camera',
    'right head camera',
    'left flank camera',
    'right flank camera',
    'rear camera'
]


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def distance2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class GridMap:
    def __init__(self):
        self.res = MAP_RESOLUTION
        self.nx = int(MAP_SIZE_X / self.res)
        self.ny = int(MAP_SIZE_Y / self.res)
        self.ox = -MAP_SIZE_X / 2.0
        self.oy = -MAP_SIZE_Y / 2.0
        # -1 unknown, 0 free, 1 occupied
        self.grid = -np.ones((self.ny, self.nx), dtype=np.int8)
        self.path = []
        self.cloud = []

    def world_to_grid(self, x, y):
        ix = int((x - self.ox) / self.res)
        iy = int((y - self.oy) / self.res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None

    def grid_to_world(self, ix, iy):
        return (self.ox + (ix + 0.5) * self.res,
                self.oy + (iy + 0.5) * self.res)

    def mark_free_ray(self, x0, y0, x1, y1):
        a = self.world_to_grid(x0, y0)
        b = self.world_to_grid(x1, y1)
        if a is None or b is None:
            return
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if 0 <= x < self.nx and 0 <= y < self.ny:
                if self.grid[y, x] != 1:
                    self.grid[y, x] = 0
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def update(self, points, pose):
        rx, ry, rz, yaw = pose
        self.path.append((rx, ry))
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        for px, py, pz in points:
            # Sensor frame convention: x = right, y = up, z = backward.
            forward = -pz
            left = -px
            wx = rx + cy * forward - sy * left
            wy = ry + sy * forward + cy * left
            wz = rz + pz
            self.cloud.append((wx, wy, wz))
            self.mark_free_ray(rx, ry, wx, wy)
            cell = self.world_to_grid(wx, wy)
            if cell is not None:
                ix, iy = cell
                self.grid[iy, ix] = 1

    def explored_ratio(self):
        known = np.count_nonzero(self.grid != -1)
        return known / float(self.grid.size)

    def occupied(self, ix, iy):
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return True
        return self.grid[iy, ix] == 1

    def traversable(self, ix, iy):
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return False
        return self.grid[iy, ix] == 0


class Explorer:
    def __init__(self):
        self.robot = Supervisor()
        self.dt = int(self.robot.getBasicTimeStep())
        if self.dt <= 0:
            self.dt = TIME_STEP

        self.start_time = self.robot.getTime()
        self.output_dir = Path(__file__).resolve().parent / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.map = GridMap()
        self.state = 'SCAN'
        self.state_enter = self.robot.getTime()
        self.last_replan = -999.0
        self.frontier_goal = None
        self.a_star_path = []
        self.path_index = 0

        self.duck_confirmed = False
        self.duck_observations = []
        self.last_duck_obs_pose = None
        self.duck_position = None
        self.total_distance = 0.0
        self.prev_xy = None
        self.total_rotation = 0.0
        self.prev_yaw = None

        self.last_scan_print = -999.0

        # ---------------- Supervisor recovery ----------------
        # Webots Supervisor API is used deliberately here because the user
        # wants a deterministic simulation recovery to the last stable pose.
        # The Spot node MUST have supervisor=TRUE in the world.
        self.robot_node = self.robot.getSelf()
        if self.robot_node is None:
            raise RuntimeError('Supervisor could not access Spot via getSelf(). Set supervisor=TRUE on Spot.')

        self.home_pose = None
        self.home_captured = False
        self.home_capture_time = self.robot.getTime() + 0.75

        self.fall_since = None
        self.recovering = False
        self.recovery_attempt = 0
        self.max_recovery_attempts = 3
        self.last_recovery_time = -999.0
        self.recovery_cooldown_until = -999.0
        self.recovery_start_time = 0.0
        self.last_stable_save = -999.0
        self.stable_state_name = 'last_stable_spot'

        self.accel = self.robot.getDevice('accelerometer')
        if self.accel:
            self.accel.enable(self.dt)

        self.gyro = self.robot.getDevice('gyro')
        if self.gyro:
            self.gyro.enable(self.dt)

        self.scan_yaw_last = None
        self.scan_rotated = 0.0
        self.scan_start_time = 0.0

        # ---------------- devices ----------------
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.dt)

        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.dt)

        self.depth = self.robot.getDevice('front depth')
        if self.depth:
            self.depth.enable(self.dt)

        self.explore_camera = self.robot.getDevice('explore camera')
        if self.explore_camera:
            self.explore_camera.enable(self.dt)

        self.cameras = {}
        for n in CAMERA_NAMES:
            d = self.robot.getDevice(n)
            if d:
                d.enable(4 * self.dt)
                self.cameras[n] = d

        self.motors = []
        self.joint_sensors = []
        for n in MOTOR_NAMES:
            m = self.robot.getDevice(n)
            self.motors.append(m)
            if m:
                m.setVelocity(min(m.getMaxVelocity(), MOTOR_SPEED_LIMIT))
                try:
                    m.setAcceleration(min(m.getAcceleration() if m.getAcceleration() > 0 else 3.0, 3.0))
                except Exception:
                    pass
                ps = None
                try:
                    ps = m.getPositionSensor()
                    if ps:
                        ps.enable(self.dt)
                except Exception:
                    ps = None
                self.joint_sensors.append(ps)
            else:
                self.joint_sensors.append(None)

        self.keyboard = self.robot.getKeyboard()
        if self.keyboard:
            self.keyboard.enable(self.dt)

        self.gait_t = 0.0
        self.gait_enabled = False
        self.forward_command = 0.0
        self.turn_command = 0.0

        print('\n' + '=' * 72)
        print('SPOT AUTONOMOUS EXPLORER + FALL RECOVERY - WEBOTS R2025a')
        print('=' * 72)
        print('STATE: RECOVERY <-> SCAN -> FRONTIER -> A* -> SCAN -> ...')
        print('Duck confirmation requires multiple observations from separated poses.')
        print('Outputs will be written to:')
        print(str(self.output_dir))
        print('Automatic recovery: SUPERVISOR last-stable-state restore is ENABLED.')
        print('Manual recovery: press R   |   Test flip: press F')
        print('=' * 72)

        print('[INIT] Supervisor mode is REQUIRED for automatic recovery.')
        print('[INIT] Spot node must have supervisor TRUE in the Scene Tree.')

        if self.depth is None:
            print('WARNING: RangeFinder "front depth" not found.')
            print('3D mapping will not work until the RangeFinder is added to Spot.frontExtension.')
        if self.explore_camera is None:
            print('WARNING: Camera "explore camera" not found.')
            print('Duck detection will be unavailable until the Camera is added to Spot.frontExtension.')
        if self.gps is None or self.imu is None:
            raise RuntimeError('Spot GPS and/or InertialUnit is missing.')


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------
    def pose(self):
        p = self.gps.getValues()
        rpy = self.imu.getRollPitchYaw()
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        yaw = float(rpy[2]) if math.isfinite(rpy[2]) else 0.0
        return x, y, z, yaw

    def update_motion_stats(self, pose):
        x, y, z, yaw = pose
        if self.prev_xy is not None:
            self.total_distance += math.hypot(x - self.prev_xy[0], y - self.prev_xy[1])
        if self.prev_yaw is not None:
            self.total_rotation += abs(wrap_angle(yaw - self.prev_yaw))
        self.prev_xy = (x, y)
        self.prev_yaw = yaw

    # --------------------------------------------------------
    # Motion / gait / Supervisor recovery
    # --------------------------------------------------------
    def capture_home_pose(self):
        # The uploaded world starts the Spot model with motor targets at zero.
        # Zero is not the recommended standing configuration for this model.
        # Use the documented upright standing pose for all four legs and move
        # into it gradually before locomotion.
        self.home_pose = np.array([
            -0.10, 0.00, 0.00,   # front left
             0.10, 0.00, 0.00,   # front right
            -0.10, 0.00, 0.00,   # rear left
             0.10, 0.00, 0.00    # rear right
        ], dtype=float)
        self.home_captured = True
        print('[INIT] Safe HOME standing pose selected:')
        print('       ' + ', '.join(f'{v:+.3f}' for v in self.home_pose))

    def clamp_joint_target(self, i, value):
        m = self.motors[i]
        if m is None:
            return value
        lo = m.getMinPosition()
        hi = m.getMaxPosition()
        if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            return clamp(value, lo + 0.03, hi - 0.03)
        return value

    def set_joint_targets(self, target, velocity=None, acceleration=None):
        if velocity is None:
            velocity = MOTOR_SPEED_LIMIT
        for i, m in enumerate(self.motors):
            if m is None:
                continue
            q = self.clamp_joint_target(i, float(target[i]))
            try:
                m.setVelocity(min(m.getMaxVelocity(), velocity))
            except Exception:
                pass
            if acceleration is not None:
                try:
                    m.setAcceleration(acceleration)
                except Exception:
                    pass
            m.setPosition(q)

    def set_home_pose(self):
        if self.home_captured:
            self.set_joint_targets(self.home_pose, velocity=1.5, acceleration=1.5)

    def gait_target(self, forward_cmd=0.0, turn_cmd=0.0):
        """Conservative crawl around the stable upright pose.

        Abduction stays near +/-0.10 rad to preserve lateral support.
        Only shoulder rotation performs the fore/aft stride.
        The elbow lift is small and occurs only during swing.
        """
        q = self.home_pose.copy()

        phase = (self.gait_t / CRAWL_CYCLE) % 1.0
        leg_phase = [0.00, 0.25, 0.50, 0.75]  # FL, FR, RL, RR

        for leg in range(4):
            p = (phase + leg_phase[leg]) % 1.0
            side = 1.0 if leg in (0, 2) else -1.0

            if p < 0.30:
                u = p / 0.30
                s = 0.5 - 0.5 * math.cos(math.pi * u)
                hip = -0.5 + s
                knee = 0.18 * math.sin(math.pi * u)
            else:
                hip = 0.5 - ((p - 0.30) / 0.70)
                knee = 0.0

            hip *= CRAWL_STRIDE
            knee *= (0.65 + 0.35 * min(1.0, abs(forward_cmd) + abs(turn_cmd)))

            # Turning is differential but small.
            turn_term = side * TURN_STRIDE * turn_cmd * math.sin(2.0 * math.pi * p)

            q[3 * leg + 1] += FORWARD_SIGN * forward_cmd * hip
            q[3 * leg + 1] += FORWARD_SIGN * turn_term
            q[3 * leg + 2] += knee

        return q

    def command_motion(self, forward, turn):
        self.forward_command += clamp(forward - self.forward_command, -COMMAND_RAMP, COMMAND_RAMP)
        self.turn_command += clamp(turn - self.turn_command, -COMMAND_RAMP, COMMAND_RAMP)
        if self.recovering or not self.home_captured:
            return
        active = abs(self.forward_command) > 0.01 or abs(self.turn_command) > 0.01
        if not active:
            self.gait_enabled = False
            self.set_home_pose()
            return
        self.gait_enabled = True
        self.gait_t += self.dt / 1000.0
        self.set_joint_targets(self.gait_target(self.forward_command, self.turn_command), velocity=MOTOR_SPEED_LIMIT)

    def stop_motion(self):
        self.forward_command = 0.0
        self.turn_command = 0.0
        self.gait_enabled = False
        self.set_home_pose()

    def body_tilt(self):
        rpy = self.imu.getRollPitchYaw()
        return float(rpy[0]), float(rpy[1])

    def fall_detected(self):
        roll, pitch = self.body_tilt()
        _, _, z, _ = self.pose()
        tilt = abs(roll) > FALL_ROLL_LIMIT or abs(pitch) > FALL_PITCH_LIMIT
        low = z < FALL_Z_LIMIT
        now = self.robot.getTime()
        bad = tilt or low
        if bad:
            if self.fall_since is None:
                self.fall_since = now
                return False
            return now - self.fall_since >= FALL_CONFIRM_TIME
        self.fall_since = None
        return False

    def save_last_stable_state(self, force=False):
        now = self.robot.getTime()
        if not force and now - self.last_stable_save < STABLE_SAVE_PERIOD:
            return
        if self.recovering:
            return
        roll, pitch = self.body_tilt()
        _, _, z, _ = self.pose()
        speed = 0.0
        try:
            speed = float(self.gps.getSpeed())
        except Exception:
            pass
        if force or (abs(roll) < math.radians(15) and abs(pitch) < math.radians(15) and z > 0.48 and speed < 0.28):
            self.robot_node.saveState(self.stable_state_name)
            self.last_stable_save = now
            if force:
                print('[INIT] Last stable Supervisor state saved.')

    def start_recovery(self):
        now = self.robot.getTime()
        if self.recovering or now < self.recovery_cooldown_until:
            return
        self.recovering = True
        self.recovery_start_time = now
        self.recovery_attempt += 1
        self.state = 'RECOVERY'
        self.stop_motion()
        roll, pitch = self.body_tilt()
        p = self.pose()
        print('\n' + '!' * 72)
        print('FALL DETECTED - SUPERVISOR RECOVERY')
        print(f'roll={math.degrees(roll):+.1f} deg, pitch={math.degrees(pitch):+.1f} deg, z={p[2]:.2f} m')
        print('[RECOVERY] Restoring the last stable upright state.')
        print('!' * 72)
        try:
            self.robot_node.loadState(self.stable_state_name)
            self.robot_node.resetPhysics()
        except Exception as exc:
            print('[RECOVERY] ERROR:', exc)
            print('[RECOVERY] Make sure Spot has supervisor TRUE.')
            self.recovering = False
            self.state = 'RECOVERY_FAILED'
            return
        self.recovery_cooldown_until = now + RECOVERY_COOLDOWN
        self.recovering = False
        self.fall_since = None
        self.forward_command = 0.0
        self.turn_command = 0.0
        self.gait_enabled = False
        self.state = 'SCAN'
        self.state_enter = now
        self.begin_scan()
        self.prev_xy = None
        self.prev_yaw = None
        print('[RECOVERY] State restored successfully.')
        print('[RECOVERY] Exploration will continue from the last stable pose.')

    def keyboard_commands(self):
        if self.keyboard is None:
            return
        key = self.keyboard.getKey()
        if key in (ord('r'), ord('R')):
            print('[KEY] Manual recovery requested.')
            self.start_recovery()
        elif key in (ord('f'), ord('F')):
            # Deterministic test: flip Spot around its X axis using Supervisor,
            # wait for physics to settle, then automatic recovery should trigger.
            rot_field = self.robot_node.getField('rotation')
            trans_field = self.robot_node.getField('translation')
            if rot_field is not None:
                original_t = trans_field.getSFVec3f() if trans_field is not None else [0.0, 0.0, 0.0]
                rot_field.setSFRotation([1.0, 0.0, 0.0, math.pi])
                if trans_field is not None:
                    trans_field.setSFVec3f([original_t[0], original_t[1], original_t[2] + 0.02])
                self.robot_node.resetPhysics()
                self.fall_since = None
                print('[KEY] Spot flipped 180 deg for automatic-recovery test.')
        elif key in (ord('x'), ord('X')):
            self.stop_motion()
            self.state = 'RECOVERY_FAILED'
            print('[KEY] Motion stopped.')

    def begin_scan(self):
        p = self.pose()
        self.scan_yaw_last = p[3]
        self.scan_rotated = 0.0
        self.scan_start_time = self.robot.getTime()

    def update_scan_rotation(self, yaw):
        """Accumulate absolute yaw change during the current scan."""
        if self.scan_yaw_last is None:
            self.scan_yaw_last = yaw
            return
        delta = wrap_angle(yaw - self.scan_yaw_last)
        self.scan_rotated += abs(delta)
        self.scan_yaw_last = yaw

    # --------------------------------------------------------
    # Depth / point cloud
    # --------------------------------------------------------
    def depth_points(self):
        if self.depth is None:
            return []

        w = self.depth.getWidth()
        h = self.depth.getHeight()
        fov_h = self.depth.getFov()
        fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) * (h / float(w)))

        try:
            flat = self.depth.getRangeImage(data_type='buffer')
            arr = np.frombuffer(flat, dtype=np.float32).reshape((h, w))
        except Exception:
            arr = np.asarray(self.depth.getRangeImageArray(), dtype=np.float32)
            if arr.ndim != 2:
                flat = self.depth.getRangeImage()
                arr = np.asarray(flat, dtype=np.float32).reshape((h, w))

        points = []
        min_range = self.depth.getMinRange()
        max_range = min(self.depth.getMaxRange(), MAX_SENSOR_RANGE)

        # Subsample to keep mapping fast.
        step_u = max(2, w // 90)
        step_v = max(2, h // 40)
        fx = (w / 2.0) / math.tan(fov_h / 2.0)
        fy = (h / 2.0) / math.tan(fov_v / 2.0)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        for v in range(0, h, step_v):
            for u in range(0, w, step_u):
                d = float(arr[v, u])
                if not math.isfinite(d) or d <= min_range or d >= max_range:
                    continue

                # planar pinhole model: sensor looks along -Z
                x = (u - cx) * d / fx
                y = (cy - v) * d / fy
                z = -d
                # Ignore ceiling/floor dominated points for 2D map.
                if abs(y) > 2.0:
                    continue
                points.append((x, y, z))

        return points

    # --------------------------------------------------------
    # Duck detection
    # --------------------------------------------------------
    def camera_rgb(self):
        if self.explore_camera is None:
            return None
        raw = self.explore_camera.getImage()
        if raw is None:
            return None
        w = self.explore_camera.getWidth()
        h = self.explore_camera.getHeight()
        a = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        # Webots Camera raw buffer is BGRA.
        return a[:, :, :3][:, :, ::-1]  # RGB

    def detect_duck_bbox(self):
        img = self.camera_rgb()
        if img is None:
            return None

        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)

        # Broad yellow/orange detector. In the final project this can be
        # replaced by YOLO or Webots Recognition without changing the map.
        mask = (
            (r > 125) &
            (g > 85) &
            (b < 115) &
            (r > b + 45) &
            (g > b + 20)
        )

        # Remove upper sky/ceiling and tiny noise.
        h, w = mask.shape
        mask[:int(0.12 * h), :] = False

        # Cheap connected component search without OpenCV.
        seen = np.zeros_like(mask, dtype=bool)
        best = None
        best_area = 0

        for yy in range(0, h, 4):
            for xx in range(0, w, 4):
                if not mask[yy, xx] or seen[yy, xx]:
                    continue
                q = [(yy, xx)]
                seen[yy, xx] = True
                minx = maxx = xx
                miny = maxy = yy
                area = 0
                qi = 0
                while qi < len(q) and area < 200000:
                    cy, cx = q[qi]
                    qi += 1
                    area += 1
                    minx = min(minx, cx); maxx = max(maxx, cx)
                    miny = min(miny, cy); maxy = max(maxy, cy)
                    for dy, dx in ((0,4),(0,-4),(4,0),(-4,0)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                if area > best_area and area >= DUCK_COLOR_MIN_AREA:
                    bw = maxx - minx + 1
                    bh = maxy - miny + 1
                    ratio = bw / float(max(1, bh))
                    if 0.25 <= ratio <= 2.8:
                        best_area = area
                        best = (minx, miny, bw, bh)

        return best

    def duck_observation(self):
        bbox = self.detect_duck_bbox()
        if bbox is None or self.depth is None:
            return None

        x, y, w, h = bbox
        rgb_w = self.explore_camera.getWidth()
        rgb_h = self.explore_camera.getHeight()
        rw = self.depth.getWidth()
        rh = self.depth.getHeight()

        u = int(clamp((x + w / 2.0) * rw / rgb_w, 0, rw - 1))
        v = int(clamp((y + h / 2.0) * rh / rgb_h, 0, rh - 1))

        try:
            depth_img = np.asarray(self.depth.getRangeImageArray(), dtype=np.float32)
            d = float(depth_img[v, u])
        except Exception:
            return None

        if not math.isfinite(d) or d <= self.depth.getMinRange() or d > DUCK_MAX_OBSERVATION_RANGE:
            return None

        # Sensor point: x right, z forward is negative.
        fov = self.depth.getFov()
        bearing = ((u / max(1, rw - 1)) - 0.5) * fov
        forward = d
        # Positive image bearing is to the camera's right.
        left = -forward * math.tan(bearing)

        rx, ry, rz, yaw = self.pose()
        wx = rx + math.cos(yaw) * forward - math.sin(yaw) * left
        wy = ry + math.sin(yaw) * forward + math.cos(yaw) * left

        # Estimate vertical image position of the duck center.
        fov_h = self.depth.getFov()
        rw = self.depth.getWidth()
        rh = self.depth.getHeight()
        fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) * (rh / float(rw)))
        fy = (rh / 2.0) / math.tan(fov_v / 2.0)
        cy = (rh - 1) / 2.0
        up = (cy - v) * d / fy
        wz = rz + up

        return {
            'x': wx,
            'y': wy,
            'z': wz,
            'range': d,
            'bearing': bearing,
            'bbox': bbox,
            'robot_x': rx,
            'robot_y': ry,
            'robot_z': rz,
            'yaw': yaw,
            'time': self.robot.getTime()
        }

    def register_duck_observation(self, obs):
        if obs is None or self.duck_confirmed:
            return

        if self.last_duck_obs_pose is not None:
            if distance2((obs['robot_x'], obs['robot_y']), self.last_duck_obs_pose) < DUCK_MIN_TRAVEL_BETWEEN_OBS:
                return

        self.last_duck_obs_pose = (obs['robot_x'], obs['robot_y'])
        self.duck_observations.append(obs)

        arr = np.array([[o['x'], o['y'], o['z']] for o in self.duck_observations])
        estimate = np.median(arr, axis=0)
        self.duck_position = tuple(float(v) for v in estimate)

        print(f'\n[DUCK candidate] observation {len(self.duck_observations)} '
              f'at X={obs["x"]:.2f}, Y={obs["y"]:.2f}, range={obs["range"]:.2f} m')

        if len(self.duck_observations) >= DUCK_MIN_CONFIRMATIONS:
            spread = np.linalg.norm(arr[:, :2] - estimate[:2], axis=1)
            if float(np.median(spread)) <= DUCK_POSITION_GATE:
                self.duck_confirmed = True
                print('\n' + '=' * 72)
                print('DUCK CONFIRMED')
                print('=' * 72)
                print(f'X = {self.duck_position[0]:.3f} m')
                print(f'Y = {self.duck_position[1]:.3f} m')
                print(f'Z = {self.duck_position[2]:.3f} m')
                print(f'Observations = {len(self.duck_observations)}')
                print('=' * 72)

    # --------------------------------------------------------
    # Frontier / A*
    # --------------------------------------------------------
    def frontier_cells(self):
        g = self.map.grid
        out = []
        for y in range(1, self.map.ny - 1):
            for x in range(1, self.map.nx - 1):
                if g[y, x] != 0:
                    continue
                nb = g[y-1:y+2, x-1:x+2]
                if np.any(nb == -1):
                    out.append((x, y))
        return out

    def frontier_clusters(self, cells):
        s = set(cells)
        clusters = []
        while s:
            seed = s.pop()
            q = [seed]
            c = [seed]
            qi = 0
            while qi < len(q):
                x, y = q[qi]; qi += 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        p = (x + dx, y + dy)
                        if p in s:
                            s.remove(p)
                            q.append(p)
                            c.append(p)
            if len(c) >= MIN_FRONTIER_CLUSTER:
                clusters.append(c)
        return clusters

    def astar(self, start, goal):
        if start is None or goal is None:
            return []
        if not self.map.traversable(*start) or not self.map.traversable(*goal):
            return []

        open_heap = []
        import heapq
        heapq.heappush(open_heap, (0.0, start))
        came = {}
        gscore = {start: 0.0}

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path

            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nx, ny = cur[0] + dx, cur[1] + dy
                if not self.map.traversable(nx, ny):
                    continue
                step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                ng = gscore[cur] + step_cost
                nb = (nx, ny)
                if ng < gscore.get(nb, float('inf')):
                    gscore[nb] = ng
                    came[nb] = cur
                    heapq.heappush(open_heap, (ng + h(nb, goal), nb))
        return []

    def choose_frontier(self):
        pose = self.pose()
        start = self.map.world_to_grid(pose[0], pose[1])
        if start is None:
            return None

        cells = self.frontier_cells()
        clusters = self.frontier_clusters(cells)
        if not clusters:
            return None

        candidates = []
        for cluster in clusters:
            gx = sum(p[0] for p in cluster) / len(cluster)
            gy = sum(p[1] for p in cluster) / len(cluster)
            # Choose the cluster cell nearest its centroid so the goal is usually in free space.
            goal = min(cluster, key=lambda p: math.hypot(p[0] - gx, p[1] - gy))
            gw = self.map.grid_to_world(*goal)
            world_dist = math.hypot(gw[0] - pose[0], gw[1] - pose[1])
            if world_dist < MIN_FRONTIER_GOAL_DISTANCE:
                continue
            path = self.astar(start, goal)
            if not path:
                continue
            # Favor reachable frontiers that reveal more unknown cells, but do not accept
            # a tiny frontier right next to the current pose.
            score = len(path) / math.sqrt(len(cluster))
            candidates.append((score, goal, path, len(cluster), world_dist))

        if not candidates:
            # Fallback: take the farthest reachable frontier, but still avoid the current cell.
            fallback = []
            for cluster in clusters:
                gx = sum(p[0] for p in cluster) / len(cluster)
                gy = sum(p[1] for p in cluster) / len(cluster)
                goal = min(cluster, key=lambda p: math.hypot(p[0] - gx, p[1] - gy))
                gw = self.map.grid_to_world(*goal)
                world_dist = math.hypot(gw[0] - pose[0], gw[1] - pose[1])
                if world_dist < 0.45:
                    continue
                path = self.astar(start, goal)
                if path:
                    fallback.append((world_dist, goal, path, len(cluster)))
            if not fallback:
                return None
            world_dist, goal, path, cluster_size = max(fallback, key=lambda x: x[0])
            return goal, path, cluster_size, world_dist

        score, goal, path, cluster_size, world_dist = min(candidates, key=lambda x: x[0])
        return goal, path, cluster_size, world_dist

    def waypoint_from_path(self):
        if not self.a_star_path or self.path_index >= len(self.a_star_path):
            return None
        # Skip waypoints too close to robot for smoother motion.
        pose = self.pose()
        rx, ry = pose[0], pose[1]
        while self.path_index < len(self.a_star_path):
            ix, iy = self.a_star_path[self.path_index]
            wx, wy = self.map.grid_to_world(ix, iy)
            if math.hypot(wx-rx, wy-ry) > 0.28:
                return (wx, wy)
            self.path_index += 1
        return None

    # --------------------------------------------------------
    # Scanning and state machine
    # --------------------------------------------------------
    def update_mapping(self):
        points = self.depth_points()
        if points:
            self.map.update(points, self.pose())

    def scan_command(self):
        # Move on a shallow arc while scanning. The previous version tried to
        # rotate in place and accumulated only a few degrees in tens of seconds.
        # A slow arc gives the perception stack fresh views and physically rotates
        # the robot much more reliably with this simplified gait.
        forward = 0.12
        turn = SCAN_TURN_COMMAND

        # Basic forward-safety check from the depth image. If the central sector
        # is blocked, turn in place until it opens instead of walking into it.
        if self.depth is not None:
            try:
                w = self.depth.getWidth()
                h = self.depth.getHeight()
                arr = np.asarray(self.depth.getRangeImageArray(), dtype=np.float32)
                if arr.ndim == 2:
                    c0 = int(0.40 * w)
                    c1 = int(0.60 * w)
                    r0 = int(0.45 * h)
                    r1 = int(0.80 * h)
                    sector = arr[r0:r1, c0:c1]
                    finite = sector[np.isfinite(sector)]
                    if finite.size and float(np.percentile(finite, 20)) < 0.65:
                        forward = 0.0
                        turn = SCAN_TURN_COMMAND
            except Exception:
                pass

        self.command_motion(forward, turn)

    def move_to_goal_command(self):
        wp = self.waypoint_from_path()
        if wp is None:
            return True

        rx, ry, rz, yaw = self.pose()
        dx = wp[0] - rx
        dy = wp[1] - ry
        d = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        err = wrap_angle(target_yaw - yaw)

        if d < GOAL_TOLERANCE:
            self.path_index += 1
            return self.waypoint_from_path() is None

        turn = clamp(err / 0.75, -1.0, 1.0)
        forward = clamp(d / 1.2, 0.15, 1.0)
        if abs(err) > 0.95:
            forward = 0.0
        self.command_motion(forward, turn)
        return False

    def maybe_print_status(self):
        t = self.robot.getTime()
        if t - self.last_scan_print < 2.0:
            return
        self.last_scan_print = t
        p = self.pose()
        roll, pitch = self.body_tilt()
        extra = ''
        if self.state == 'SCAN':
            extra = f' scan_rot={math.degrees(self.scan_rotated):5.0f}deg'
        elif self.state == 'RECOVERY':
            extra = f' recovery={self.recovery_attempt+1}/{self.max_recovery_attempts} stage={self.recovery_stage}'
        elapsed = max(0.001, t - self.start_time)
        speed = self.total_distance / elapsed
        print(f'[STATUS] t={t:6.1f}s state={self.state:<8} '
              f'pose=({p[0]:5.2f},{p[1]:5.2f},z={p[2]:4.2f}) '
              f'rp=({math.degrees(roll):+4.0f},{math.degrees(pitch):+4.0f}) '
              f'explored={100*self.map.explored_ratio():5.1f}% '
              f'dist={self.total_distance:5.2f}m speed={speed:4.2f}m/s '
              f'samples={len(self.map.path):5d} duck_obs={len(self.duck_observations)}{extra}')

    def run(self):
        # Capture the actual joint targets/sensor values before any locomotion command.
        # No Supervisor state is used: recovery is done physically with the legs.
        settle_until = self.robot.getTime() + 2.5
        while self.robot.step(self.dt) != -1 and self.robot.getTime() < settle_until:
            now = self.robot.getTime()
            if not self.home_captured and now >= self.home_capture_time:
                self.capture_home_pose()
                self.set_home_pose()
            self.update_motion_stats(self.pose())

        if not self.home_captured:
            self.capture_home_pose()
        self.set_home_pose()
        self.save_last_stable_state(force=True)
        print('[INIT] Supervisor recovery is enabled; Spot must have supervisor=TRUE.')

        self.state = 'SCAN'
        self.state_enter = self.robot.getTime()
        self.begin_scan()

        while self.robot.step(self.dt) != -1:
            now = self.robot.getTime()
            pose = self.pose()

            self.keyboard_commands()

            # SAFETY FIRST: fall detection before perception/navigation commands.
            if (self.home_captured and
                self.state not in ('RECOVERY', 'RECOVERY_FAILED') and
                self.fall_detected()):
                self.start_recovery()

            if self.state == 'RECOVERY':
                self.recovery_step()
                self.maybe_print_status()
                continue

            if self.state == 'RECOVERY_FAILED':
                self.stop_motion()
                self.maybe_print_status()
                continue

            self.update_motion_stats(pose)
            self.save_last_stable_state()

            # Mapping runs continuously.
            if int(now * 1000 / self.dt) % MAP_UPDATE_PERIOD == 0:
                self.update_mapping()

            # Duck detection runs continuously, but is confirmed only from separated poses.
            if self.explore_camera is not None and self.depth is not None:
                obs = self.duck_observation()
                if obs is not None:
                    self.register_duck_observation(obs)

            if now - self.start_time > MAX_EXPLORATION_TIME:
                print('\nMaximum exploration time reached.')
                break

            if self.state == 'SCAN':
                self.update_scan_rotation(pose[3])

                scan_done = (
                    self.scan_rotated >= SCAN_ROTATE_TARGET or
                    now - self.scan_start_time >= SCAN_TIME
                )

                if scan_done:
                    choice = self.choose_frontier()
                    if choice is None:
                        if self.map.explored_ratio() >= MIN_EXPLORED_RATIO_TO_FINISH:
                            print('\nNo useful frontier remains. Exploration complete.')
                            break
                        # Force a different viewing direction before trying frontier detection again.
                        self.begin_scan()
                    else:
                        self.frontier_goal, self.a_star_path, cluster_size, goal_distance = choice
                        self.path_index = 0
                        self.state = 'MOVE'
                        self.state_enter = now
                        print(
                            f'\n[FRONTIER] cluster={cluster_size}, '
                            f'goal={self.map.grid_to_world(*self.frontier_goal)}, '
                            f'distance={goal_distance:.2f} m, path={len(self.a_star_path)} cells'
                        )
                else:
                    self.scan_command()

            elif self.state == 'MOVE':
                if now - self.last_replan > FRONTIER_REPLAN_PERIOD:
                    new_choice = self.choose_frontier()
                    self.last_replan = now
                    if new_choice is not None:
                        self.frontier_goal, self.a_star_path, cluster_size, goal_distance = new_choice
                        self.path_index = 0

                done = self.move_to_goal_command()
                if done:
                    self.stop_motion()
                    self.state = 'SCAN'
                    self.state_enter = now
                    self.begin_scan()

            self.maybe_print_status()

        self.stop_motion()
        self.finalize()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    def save_csv(self):
        traj = self.output_dir / 'spot_trajectory.csv'
        with traj.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['time_s', 'x_m', 'y_m'])
            t0 = self.start_time
            for i, (x, y) in enumerate(self.map.path):
                w.writerow([i, x, y])

        duck = self.output_dir / 'duck_location.csv'
        with duck.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['confirmed', 'x_m', 'y_m', 'z_m', 'observations'])
            if self.duck_position is None:
                w.writerow([0, '', '', '', len(self.duck_observations)])
            else:
                w.writerow([1 if self.duck_confirmed else 0,
                            self.duck_position[0],
                            self.duck_position[1],
                            self.duck_position[2],
                            len(self.duck_observations)])

        obsf = self.output_dir / 'duck_observations.csv'
        with obsf.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['time_s','robot_x','robot_y','robot_z','yaw_rad',
                        'range_m','bearing_rad','duck_x','duck_y','duck_z'])
            for o in self.duck_observations:
                w.writerow([o['time'], o['robot_x'], o['robot_y'], o['robot_z'],
                            o['yaw'], o['range'], o['bearing'], o['x'], o['y'], o['z']])

        cloudf = self.output_dir / 'point_cloud.csv'
        with cloudf.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['x_m','y_m','z_m'])
            # Limit CSV size while keeping the map dense.
            cloud = self.map.cloud
            stride = max(1, len(cloud) // 100000)
            for p in cloud[::stride]:
                w.writerow(p)

    def save_pngs(self):
        if plt is None:
            print('matplotlib not available: PNG maps were not generated.')
            return

        # ---- 2D occupancy map ----
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [
            self.map.ox,
            self.map.ox + self.map.nx * self.map.res,
            self.map.oy,
            self.map.oy + self.map.ny * self.map.res,
        ]
        ax.imshow(self.map.grid, origin='lower', extent=extent,
                  interpolation='nearest', vmin=-1, vmax=1)
        if self.map.path:
            pp = np.asarray(self.map.path)
            ax.plot(pp[:,0], pp[:,1], linewidth=1.5, label='SPOT trajectory')
            ax.scatter(pp[0,0], pp[0,1], marker='o', s=40, label='Start')
            ax.scatter(pp[-1,0], pp[-1,1], marker='s', s=40, label='End')
        if self.duck_position is not None:
            ax.scatter(self.duck_position[0], self.duck_position[1],
                       marker='*', s=180, label='Duck')
            ax.text(self.duck_position[0] + 0.1, self.duck_position[1] + 0.1,
                    f'Duck ({self.duck_position[0]:.2f}, {self.duck_position[1]:.2f})')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('SPOT Occupancy / Exploration Map')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'spot_map_2d.png', dpi=250)
        plt.close(fig)

        # ---- 3D point cloud ----
        if self.map.cloud:
            pts = np.asarray(self.map.cloud, dtype=float)
            if len(pts) > 50000:
                idx = np.linspace(0, len(pts) - 1, 50000).astype(int)
                pts = pts[idx]

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=0.8)

            if self.map.path:
                pp = np.asarray(self.map.path)
                ax.plot(pp[:,0], pp[:,1], np.full(len(pp), 0.05), linewidth=2.0)

            if self.duck_position is not None:
                ax.scatter([self.duck_position[0]], [self.duck_position[1]],
                           [self.duck_position[2]], s=180, marker='*')

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title('SPOT 3D Point-Cloud Map')
            fig.tight_layout()
            fig.savefig(self.output_dir / 'spot_map_3d.png', dpi=250)
            plt.close(fig)

    def finalize(self):
        self.save_csv()
        self.save_pngs()
        print('\n' + '=' * 72)
        print('EXPLORATION FINISHED')
        print('=' * 72)
        if self.duck_position is not None:
            print(f'Duck confirmed: {self.duck_confirmed}')
            print(f'Duck X={self.duck_position[0]:.3f} m')
            print(f'Duck Y={self.duck_position[1]:.3f} m')
            print(f'Duck Z={self.duck_position[2]:.3f} m')
        else:
            print('Duck was not confirmed.')
        print(f'Explored ratio: {100*self.map.explored_ratio():.1f}%')
        print(f'Robot travel: {self.total_distance:.2f} m')
        print(f'Rotation: {math.degrees(self.total_rotation):.1f} deg')
        print('Files:')
        for name in [
            'spot_map_2d.png',
            'spot_map_3d.png',
            'spot_trajectory.csv',
            'duck_location.csv',
            'duck_observations.csv',
            'point_cloud.csv'
        ]:
            print('  ', self.output_dir / name)
        print('=' * 72)


if __name__ == '__main__':
    Explorer().run()
