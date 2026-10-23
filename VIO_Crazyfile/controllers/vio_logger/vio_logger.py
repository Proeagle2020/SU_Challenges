
from controller import Robot
from pathlib import Path
import csv
import math
import sys

import numpy as np
import cv2

robot = Robot()
step = int(robot.getBasicTimeStep())

# Sensors
camera = robot.getDevice("camera")
gyro = robot.getDevice("gyro")
accel = robot.getDevice("accelerometer")
imu = robot.getDevice("inertial_unit")
gps = robot.getDevice("gps")  # Optional

# Enable sensors
camera.enable(step)
gyro.enable(step)
accel.enable(step)
imu.enable(step)

if gps is not None:
    gps.enable(step)
    print("GPS enabled for evaluation only.")
else:
    print("GPS not present. Continuing without GPS.")

# Project root:
# .../VIO_Crazyfile/controllers/vio_logger/vio_logger.py
project_root = Path(__file__).resolve().parents[2]
out = project_root / "output"
images_dir = out / "images"

out.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

csv_file = out / "vio_dataset.csv"

print("Project root:", project_root)
print("Output folder:", out)
print("Images folder:", images_dir)
print("CSV file:", csv_file)

frame_period = 0.033  # approximately 30 Hz
next_frame_time = 0.0
frame_id = 0
last_reported_frame_count = 0

try:
    with csv_file.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "roll", "pitch", "yaw",
            "gt_x", "gt_y", "gt_z",
            "image"
        ])
        f.flush()

        while robot.step(step) != -1:
            t = robot.getTime()

            a = accel.getValues()
            g = gyro.getValues()
            rpy = imu.getRollPitchYaw()

            if gps is not None:
                gt = gps.getValues()
            else:
                gt = (
                    float("nan"),
                    float("nan"),
                    float("nan")
                )

            image_name = ""

            if t >= next_frame_time:
                image = camera.getImage()

                if image is not None:
                    width = camera.getWidth()
                    height = camera.getHeight()

                    bgra = np.frombuffer(
                        image, dtype=np.uint8
                    ).reshape((height, width, 4))

                    bgr = cv2.cvtColor(
                        bgra, cv2.COLOR_BGRA2BGR
                    )

                    image_name = f"frame_{frame_id:06d}.png"
                    image_path = images_dir / image_name

                    saved = cv2.imwrite(
                        str(image_path), bgr
                    )

                    if saved:
                        frame_id += 1
                    else:
                        print(
                            "ERROR: Could not save image:",
                            image_path
                        )
                        image_name = ""

                next_frame_time = t + frame_period

            writer.writerow([
                f"{t:.6f}",
                *a, *g, *rpy,
                *gt,
                image_name
            ])

            # Make CSV data visible during simulation.
            if int(t * 10) != int((t - step / 1000.0) * 10):
                f.flush()

            if frame_id >= last_reported_frame_count + 100:
                print(
                    f"t={t:.2f}s, frames={frame_id}, "
                    f"CSV exists={csv_file.exists()}"
                )
                last_reported_frame_count = frame_id

except Exception as exc:
    print("LOGGER ERROR:", repr(exc))
    raise

print("Dataset saved:", csv_file)
print("Total images saved:", frame_id)
