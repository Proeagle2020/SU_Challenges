from pathlib import Path
import csv
import math
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Run this script from its folder. Expected structure:
# ./output/vio_dataset.csv and ./output/images/frame_XXXXXX.png
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "output"
CSV_PATH = DATA / "vio_dataset.csv"
IMG_DIR = DATA / "images"
TRAJ_CSV = DATA / "visual_trajectory.csv"
PLOT_PNG = DATA / "visual_trajectory.png"
DIAG_TXT = DATA / "vio_diagnostics.txt"

# Camera model from Crazyflie PROTO: 324x324, horizontal FOV 0.87 rad.
# Webots simulated pinhole camera is treated as zero-distortion for this baseline.
IMAGE_WIDTH = 324
IMAGE_HEIGHT = 324
FOV_RAD = 0.87
FOCAL = IMAGE_WIDTH / (2.0 * math.tan(FOV_RAD / 2.0))
K = np.array([[FOCAL, 0.0, IMAGE_WIDTH / 2.0],
              [0.0, FOCAL, IMAGE_HEIGHT / 2.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)

MIN_MATCHES = 25
MIN_INLIERS = 15
MAX_FEATURES = 1800
RATIO_TEST = 0.75

def load_dataset():
    if not CSV_PATH.exists():
        raise SystemExit(f"Cannot find {CSV_PATH}")
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    frames = []
    for row in rows:
        name = (row.get("image") or "").strip()
        if not name:
            continue
        try:
            t = float(row["time_s"])
        except (ValueError, KeyError):
            continue
        p = IMG_DIR / name
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            frames.append({"time": t, "name": name, "path": p, "gray": im, "row": row})
    frames.sort(key=lambda x: x["time"])
    if len(frames) < 2:
        raise SystemExit(f"Need at least 2 readable frames; found {len(frames)} in {IMG_DIR}")
    return rows, frames

def rotation_from_rpy(roll, pitch, yaw):
    # ZYX convention; used only for orientation diagnostics/world alignment.
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=float)
    return Rz @ Ry @ Rx

def finite_rpy(row):
    try:
        return [float(row[k]) for k in ("roll","pitch","yaw")]
    except (ValueError, TypeError, KeyError):
        return [0.0, 0.0, 0.0]

def main():
    rows, frames = load_dataset()
    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    points = [np.zeros(3, dtype=float)]
    visual_rotations = [np.eye(3)]
    frame_records = [{
        "time_s": frames[0]["time"],
        "image": frames[0]["name"],
        "x_rel": 0.0, "y_rel": 0.0, "z_rel": 0.0,
        "matches": 0, "inliers": 0, "status": "origin"
    }]
    diagnostics = []
    prev_gray = frames[0]["gray"]
    prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
    R_world_cam = np.eye(3)
    p_world = np.zeros(3, dtype=float)
    successful = 0

    for i in range(1, len(frames)):
        cur = frames[i]
        gray = cur["gray"]
        kp, desc = orb.detectAndCompute(gray, None)
        status = "ok"
        match_count = 0
        inlier_count = 0

        if prev_desc is None or desc is None or len(prev_kp) < 8 or len(kp) < 8:
            status = "no_descriptors"
        else:
            pairs = matcher.knnMatch(prev_desc, desc, k=2)
            good = [m for pair in pairs if len(pair) == 2
                    for m, n in [pair] if m.distance < RATIO_TEST * n.distance]
            match_count = len(good)
            if len(good) < MIN_MATCHES:
                status = "few_matches"
            else:
                p1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                p2 = np.float32([kp[m.trainIdx].pt for m in good])
                E, mask = cv2.findEssentialMat(
                    p1, p2, K, method=cv2.RANSAC,
                    prob=0.999, threshold=1.5
                )
                if E is None:
                    status = "essential_failed"
                else:
                    try:
                        inlier_count, R_rel, t_rel, pose_mask = cv2.recoverPose(
                            E, p1, p2, K, mask=mask
                        )
                        inlier_count = int(inlier_count)
                        if inlier_count < MIN_INLIERS:
                            status = "few_inliers"
                        else:
                            # Monocular recoverPose translation has unknown scale.
                            # Integrate a relative direction with unit step, NOT meters.
                            # Convert camera-relative motion to an arbitrary relative path.
                            t_vec = t_rel.reshape(3)
                            delta_world = R_world_cam @ (-R_rel.T @ t_vec)
                            norm = np.linalg.norm(delta_world)
                            if norm > 1e-9:
                                delta_world = delta_world / norm
                                p_world = p_world + delta_world
                                R_world_cam = R_world_cam @ R_rel.T
                                successful += 1
                            else:
                                status = "zero_translation"
                    except cv2.error:
                        status = "recover_pose_failed"

        points.append(p_world.copy())
        visual_rotations.append(R_world_cam.copy())
        frame_records.append({
            "time_s": cur["time"],
            "image": cur["name"],
            "x_rel": float(p_world[0]),
            "y_rel": float(p_world[1]),
            "z_rel": float(p_world[2]),
            "matches": match_count,
            "inliers": inlier_count,
            "status": status
        })
        diagnostics.append(
            f"{cur['time']:.4f}s {cur['name']}: "
            f"matches={match_count}, inliers={inlier_count}, status={status}"
        )
        prev_gray, prev_kp, prev_desc = gray, kp, desc

        if i % 100 == 0:
            print(f"Processed {i}/{len(frames)-1} frame pairs; successful={successful}")

    DATA.mkdir(parents=True, exist_ok=True)
    with TRAJ_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time_s","image","x_rel","y_rel","z_rel","matches","inliers","status"
        ])
        writer.writeheader()
        writer.writerows(frame_records)

    xyz = np.asarray(points)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], linewidth=1.5)
    ax.scatter(xyz[0,0], xyz[0,1], xyz[0,2], marker="o", label="start")
    ax.scatter(xyz[-1,0], xyz[-1,1], xyz[-1,2], marker="x", label="end")
    ax.set_xlabel("X (relative units)")
    ax.set_ylabel("Y (relative units)")
    ax.set_zlabel("Z (relative units)")
    ax.set_title("Monocular visual-odometry trajectory (scale unknown)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=160)
    plt.close(fig)

    # IMU/RPY sanity summary
    finite_imu = 0
    for r in rows:
        try:
            vals = [float(r[k]) for k in ("ax","ay","az","gx","gy","gz","roll","pitch","yaw")]
            if all(math.isfinite(v) for v in vals):
                finite_imu += 1
        except (ValueError, TypeError, KeyError):
            pass

    summary = [
        "VIO PROJECT - STAGE 2-4 DIAGNOSTICS",
        f"CSV rows: {len(rows)}",
        f"Readable camera frames: {len(frames)}",
        f"Finite IMU/RPY rows: {finite_imu}",
        f"Processed frame pairs: {len(frames)-1}",
        f"Pairs with accepted visual pose: {successful}",
        f"Accepted-pose ratio: {successful/max(1,len(frames)-1):.3f}",
        "",
        "IMPORTANT LIMITATION:",
        "This is a monocular visual-odometry baseline with IMU data quality checked,",
        "not a tightly-coupled, metric Visual-Inertial Odometry estimator.",
        "recoverPose() estimates translation direction only; monocular scale is unknown.",
        "The trajectory axes are relative units, not meters. IMU is not yet fused into",
        "a metric state estimator. Do not use this output as a metric flight estimate.",
        "",
        "Per-frame diagnostics:",
        *diagnostics
    ]
    DIAG_TXT.write_text("\n".join(summary), encoding="utf-8")

    print("\n=== PIPELINE COMPLETE ===")
    print("Trajectory CSV:", TRAJ_CSV)
    print("Trajectory plot:", PLOT_PNG)
    print("Diagnostics:", DIAG_TXT)
    print(f"Readable frames: {len(frames)}")
    print(f"Accepted visual poses: {successful}/{len(frames)-1}")
    print("Trajectory scale is arbitrary; this is the visual-odometry baseline.")

if __name__ == "__main__":
    main()
