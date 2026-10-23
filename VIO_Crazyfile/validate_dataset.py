from pathlib import Path
import csv
import math
import statistics
import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "output"
CSV_PATH = DATA / "vio_dataset.csv"
IMG_DIR = DATA / "images"

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    rows = []
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"time_s", "ax", "ay", "az", "gx", "gy", "gz",
                    "roll", "pitch", "yaw", "image"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)

    image_rows = [r for r in rows if (r.get("image") or "").strip()]
    readable, unreadable = 0, []
    for row in image_rows:
        p = IMG_DIR / row["image"]
        im = cv2.imread(str(p))
        if im is None:
            unreadable.append(str(p))
        else:
            readable += 1
            if readable == 1:
                h, w = im.shape[:2]
                print(f"First readable image: {p} ({w}x{h})")

    times = []
    numeric_ok = 0
    for r in rows:
        try:
            vals = [float(r[k]) for k in ("time_s","ax","ay","az","gx","gy","gz","roll","pitch","yaw")]
            if all(math.isfinite(v) for v in vals):
                numeric_ok += 1
            times.append(float(r["time_s"]))
        except (ValueError, TypeError, KeyError):
            pass

    print("\n=== DATASET CHECK ===")
    print(f"CSV: {CSV_PATH}")
    print(f"CSV rows: {len(rows)}")
    print(f"Rows with finite time + IMU + RPY values: {numeric_ok}")
    print(f"Rows containing an image filename: {len(image_rows)}")
    print(f"Readable images: {readable}")
    print(f"Unreadable/missing images: {len(unreadable)}")
    if times:
        print(f"Time range: {min(times):.3f} to {max(times):.3f} s")
        dts = [b-a for a,b in zip(times, times[1:]) if b > a]
        if dts:
            print(f"Median CSV timestep: {statistics.median(dts):.6f} s")
            print(f"Approximate CSV sampling rate: {1/statistics.median(dts):.2f} Hz")
    if image_rows:
        image_times = [float(r["time_s"]) for r in image_rows]
        dti = [b-a for a,b in zip(image_times, image_times[1:]) if b > a]
        if dti:
            print(f"Median image interval: {statistics.median(dti):.4f} s")
            print(f"Approximate image rate: {1/statistics.median(dti):.2f} Hz")
    if unreadable:
        print("\nFirst missing/unreadable image paths:")
        for p in unreadable[:10]:
            print("  ", p)
    if len(rows) < 2 or readable < 2:
        print("\nNOT READY: need at least two readable images and CSV rows.")
    else:
        print("\nDataset is structurally ready for the visual-odometry baseline.")
        print("This check validates files/timestamps; it does not prove metric VIO accuracy.")

if __name__ == "__main__":
    main()
