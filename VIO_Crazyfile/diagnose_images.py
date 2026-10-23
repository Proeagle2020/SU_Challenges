from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "output" / "images"
OUT_DIR = ROOT / "output"

images = sorted(IMAGE_DIR.glob("frame_*.png"))

if len(images) < 2:
    raise SystemExit(f"Not enough images in: {IMAGE_DIR}")

# چند تصویر با فاصله زمانی بیشتر برای تشخیص بررسی می‌شوند
indices = sorted(set([0, 1, 5, 10, 20, len(images) - 1]))
indices = [i for i in indices if i < len(images)]

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

print("Total images:", len(images))
print("\nIMAGE QUALITY")

loaded = {}

for i in indices:
    img = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(i, "ERROR: unreadable")
        continue

    loaded[i] = img
    kp, _ = orb.detectAndCompute(img, None)

    print(
        f"Frame {i:4d} | size={img.shape} | "
        f"mean={img.mean():.2f} | std={img.std():.2f} | "
        f"ORB keypoints={len(kp) if kp else 0}"
    )

print("\nPAIRWISE FEATURE MATCHING")

if 0 not in loaded:
    raise SystemExit("First image could not be loaded.")

base = loaded[0]
kp1, des1 = orb.detectAndCompute(base, None)

for i, img in loaded.items():
    if i == 0:
        continue

    kp2, des2 = orb.detectAndCompute(img, None)

    if des1 is None or des2 is None:
        print(f"Frame 0 -> {i}: descriptors missing")
        continue

    pairs = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]

    print(
        f"Frame 0 -> {i}: "
        f"keypoints={len(kp1)}/{len(kp2)}, "
        f"ratio-test matches={len(good)}"
    )

    if i in (1, 10, 20):
        vis = cv2.drawMatches(
            base, kp1, img, kp2, good[:100], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        target = OUT_DIR / f"matches_frame0_{i}.png"
        cv2.imwrite(str(target), vis)
        print("  Saved:", target)

print("\nDiagnostic finished.")