# Crazyflie Webots VIO project: stages 2-4

## Put these files here
Copy `validate_dataset.py` and `run_vo_pipeline.py` into:
`E:\Mywebots\VIO_Crazyflie\`

Expected input:
- `output\vio_dataset.csv`
- `output\images\frame_XXXXXX.png`

## Stage 2: validate the dataset
Open PowerShell:
```powershell
cd E:\Mywebots\VIO_Crazyflie
python validate_dataset.py
```
This checks CSV columns, finite sensor values, image readability, and timestamp rates.

## Install processing dependencies
Use the same Python 3.13 interpreter that runs your controller:
```powershell
C:\Users\Redmon\AppData\Local\Programs\Python\Python313\python.exe -m pip install numpy opencv-python matplotlib
```

## Stage 3: run the visual-odometry baseline
```powershell
cd E:\Mywebots\VIO_Crazyflie
C:\Users\Redmon\AppData\Local\Programs\Python\Python313\python.exe run_vo_pipeline.py
```

## Stage 4: inspect outputs
- `output\visual_trajectory.csv`
- `output\visual_trajectory.png`
- `output\vio_diagnostics.txt`

## Important scientific limitation
This first implementation is a monocular visual-odometry baseline. It uses ORB feature matches and OpenCV's essential-matrix / recoverPose pipeline. It validates IMU data but does not yet tightly fuse IMU into a metric estimator. Monocular translation scale is unknown, so trajectory axes are relative units, not meters. It is an end-to-end baseline, not a complete research-grade VIO system.
