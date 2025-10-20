# demo.py
from auto_slam import AutoSLAM
import cv2

config = {
    'use_clahe': True,
    'feature_level': 'fast'
}

# slam = AutoSLAM('calibration.npz', config)
slam = AutoSLAM(config)

cap = cv2.VideoCapture('/hpc2hdd/home/txu647/code/video_data/driving_cases2/high_interactive1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pose = slam.process_frame(gray)
    
    # 可视化轨迹
    print(f"Estimated pose:\n{pose}")
    
cap.release()
