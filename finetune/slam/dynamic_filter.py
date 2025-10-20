# dynamic_filter.py
import cv2
import numpy as np

class DynamicObjectFilter:
    def __init__(self, threshold=2.5):
        self.prev_flow = None
        self.threshold = threshold
        
    def get_static_mask(self, frame_gray):
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = frame_gray
            return np.ones(frame_gray.shape[:2], dtype=np.uint8) * 255

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, frame_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # 计算运动幅度
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        # 自适应阈值处理
        motion_mask = (mag < self.threshold * np.median(mag)).astype(np.uint8) * 255
        
        self.prev_frame = frame_gray  # 更新上一帧
        return motion_mask
