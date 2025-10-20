# sensor_fusion.py
import numpy as np
import cv2

class MotionModelFuser:
    def __init__(self):
        # 初始化卡尔曼滤波器
        self.kf = cv2.KalmanFilter(dynamParams=6, measureParams=3)
        self._init_kf()
        
    def _init_kf(self):
        """初始化卡尔曼滤波器参数"""
        # 状态转移矩阵 (简单匀速模型)
        self.kf.transitionMatrix = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ], np.float32)
        
        # 测量矩阵
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
        ], np.float32)
        
        # 噪声协方差
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        
    def update(self, pose_delta):
        """融合运动模型预测与观测"""
        # 提取平移部分作为测量
        measurement = pose_delta[:3,3]
        
        # 卡尔曼滤波更新
        self.kf.predict()
        self.kf.correct(measurement)
        
        # 获取融合后状态
        state = self.kf.statePost
        return self._state_to_pose(state)
    
    def _state_to_pose(self, state):
        """将状态向量转换为变换矩阵"""
        # 这里需要根据实际运动模型实现
        # 简化版返回单位矩阵
        return np.eye(4)
