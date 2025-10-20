# auto_slam.py
import cv2
import numpy as np
from collections import deque
from dynamic_filter import DynamicObjectFilter
from fast_vo import FastVisualOdometry
from sensor_fusion import MotionModelFuser

class AutoSLAM:
    def __init__(self, config, calib_file=None):
        """初始化SLAM系统
        Args:
            config: 配置参数字典
            calib_file: 相机标定文件路径
        """
        # 初始化组件
        self.vo = FastVisualOdometry(calib_file)  # 快速视觉里程计
        self.dyn_filter = DynamicObjectFilter()   # 动态物体过滤
        self.motion_model = MotionModelFuser()    # 运动模型融合
        
        # 系统参数
        self.frame_buffer = deque(maxlen=2)       # 帧缓存
        self.trajectory = []
        self.config = config

    def process_frame(self, frame):
        """处理单帧图像
        Returns:
            pose: 相机位姿 (4x4变换矩阵)
        """
        # 1. 图像预处理
        processed_frame = self._preprocess(frame)
        
        # 2. 动态物体检测
        static_mask = self.dyn_filter.get_static_mask(processed_frame['gray'])
        
        # 3. 视觉里程计计算
        if len(self.frame_buffer) >= 1:
            prev_frame = self.frame_buffer[-1]
            pose_delta = self.vo.compute_pose(
                prev_frame['features'], 
                processed_frame['features'],
                static_mask  # 仅使用静态区域特征
            )
            
            # 4. 传感器融合
            refined_pose = self.motion_model.update(pose_delta)
            self.trajectory.append(refined_pose)
        
        # 缓存当前帧
        self.frame_buffer.append(processed_frame)
        
        return self.trajectory[-1] if self.trajectory else np.eye(4)

    def _preprocess(self, frame):
        """图像预处理流水线"""
        # 特征提取
        features = cv2.ORB_create().detectAndCompute(frame, None)
        
        # 添加预处理步骤
        if self.config.get('use_clahe', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            frame = clahe.apply(frame)
            
        return {
            'gray': frame,
            'features': features
        }

    def get_trajectory(self):
        """获取完整轨迹"""
        return np.array(self.trajectory)
