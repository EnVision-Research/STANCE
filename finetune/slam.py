import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


class VisualSLAM:
    def __init__(self, focal_length=1000, pp=(256, 128)):
        """
        初始化Visual SLAM系统
        
        参数:
        focal_length: 焦距（像素）
        pp: 主点坐标 (cx, cy)
        """
        self.focal_length = focal_length
        self.pp = pp
        
        # 相机内参矩阵
        self.K = np.array([
            [focal_length, 0, pp[0]],
            [0, focal_length, pp[1]],
            [0, 0, 1]
        ])
        
        # 特征检测器
        self.orb = cv2.ORB_create(
            nfeatures=3000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # FLANN匹配器
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    def detect_features(self, frame):
        """检测关键点和计算描述符"""
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
        
    def match_features(self, desc1, desc2, ratio=0.7):
        """特征匹配"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
            
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        try:
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good.append(m)
        except ValueError:
            pass
        return good
        
    def estimate_motion(self, kp1, kp2, matches, mask_ratio=0.99):
        """估计两帧之间的运动"""
        if len(matches) < 8:
            return None, None, None
            
        # 获取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # 使用RANSAC估计本质矩阵
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            return None, None, None
            
        # 从本质矩阵恢复R和t
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        return R, t, mask
        
    def process_sequence(self, video_tensor):
        """
        处理视频序列并估计相机轨迹
        
        参数:
        video_tensor: shape (B, T, C, H, W)的视频张量
        
        返回:
        trajectories: 列表，包含每个批次的相机轨迹
        """
        B, T, C, H, W = video_tensor.shape
        trajectories = []
        
        for b in range(B):
            # 初始化轨迹
            trajectory = [np.eye(4)]  # 从单位矩阵开始
            R_total = np.eye(3)
            t_total = np.zeros((3, 1))
            
            # 处理每一帧
            prev_frame = None
            prev_kp = None
            prev_desc = None
            
            for t in range(T):
                # 转换当前帧为uint8格式
                current_frame = (video_tensor[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                
                if prev_frame is None:
                    prev_frame = current_frame_gray
                    prev_kp, prev_desc = self.detect_features(prev_frame)
                    continue
                
                # 检测特征点
                curr_kp, curr_desc = self.detect_features(current_frame_gray)
                
                # 匹配特征点
                matches = self.match_features(prev_desc, curr_desc)
                
                # 估计运动
                R, t, mask = self.estimate_motion(prev_kp, curr_kp, matches)
                
                if R is not None and t is not None:
                    # 更新累积运动
                    R_total = R @ R_total
                    t_total = R @ t_total + t
                    
                    # 创建变换矩阵
                    transform = np.eye(4)
                    transform[:3, :3] = R_total
                    transform[:3, 3:4] = t_total
                    
                    trajectory.append(transform)
                else:
                    # 如果运动估计失败，使用上一帧的变换
                    trajectory.append(trajectory[-1])
                
                # 更新前一帧
                prev_frame = current_frame_gray
                prev_kp = curr_kp
                prev_desc = curr_desc
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def visualize_trajectory(self, trajectory, title="Camera Trajectory"):
        """可视化相机轨迹"""
        positions = np.array([transform[:3, 3] for transform in trajectory])
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Camera Path')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', label='End')
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 添加图例
        ax.legend()
        
        # 设置相同的缩放比例
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
        
        return positions


# 使用示例
if __name__ == "__main__":
    # 创建SLAM系统
    slam = VisualSLAM()
    
    # 假设我们有一个视频张量 (1, 25, 3, 256, 512)
    video = torch.randn(1, 25, 3, 256, 512)
    
    # 处理视频序列
    trajectories = slam.process_sequence(video)
    
    # 可视化第一个序列的轨迹
    positions = slam.visualize_trajectory(trajectories[0], "Vehicle Trajectory")
    
    # 打印轨迹信息
    print("轨迹统计信息:")
    print(f"总位移: {np.linalg.norm(positions[-1] - positions[0]):.2f} 米")
    print(f"平均速度: {np.linalg.norm(positions[-1] - positions[0]) / (len(positions) / 30):.2f} 米/秒")  # 假设30fps
