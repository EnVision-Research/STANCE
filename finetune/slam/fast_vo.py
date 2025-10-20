# fast_vo.py
import cv2
import numpy as np

class FastVisualOdometry:
    def __init__(self, calib_file=None):
        # 加载相机参数
        if calib_file is None:  # tianshuo, only for test
            self.K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
            self.dist = np.zeros((4,1))
        else:
            self.K = np.load(calib_file)['K']
            self.dist = np.load(calib_file)['dist']
        
        # ORB特征参数
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
    def compute_pose(self, prev_features, curr_features, mask=None):
        """计算相对位姿"""
        # 特征匹配
        matches = self.bf.knnMatch(prev_features[1], curr_features[1], k=2)
        
        # Lowe's比率测试
        good_matches = [m for m,n in matches if m.distance < 0.75*n.distance]
        
        # 获取匹配点对
        query_indices = np.array([m.queryIdx for m in good_matches])  # 前一帧关键点索引
        train_indices = np.array([m.trainIdx for m in good_matches])   # 当前帧关键点索引

        # 根据索引获取坐标
        prev_pts = cv2.KeyPoint_convert(prev_features[0])[query_indices]  # 前一帧关键点坐标
        curr_pts = cv2.KeyPoint_convert(curr_features[0])[train_indices]   # 当前帧关键点坐标

        # 结合掩码过滤动态点（如果启用了动态物体过滤）
        if mask is not None:
            # 提取掩码内有效的点
            valid_idx = []
            for i, pt in enumerate(curr_pts):
                x, y = map(int, pt)
                if mask[y, x] > 0:  # 检查是否在静态区域
                    valid_idx.append(i)
            prev_pts = prev_pts[valid_idx]
            curr_pts = curr_pts[valid_idx]
            
        # 五点法计算本质矩阵
        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts, self.K, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        # 恢复位姿
        _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, self.K)
        
        return self._to_transformation_matrix(R, t)
    
    def _to_transformation_matrix(self, R, t):
        """转换为齐次变换矩阵"""
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.ravel()
        return T
