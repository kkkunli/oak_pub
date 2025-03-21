import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

class CalibrationValidator:
    def __init__(self, calib_file):
        with open(calib_file) as f:
            data = json.load(f)
            self.mtx = np.array(data['camera_matrix'])
            self.dist = np.array(data['distortion_coefficients'])
            self.resolution = (data['resolution']['width'], 
                              data['resolution']['height'])
        
        # 生成验证模式
        self.patterns = {
            'grid': self._create_grid(),
            'diagonal': self._create_diagonal_lines()
        }
    
    def _create_grid(self, spacing=100):
        img = np.ones((self.resolution[1], self.resolution[0],3), dtype=np.uint8)*255
        # 绘制网格线
        for x in range(0, img.shape[1], spacing):
            cv2.line(img, (x,0), (x,img.shape[0]), (0,0,0), 2)
        for y in range(0, img.shape[0], spacing):
            cv2.line(img, (0,y), (img.shape[1],y), (0,0,0), 2)
        return img
    
    def _create_diagonal_lines(self):
        img = np.ones((self.resolution[1], self.resolution[0],3), dtype=np.uint8)*255
        cv2.line(img, (0,0), (img.shape[1],img.shape[0]), (0,0,0), 10)
        cv2.line(img, (img.shape[1],0), (0,img.shape[0]), (0,0,0), 10)
        return img
    
    def analyze_distortion(self, pattern_name='grid'):
        original = self.patterns[pattern_name]
        undistorted = cv2.undistort(original, self.mtx, self.dist)
        
        # 计算形变误差
        diff = cv2.absdiff(original, undistorted)
        error = np.mean(diff)
        
        # 可视化对比
        plt.figure(figsize=(15,6))
        plt.subplot(131), plt.imshow(original), plt.title('Original')
        plt.subplot(132), plt.imshow(undistorted), plt.title('Undistorted')
        plt.subplot(133), plt.imshow(diff), plt.title(f'Diff (Error: {error:.2f})')
        plt.show()
        
        return error

# 使用示例
if __name__ == "__main__":
    validator = CalibrationValidator("my_calibration.json")
    
    print("=== 网格形变验证 ===")
    grid_error = validator.analyze_distortion('grid')
    
    print("\n=== 对角线直线验证 ===")
    line_error = validator.analyze_distortion('diagonal')
    
    print(f"\n验证结果：\n网格误差：{grid_error:.2f}\n直线误差：{line_error:.2f}")
    print("标准：误差值应<15（0-255尺度）")
