#!/usr/bin/env python3

"""
DepthAI相机校准节点 (OpenCV 4.7+兼容版)
- 通过Charuco棋盘格进行相机校准
"""

import depthai as dai
import cv2
import numpy as np
import time
import sys
import argparse
from pathlib import Path
import json

# ROS2相关导入
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class CalibrationPublisher(Node):
    def __init__(self, node_name="depthai_calibration", camera_name="oak"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.camera_name = camera_name
        
        self.calib_pub = self.create_publisher(
            CameraInfo, 
            f'/{camera_name}/calibration', 
            10
        )

    def publish_calibration(self, camera_info):
        if camera_info is not None:
            self.calib_pub.publish(camera_info)

class CharucoCalibrator:
    def __init__(self, squaresX=9, squaresY=6, squareSize=0.024, markerSize=0.018):
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.squareSize = squareSize
        self.markerSize = markerSize
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.board = cv2.aruco.CharucoBoard(
            (squaresX, squaresY), 
            squareSize, 
            markerSize, 
            self.dictionary
        )
        
        self.calib_data = []
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detect_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        
        if len(corners) > 0:
            ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board)
            return ret, c_corners, c_ids
        return False, None, None

    def add_calib_data(self, frame):
        ret, corners, ids = self.detect_corners(frame)
        if ret and len(corners) >= 4:
            self.calib_data.append((corners, ids))
            return True
        return False

    def calibrate(self, frame_size):
        obj_points = []
        img_points = []
        
        for corners, ids in self.calib_data:
            objp = self.board.getChessboardCorners()[ids]
            img_points.append(corners)
            obj_points.append(objp)
        
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + 
                cv2.CALIB_FIX_ASPECT_RATIO + 
                cv2.CALIB_FIX_PRINCIPAL_POINT)
        
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            img_points, obj_points, self.board, frame_size, None, None)
        
        return ret, mtx, dist

def create_pipeline(args):
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    
    # 分辨率配置
    resolution_map = {
        "4K": dai.ColorCameraProperties.SensorResolution.THE_4_K,
        "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
        "720p": dai.ColorCameraProperties.SensorResolution.THE_720_P
    }
    cam_rgb.setResolution(resolution_map.get(args.resolution, dai.ColorCameraProperties.SensorResolution.THE_1080_P))
    
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(args.fps)
    
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.video.link(xout.input)
    
    return pipeline

def main():
    parser = argparse.ArgumentParser(description='DepthAI相机校准')
    parser.add_argument('-r', '--resolution', default='720p', 
                        choices=['720p', '1080p', '4K'])
    parser.add_argument('-f', '--fps', type=float, default=15.0)
    parser.add_argument('-n', '--name', default='oak')
    parser.add_argument('-s', '--save', default='calibration.json')
    parser.add_argument('--square-size', type=float, default=0.024)
    parser.add_argument('--squares-x', type=int, default=9)
    parser.add_argument('--squares-y', type=int, default=6)
    parser.add_argument('--manual', type=bool, default=True)
    args = parser.parse_args()

    # 初始化ROS2
    ros_node = None
    if ROS2_AVAILABLE:
        rclpy.init()
        ros_node = CalibrationPublisher(camera_name=args.name)
    
    # 创建校准器
    calibrator = CharucoCalibrator(
        squaresX=args.squares_x,
        squaresY=args.squares_y,
        squareSize=args.square_size,
        markerSize=args.square_size*0.75
    )
    
    # 创建DepthAI管道
    pipeline = create_pipeline(args)
    with dai.Device(pipeline) as device:
        video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        
        print("开始校准采集 - 需要至少20张有效图像")
        print("请多角度移动棋盘格...")
        
        collected = 0
        manual_mode = args.manual

        while collected < 20:
            frame = video_queue.get().getCvFrame()
            
            # 检测角点
            ret, corners, ids = calibrator.detect_corners(frame)
            
            # 实时显示检测状态
            status_color = (0,255,0) if ret else (0,0,255)
            cv2.putText(frame, f"Valid: {str(ret)}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # 手动触发模式
            if manual_mode:
                cv2.putText(frame, "[SPACE] Capture | [Q] Quit", (20,120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                key = cv2.waitKey(1)
                if key == ord(' ') and ret:
                    if calibrator.add_calib_data(frame):
                        collected +=1
                        print(f"手动采集成功: {collected}/20")
                elif key == ord('q'):
                    break
            else:  # 自动模式
                if ret and time.time() - last_capture > 1.0:  # 1秒间隔
                    if calibrator.add_calib_data(frame):
                        collected +=1
                        print(f"自动采集: {collected}/20")
                        last_capture = time.time()

            # 显示剩余需采集数量
            cv2.putText(frame, f"Collected: {collected}/20", (20,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Calibration", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        # 执行校准
        print("\n开始计算校准参数...")
        ret, mtx, dist = calibrator.calibrate(frame.shape[:2][::-1])
        
        if ret:
            print("校准成功!")
            print(f"内参矩阵:\n{mtx}")
            print(f"畸变系数:\n{dist}")
            
            # 保存校准结果
            calib_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
                "resolution": {
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                }
            }
            with open(args.save, 'w') as f:
                json.dump(calib_data, f)
            
            # 发布ROS2参数
            if ros_node:
                info = CameraInfo()
                info.width = frame.shape[1]
                info.height = frame.shape[0]
                info.k = mtx.flatten().tolist()
                info.d = dist.flatten().tolist()
                ros_node.publish_calibration(info)
        
        else:
            print("校准失败，请重新采集数据")
        
        cv2.destroyAllWindows()
    
    if ros_node:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
