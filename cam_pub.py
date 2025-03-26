#!/usr/bin/env python3

"""
DepthAI摄像头ROS2发布节点
- 持续发布高质量图像到ROS2话题
- 支持参数化配置分辨率和图像质量
- 针对Ubuntu 22.04 + ROS2 Humble环境优化
- 增加畸变校正功能
"""

import depthai as dai
import cv2
import numpy as np
import time
import sys
import signal
import argparse

# ROS2相关导入
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
    print("ROS2依赖库已找到")
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2依赖库未找到，ROS2功能将被禁用")

# ROS2图像发布器类
class ROS2ImagePublisher(Node):
    def __init__(self, node_name="depthai_camera", camera_name="oak"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.camera_name = camera_name
        
        # 创建图像发布器
        self.image_pub = self.create_publisher(
            Image, 
            f'/{camera_name}/image_raw', 
            10
        )
        
        # 创建相机信息发布器
        self.camera_info_pub = self.create_publisher(
            CameraInfo, 
            f'/{camera_name}/camera_info', 
            10
        )
        
        self.get_logger().info(f"ROS2图像发布节点已初始化，发布到话题 /{camera_name}/image_raw")
        
    def publish_image(self, cv_image, camera_info=None):
        """发布OpenCV图像到ROS2话题"""
        if cv_image is not None:
            try:
                # 确定正确的编码格式
                if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                    encoding = "bgr8"
                else:
                    encoding = "mono8"
                    
                # 创建图像消息
                ros_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
                timestamp = self.get_clock().now().to_msg()
                ros_msg.header.stamp = timestamp
                ros_msg.header.frame_id = self.camera_name
                
                # 发布图像
                self.image_pub.publish(ros_msg)
                
                # 如果有相机信息，也发布它
                if camera_info is not None:
                    camera_info.header.stamp = timestamp
                    camera_info.header.frame_id = self.camera_name
                    self.camera_info_pub.publish(camera_info)
                    
            except Exception as e:
                self.get_logger().error(f"发布图像失败: {str(e)}")

def get_camera_info(frame, intrinsic_matrix, dist_coeffs=None):
    """创建相机信息消息"""
    if not ROS2_AVAILABLE:
        return None
        
    from sensor_msgs.msg import CameraInfo
    
    camera_info = CameraInfo()
    if frame is not None:
        camera_info.width = frame.shape[1]
        camera_info.height = frame.shape[0]
    
    # 如果有内参矩阵，填充相机信息
    if intrinsic_matrix is not None:
        # 相机矩阵 (fx, 0, cx, 0, fy, cy, 0, 0, 1)
        camera_info.k[0] = float(intrinsic_matrix[0][0])  # fx
        camera_info.k[2] = float(intrinsic_matrix[0][2])  # cx
        camera_info.k[4] = float(intrinsic_matrix[1][1])  # fy
        camera_info.k[5] = float(intrinsic_matrix[1][2])  # cy
        camera_info.k[8] = 1.0
        
        # 投影矩阵 (fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0)
        camera_info.p[0] = float(intrinsic_matrix[0][0])  # fx
        camera_info.p[2] = float(intrinsic_matrix[0][2])  # cx
        camera_info.p[5] = float(intrinsic_matrix[1][1])  # fy
        camera_info.p[6] = float(intrinsic_matrix[1][2])  # cy
        camera_info.p[10] = 1.0
        
    # 如果有畸变系数，也添加到相机信息中
    if dist_coeffs is not None:
        for i in range(min(len(dist_coeffs), 5)):
            camera_info.d.append(float(dist_coeffs[i]))
    
    return camera_info

def create_pipeline(args):
    """创建DepthAI管道"""
    pipeline = dai.Pipeline()
    
    # 创建彩色相机节点
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    
    # 设置相机属性
    if args.resolution == "4K":
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    elif args.resolution == "1080p":
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    elif args.resolution == "720p":
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    else:
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    # 设置输出尺寸
    width = args.width
    height = args.height
    
    # 设置视频和预览尺寸 (正确的API方法)
    cam_rgb.setVideoSize(width, height)
    cam_rgb.setPreviewSize(640, 360)  # 低分辨率预览
    
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(args.fps)
    
    # 启用畸变校正
    try:
        cam_rgb.initialControl.setManualFocus(130)  # 适当的对焦值
        cam_rgb.setIspScale(1, 1)  # 确保不改变分辨率
        # 尝试设置畸变系数 - 如果此方法可用
        if hasattr(cam_rgb, 'setDistortionCoefficients'):
            cam_rgb.setDistortionCoefficients([0, 0, 0, 0, 0])  # 初始设置为0，让设备使用校准参数
    except Exception as e:
        print(f"设置相机校正参数时出错: {e}")
        print("将使用OpenCV进行后处理畸变校正")

    # 创建控制输入队列
    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName("control")
    controlIn.out.link(cam_rgb.inputControl)
    
    # 创建输出
    xout_isp = pipeline.create(dai.node.XLinkOut)
    xout_isp.setStreamName("isp")
    cam_rgb.isp.link(xout_isp.input)
    
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")
    cam_rgb.preview.link(xout_preview.input)
    
    return pipeline

def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='DepthAI ROS2 图像发布节点')
    parser.add_argument('-r', '--resolution', type=str, default='720p', 
                        choices=['720p', '1080p', '4K'],
                        help='相机分辨率')
    parser.add_argument('-f', '--fps', type=float, default=30.0,
                        help='帧率')
    parser.add_argument('-d', '--device', type=str, default='',
                        help='设备MX ID (如有多个设备)')
    parser.add_argument('-n', '--name', type=str, default='oak',
                        help='ROS2话题名称前缀')
    parser.add_argument('-s', '--show', action='store_true', default=True,
                        help='显示图像窗口')
    parser.add_argument('--no-ros', action='store_true',
                        help='禁用ROS2发布')
    parser.add_argument('--undistort', action='store_true', default=True,
                        help='启用畸变校正')
    
    args = parser.parse_args()
    
    # 优化节点名称，保证符合ROS2命名规范
    node_name = args.name.replace('-', '_').replace(' ', '_')
    
    # 根据分辨率和大小的默认值
    if args.resolution == '720p':
        args.width = 1280
        args.height = 720
    if args.resolution == '1080p':
        args.width = 1920
        args.height = 1080
    elif args.resolution == '4K':
        args.width = 3840
        args.height = 2160
    
    # 初始化ROS2（如果启用）
    ros_node = None
    if not args.no_ros and ROS2_AVAILABLE:
        try:
            print("正在初始化ROS2...")
            rclpy.init(args=None)
            ros_node = ROS2ImagePublisher(node_name=f"depthai_camera", camera_name=node_name)
            print(f"ROS2节点已初始化，发布到话题 /{node_name}/image_raw")
        except Exception as e:
            print(f"初始化ROS2失败: {e}")
            if 'rclpy' in sys.modules and rclpy.ok():
                try:
                    rclpy.shutdown()
                except:
                    pass
            ros_node = None
    
    try:
        # 创建DepthAI管道
        pipeline = create_pipeline(args)
        
        # 连接到设备
        print("连接到设备...")
        device_info = dai.DeviceInfo(args.device) if args.device else None
        with dai.Device(pipeline, device_info) as device:
            # 获取设备信息
            device_name = device.getDeviceInfo().getMxId()
            print(f"已连接到设备: {device_name}")
            
            # 获取相机校准参数
            calib = device.readCalibration()
            intrinsic_matrix = None
            dist_coeffs = None
            
            # 尝试获取内参和畸变系数
            try:
                # 首先尝试RGB接口
                try:
                    intrinsic_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 
                                                                        args.width, args.height))
                    dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
                    print(f"成功获取RGB相机内参和畸变系数")
                except:
                    # 然后尝试CAM_A (常见于OAK-D和其他设备)
                    try:
                        intrinsic_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 
                                                                            args.width, args.height))
                        dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
                        print(f"成功获取CAM_A相机内参和畸变系数")
                    except:
                        print("无法获取相机内参和畸变系数，将使用默认值")
            except Exception as e:
                print(f"获取相机参数时出错: {e}")
            
            if intrinsic_matrix is not None:
                print(f"相机内参矩阵:\n{intrinsic_matrix}")
            if dist_coeffs is not None:
                print(f"畸变系数: {dist_coeffs}")

            # 获取输出队列
            q_isp = device.getOutputQueue(name="isp", maxSize=4, blocking=False)
            q_preview = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
            
            print(f"开始处理图像，分辨率: {args.width}x{args.height} @ {args.fps}fps")
            print("按Ctrl+C停止")
            
            # 使用时间差计算 FPS，不依赖帧数计数
            start_time = time.time()
            last_fps_print = start_time
            
            # 如果启用畸变校正且有必要的参数
            use_opencv_undistort = args.undistort and intrinsic_matrix is not None and dist_coeffs is not None
            if use_opencv_undistort:
                print("使用OpenCV进行额外的畸变校正")
                # 计算新的相机矩阵用于校正
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    intrinsic_matrix, dist_coeffs, (args.width, args.height), 0, (args.width, args.height))

            # 控制相机参数
            controlQueue = device.getInputQueue('control')
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(130)  # 设置对焦
            controlQueue.send(ctrl)

            # 主循环中
            while True:
                # 获取高分辨率帧
                in_isp = q_isp.tryGet()
                if in_isp is not None:
                    frame = in_isp.getCvFrame()
                    
                    # 应用OpenCV畸变校正（如果启用）
                    if use_opencv_undistort:
                        # 使用OpenCV的undistort函数校正畸变
                        frame = cv2.undistort(frame, intrinsic_matrix, dist_coeffs, None, newcameramtx)
                    
                    # 计算和显示FPS (每秒更新一次)
                    current_time = time.time()
                    if current_time - last_fps_print >= 1.0:
                        fps = 1.0 / (current_time - start_time) if (current_time != start_time) else 0
                        print(f"FPS: {fps:.2f}, 分辨率: {frame.shape[1]}x{frame.shape[0]}")
                        last_fps_print = current_time
                    start_time = current_time
                    
                    # 显示图像（如果需要）
                    if args.show:
                        # 显示低分辨率预览
                        # preview = q_preview.tryGet()
                        # if preview is not None:
                        #     preview_frame = preview.getCvFrame()
                        #     # 对预览图也应用畸变校正（如果启用）
                        #     if use_opencv_undistort and preview_frame.shape[0] > 0:
                        #         preview_h, preview_w = preview_frame.shape[:2]
                        #         # 调整相机矩阵用于预览分辨率
                        #         scale_x = preview_w / args.width
                        #         scale_y = preview_h / args.height
                        #         preview_matrix = intrinsic_matrix.copy()
                        #         preview_matrix[0,0] *= scale_x  # fx
                        #         preview_matrix[0,2] *= scale_x  # cx
                        #         preview_matrix[1,1] *= scale_y  # fy
                        #         preview_matrix[1,2] *= scale_y  # cy
                        #         preview_newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                        #             preview_matrix, dist_coeffs, (preview_w, preview_h), 0, (preview_w, preview_h))
                        #         preview_frame = cv2.undistort(preview_frame, preview_matrix, 
                        #                                     dist_coeffs, None, preview_newcameramtx)
                            
                        #     cv2.imshow("Preview", preview_frame)
                        
                        # 显示高分辨率图像
                        cv2.imshow("Video", frame)
                        cv2.waitKey(1)  # 确保调用 waitKey 以刷新窗口
                    
                    # 发布到ROS2
                    if ros_node:
                        # 创建相机信息
                        camera_info = get_camera_info(frame, intrinsic_matrix, dist_coeffs)
                        # 发布图像和相机信息
                        ros_node.publish_image(frame, camera_info)
                        # 处理ROS2回调
                        rclpy.spin_once(ros_node, timeout_sec=0.001)
                
                # 检查键盘输入
                if args.show and cv2.waitKey(1) == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("用户中断，程序停止")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if args.show:
            cv2.destroyAllWindows()
        
        if ros_node:
            try:
                ros_node.destroy_node()
                rclpy.shutdown()
                print("ROS2节点已关闭")
            except Exception as e:
                print(f"关闭ROS2时出错: {e}")
        
        print("程序已退出")

if __name__ == "__main__":
    main()
