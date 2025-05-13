import depthai as dai
import cv2
import numpy as np
import time
import argparse
import threading

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
    print("ROS2依赖库已找到")
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2依赖库未找到")

class ROS2ImagePublisher(Node):
    def __init__(self, node_name="depthai_camera", camera_name="oak"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.camera_name = camera_name
        self.lock = threading.Lock()
        
        # 使用与AprilTag节点兼容的QoS设置
        qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST
        )
        
        self.image_pub = self.create_publisher(Image, f'/{camera_name}/image_raw', qos)
        self.camera_info_pub = self.create_publisher(CameraInfo, f'/{camera_name}/camera_info', qos)
        
        # 发布频率统计
        self.frame_count = 0
        self.last_stat_time = time.time()
        
        self.get_logger().info("ROS2图像发布器已初始化")

    def publish_image(self, frame, camera_info):
        """直接发布图像和相机信息，保持同步"""
        try:
            # 使用锁保证线程安全
            with self.lock:
                # 转换为灰度图像以减少数据量
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
                
                # 创建消息
                msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
                
                # 使用相同的时间戳
                timestamp = self.get_clock().now().to_msg()
                msg.header.stamp = timestamp
                msg.header.frame_id = self.camera_name
                camera_info.header.stamp = timestamp
                camera_info.header.frame_id = self.camera_name
                
                # 先发布相机信息，再发布图像
                self.camera_info_pub.publish(camera_info)
                self.image_pub.publish(msg)
                
                # 统计发布频率
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_stat_time >= 1.0:
                    fps = self.frame_count / (current_time - self.last_stat_time)
                    self.get_logger().info(f"当前发布频率: {fps:.1f} Hz")
                    self.frame_count = 0
                    self.last_stat_time = current_time
                    
        except Exception as e:
            self.get_logger().error(f"发布错误: {str(e)}")

def get_camera_info(frame, intrinsic_matrix, dist_coeffs=None):
    from sensor_msgs.msg import CameraInfo
    camera_info = CameraInfo()
    camera_info.width = frame.shape[1]
    camera_info.height = frame.shape[0]
    if intrinsic_matrix is not None:
        camera_info.k[0] = float(intrinsic_matrix[0][0])
        camera_info.k[2] = float(intrinsic_matrix[0][2])
        camera_info.k[4] = float(intrinsic_matrix[1][1])
        camera_info.k[5] = float(intrinsic_matrix[1][2])
        camera_info.k[8] = 1.0
        camera_info.p[0] = float(intrinsic_matrix[0][0])
        camera_info.p[2] = float(intrinsic_matrix[0][2])
        camera_info.p[5] = float(intrinsic_matrix[1][1])
        camera_info.p[6] = float(intrinsic_matrix[1][2])
        camera_info.p[10] = 1.0
    if dist_coeffs is not None:
        for i in range(min(len(dist_coeffs), 5)):
            camera_info.d.append(float(dist_coeffs[i]))
    return camera_info

def create_pipeline(args):
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    if args.resolution == "4K":
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    elif args.resolution == "1080p":
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    else:
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam_rgb.setVideoSize(args.width, args.height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(args.fps)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.video.link(xout.input)
    return pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', default='720p', choices=['720p', '1080p', '4K'])
    parser.add_argument('-f', '--fps', type=float, default=30.0,
                        help='相机采集帧率 (Hz)')
    parser.add_argument('-d', '--device', default='')
    parser.add_argument('-n', '--name', default='oak')
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('--no-ros', action='store_true')
    parser.add_argument('--undistort', action='store_true', default=False)
    args = parser.parse_args()

    if args.resolution == '720p': args.width, args.height = 1280, 720
    elif args.resolution == '1080p': args.width, args.height = 1920, 1080
    else: args.width, args.height = 3840, 2160

    ros_node = None
    if not args.no_ros and ROS2_AVAILABLE:
        try:
            rclpy.init()
            ros_node = ROS2ImagePublisher(
                node_name='depthai_camera', 
                camera_name=args.name
            )
            
            # 创建独立线程处理ROS2消息
            ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
            ros_spin_thread.start()
            
        except Exception as e:
            print(f"ROS2初始化失败: {e}")
            ros_node = None

    try:
        pipeline = create_pipeline(args)
        device_info = dai.DeviceInfo(args.device) if args.device else None
        with dai.Device(pipeline, device_info) as device:
            try:
                calib = device.readCalibration()
                intrinsic = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, args.width, args.height))
                dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
                print("成功获取内参")
            except Exception as e:
                print(f"获取相机参数失败: {e}")
                print("使用默认内参")
                intrinsic = np.array([[1000, 0, args.width/2], [0, 1000, args.height/2], [0, 0, 1]])
                dist = np.array([0, 0, 0, 0, 0])

            map1, map2 = None, None
            if args.undistort:
                map1, map2 = cv2.initUndistortRectifyMap(
                    intrinsic, dist, None, intrinsic, (args.width, args.height), cv2.CV_16SC2)

            q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
            print("按 Ctrl+C 停止")
            last_time = time.time()
            frame_count = 0

            while True:
                in_frame = q.tryGet()
                if in_frame is not None:
                    frame = in_frame.getCvFrame()
                    if args.undistort and map1 is not None:
                        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

                    now = time.time()
                    frame_count += 1
                    if now - last_time >= 1.0:  # 每秒更新一次FPS
                        fps = frame_count / (now - last_time)
                        print(f"相机FPS: {fps:.2f}, 分辨率: {frame.shape[1]}x{frame.shape[0]}")
                        frame_count = 0
                        last_time = now

                    if args.show:
                        cv2.imshow("Video", frame)
                        if cv2.waitKey(1) == ord('q'):
                            break

                    if ros_node:
                        info = get_camera_info(frame, intrinsic, dist)
                        ros_node.publish_image(frame, info)

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if args.show:
            cv2.destroyAllWindows()
        if ros_node:
            try:
                ros_node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                print("ROS2节点已关闭")
            except Exception as e:
                print(f"关闭ROS2时出错: {e}")
        print("程序已正常退出")

if __name__ == '__main__':
    main()
