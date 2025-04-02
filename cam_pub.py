import depthai as dai
import cv2
import numpy as np
import time
import sys
import argparse
import threading
import queue

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
        
        # 使用与AprilTag节点兼容的QoS设置
        # 注意：AprilTag节点使用rmw_qos_profile_sensor_data
        qos = rclpy.qos.QoSProfile(
            depth=10,  # 增加队列深度
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,  # 改为可靠传输
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST
        )
        
        self.image_pub = self.create_publisher(Image, f'/{camera_name}/image_raw', qos)
        self.camera_info_pub = self.create_publisher(CameraInfo, f'/{camera_name}/camera_info', qos)
        self.frame_queue = queue.Queue(maxsize=4)
        self.publish_lock = threading.Lock()
        self.running = True
        self.publish_thread = threading.Thread(target=self._publish_loop) 
        self.publish_thread.start()
        
        # 添加丢帧计数器
        self.dropped_frames = 0
        self.total_frames = 0
        
        # 添加降采样计数器
        self.frame_count = 0
        self.publish_every_n_frames = 3  # 每3帧发布一次，降低处理负担

    def publish_image_async(self, frame, camera_info):
        self.total_frames += 1
        self.frame_count += 1
        
        # 降采样发布
        if self.frame_count % self.publish_every_n_frames != 0:
            return
            
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.dropped_frames += 1
                if self.total_frames % 30 == 0:
                    self.get_logger().info(f"丢帧率: {self.dropped_frames/self.total_frames:.2f}")
            except queue.Empty:
                pass
        self.frame_queue.put((frame, camera_info), block=False)

    def _publish_loop(self):
        while self.running:
            try:
                frame, camera_info = self.frame_queue.get(timeout=0.1)
                
                # 转换为灰度图像以减少数据量
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
                
                # 创建消息
                msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
                timestamp = self.get_clock().now().to_msg()
                
                # 确保时间戳完全一致
                msg.header.stamp = timestamp
                msg.header.frame_id = self.camera_name
                camera_info.header.stamp = timestamp
                camera_info.header.frame_id = self.camera_name
                
                # 同时发布两个消息，确保同步
                with self.publish_lock:
                    self.image_pub.publish(msg)
                    self.camera_info_pub.publish(camera_info)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"发布错误: {str(e)}")

    def stop(self):
        self.running = False
        self.publish_thread.join()


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
    parser.add_argument('-f', '--fps', type=float, default=10.0)
    parser.add_argument('-d', '--device', default='')
    parser.add_argument('-n', '--name', default='oak')
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('--no-ros', action='store_true')
    parser.add_argument('--undistort', action='store_true', default=True)
    args = parser.parse_args()

    if args.resolution == '720p': args.width, args.height = 1280, 720
    elif args.resolution == '1080p': args.width, args.height = 1920, 1080
    else: args.width, args.height = 3840, 2160

    ros_node = None
    if not args.no_ros and ROS2_AVAILABLE:
        rclpy.init()
        ros_node = ROS2ImagePublisher(node_name='depthai_camera', camera_name=args.name)

    try:
        pipeline = create_pipeline(args)
        device_info = dai.DeviceInfo(args.device) if args.device else None
        with dai.Device(pipeline, device_info) as device:
            calib = device.readCalibration()
            intrinsic = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, args.width, args.height))
            dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
            print("成功获取内参")

            map1, map2 = None, None
            if args.undistort:
                map1, map2 = cv2.initUndistortRectifyMap(
                    intrinsic, dist, None, intrinsic, (args.width, args.height), cv2.CV_16SC2)

            q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
            print("按 Ctrl+C 停止")
            last_time = time.time()

            while True:
                in_frame = q.tryGet()
                if in_frame is not None:
                    frame = in_frame.getCvFrame()
                    if args.undistort and map1 is not None:
                        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

                    now = time.time()
                    fps = 1.0 / (now - last_time)
                    print(f"FPS: {fps:.2f}, 分辨率: {frame.shape[1]}x{frame.shape[0]}")
                    last_time = now

                    if args.show:
                        cv2.imshow("Video", frame)
                        if cv2.waitKey(1) == ord('q'):
                            break

                    if ros_node:
                        info = get_camera_info(frame, intrinsic, dist)
                        ros_node.publish_image_async(frame, info)

    except KeyboardInterrupt:
        print("User interrupt")
    finally:
        if args.show:
            cv2.destroyAllWindows()
        if ros_node:
            ros_node.stop()
            ros_node.destroy_node()
            time.sleep(0.2)
            rclpy.shutdown()
            print("ROS2节点已关闭")
        print("Exit cleanly")

if __name__ == '__main__':
    main()
