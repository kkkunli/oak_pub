import cv2
import numpy as np

def digital_verification(image_path, params):
    # 加载数字图像
    digital_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 创建检测器
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    # 执行检测（添加类型转换）
    corners, ids, _ = detector.detectMarkers(digital_img)
    
    # 转换数据类型以适应OpenCV 4.10.0
    if ids is not None:
        ids = ids.astype(np.int32)  # 显式转换为int32类型
    
    # 理论参数
    verification = {
        'markers_detected': len(ids) if ids is not None else 0,
        'markers_expected': (params["squares_x"]-1)*(params["squares_y"]-1),
        'pixel_integrity': 0.0
    }
    
    # 仅当检测到标记时执行分析
    if ids is not None and len(ids) > 0:
        # 创建临时绘制图像
        marker_img = cv2.cvtColor(digital_img, cv2.COLOR_GRAY2BGR)
        
        # 修正绘制参数（OpenCV 4.10.0要求）
        # 将单个marker的corners转换为列表的列表
        draw_corners = [np.array(corners[0], dtype=np.int32)]
        draw_ids = np.array([ids[0]], dtype=np.int32)
        
        cv2.aruco.drawDetectedMarkers(marker_img, draw_corners, draw_ids)
        
        # 像素完整性分析
        x,y,w,h = cv2.boundingRect(corners[0])
        center_roi = digital_img[y+7:y+12, x+7:x+12]
        verification['pixel_integrity'] = np.mean(np.abs(center_roi - 128)) / 128
    
    return verification

if __name__ == "__main__":
    params = {
        "squares_x": 9,
        "squares_y": 6,
        "square_size": 24/1000,
        "marker_size": 18/1000,
        "dpi": 300
    }
    
    result = digital_verification("charuco_board_A4.png", params)
    
    # 控制台输出
    print("\n数字验证报告：")
    print("="*40)
    print(f"检测到标记数\t: {result['markers_detected']}/{result['markers_expected']}")
    print(f"像素纯度\t: {result['pixel_integrity']:.2f} (1.0为最佳)")
    
    if result['markers_detected'] < result['markers_expected']:
        print("\n问题诊断：")
        print("1. 生成图像可能存在抗锯齿效应（检查imwrite是否启用压缩）")
        print("2. OpenCV版本差异导致检测失败（4.10.0需要显式类型转换）")
        print("3. 建议添加以下生成后处理：")
        print("   img = cv2.medianBlur(img, 3)")
        print("   _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)")
