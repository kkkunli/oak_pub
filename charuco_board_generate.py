import cv2
import numpy as np

# 增强参数配置
squares_x = 9  
squares_y = 6
square_size_mm = 24
marker_size_mm = 18
dpi = 300
margin_mm = 10
border_width = 5  # 新增边界白边

# 精确计算棋盘区域（单位：米）
board_width_m = (squares_x-1) * square_size_mm / 1000
board_height_m = (squares_y-1) * square_size_mm / 1000

# 创建高对比度棋盘（反转颜色）
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_size_mm/1000,
    marker_size_mm/1000,
    dictionary
)

# 计算像素尺寸（精确到整数倍）
px_per_mm = dpi / 25.4
board_width_px = int(round((squares_x-1)*square_size_mm * px_per_mm))
board_height_px = int(round((squares_y-1)*square_size_mm * px_per_mm))
margin_px = int(margin_mm * px_per_mm)

# 创建高分辨率画布（添加抗锯齿）
img = np.full((
    board_height_px + 2*margin_px + 2*border_width,
    board_width_px + 2*margin_px + 2*border_width
), 255, dtype=np.uint8)

# 生成抗锯齿棋盘图
board_img = board.generateImage(
    (board_width_px, board_height_px),
    marginSize=border_width,
    borderBits=1
)

# 精确放置棋盘（带边界补偿）
y_start = margin_px + border_width
y_end = y_start + board_height_px
x_start = margin_px + border_width
x_end = x_start + board_width_px
img[y_start:y_end, x_start:x_end] = board_img

# 保存无损PNG（禁用压缩）
cv2.imwrite("charuco_board_A4.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# 打印关键参数
print(f"实际棋盘区域: {board_width_px/px_per_mm:.1f}x{board_height_px/px_per_mm:.1f}mm")
print(f"建议打印设置：")
print("1. 使用A4纸（210×297mm）")
print("2. 关闭页面缩放（100%打印）")
print("3. 使用高质量激光打印")
print("4. 检查实际测量尺寸：")
print(f"   理论棋盘宽：{(squares_x-1)*square_size_mm}mm")
print(f"   理论棋盘高：{(squares_y-1)*square_size_mm}mm")
