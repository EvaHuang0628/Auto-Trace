import cv2
import numpy as np

def find_track_boundaries(track_mask):
    """
    找出轨道的左右边界线。
    
    使用Canny边缘检测和霍夫变换检测直线，然后根据斜率将直线分为左右边界。
    对每边的多条线段取平均值得到最终的边界线。
    
    参数：
        track_mask: numpy.ndarray，二值化的轨道掩码图像
        
    返回：
        tuple: (left_boundary, right_boundary) 或 None
            - left_boundary: [x1, y1, x2, y2] 左边界线的两个端点坐标
            - right_boundary: [x1, y1, x2, y2] 右边界线的两个端点坐标
            - 如果未检测到足够的边界线，返回None
    """
    # 使用Canny边缘检测
    edges = cv2.Canny(track_mask, 50, 150)
    
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=100, maxLineGap=50)
    
    if lines is None or len(lines) < 2:
        return None
    
    # 将线段按照斜率分组（左边界和右边界）
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:  # 避免除以零
            continue
        slope = (y2 - y1) / (x2 - x1)
        
        # 根据斜率判断是左边界还是右边界
        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])
    
    # 如果没有足够的线段，返回None
    if not left_lines or not right_lines:
        return None
    
    # 对每边取平均得到最终的边界线
    left_boundary = np.mean(left_lines, axis=0, dtype=np.int32)
    right_boundary = np.mean(right_lines, axis=0, dtype=np.int32)
    
    return left_boundary, right_boundary

def calculate_perpendicular_distance(point, line):
    """
    计算点到直线的垂直距离。
    
    使用点到直线距离公式：|ax₀ + by₀ + c|/√(a² + b²)
    其中直线方程为：ax + by + c = 0
    
    参数：
        point: tuple，(x, y) 点的坐标
        line: list，[x1, y1, x2, y2] 直线的两个端点坐标
        
    返回：
        float: 点到直线的垂直距离
    """
    x0, y0 = point
    x1, y1, x2, y2 = line
    
    # 直线方程：ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    
    # 点到直线的距离公式
    distance = abs(a*x0 + b*y0 + c) / np.sqrt(a*a + b*b)
    
    return distance

def calculate_direction(point, line1, line2):
    """
    计算从点到两条边界线的垂直方向。
    
    首先计算两条边界线的中点，然后计算轨道的大致方向（从中点1到中点2），
    最后计算垂直于轨道方向的角度（顺时针旋转90度）。
    
    参数：
        point: tuple，(x, y) 点的坐标（通常是小车位置）
        line1: list，[x1, y1, x2, y2] 第一条边界线的两个端点坐标
        line2: list，[x1, y1, x2, y2] 第二条边界线的两个端点坐标
        
    返回：
        float: 方向角度（0-360度），表示小车应该行驶的方向
    """
    # 计算两条线的中点
    mid_point1 = ((line1[0] + line1[2])//2, (line1[1] + line1[3])//2)
    mid_point2 = ((line2[0] + line2[2])//2, (line2[1] + line2[3])//2)
    
    # 计算轨道的大致方向（从中点1到中点2的方向）
    track_direction = np.arctan2(mid_point2[1] - mid_point1[1],
                                mid_point2[0] - mid_point1[0])
    
    # 垂直方向（顺时针旋转90度）
    perpendicular_direction = track_direction + np.pi/2
    
    # 转换为度数
    angle_deg = np.degrees(perpendicular_direction)
    
    # 确保角度在0-360度范围内
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg

def process_image_boundary(image_path):
    """
    使用边界线方法处理图像并确定小车的最佳行驶方向。
    
    处理步骤：
    1. 读取图像并转换为HSV颜色空间
    2. 使用颜色阈值分割出轨道区域
    3. 应用形态学操作减少噪声
    4. 检测左右边界线
    5. 计算小车（图像中心）到边界线的距离
    6. 根据距离比例确定行驶方向
    7. 在图像上绘制可视化结果
    
    参数：
        image_path: str，输入图像的路径
        
    返回：
        tuple: (result_img, direction) 或 (None, None)
            - result_img: numpy.ndarray，带有边界线和方向指示的结果图像
            - direction: float，最佳行驶方向（0-360度）
            - 如果处理失败，返回(None, None)
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义轨道颜色的HSV范围（默认适用于棕色/黄褐色轨道）
    # H: 10-30 (色调，棕色/黄褐色范围)
    # S: 50-255 (饱和度，排除灰白色)
    # V: 50-255 (亮度，排除过暗区域)
    lower_track = np.array([10, 50, 50])
    upper_track = np.array([30, 255, 255])
    
    # 创建轨道的掩码
    track_mask = cv2.inRange(hsv, lower_track, upper_track)
    
    # 应用形态学操作来减少噪声
    kernel = np.ones((5, 5), np.uint8)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找出边界线
    boundaries = find_track_boundaries(track_mask)
    if boundaries is None:
        print("未能检测到足够的边界线")
        return None, None
    
    left_boundary, right_boundary = boundaries
    
    # 图像中心点（假设小车在图像中心）
    height, width = img.shape[:2]
    center_point = (width // 2, height // 2)
    
    # 计算到两条边界线的距离
    d1 = calculate_perpendicular_distance(center_point, left_boundary)
    d2 = calculate_perpendicular_distance(center_point, right_boundary)
    
    # 根据距离比例决定行进方向
    if 0.8 * d2 <= d1 <= 1.3 * d2:
        # 距离相近，沿着垂直方向行驶
        direction = calculate_direction(center_point, left_boundary, right_boundary)
    else:
        # 距离相差较大，需要先调整位置
        # 计算到边界线的垂点连线方向
        direction = calculate_direction(center_point, left_boundary, right_boundary)
        if d1 > d2:
            direction = (direction + 180) % 360  # 反向
    
    # 在图像上绘制结果
    result_img = img.copy()
    
    # 绘制边界线
    cv2.line(result_img, (left_boundary[0], left_boundary[1]),
             (left_boundary[2], left_boundary[3]), (0, 0, 255), 2)
    cv2.line(result_img, (right_boundary[0], right_boundary[1]),
             (right_boundary[2], right_boundary[3]), (0, 0, 255), 2)
    
    # 绘制中心点
    cv2.circle(result_img, center_point, 5, (0, 255, 0), -1)
    
    # 绘制方向箭头
    arrow_length = 100
    end_point = (int(center_point[0] + arrow_length * np.cos(np.radians(direction))),
                int(center_point[1] + arrow_length * np.sin(np.radians(direction))))
    cv2.arrowedLine(result_img, center_point, end_point, (255, 0, 0), 2)
    
    return result_img, direction