import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def process_image(image_path):
    """
    处理图像并确定小车的最佳行驶方向
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    # 转换为HSV颜色空间，更容易进行颜色分割
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义轨道颜色的HSV范围（需要根据实际轨道颜色调整）
    # 这里假设轨道是棕色/黄褐色
    lower_track = np.array([4, 171, 90])
    upper_track = np.array([24, 255, 255])
    
    # 创建轨道的掩码
    track_mask = cv2.inRange(hsv, lower_track, upper_track)
    
    # 应用形态学操作来减少噪声
    kernel = np.ones((5, 5), np.uint8)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel)
    
    # 获取轨道区域
    track = cv2.bitwise_and(img, img, mask=track_mask)
    
    # 图像中心点（假设小车在图像中心）
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # 检查小车是否脱轨
    center_region = track_mask[center_y-20:center_y+20, center_x-20:center_x+20]
    if np.sum(center_region) == 0:
        print("检测到小车脱轨！计算返回轨道的最短路径...")
        return find_shortest_path_to_track(track_mask, center_x, center_y, img)
    
    # 分析轨道方向
    return analyze_track_direction(track_mask, center_x, center_y, img)

def find_shortest_path_to_track(track_mask, center_x, center_y, img):
    """
    当小车脱轨时，计算返回轨道的最短路径
    """
    # 使用距离变换找到最近的轨道点
    dist_transform = cv2.distanceTransform(255 - track_mask, cv2.DIST_L2, 5)
    
    # 创建搜索区域（以小车为中心的圆形区域）
    search_radius = 300  # 搜索半径
    y, x = np.ogrid[-center_y:track_mask.shape[0]-center_y, -center_x:track_mask.shape[1]-center_x]
    mask = x*x + y*y <= search_radius*search_radius
    
    # 在搜索区域内找到最近的轨道点
    masked_dist = np.ma.masked_array(dist_transform, ~mask)
    min_idx = np.unravel_index(np.argmin(masked_dist), dist_transform.shape)
    nearest_y, nearest_x = min_idx
    
    # 计算方向角度（从小车指向最近轨道点）
    dx = nearest_x - center_x
    dy = nearest_y - center_y
    angle = math.degrees(math.atan2(dy, dx))
    
    # 在图像上绘制指示线和箭头
    result_img = img.copy()
    cv2.line(result_img, (center_x, center_y), (nearest_x, nearest_y), (0, 0, 255), 2)
    draw_arrow(result_img, center_x, center_y, angle, 50, (0, 0, 255))
    
    # 在图像上标注信息
    cv2.putText(result_img, f"脱轨! 返回角度: {angle:.1f}度", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return result_img, angle

def analyze_track_direction(track_mask, center_x, center_y, img):
    """
    分析轨道方向，确定小车应该行驶的方向
    """
    # 创建扇形区域进行方向分析
    angles = []
    max_density = 0
    best_angle = 0
    prev_best_angle = None  # 用于防止原地打转
    
    # 检测轨道边界
    contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img.copy(), 0
    
    # 找到最大的轨道轮廓
    track_contour = max(contours, key=cv2.contourArea)
    
    # 计算轨道中心
    M = cv2.moments(track_contour)
    if M["m00"] != 0:
        track_center_x = int(M["m10"] / M["m00"])
        track_center_y = int(M["m01"] / M["m00"])
    else:
        track_center_x, track_center_y = center_x, center_y
    
    # 在小车周围360度范围内，每15度检查一次轨道密度和边界距离
    radius = 150  # 检测半径
    angle_scores = []  # 存储每个角度的综合评分
    
    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        end_x = int(center_x + radius * math.cos(rad))
        end_y = int(center_y + radius * math.sin(rad))
        
        # 创建线段上的点
        points = create_line_points(center_x, center_y, end_x, end_y)
        
        # 计算该方向上的轨道密度
        track_density = calculate_track_density(track_mask, points)
        
        # 计算到轨道边界的距离
        edge_distance = calculate_edge_distance(track_contour, center_x, center_y, angle, radius)
        
        # 计算是否为逆时针方向（相对于轨道中心）
        is_counterclockwise = is_direction_counterclockwise(center_x, center_y, track_center_x, track_center_y, angle)
        
        # 综合评分：轨道密度 + 边界距离适中性 + 逆时针方向优先
        # 边界距离评分：距离太近或太远都不好，最佳距离是半径的30%-70%之间
        optimal_distance_ratio = 0.5  # 最佳距离比例（相对于半径）
        distance_score = 1.0 - abs(edge_distance / radius - optimal_distance_ratio)
        
        # 逆时针方向加分
        counterclockwise_bonus = 0.3 if is_counterclockwise else 0
        
        # 综合评分 = 轨道密度(0-1) * 0.5 + 距离评分(0-1) * 0.3 + 逆时针加分(0/0.3) * 0.2
        score = track_density * 0.5 + distance_score * 0.3 + counterclockwise_bonus * 0.2
        
        # 记录角度、密度和评分
        angles.append((angle, track_density))
        angle_scores.append((angle, score))
        
        # 更新最佳方向
        if score > max_density:
            max_density = score
            best_angle = angle
    
    # 防止原地打转：如果新方向与上一次方向相差太大，进行平滑处理
    if prev_best_angle is not None:
        angle_diff = abs(best_angle - prev_best_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > 90:  # 如果方向变化超过90度，进行平滑
            best_angle = (best_angle + prev_best_angle) // 2  # 取平均值
    
    prev_best_angle = best_angle  # 更新上一次的最佳方向
    
    # 将角度转换为-180到180度范围
    if best_angle > 180:
        best_angle -= 360
    
    # 在图像上绘制指示线和箭头
    result_img = img.copy()
    
    # 绘制所有检测的方向（透明度较低）
    for angle, density in angles:
        rad = math.radians(angle)
        end_x = int(center_x + radius * math.cos(rad))
        end_y = int(center_y + radius * math.sin(rad))
        # 根据密度设置颜色（绿色到红色）
        color_intensity = int(255 * density / max_density) if max_density > 0 else 0
        line_color = (0, color_intensity, 255 - color_intensity)
        cv2.line(result_img, (center_x, center_y), (end_x, end_y), line_color, 1)
    
    # 绘制最佳方向（粗线和箭头）
    rad = math.radians(best_angle)
    end_x = int(center_x + radius * math.cos(rad))
    end_y = int(center_y + radius * math.sin(rad))
    cv2.line(result_img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
    draw_arrow(result_img, center_x, center_y, best_angle, 50, (0, 255, 0))
    
    # 在图像上标注信息
    cv2.putText(result_img, f"最佳方向: {best_angle}度", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result_img, best_angle

def create_line_points(x1, y1, x2, y2):
    """
    创建两点之间的线段上的所有点
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points

def calculate_track_density(track_mask, points):
    """
    计算给定点集在轨道上的密度
    """
    if not points:
        return 0
    
    track_points = 0
    for x, y in points:
        if 0 <= y < track_mask.shape[0] and 0 <= x < track_mask.shape[1]:
            if track_mask[y, x] > 0:
                track_points += 1
    
    return track_points / len(points)

def draw_arrow(img, x, y, angle, length, color):
    """
    在图像上绘制箭头
    """
    rad = math.radians(angle)
    end_x = int(x + length * math.cos(rad))
    end_y = int(y + length * math.sin(rad))
    
    # 绘制箭头主体
    cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 2, tipLength=0.3)

def main():
    # 测试图像路径
    image_path = 'test_image.jpg'  # 替换为实际图像路径
    
    # 处理图像
    result_img, best_angle = process_image(image_path)
    
    if result_img is not None:
        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'最佳行驶方向: {best_angle}度')
        plt.axis('off')
        plt.show()
        
        # 保存结果
        cv2.imwrite('result.jpg', result_img)
        print(f"最佳行驶方向: {best_angle}度")
        print("结果已保存为 'result.jpg'")
    else:
        print("处理失败")

if __name__ == "__main__":
    main()


def calculate_edge_distance(contour, center_x, center_y, angle, radius):
    """
    计算从中心点沿指定角度方向到轨道边界的距离
    """
    # 从中心点沿指定角度方向创建一条射线
    rad = math.radians(angle)
    ray_end_x = int(center_x + radius * 2 * math.cos(rad))
    ray_end_y = int(center_y + radius * 2 * math.sin(rad))
    
    # 创建射线上的点
    ray_points = create_line_points(center_x, center_y, ray_end_x, ray_end_y)
    
    # 找到射线与轮廓的交点
    min_distance = radius * 2  # 初始化为最大可能距离
    
    for i, point in enumerate(ray_points):
        x, y = point
        # 检查点是否在图像范围内
        if 0 <= y < 1000 and 0 <= x < 1000:  # 假设图像最大尺寸为1000x1000
            # 检查点是否在轮廓边界上或附近
            distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
            if abs(distance) < 5:  # 如果点在轮廓边界附近（距离小于5像素）
                # 计算从中心点到该点的距离
                point_distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if point_distance < min_distance:
                    min_distance = point_distance
    
    return min_distance

def is_direction_counterclockwise(center_x, center_y, track_center_x, track_center_y, angle):
    """
    判断给定角度是否为逆时针方向（相对于轨道中心）
    """
    # 计算从小车到轨道中心的向量
    to_track_center_x = track_center_x - center_x
    to_track_center_y = track_center_y - center_y
    
    # 计算指定角度的向量
    rad = math.radians(angle)
    direction_x = math.cos(rad)
    direction_y = math.sin(rad)
    
    # 计算叉积，判断是否为逆时针方向
    # 如果叉积为正，则方向向量在轨道中心向量的逆时针方向
    cross_product = to_track_center_x * direction_y - to_track_center_y * direction_x
    
    return cross_product > 0