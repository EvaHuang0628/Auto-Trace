import time
import math
import mss
import numpy as np
import cv2
from android_control import focus_scrcpy_window, swipe, wait
from boundary_trace import process_image_boundary_for_Control
from auto_trace import process_image_for_Control

def angle_to_coordinates(angle_degrees, radius=50):
    """
    将角度转换为圆周上的坐标
    :param angle_degrees: 角度值，0-360，正右为0度，逆时针增加
    :param radius: 圆的半径，默认为50px
    :return: (x, y) 坐标元组
    """
    # 将角度转换为弧度（并调整坐标系，使0度在正右，逆时针增加）
    angle_radians = math.radians(angle_degrees)
    
    # 计算坐标
    x = radius * math.cos(angle_radians)
    y = -radius * math.sin(angle_radians)  # 使用负号是因为屏幕坐标系Y轴向下
    
    return round(x), round(y)

def get_screen_image(left_x=None, top_y=None, right_x=None, bottom_y=None):
    """
    获取当前屏幕图像
    :param left_x: 截图区域的左上角x坐标，None表示使用整个屏幕
    :param top_y: 截图区域的左上角y坐标，None表示使用整个屏幕
    :param right_x: 截图区域的右下角x坐标，None表示使用整个屏幕
    :param bottom_y: 截图区域的右下角y坐标，None表示使用整个屏幕
    :return: 屏幕图像的numpy数组
    """
    try:
        with mss.mss() as sct:
            # 获取主显示器
            monitor = sct.monitors[1]  # 主显示器索引为1
            # 截取屏幕
            if left_x is None or top_y is None or right_x is None or bottom_y is None:
                # 如果任一参数为None，截取整个屏幕
                screenshot = sct.grab(monitor)
            else:
                # 截取指定区域
                monitor = {
                    'left': left_x,
                    'top': top_y,
                    'width': right_x - left_x,
                    'height': bottom_y - top_y
                }
                screenshot = sct.grab(monitor)

            # 转换为numpy数组并转为BGR格式（OpenCV使用的格式）
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
            return image
    except Exception as e:
        print(f"截图失败: {e}")
        return None

def auto_control_loop(interval=1):
    """
    自动控制主循环
    :param interval: 每次处理的间隔时间（秒），默认为1秒
    """
    # 切换到串流画面
    focus_scrcpy_window()
    
    # 假设圆心位置
    circle_x = 146
    circle_y = 669

    isRealTime = True

    # 实时画面截屏坐标
    left_x = 1 if isRealTime else 978
    top_y = 297 if isRealTime else 79
    right_x = 934 if isRealTime else 1915
    bottom_y = 763 if isRealTime else 547
    

    while True:

        # 获取当前时间戳（用于性能测量）
        start_time = time.time()

    
        # 获取画面内容存储成变量
        screen_image = get_screen_image(left_x, top_y, right_x, bottom_y)
        
        if screen_image is not None:
            # 处理图像获取方向
            direction_result = process_image_boundary_for_Control(screen_image)
            # direction_result = process_image_for_Control(screen_image)

            # 检查返回值是否为元组，如果是元组则提取角度值
            if isinstance(direction_result, tuple):
                # 元组格式为 (result_img, angle)
                _, direction_angle = direction_result
            else:
                # 直接返回角度值
                direction_angle = direction_result

            if direction_angle is not None:
                # 将方向转换为圆周坐标
                offset_x, offset_y = angle_to_coordinates(direction_angle)
                target_x = circle_x + offset_x
                target_y = circle_y + offset_y
                
                print(f"方向角度: {direction_angle}°, 目标坐标: ({target_x}, {target_y})")
                
                # 执行滑动操作
                swipe(circle_x, circle_y, int(target_x), int(target_y), 0.5)
            else:
                print("无法确定方向角度")
        else:
            print("获取屏幕图像失败")
        
        # 计算处理耗时并等待剩余时间
        # processing_time = time.time() - start_time
        # sleep_time = max(0, interval - processing_time)
        # time.sleep(sleep_time)


if __name__ == "__main__":
    auto_control_loop()