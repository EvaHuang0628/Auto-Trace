import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from auto_trace import process_image, draw_arrow
from sklearn.cluster import KMeans

def adjust_hsv_thresholds(image_path, interactive=True):
    """
    调整HSV阈值以适应不同的轨道颜色
    
    参数:
        image_path: 图像路径
        interactive: 是否使用交互式调整，如果为False则自动调整
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 初始HSV阈值（与auto_trace.py中相同）
    h_min, s_min, v_min = 10, 50, 50
    h_max, s_max, v_max = 30, 255, 255
    
    if interactive:
        # 创建窗口
        cv2.namedWindow('调整HSV阈值')
        
        # 创建滑动条
        def nothing(x):
            pass
        
        cv2.createTrackbar('H_min', '调整HSV阈值', h_min, 179, nothing)
        cv2.createTrackbar('H_max', '调整HSV阈值', h_max, 179, nothing)
        cv2.createTrackbar('S_min', '调整HSV阈值', s_min, 255, nothing)
        cv2.createTrackbar('S_max', '调整HSV阈值', s_max, 255, nothing)
        cv2.createTrackbar('V_min', '调整HSV阈值', v_min, 255, nothing)
        cv2.createTrackbar('V_max', '调整HSV阈值', v_max, 255, nothing)
        
        while True:
            # 获取当前滑动条的值
            h_min = cv2.getTrackbarPos('H_min', '调整HSV阈值')
            h_max = cv2.getTrackbarPos('H_max', '调整HSV阈值')
            s_min = cv2.getTrackbarPos('S_min', '调整HSV阈值')
            s_max = cv2.getTrackbarPos('S_max', '调整HSV阈值')
            v_min = cv2.getTrackbarPos('V_min', '调整HSV阈值')
            v_max = cv2.getTrackbarPos('V_max', '调整HSV阈值')
            
            # 创建掩码
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # 应用形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 显示结果
            result = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow('原图', img)
            cv2.imshow('掩码', mask)
            cv2.imshow('结果', result)
            
            # 按ESC键退出
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        
        cv2.destroyAllWindows()
        
        print("\n最终HSV阈值:")
        print(f"lower_track = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_track = np.array([{h_max}, {s_max}, {v_max}])")
    else:
        # 自动调整HSV阈值（基于图像分析）
        # 这里使用简单的颜色聚类来找到主要的轨道颜色
        # 将图像转换为一维数组
        pixels = hsv.reshape(-1, 3)
        
        # 使用K-means聚类找到主要颜色
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(pixels)
        colors = kmeans.cluster_centers_
        
        # 找到最可能的轨道颜色（假设轨道颜色在黄色/棕色范围内）
        track_color_idx = -1
        for i, color in enumerate(colors):
            h, s, v = color
            # 检查是否在黄色/棕色范围内
            if 5 <= h <= 40 and s >= 30 and v >= 30:
                track_color_idx = i
                break
        
        if track_color_idx != -1:
            # 找到了可能的轨道颜色
            h, s, v = colors[track_color_idx]
            # 设置HSV范围（给一定的容差）
            h_min = max(0, int(h - 10))
            h_max = min(179, int(h + 10))
            s_min = max(0, int(s - 30))
            s_max = 255
            v_min = max(0, int(v - 30))
            v_max = 255
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        print(f"自动调整HSV阈值: [{h_min}, {s_min}, {v_min}] - [{h_max}, {s_max}, {v_max}]")
    
    return lower, upper

def modify_auto_trace_thresholds(lower, upper):
    """
    修改auto_trace.py中的HSV阈值
    """
    try:
        with open('auto_trace.py', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 替换HSV阈值
        import re
        content = re.sub(r'lower_track = np.array\(\[\d+, \d+, \d+\]\)', 
                         f'lower_track = np.array([{lower[0]}, {lower[1]}, {lower[2]}])', 
                         content)
        content = re.sub(r'upper_track = np.array\(\[\d+, \d+, \d+\]\)', 
                         f'upper_track = np.array([{upper[0]}, {upper[1]}, {upper[2]}])', 
                         content)
        
        with open('auto_trace.py', 'w', encoding='utf-8') as file:
            file.write(content)
        
        print("已更新auto_trace.py中的HSV阈值")
    except Exception as e:
        print(f"更新HSV阈值失败: {e}")

def test_with_image(image_path, save_result=True):
    """
    使用指定图像测试循迹算法
    """
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
        if save_result:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = f'result_{base_name}.jpg'
            cv2.imwrite(result_path, result_img)
            print(f"最佳行驶方向: {best_angle}度")
            print(f"结果已保存为 '{result_path}'")
    else:
        print("处理失败")

def process_video(video_path, start_time_sec=85, frame_interval=3, save_result=False, adjust_hsv=False):
    """
    处理视频文件，从指定时间点开始，每隔指定帧数提取一帧进行处理
    
    参数:
        video_path: 视频文件路径
        start_time_sec: 开始处理的时间点（秒）
        frame_interval: 帧间隔，每隔多少帧处理一次
        save_result: 是否保存结果视频
        adjust_hsv: 是否定期更新HSV阈值
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频FPS: {fps}")
    print(f"总帧数: {frame_count}")
    print(f"分辨率: {width}x{height}")
    
    # 设置开始帧位置
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 创建窗口
    cv2.namedWindow('视频处理结果', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('视频处理结果', width, height)
    
    # 如果需要保存结果视频
    out = None
    if save_result:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.splitext(video_path)[0] + '_hsv_result.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps/frame_interval, (width, height))
    
    frame_idx = 0
    last_hsv_update_time = 0  # 上次更新HSV阈值的时间
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("视频处理完成或读取错误")
                break
            
            # 计算当前时间
            current_time = (start_frame + frame_idx) / fps
            
            # 每隔frame_interval帧处理一次
            if frame_idx % frame_interval == 0:
                # 将当前帧保存为临时图像文件
                temp_img_path = 'temp_frame.jpg'
                cv2.imwrite(temp_img_path, frame)
                
                # 如果启用了HSV阈值调整且距离上次更新已经过去3秒
                if adjust_hsv and (current_time - last_hsv_update_time >= 3):
                    print("更新HSV阈值...")
                    # 使用非交互式模式调整HSV阈值
                    lower, upper = adjust_hsv_thresholds(temp_img_path, interactive=False)
                    if lower is not None and upper is not None:
                        modify_auto_trace_thresholds(lower, upper)
                        last_hsv_update_time = current_time
                
                # 处理图像
                result_img, best_angle = process_image(temp_img_path)
                
                if result_img is not None:
                    # 显示结果
                    cv2.imshow('视频处理结果', result_img)
                    
                    # 保存到结果视频
                    if save_result and out is not None:
                        out.write(result_img)
                    
                    # 显示当前帧信息
                    print(f"当前时间: {current_time:.2f}秒, 方向: {best_angle}度")
                
                # 删除临时文件
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
    
    finally:
        # 释放资源
        cap.release()
        if save_result and out is not None:
            out.write(result_img)
            out.release()
        cv2.destroyAllWindows()
        
        # 删除可能存在的临时文件
        if os.path.exists('temp_frame.jpg'):
            os.remove('temp_frame.jpg')
        
        if save_result:
            print(f"结果视频已保存为: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='测试循迹小车导航系统')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--video', type=str, help='输入视频路径')
    parser.add_argument('--start-time', type=float, default=85, help='视频开始处理的时间点（秒）')
    parser.add_argument('--frame-interval', type=int, default=3, help='视频处理的帧间隔')
    parser.add_argument('--adjust', action='store_true', help='调整HSV阈值')
    parser.add_argument('--save', action='store_true', default=True, help='保存结果图像/视频')
    
    args = parser.parse_args()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"视频不存在: {args.video}")
            return
        process_video(args.video, args.start_time, args.frame_interval, args.save, args.adjust)
        return
    
    if args.image is None:
        print("请提供图像路径或视频路径，例如:")
        print("python test_auto_trace.py --image test_image.jpg")
        print("python test_auto_trace.py --video 1.mp4 --start-time 85")
        return
    
    if not os.path.exists(args.image):
        print(f"图像不存在: {args.image}")
        return
    
    if args.adjust:
        lower, upper = adjust_hsv_thresholds(args.image)
        choice = input("是否更新auto_trace.py中的HSV阈值? (y/n): ")
        if choice.lower() == 'y':
            modify_auto_trace_thresholds(lower, upper)
    
    test_with_image(args.image, args.save)

if __name__ == "__main__":
    main()