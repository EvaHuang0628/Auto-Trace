import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from boundary_trace import process_image_boundary

def test_with_image(image_path, save_result=False):
    """
    使用指定图像测试边界线导航算法。
    
    处理步骤：
    1. 调用process_image_boundary处理输入图像
    2. 使用matplotlib显示处理结果
    3. 可选保存结果图像
    
    参数：
        image_path: str，输入图像的路径
        save_result: bool，是否保存处理结果，默认为True
    
    输出：
        - 显示带有边界线和方向指示的结果图像
        - 打印最佳行驶方向
        - 如果save_result为True，保存结果图像
    """
    # 处理图像
    result_img, best_angle = process_image_boundary(image_path)
    
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
            result_path = f'result_boundary_{base_name}.jpg'
            cv2.imwrite(result_path, result_img)
            print(f"最佳行驶方向: {best_angle}度")
            print(f"结果已保存为 '{result_path}'")
    else:
        print("处理失败")

def process_video(video_path, start_time_sec=85, frame_interval=3, save_result=False):
    """
    处理视频文件，从指定时间点开始，每隔指定帧数提取一帧进行处理。
    
    处理步骤：
    1. 打开视频文件并获取基本信息（FPS、分辨率等）
    2. 从指定时间点开始处理
    3. 每隔指定帧数提取一帧进行边界线检测
    4. 实时显示处理结果
    5. 可选保存处理后的视频
    
    参数：
        video_path: str，输入视频的路径
        start_time_sec: float，开始处理的时间点（秒），默认为85秒
        frame_interval: int，处理帧的间隔，默认为3帧
        save_result: bool，是否保存处理结果视频，默认为False
    
    输出：
        - 实时显示处理结果
        - 打印每帧的时间和方向信息
        - 如果save_result为True，保存处理后的视频
    
    注意：
        - 按'q'键可以退出处理
        - 处理过程中会创建临时文件，程序会在结束时自动清理
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
        output_path = os.path.splitext(video_path)[0] + '_boundary_result.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps/frame_interval, (width, height))
    
    frame_idx = 0
    
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
                
                # 处理图像
                result_img, best_angle = process_image_boundary(temp_img_path)
                
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
            out.release()
        cv2.destroyAllWindows()
        
        # 删除可能存在的临时文件
        if os.path.exists('temp_frame.jpg'):
            os.remove('temp_frame.jpg')
        
        if save_result:
            print(f"结果视频已保存为: {output_path}")

def main():
    """
    主函数：解析命令行参数并执行相应的处理流程。
    
    支持两种处理模式：
    1. 图像处理模式：处理单张图像
    2. 视频处理模式：处理视频文件
    
    使用示例：
        处理图像：python test_boundary_trace.py --image test_image.jpg
        处理视频：python test_boundary_trace.py --video 1.mp4 --start-time 85
    """
    parser = argparse.ArgumentParser(description='测试边界线导航算法')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--video', type=str, help='输入视频路径')
    parser.add_argument('--start-time', type=float, default=85, help='视频开始处理的时间点（秒）')
    parser.add_argument('--frame-interval', type=int, default=3, help='视频处理的帧间隔')
    parser.add_argument('--save', action='store_true', default=True, help='保存结果图像/视频')
    
    args = parser.parse_args()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"视频不存在: {args.video}")
            return
        process_video(args.video, args.start_time, args.frame_interval, args.save)
        return
    
    if args.image is None:
        print("请提供图像路径或视频路径，例如:")
        print("python test_boundary_trace.py --image test_image.jpg")
        print("python test_boundary_trace.py --video 1.mp4 --start-time 85")
        return
    
    if not os.path.exists(args.image):
        print(f"图像不存在: {args.image}")
        return
    
    test_with_image(args.image, args.save)

if __name__ == "__main__":
    main()