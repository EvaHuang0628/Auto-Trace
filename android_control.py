import pyautogui
import time
import sys

"""
安卓手机屏幕控制脚本
通过pyautogui库实现对通过scrcpy连接到电脑的安卓手机的控制
包含基本的点击、拖动等操作功能

使用前请确保：
1. 已安装pyautogui库 (pip install pyautogui)
2. 已通过scrcpy成功连接安卓手机到电脑
3. scrcpy窗口处于可见状态
"""

# 设置pyautogui的安全措施，将鼠标移动到屏幕左上角将中断程序
pyautogui.FAILSAFE = True

# 设置操作间隔时间（秒），防止操作过快
pyautogui.PAUSE = 0.5


def get_screen_info():
    """
    获取屏幕分辨率信息
    """
    width, height = pyautogui.size()
    print(f"屏幕分辨率: {width}x{height}")
    return width, height


def get_mouse_position():
    """
    获取当前鼠标位置
    用于调试和定位坐标
    """
    x, y = pyautogui.position()
    print(f"当前鼠标位置: ({x}, {y})")
    return x, y


def click_at(x, y, clicks=1, interval=0.25):
    """
    在指定位置点击
    
    参数:
        x, y: 点击坐标
        clicks: 点击次数，默认为1
        interval: 多次点击时的间隔时间，默认为0.25秒
    """
    try:
        pyautogui.click(x=x, y=y, clicks=clicks, interval=interval)
        print(f"已在位置 ({x}, {y}) 点击 {clicks} 次")
        return True
    except Exception as e:
        print(f"点击操作失败: {e}")
        return False


def long_press(x, y, duration=1.0):
    """
    在指定位置长按
    
    参数:
        x, y: 长按坐标
        duration: 长按时间（秒），默认为1秒
    """
    try:
        pyautogui.mouseDown(x=x, y=y)
        time.sleep(duration)
        pyautogui.mouseUp()
        print(f"已在位置 ({x}, {y}) 长按 {duration} 秒")
        return True
    except Exception as e:
        print(f"长按操作失败: {e}")
        # 确保鼠标释放
        pyautogui.mouseUp()
        return False


def drag(start_x, start_y, end_x, end_y, duration=0.5):
    """
    从起始位置拖动到目标位置
    
    参数:
        start_x, start_y: 起始坐标
        end_x, end_y: 目标坐标
        duration: 拖动持续时间（秒），默认为0.5秒
    """
    # try:
    pyautogui.moveTo(start_x, start_y)
    pyautogui.dragTo(end_x, end_y, duration=duration)
    print(f"已从 ({start_x}, {start_y}) 拖动到 ({end_x}, {end_y})")
    return True
    # except Exception as e:
    #     print(f"拖动操作失败: {e}")
    #     return False


def swipe(start_x, start_y, end_x, end_y, duration=0.3):
    """
    滑动操作（与拖动类似，但通常用于模拟手指滑动）
    
    参数:
        start_x, start_y: 起始坐标
        end_x, end_y: 目标坐标
        duration: 滑动持续时间（秒），默认为0.3秒
    """
    return drag(start_x, start_y, end_x, end_y, duration)


def scroll(x, y, clicks=10, direction="down"):
    """
    在指定位置滚动
    
    参数:
        x, y: 滚动位置的坐标
        clicks: 滚动的数量，正数向下滚动，负数向上滚动
        direction: 滚动方向，"up"或"down"，默认为"down"
    """
    try:
        pyautogui.moveTo(x, y)
        if direction.lower() == "up":
            pyautogui.scroll(abs(clicks))  # 向上滚动为正数
        else:
            pyautogui.scroll(-abs(clicks))  # 向下滚动为负数
        print(f"已在位置 ({x}, {y}) 向{direction}滚动 {abs(clicks)} 次")
        return True
    except Exception as e:
        print(f"滚动操作失败: {e}")
        return False


def double_click(x, y):
    """
    在指定位置双击
    
    参数:
        x, y: 双击坐标
    """
    return click_at(x, y, clicks=2)


def wait(seconds):
    """
    等待指定的秒数
    
    参数:
        seconds: 等待时间（秒）
    """
    print(f"等待 {seconds} 秒...")
    time.sleep(seconds)


def focus_scrcpy_window(window_title="MI 6X"):
    """
    尝试将焦点切换到scrcpy窗口
    注意：这个功能在某些系统上可能不可靠
    
    参数:
        window_title: scrcpy窗口的标题，默认为"scrcpy"
    """
    try:
        # 尝试查找并激活窗口
        window = pyautogui.getWindowsWithTitle(window_title)
        if window:
            window[0].activate()
            print(f"已切换到 {window_title} 窗口")
            # 给窗口一点时间来获取焦点
            time.sleep(0.5)
            return True
        else:
            print(f"未找到标题为 {window_title} 的窗口")
            return False
    except Exception as e:
        print(f"切换窗口失败: {e}")
        return False


def demo():
    """
    演示基本功能的示例函数
    """
    print("=== 安卓屏幕控制演示 ===")
    
    # 获取屏幕信息
    screen_width, screen_height = get_screen_info()
    
    # 尝试切换到scrcpy窗口
    focus_scrcpy_window()
    
    # 等待用户准备
    print("请确保scrcpy窗口可见并处于活动状态")
    print("演示将在3秒后开始...")
    wait(3)
    
    # 获取当前鼠标位置（用于调试）
    current_x, current_y = get_mouse_position()
    
    # 执行一些基本操作示例
    # 注意：以下坐标需要根据实际情况调整
    
    # 示例：点击屏幕中心
    # center_x = screen_width // 2
    # center_y = screen_height // 2

    # 轮盘位置，用python android_control.py获取
    center_x = 150
    center_y = 673
    print("点击屏幕中心")
    click_at(center_x, center_y)
    wait(1)
    
    # 示例：从屏幕中心向上滑动
    print("执行向上滑动")
    swipe(center_x, center_y, center_x, center_y - 200)
    wait(1)
    
    # 示例：从屏幕中心向下滑动
    print("执行向下滑动")
    swipe(center_x, center_y, center_x, center_y + 200)
    wait(1)
    
    print("演示完成")


if __name__ == "__main__":
    print("安卓手机屏幕控制工具已启动")
    print("提示: 将鼠标移动到屏幕左上角可以紧急中断程序")
    
    # 如果直接运行此脚本，执行演示
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        print("您可以导入此模块使用其功能，或使用 --demo 参数运行演示")
        print("例如: python android_control.py --demo")
        
        # 显示当前鼠标位置，帮助用户确定坐标
        try:
            print("\n按Ctrl+C退出坐标显示")
            print("移动鼠标到scrcpy窗口中需要操作的位置以获取坐标...")
            while True:
                x, y = get_mouse_position()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n已退出坐标显示")