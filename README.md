# 循迹小车导航系统

这个项目实现了一个智能循迹小车导航系统，提供两种导航算法：基于颜色阈值分割的区域分析算法和基于边界线检测的导航算法。系统能够分析小车周围的环境，识别轨道和非轨道区域，并计算小车应该行驶的最佳方向。

## 功能特点

### 区域分析算法
- 基于颜色阈值分割识别轨道
- 计算小车最佳行驶方向（-180°到180°）
- 检测小车是否脱轨，并计算返回轨道的最短路径
- 在图像上绘制指示线和箭头，直观显示行驶方向
- 提供交互式工具调整HSV阈值，适应不同的轨道颜色

### 边界线导航算法
- 自动检测轨道左右边界线
- 基于边界线计算最佳行驶方向（0°到360°）
- 根据小车到边界线的距离动态调整行驶方向
- 支持图像和视频的实时处理
- 在图像上绘制边界线和导航方向箭头

## 安装依赖

```bash
pip install opencv-python numpy matplotlib
```

## 使用方法

### 基本使用

#### 区域分析算法
1. 处理单张图片：

```bash
python test_auto_trace.py --image 图片路径.jpg
```

2. 处理视频文件：
```bash
python test_auto_trace.py --video 视频路径.mp4 --start-time 85
```

#### 边界线导航算法
1. 处理单张图片：
```bash
python test_boundary_trace.py --image 图片路径.jpg
```

2. 处理视频文件：
```bash
python test_boundary_trace.py --video 视频路径.mp4 --start-time 85 --frame-interval 3
```

参数说明：
- `--start-time`：视频开始处理的时间点（秒）
- `--frame-interval`：视频处理的帧间隔
- `--save`：保存处理结果（默认开启）

### 调整颜色阈值

如果默认的颜色阈值不能很好地识别轨道，可以使用交互式工具调整HSV阈值：

```bash
python test_auto_trace.py --image 图片路径.jpg --adjust
```

调整滑动条，直到掩码图像中轨道区域被清晰地标识出来。按ESC键退出调整界面，程序会显示最终的HSV阈值，并询问是否更新到主程序中。

## 算法原理

### 区域分析算法
1. **颜色阈值分割**：将图像转换为HSV颜色空间，使用预设的颜色阈值分割出轨道区域
2. **形态学处理**：使用开闭运算减少噪声，获得更清晰的轨道区域
3. **方向分析**：
   - 正常情况：在小车周围360度范围内，每15度检查一次轨道密度，选择密度最高的方向作为最佳行驶方向
   - 脱轨情况：计算小车到最近轨道点的距离和方向，作为返回轨道的路径
4. **可视化**：在图像上绘制指示线和箭头，直观显示行驶方向

### 边界线导航算法
1. **预处理**：
   - 将图像转换为HSV颜色空间
   - 使用颜色阈值分割出轨道区域
   - 应用形态学操作减少噪声
2. **边界检测**：
   - 使用Canny边缘检测器提取边缘
   - 应用霍夫变换检测直线
   - 根据斜率将直线分为左右边界
3. **方向计算**：
   - 计算小车（图像中心）到两条边界线的垂直距离
   - 根据距离比例决定行驶方向
   - 当距离相近时，沿着垂直方向行驶
   - 当距离相差较大时，调整方向以保持在轨道中心
4. **可视化**：
   - 绘制检测到的左右边界线
   - 在小车位置绘制方向箭头
   - 显示最佳行驶方向角度

## 文件说明

- `auto_trace.py`：区域分析算法的主程序，包含图像处理和方向计算的核心算法
- `test_auto_trace.py`：区域分析算法的测试脚本，提供命令行接口和交互式HSV阈值调整工具
- `boundary_trace.py`：边界线导航算法的主程序，包含边界检测和方向计算的核心算法
- `test_boundary_trace.py`：边界线导航算法的测试脚本，支持图像和视频处理

## 注意事项

- 默认的HSV阈值适用于棕色/黄褐色轨道，如果轨道颜色不同，请使用`--adjust`参数调整阈值
- 图像中心点被假设为小车位置，请确保小车在图像中心
- 检测半径和搜索半径可以在代码中调整，以适应不同的场景