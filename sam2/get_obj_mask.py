import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
import cv2
import tkinter as tk
from PIL import ImageTk
import sys
from pathlib import Path
import glob
import shutil

# 导入VideoReader类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from video_process import VideoReader

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nMPS设备支持尚处于初级阶段。SAM 2在CUDA上训练，可能在MPS上给出数值上的差异。"
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "/home/xinyu/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# 保存mask的函数
def save_mask(mask, save_path):
    """
    保存mask为二值图像
    mask: 二值mask (布尔值或0/1)
    save_path: 保存路径
    """
    try:
        # 确保mask是正确的尺寸和类型
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 确保mask是2D数组
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # 转换为uint8
        mask_uint8 = (mask.astype(np.uint8) * 255)
        
        # 使用PIL保存图像
        Image.fromarray(mask_uint8).save(save_path)
    except Exception as e:
        print(f"保存mask失败: {e}")
        try:
            np.save(save_path.replace('.png', '.npy'), mask)
            print(f"已将mask保存为numpy格式: {save_path.replace('.png', '.npy')}")
        except Exception as e2:
            print(f"保存numpy格式也失败: {e2}")

# 修改交互式选点函数，添加帧浏览功能
def select_point_interactive(img_array, window_name="选择点击位置", already_rgb=True, video_path=None, start_frame=0):
    """
    打开图像并让用户通过鼠标点击选择点的位置，支持在视频的多个帧中选择
    
    Args:
        img_array: 图像数组（numpy数组）- 初始帧
        window_name: 窗口名称
        already_rgb: 图像是否已经是RGB格式（而不是BGR）
        video_path: 视频文件路径，如果提供则允许浏览不同帧
        start_frame: 开始的帧索引
        
    Returns:
        (x, y, frame_idx): 用户点击的坐标和帧索引
    """
    # 全局变量存储点击的坐标
    click_point = []
    current_frame_idx = start_frame
    current_frame = img_array.copy()
    
    # 如果提供了视频路径，打开视频文件
    cap = None
    total_frames = 0
    if video_path and os.path.exists(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # 确保开始帧在有效范围内
                current_frame_idx = max(0, min(start_frame, total_frames - 1))
                # 设置到指定帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        except Exception as e:
            print(f"打开视频时出错: {e}")
            if cap:
                cap.release()
            cap = None
    
    # 尝试使用PIL和Tkinter
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
        
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title(f"{window_name} - 帧 {current_frame_idx}")
        
        # 函数：更新显示的帧
        def update_frame(new_frame_array):
            nonlocal tk_img, current_frame
            current_frame = new_frame_array.copy()
            
            # 将numpy数组转换为PIL图像
            if new_frame_array.shape[2] == 3:  # 彩色图像
                if already_rgb:
                    pil_img = Image.fromarray(new_frame_array)
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(new_frame_array, cv2.COLOR_BGR2RGB))
            else:  # 灰度图像
                pil_img = Image.fromarray(new_frame_array)
            
            # 调整大小如果需要
            width, height = pil_img.size
            max_dim = 800
            scale = 1.0
            if width > max_dim or height > max_dim:
                scale = max_dim / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 更新Tkinter图像
            tk_img = ImageTk.PhotoImage(pil_img)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            
            # 更新窗口标题
            root.title(f"{window_name} - 帧 {current_frame_idx}")
            
            # 更新状态文本
            if cap:
                status_label.config(text=f"Frame: {current_frame_idx} / {total_frames - 1}")
        
        # 函数：读取特定帧
        def read_frame(frame_idx):
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb
            return None
        
        # 函数：前进到下一帧
        def next_frame():
            nonlocal current_frame_idx
            if cap and current_frame_idx < total_frames - 1:
                current_frame_idx += 1
                new_frame = read_frame(current_frame_idx)
                if new_frame is not None:
                    update_frame(new_frame)
                    click_point.clear()  # 清除之前的点
        
        # 函数：后退到上一帧
        def prev_frame():
            nonlocal current_frame_idx
            if cap and current_frame_idx > 0:
                current_frame_idx -= 1
                new_frame = read_frame(current_frame_idx)
                if new_frame is not None:
                    update_frame(new_frame)
                    click_point.clear()  # 清除之前的点
        
        # 函数：前进多帧
        def forward_frames():
            nonlocal current_frame_idx
            if cap and current_frame_idx < total_frames - 10:
                current_frame_idx += 10
                new_frame = read_frame(current_frame_idx)
                if new_frame is not None:
                    update_frame(new_frame)
                    click_point.clear()  # 清除之前的点
        
        # 函数：后退多帧
        def backward_frames():
            nonlocal current_frame_idx
            if cap and current_frame_idx >= 10:
                current_frame_idx -= 10
                new_frame = read_frame(current_frame_idx)
                if new_frame is not None:
                    update_frame(new_frame)
                    click_point.clear()  # 清除之前的点
        
        # 函数：跳转到第一帧
        def goto_first_frame():
            nonlocal current_frame_idx
            if cap and current_frame_idx > 0:
                current_frame_idx = 0
                new_frame = read_frame(current_frame_idx)
                if new_frame is not None:
                    update_frame(new_frame)
                    click_point.clear()  # 清除之前的点
        
        # 函数：跳转到最后一帧
        def goto_last_frame():
            nonlocal current_frame_idx
            if cap and current_frame_idx < total_frames - 1:
                current_frame_idx = total_frames - 1
                new_frame = read_frame(current_frame_idx)
                if new_frame is not None:
                    update_frame(new_frame)
                    click_point.clear()  # 清除之前的点
        
        # 准备初始帧
        if current_frame.shape[2] == 3:  # 彩色图像
            if already_rgb:
                pil_img = Image.fromarray(current_frame)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        else:  # 灰度图像
            pil_img = Image.fromarray(current_frame)
        
        # 调整大小如果需要
        width, height = pil_img.size
        max_dim = 800
        scale = 1.0
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Image resized to {new_width}x{new_height}")
        
        # 转换为Tkinter可用格式
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # 创建画布并显示图像
        canvas = tk.Canvas(root, width=tk_img.width(), height=tk_img.height())
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        
        # 添加帧导航控件
        if cap:
            # 创建第一行导航按钮 - 主要导航按钮
            nav_frame = tk.Frame(root)
            nav_frame.pack(pady=5)
            
            # 添加导航按钮
            tk.Button(nav_frame, text="<< -10", command=backward_frames).pack(side=tk.LEFT, padx=5)
            tk.Button(nav_frame, text="< Prev", command=prev_frame).pack(side=tk.LEFT, padx=5)
            tk.Button(nav_frame, text="Next >", command=next_frame).pack(side=tk.LEFT, padx=5)
            tk.Button(nav_frame, text="+10 >>", command=forward_frames).pack(side=tk.LEFT, padx=5)
            
            # 创建第二行导航按钮 - 跳转到首帧和末帧
            extra_nav_frame = tk.Frame(root)
            extra_nav_frame.pack(pady=2)
            
            # 添加跳转按钮
            tk.Button(extra_nav_frame, text="First Frame", command=goto_first_frame,
                      bg="#E3F2FD", fg="black").pack(side=tk.LEFT, padx=10)
            tk.Button(extra_nav_frame, text="Last Frame", command=goto_last_frame,
                      bg="#E3F2FD", fg="black").pack(side=tk.LEFT, padx=10)
            
            # 状态标签
            status_label = tk.Label(root, text=f"Frame: {current_frame_idx} / {total_frames - 1}")
            status_label.pack(pady=2)
        
        # 点击处理函数
        def on_click(event):
            x, y = event.x, event.y
            click_point.clear()  # 确保只保存最新的点
            click_point.append((x, y, current_frame_idx))
            print(f"Selected point: ({x}, {y}) on frame {current_frame_idx}")
            # 绘制绿色圆点标记所选位置
            canvas.create_oval(x-5, y-5, x+5, y+5, fill='green')
        
        # 绑定鼠标点击事件
        canvas.bind("<Button-1>", on_click)
        
        # 添加确认和重置按钮
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        def confirm():
            root.quit()
        
        def reset():
            click_point.clear()
            # 重新显示干净的图像
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            print("Selection reset, please click again")
        
        # 创建确认和重置按钮
        tk.Button(button_frame, text="Confirm", command=confirm, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Reset", command=reset).pack(side=tk.LEFT)
        
        # 添加说明标签
        instruction_label = tk.Label(root, 
                                    text="Click on the image to select a point. Use navigation buttons to browse frames.")
        instruction_label.pack(pady=5)
        
        # 主循环
        root.mainloop()
        
        # 关闭窗口和视频
        try:
            root.destroy()
        except:
            pass
        
        if cap:
            cap.release()
        
        if not click_point:
            print("No point selected, will use default coordinates")
            return None
        
        # 如果图像被缩放，映射回原始坐标
        if scale != 1.0:
            x, y, frame = click_point[-1]
            x = int(x / scale)
            y = int(y / scale)
            print(f"Mapped back to original coordinates: ({x}, {y}) on frame {frame}")
            return (x, y, frame)
        
        return click_point[-1]  # 返回(x, y, frame_idx)
        
    except Exception as e:
        print(f"Tkinter method failed: {e}")
        print("Falling back to default coordinates...")
        if cap:
            cap.release()
        return None

# 检查目录是否包含视频文件
def is_video_dir(dir_path):
    """检查目录是否包含.mkv文件"""
    return any(os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.mkv') and not f.endswith('.depth.mkv') for f in os.listdir(dir_path))

# 获取目录中的所有相机文件
def find_camera_videos(dir_path):
    """
    在目录中查找所有RGB相机视频文件（排除深度视频文件）
    返回字典 {相机名称: 相机文件路径}
    """
    camera_files = {}
    for file in os.listdir(dir_path):
        if file.endswith('.mkv') and not file.endswith('.depth.mkv'):
            # 提取相机名称 (例如 "camera_1")
            camera_name = os.path.splitext(file)[0]
            # 确保不是深度相机
            if not camera_name.endswith('.depth'):
                camera_files[camera_name] = os.path.join(dir_path, file)
    return camera_files

# 查找所有包含视频的任务目录
def find_task_dirs(base_dir):
    """
    查找所有包含视频文件的任务目录
    """
    task_dirs = []
    
    # 如果基础目录本身包含视频文件，将其视为单个任务
    if is_video_dir(base_dir):
        return [base_dir]
    
    # 否则，查找包含视频的子目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if is_video_dir(item_path):
                task_dirs.append(item_path)
            elif "pour" in item.lower():  # 查找可能的pour任务子目录
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and is_video_dir(subitem_path):
                        task_dirs.append(subitem_path)
    
    return task_dirs

# 自定义的视频读取函数 - 只用于RGB视频，避免使用VideoReader
def read_video_frames(video_path, max_frames=None):
    """
    读取视频帧，不依赖VideoReader类
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大读取帧数，None表示读取所有帧
        
    Returns:
        frames: 帧列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return []
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已读取 {frame_count} 帧")
            
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    return frames

def process_video(video_dir, camera_name, source_point=None, target_point=None):
    """
    处理单个RGB视频并生成mask
    
    Args:
        video_dir: 包含视频的目录
        camera_name: 相机名称
        source_point: 源物体的点坐标 (x,y)
        target_point: 目标物体的点坐标 (x,y)
    """
    # 构建MKV文件路径
    video_path = os.path.join(video_dir, f"{camera_name}.mkv")
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 {video_path} 不存在")
        return False
    
    # 创建保存目录
    frames_dir = os.path.join(video_dir, "frames", camera_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # 创建保存mask的目录
    mask_save_dir = os.path.join(frames_dir, "object_masks")
    
    # 如果目录已存在，清除旧的mask文件
    if os.path.exists(mask_save_dir):
        source_dir = os.path.join(mask_save_dir, "source")
        target_dir = os.path.join(mask_save_dir, "target")
        
        # 清除旧的source masks
        if os.path.exists(source_dir):
            for file in os.listdir(source_dir):
                if file.endswith('.png') or file.endswith('.npy'):
                    os.remove(os.path.join(source_dir, file))
        
        # 清除旧的target masks
        if os.path.exists(target_dir):
            for file in os.listdir(target_dir):
                if file.endswith('.png') or file.endswith('.npy'):
                    os.remove(os.path.join(target_dir, file))
    
    # 创建或确保目录存在
    os.makedirs(mask_save_dir, exist_ok=True)
    
    # 创建source和target子目录
    source_mask_dir = os.path.join(mask_save_dir, "source")
    os.makedirs(source_mask_dir, exist_ok=True)
    
    target_mask_dir = os.path.join(mask_save_dir, "target")
    os.makedirs(target_mask_dir, exist_ok=True)
    
    # 直接使用OpenCV读取第一帧用于交互
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return False
    
    ret, first_frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        print(f"错误: 无法读取视频第一帧 {video_path}")
        return False
    
    # 转换BGR到RGB
    first_frame = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    
    # 打印第一帧信息
    print(f"视频第一帧尺寸: {first_frame.shape}")
    
    # 初始化SAM2预测器
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # 创建一个临时目录来存储帧
    temp_frames_dir = os.path.join(frames_dir, "temp_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    # 使用OpenCV逐帧读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return False
    
    frame_paths = []
    frame_idx = 0
    
    print("正在预处理视频帧...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存帧到临时目录
        frame_path = os.path.join(temp_frames_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)  # 直接保存BGR格式
        frame_paths.append(frame_path)
        
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx} 帧")
        
        frame_idx += 1
    
    cap.release()
    print(f"视频共有 {len(frame_paths)} 帧")
    
    # 初始化SAM2状态
    inference_state = predictor.init_state(video_path=temp_frames_dir)
    
    # 创建一个字典来存储不同物体的提示信息
    prompts = {}
    
    # 记录用于source和target的帧索引
    source_frame_idx = 0
    target_frame_idx = 0
    
    # 第一个物体（id=1）
    ann_obj_id = 1
    
    # 如果提供了命令行坐标参数，使用它们；否则进行交互式选择
    if source_point:
        try:
            x, y = source_point
            points = np.array([[x, y]], dtype=np.float32)
            print(f"使用提供的source点: ({x}, {y})")
        except:
            print(f"无法解析source点坐标: {source_point}")
            print("请使用正确的格式，例如: '500,500'")
            shutil.rmtree(temp_frames_dir)
            return False
    else:
        # 交互式选择第一个物体的点，允许浏览不同帧
        print("请为第一个物体(source)选择一个点:")
        try:
            result = select_point_interactive(first_frame, already_rgb=True, 
                                            video_path=video_path, start_frame=0)
            if result:
                x, y, source_frame_idx = result
                points = np.array([[x, y]], dtype=np.float32)
                print(f"在帧 {source_frame_idx} 上选择了点: ({x}, {y})")
            else:
                # 使用默认坐标
                points = np.array([[500, 500]], dtype=np.float32)
                print(f"使用默认点: {points}")
        except Exception as e:
            print(f"选择点时发生错误: {e}")
            # 使用默认坐标
            points = np.array([[500, 500]], dtype=np.float32)
            print(f"使用默认点: {points}")
    
    # 对于标签，`1`表示正点击，`0`表示负点击
    labels = np.array([1], np.int32)
    prompts[ann_obj_id] = (points, labels)
    
    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=source_frame_idx,  # 使用实际选择的帧索引
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
    except Exception as e:
        print(f"添加第一个点时出错: {e}")
        shutil.rmtree(temp_frames_dir)
        return False
    
    # 第二个物体（id=2）
    ann_obj_id = 2
    
    # 如果提供了命令行坐标参数，使用它们；否则进行交互式选择
    if target_point:
        try:
            x, y = target_point
            points = np.array([[x, y]], dtype=np.float32)
            print(f"使用提供的target点: ({x}, {y})")
        except:
            print(f"无法解析target点坐标: {target_point}")
            print("请使用正确的格式，例如: '300,450'")
            shutil.rmtree(temp_frames_dir)
            return False
    else:
        # 交互式选择第二个物体的点，从第一个物体选择的帧开始
        print("请为第二个物体(target)选择一个点:")
        try:
            # 读取source选择的帧，作为target选择的起始帧
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
                ret, source_frame_bgr = cap.read()
                cap.release()
                if ret:
                    source_frame = cv2.cvtColor(source_frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    source_frame = first_frame
            else:
                source_frame = first_frame
            
            result = select_point_interactive(source_frame, already_rgb=True,
                                            video_path=video_path, start_frame=source_frame_idx)
            if result:
                x, y, target_frame_idx = result
                points = np.array([[x, y]], dtype=np.float32)
                print(f"在帧 {target_frame_idx} 上选择了点: ({x}, {y})")
            else:
                # 使用默认坐标
                points = np.array([[300, 450]], dtype=np.float32)
                target_frame_idx = source_frame_idx  # 如果未选择，使用与source相同的帧
                print(f"使用默认点: {points}")
        except Exception as e:
            print(f"选择点时发生错误: {e}")
            # 使用默认坐标
            points = np.array([[300, 450]], dtype=np.float32)
            target_frame_idx = source_frame_idx  # 如果未选择，使用与source相同的帧
            print(f"使用默认点: {points}")
    
    labels = np.array([1], np.int32)
    prompts[ann_obj_id] = (points, labels)
    
    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=target_frame_idx,  # 使用实际选择的帧索引
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
    except Exception as e:
        print(f"添加第二个点时出错: {e}")
        shutil.rmtree(temp_frames_dir)
        return False
    
    # 运行传播并收集结果
    print("开始在视频中传播mask...")
    video_segments = {}
    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if out_frame_idx % 100 == 0:
                print(f"已处理 {out_frame_idx} 帧")
    except Exception as e:
        print(f"视频传播过程中出错: {e}")
        shutil.rmtree(temp_frames_dir)
        return False
    
    # 保存所有帧的所有物体mask，包括可能没有被传播覆盖到的帧
    print("保存所有帧的mask...")
    
    # 获取视频的总帧数
    total_frames = len(frame_paths)
    
    # 创建空mask模板 - 与第一帧大小相同的全零数组
    empty_mask_shape = first_frame.shape[:2]  # 高度和宽度
    empty_mask = np.zeros(empty_mask_shape, dtype=np.uint8)
    
    # 遍历所有可能的帧索引
    for frame_idx in range(total_frames):
        # 检查该帧是否在传播结果中
        if frame_idx in video_segments:
            # 有传播结果的帧，保存实际的mask
            frame_masks = video_segments[frame_idx]
            
            # 保存source物体的mask（如果存在）
            if 1 in frame_masks:
                mask_save_path = os.path.join(source_mask_dir, f"{frame_idx:05d}.png")
                save_mask(frame_masks[1], mask_save_path)
            else:
                # 如果该帧没有source物体的mask，保存空mask
                mask_save_path = os.path.join(source_mask_dir, f"{frame_idx:05d}.png")
                save_mask(empty_mask, mask_save_path)
            
            # 保存target物体的mask（如果存在）
            if 2 in frame_masks:
                mask_save_path = os.path.join(target_mask_dir, f"{frame_idx:05d}.png")
                save_mask(frame_masks[2], mask_save_path)
            else:
                # 如果该帧没有target物体的mask，保存空mask
                mask_save_path = os.path.join(target_mask_dir, f"{frame_idx:05d}.png")
                save_mask(empty_mask, mask_save_path)
        else:
            # 没有传播结果的帧，保存空mask
            mask_save_path = os.path.join(source_mask_dir, f"{frame_idx:05d}.png")
            save_mask(empty_mask, mask_save_path)
            
            mask_save_path = os.path.join(target_mask_dir, f"{frame_idx:05d}.png")
            save_mask(empty_mask, mask_save_path)
        
        # 显示进度
        if frame_idx % 100 == 0:
            print(f"已保存 {frame_idx}/{total_frames} 帧的mask")
    
    # 清理临时文件
    print("清理临时文件...")
    shutil.rmtree(temp_frames_dir)
    
    print(f"处理完成! 所有mask已保存到:")
    print(f"Source masks: {source_mask_dir}")
    print(f"Target masks: {target_mask_dir}")
    return True

# 添加预览视频中间帧的函数
def preview_video_frame(video_path, frame_index=None, window_name="Video Preview"):
    """
    Preview a specific frame from the video (middle frame by default)
    
    Args:
        video_path: Path to the video file
        frame_index: Index of the frame to preview, None means middle frame
        window_name: Window title
    
    Returns:
        True when preview window is closed
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Error: Cannot get frame count for {video_path}")
        cap.release()
        return False
    
    # Determine frame to preview
    if frame_index is None:
        # Use middle frame
        frame_index = total_frames // 2
    
    # Adjust if out of range
    frame_index = max(0, min(frame_index, total_frames - 1))
    
    # Move to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Cannot read frame {frame_index}")
        return False
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Use Tkinter to display the image
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
        
        # Create Tkinter window
        root = tk.Tk()
        root.title(f"{window_name} - Frame {frame_index} of {total_frames}")
        
        # Convert numpy array to PIL image
        pil_img = Image.fromarray(frame_rgb)
        
        # Resize if needed
        width, height = pil_img.size
        max_dim = 800
        scale = 1.0
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter compatible format
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Create canvas and display image
        canvas = tk.Canvas(root, width=tk_img.width(), height=tk_img.height())
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        
        # Show instructions
        info_text = "Please observe the image to identify source and target objects"
        label = tk.Label(root, text=info_text, font=("Arial", 12))
        label.pack(pady=5)
        
        # Add close button
        def close_preview():
            root.quit()
        
        close_button = tk.Button(root, text="Continue to Selection", command=close_preview, 
                              font=("Arial", 12), bg="#4CAF50", fg="white", 
                              padx=20, pady=10)
        close_button.pack(pady=10)
        
        # Main loop
        root.mainloop()
        
        # Close window
        try:
            root.destroy()
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"Preview failed: {e}")
        return False

# 添加检查任务是否已处理过的函数
def is_task_processed(video_dir, camera_name):
    """
    检查特定摄像机视频是否已经被处理过，判断依据为:
    1. 是否存在source和target文件夹
    2. 这些文件夹中的mask数量是否与视频帧数相匹配
    
    Args:
        video_dir: 任务目录
        camera_name: 相机名称
        
    Returns:
        (bool, str): 是否已处理和原因描述
    """
    video_path = os.path.join(video_dir, f"{camera_name}.mkv")
    if not os.path.exists(video_path):
        return False, f"视频文件 {video_path} 不存在"
    
    # 获取视频的总帧数
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"无法打开视频 {video_path}"
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 检查目录结构
    frames_dir = os.path.join(video_dir, "frames", camera_name)
    mask_dir = os.path.join(frames_dir, "object_masks")
    source_dir = os.path.join(mask_dir, "source")
    target_dir = os.path.join(mask_dir, "target")
    
    # 检查所有必要的目录是否存在
    if not os.path.exists(source_dir) or not os.path.exists(target_dir):
        return False, "缺少source或target目录"
    
    # 获取mask文件数量
    source_masks = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    target_masks = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    
    # 检查mask文件数量是否与帧数一致
    if len(source_masks) != total_frames:
        return False, f"source mask数量 ({len(source_masks)}) 与视频帧数 ({total_frames}) 不一致"
    
    if len(target_masks) != total_frames:
        return False, f"target mask数量 ({len(target_masks)}) 与视频帧数 ({total_frames}) 不一致"
    
    # 一切正常，视频已经处理过
    return True, f"已有完整的mask文件（{total_frames}帧）"

# 修改main函数，添加处理状态检查
def main():
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="从MKV视频直接提取mask")
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='包含MKV视频的目录路径或上级目录')
    parser.add_argument('--camera', type=str, default=None, 
                       help='相机名称，如 "camera_2"，不指定则处理所有相机')
    parser.add_argument('--source_point', type=str, default=None, 
                       help='第一个物体点坐标，格式为 "x,y"')
    parser.add_argument('--target_point', type=str, default=None, 
                       help='第二个物体点坐标，格式为 "x,y"')
    parser.add_argument('--force', action='store_true',
                       help='强制重新处理已处理过的视频')
    
    args = parser.parse_args()
    
    # 视频基础目录
    base_dir = args.video_dir
    force_process = args.force
    
    # 如果提供了点坐标，解析它们
    source_coords = None
    if args.source_point:
        try:
            x, y = map(int, args.source_point.split(','))
            source_coords = (x, y)
        except:
            print(f"无法解析source点坐标: {args.source_point}")
            print("请使用正确的格式，例如: '500,500'")
            return
    
    target_coords = None
    if args.target_point:
        try:
            x, y = map(int, args.target_point.split(','))
            target_coords = (x, y)
        except:
            print(f"无法解析target点坐标: {args.target_point}")
            print("请使用正确的格式，例如: '300,450'")
            return
    
    # 查找要处理的任务目录
    task_dirs = find_task_dirs(base_dir)
    
    if not task_dirs:
        print(f"错误: 在 {base_dir} 中没有找到包含视频的任务目录")
        return
    
    print(f"找到 {len(task_dirs)} 个任务目录:")
    for task_dir in task_dirs:
        print(f"  - {task_dir}")
    
    # 统计信息
    total_tasks = 0
    processed_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    
    # 遍历每个任务目录
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        print(f"\n===== 开始处理任务: {task_name} =====")
        
        # 如果指定了相机，则只处理该相机
        if args.camera:
            camera_list = [args.camera]
        else:
            # 否则找出所有相机
            camera_files = find_camera_videos(task_dir)
            camera_list = list(camera_files.keys())
            camera_list.sort()  # 排序以保持一致的处理顺序
        
        print(f"将处理以下相机: {', '.join(camera_list)}")
        
        # 处理每个相机
        for camera_name in camera_list:
            total_tasks += 1
            print(f"\n----- 处理 {task_name} 的 {camera_name} -----")
            
            # 检查是否已处理过
            is_processed, reason = is_task_processed(task_dir, camera_name)
            
            if is_processed and not force_process:
                print(f"跳过已处理的视频: {reason}")
                skipped_tasks += 1
                continue
            elif is_processed and force_process:
                print(f"强制重新处理视频: {reason}")
            else:
                print(f"处理视频: {reason}")
            
            success = process_video(task_dir, camera_name, source_coords, target_coords)
            if success:
                print(f"{camera_name} 处理成功!")
                processed_tasks += 1
            else:
                print(f"{camera_name} 处理失败!")
                failed_tasks += 1
        
        print(f"===== 完成处理任务: {task_name} =====\n")
    
    # 打印统计信息
    print("\n===== 处理统计 =====")
    print(f"总任务数: {total_tasks}")
    print(f"成功处理: {processed_tasks}")
    print(f"跳过处理: {skipped_tasks}")
    print(f"处理失败: {failed_tasks}")
    print("所有任务处理完成!")

if __name__ == "__main__":
    main() 

'''
python get_obj_mask.py --video_dir /home/xinyu/mocap_poem_ws/data/record_new
'''
