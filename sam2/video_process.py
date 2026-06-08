import ffmpeg
import os
import numpy as np
import cv2
import argparse
from pathlib import Path

class VideoReader:
    def __init__(self, data_dir):
        """
        初始化视频读取器
        
        参数:
            data_dir: 包含.mkv视频文件的目录路径
        """
        self.stream_filedir = data_dir
        self.color_proc = {}
        # 视频分辨率
        self.VIDEO_SHAPE = (1280, 720)  # (宽度, 高度)
    
    def start_stream(self, color_view):
        """
        启动视频流处理
        
        参数:
            color_view: RGB视频文件名（不含扩展名）
        """
        # 初始化RGB视频流
        self.color_proc[color_view] = (
            ffmpeg.input(os.path.join(self.stream_filedir, f"{color_view}.mkv"))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .global_args("-loglevel", "error")
            .global_args("-y")
            .run_async(pipe_stdout=True, pipe_stdin=True)
        )
        
        print(f"已成功启动视频流：{color_view}.mkv")
    
    def read_frame(self, color_view):
        """
        读取一帧RGB数据
        
        参数:
            color_view: RGB视频文件名（不含扩展名）
        
        返回:
            rgb_img: RGB图像数组
        """
        # 读取RGB帧
        try:
            rgb_buf = self.color_proc[color_view].stdout.read(self.VIDEO_SHAPE[0] * self.VIDEO_SHAPE[1] * 3)
            if len(rgb_buf) == 0:
                return None  # 视频结束
            
            rgb_img = np.frombuffer(rgb_buf, dtype=np.uint8).reshape((self.VIDEO_SHAPE[1], self.VIDEO_SHAPE[0], 3))
            return rgb_img
        except Exception as e:
            print(f"读取帧时出错: {e}")
            return None
    
    def close(self):
        """关闭所有视频流"""
        try:
            for proc in self.color_proc.values():
                proc.terminate()
            print("已关闭所有视频流")
        except Exception as e:
            print(f"关闭视频流时出错: {e}")

def save_frames_as_images(data_dir, color_filename, output_dir, camera_name=None, skip_existing=False):
    """
    将视频的每一帧保存为图像文件
    
    参数:
        data_dir: 视频文件所在目录
        color_filename: RGB视频文件名（不含扩展名）
        output_dir: 输出图像的基础目录
        camera_name: 相机名称，用于创建子目录
        skip_existing: 是否跳过已存在的帧
    """
    # 如果提供了相机名称，创建对应的子目录
    if camera_name:
        camera_output_dir = os.path.join(output_dir, camera_name)
    else:
        camera_output_dir = output_dir
    
    # 创建输出目录结构
    rgb_output_dir = os.path.join(camera_output_dir, 'rgb')
    os.makedirs(rgb_output_dir, exist_ok=True)
    
    # 检查是否已有处理过的帧
    if skip_existing:
        # 获取已存在的最大帧索引
        existing_frames = [int(f.split('.')[0]) for f in os.listdir(rgb_output_dir) if f.endswith('.png')]
        if existing_frames:
            max_existing_frame = max(existing_frames)
            print(f"在 {rgb_output_dir} 中发现 {len(existing_frames)} 个已处理帧，将从第 {max_existing_frame + 1} 帧继续")
            start_frame = max_existing_frame + 1
        else:
            start_frame = 0
    else:
        start_frame = 0
    
    # 如果要跳过已处理的帧，并且已有处理结果
    if skip_existing and start_frame > 0:
        # 我们需要快进视频到指定帧
        reader = VideoReader(data_dir)
        reader.start_stream(color_filename)
        
        print(f"快进视频到第 {start_frame} 帧...")
        # 快进到指定帧
        for _ in range(start_frame):
            rgb_img = reader.read_frame(color_filename)
            if rgb_img is None:
                print(f"视频快进过程中结束，没有新帧可提取")
                reader.close()
                return 0
    else:
        # 正常启动读取器
        reader = VideoReader(data_dir)
        reader.start_stream(color_filename)
    
    print(f"开始提取 {color_filename} 的帧...")
    try:
        frame_idx = start_frame
        processed_count = 0
        
        while True:
            rgb_img = reader.read_frame(color_filename)
            if rgb_img is None:
                print(f"{color_filename} 视频处理结束")
                break
                
            # 保存RGB图像 - 使用新的命名格式 (00000.png 而不是 frame_000000.png)
            rgb_filename = os.path.join(rgb_output_dir, f'{frame_idx:05d}.png')
            cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            # 每20帧显示进度
            if frame_idx % 20 == 0:
                print(f"{color_filename}: 已处理 {frame_idx} 帧")
                
            frame_idx += 1
            processed_count += 1
            
    except Exception as e:
        print(f"处理 {color_filename} 视频时出错: {e}")
    finally:
        reader.close()
        if start_frame > 0:
            print(f"{color_filename}: 跳过了 {start_frame} 帧，新处理了 {processed_count} 帧，总共有 {frame_idx} 帧图像在 {camera_output_dir}")
        else:
            print(f"{color_filename}: 总共保存了 {frame_idx} 帧图像到 {camera_output_dir}")
    
    return processed_count

def process_all_cameras(data_dir, output_dir, skip_existing=False):
    """
    处理所有相机视角的视频
    
    参数:
        data_dir: 数据目录路径
        output_dir: 输出目录路径
        skip_existing: 是否跳过已处理的帧
    """
    total_frames = 0
    
    # 获取目录下所有mkv文件
    mkv_files = [f for f in os.listdir(data_dir) if f.endswith('.mkv')]
    
    # 筛选出RGB视频文件（不含.depth的文件）
    rgb_files = [f for f in mkv_files if '.depth.' not in f and not f.endswith('.depth.mkv')]
    
    if not rgb_files:
        print(f"警告: 在 {data_dir} 中没有找到RGB视频文件")
        return total_frames
    
    print(f"找到以下RGB视频文件: {rgb_files}")
    
    for rgb_file in rgb_files:
        # 提取基本文件名（不含扩展名）
        color_filename = os.path.splitext(rgb_file)[0]
        
        print(f"\n开始处理 {color_filename} 视角...")
        frames = save_frames_as_images(data_dir, color_filename, output_dir, color_filename, skip_existing)
        total_frames += frames
    
    print(f"\n所有视角处理完成，总共新提取了 {total_frames} 帧图像")

def main():
    parser = argparse.ArgumentParser(description='将MKV格式的RGB视频转换为图像序列')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--color', type=str, default=None, help='RGB视频文件名（不含扩展名），不指定则处理所有相机')
    parser.add_argument('--output_dir', type=str, default=None, help='输出图像的目录')
    parser.add_argument('--process_all', action='store_true', help='处理数据目录下的所有子文件夹')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已处理的帧')
    
    args = parser.parse_args()
    
    # 确保数据目录存在
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"错误：数据目录 '{args.data_dir}' 不存在")
        return
    
    if args.process_all:
        # 处理所有子文件夹
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not subdirs:
            print(f"警告：在 '{args.data_dir}' 中没有找到子文件夹")
            return
            
        print(f"找到 {len(subdirs)} 个子文件夹，将依次处理...")
        for subdir in subdirs:
            print(f"\n========== 开始处理文件夹: {subdir.name} ==========")
            # 对每个子文件夹，在其中创建frames目录
            output_dir = os.path.join(subdir, 'frames')
            process_all_cameras(str(subdir), output_dir, args.skip_existing)
            print(f"========== 完成处理文件夹: {subdir.name} ==========\n")
    else:
        # 确定输出目录
        if args.output_dir is None:
            # 如果没有指定输出目录，使用输入目录下的frames子目录
            output_dir = os.path.join(args.data_dir, 'frames')
        else:
            output_dir = args.output_dir
        
        # 处理所有相机视角还是单个视角
        if args.color is None:
            # 处理所有相机视角
            print(f"将处理所有相机视角的视频")
            print(f"图像将保存到：{output_dir}")
            process_all_cameras(args.data_dir, output_dir, args.skip_existing)
        else:
            # 处理单个相机视角
            # 确保视频文件存在
            rgb_file = data_path / f"{args.color}.mkv"
            
            if not rgb_file.exists():
                print(f"错误：RGB视频文件 '{rgb_file}' 不存在")
                return
                
            print(f"正在将视频转换为图像：{rgb_file}")
            print(f"图像将保存到：{output_dir}/{args.color}")
            save_frames_as_images(args.data_dir, args.color, output_dir, args.color, args.skip_existing)

if __name__ == "__main__":
    main()

'''
处理一个episode：
python video_process.py --data_dir /mnt/homes/yifu-ldap/lab/im2flow2act/Hyper_Channel/data/pour__2025_0319_1613_20

处理所有episode：加--process_all
python video_process.py --data_dir /mnt/homes/yifu-ldap/lab/im2flow2act/Hyper_Channel/data --process_all

跳过已处理的帧：加--skip_existing
python video_process.py --data_dir /mnt/homes/yifu-ldap/lab/im2flow2act/Hyper_Channel/data --process_all --skip_existing
'''