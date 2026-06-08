import os
import cv2
import argparse
from pathlib import Path

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

# 查找所有包含视频的pour任务目录
def find_pour_task_dirs(base_dir):
    """
    查找所有包含视频文件的pour任务目录
    """
    pour_task_dirs = []
    
    # 遍历基础目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 直接检查是否为pour任务目录
            if "pour" in item.lower() and is_video_dir(item_path):
                pour_task_dirs.append(item_path)
            # 检查子目录
            elif "pour" in item.lower():
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and is_video_dir(subitem_path):
                        pour_task_dirs.append(subitem_path)
    
    return pour_task_dirs

# 检查任务是否已处理过的函数
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
        return False, f"视频文件不存在"
    
    # 获取视频的总帧数
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"无法打开视频"
    
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
        return False, f"source mask数量不匹配"
    
    if len(target_masks) != total_frames:
        return False, f"target mask数量不匹配"
    
    # 一切正常，视频已经处理过
    return True, f"已有完整mask"

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="统计已标注mask的pour任务数量")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='数据根目录路径')
    
    args = parser.parse_args()
    base_dir = args.data_dir
    
    # 查找所有pour任务目录
    pour_task_dirs = find_pour_task_dirs(base_dir)
    
    if not pour_task_dirs:
        print(f"错误: 在 {base_dir} 中没有找到pour任务目录")
        return
    
    print(f"正在统计 {len(pour_task_dirs)} 个pour任务的标注情况...")
    
    # 统计结果
    stats = []
    task_stats = {}  # 记录每个任务的标注状态
    
    # 遍历每个pour任务目录
    for task_dir in pour_task_dirs:
        task_name = os.path.basename(task_dir)
        
        # 找出该任务的所有相机
        camera_files = find_camera_videos(task_dir)
        camera_list = list(camera_files.keys())
        camera_list.sort()  # 排序以保持一致的处理顺序
        
        task_cameras_total = len(camera_list)
        task_cameras_processed = 0
        
        # 初始化任务统计信息
        task_stats[task_dir] = {
            "任务名称": task_name,
            "总相机数": task_cameras_total,
            "已标注相机数": 0,
            "相机列表": {}
        }
        
        # 检查每个相机
        for camera_name in camera_list:
            # 检查是否已处理过
            is_processed, reason = is_task_processed(task_dir, camera_name)
            
            # 更新任务统计信息
            task_stats[task_dir]["相机列表"][camera_name] = {
                "是否已标注": is_processed,
                "备注": reason
            }
            
            if is_processed:
                task_cameras_processed += 1
                task_stats[task_dir]["已标注相机数"] += 1
            
            # 记录统计信息
            stats.append({
                "任务目录": task_dir,
                "任务名称": task_name,
                "相机名称": camera_name,
                "是否已标注": is_processed,
                "备注": reason
            })
    
    # 统计总数
    total_tasks = len(pour_task_dirs)
    total_cameras = len(stats)
    processed_cameras = sum(1 for item in stats if item["是否已标注"])
    
    # 计算每个任务的标注完成度
    fully_processed_tasks = 0
    partially_processed_tasks = 0
    not_processed_tasks = 0
    
    for task_dir, task_data in task_stats.items():
        if task_data["已标注相机数"] == task_data["总相机数"]:
            fully_processed_tasks += 1
        elif task_data["已标注相机数"] > 0:
            partially_processed_tasks += 1
        else:
            not_processed_tasks += 1
    
    # 打印统计结果
    print("\n===== pour任务标注统计结果 =====")
    print(f"总pour任务数: {total_tasks}")
    print(f"已完全标注的任务数: {fully_processed_tasks}")
    print(f"部分标注的任务数: {partially_processed_tasks}")
    print(f"未标注的任务数: {not_processed_tasks}")
    print(f"任务完成率: {fully_processed_tasks}/{total_tasks} ({fully_processed_tasks/total_tasks*100:.2f}%)")
    print(f"总相机数: {total_cameras}")
    print(f"已标注的相机数: {processed_cameras}")
    print(f"相机标注率: {processed_cameras}/{total_cameras} ({processed_cameras/total_cameras*100:.2f}%)")

if __name__ == "__main__":
    main()

'''
使用方法示例:
python count_pour_masks.py --data_dir /mnt/homes/yifu-ldap/lab/im2flow2act/Hyper_Channel/data
''' 