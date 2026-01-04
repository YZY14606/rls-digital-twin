import numpy as np
from pathlib import Path
import os
from transforms3d.euler import quat2euler,euler2quat
import torch
import math
import matplotlib.pyplot as plt
import pandas as pd


def calculate_theta_error(pushed_pose,future_pose):
    Fall =  False
    # 计算旋转角度的误差
    #计算物体的欧拉角(绕z轴)
    euler_object = quat2euler(pushed_pose[3:7])[2]
    target_euler = torch.tensor(quat2euler(future_pose[3:7]))
    #转换为正值
    if euler_object < 0:
        euler_object = math.pi*2 + euler_object
    if target_euler[2] <0:
        target_euler[2] = math.pi*2 + target_euler[2]
    # 计算差值
    theta_diff = torch.abs(euler_object - target_euler[2])
    # 将差值转换到0-pi之间
    processed_diff = min(theta_diff, 2 * np.pi - theta_diff)


    euler_now_np = [abs(x) for x in quat2euler(pushed_pose[3:7])]
    if euler_now_np[0] > np.pi/6  or euler_now_np[1] > np.pi/6:
        Fall = True

    return np.array(processed_diff),Fall


def calculate_error(all_pose):
    """"Input: 
        all_pose: 物体被push后的pose与目标单步pose.
        all_pose[0]: 被push后的pose.
        all_pose[1]: 目标的单步pose;

        Output: pose_error, Shape(2).
    """
    pushed_pose = all_pose[0]
    future_pose = all_pose[1]
    # 计算xy平面上的距离差距
    distance = np.linalg.norm(pushed_pose[:2] - future_pose[:2])
    # 计算角度差距
    theta_diff, Fall = calculate_theta_error(pushed_pose,future_pose)
    # 进行结果的合并
    pose_error = np.array([distance, theta_diff])

    return pose_error ,Fall


def calculate_save_pose_error(data_dir):
    # 进行traj_i路径的读取
    pose_dir = Path(data_dir)
    traj_paths = list(pose_dir.glob("**/traj_*/")) 

    # 对于每个traj，计算pose误差
    for traj in traj_paths:
        # npy文件的路径
        record_npy_path = os.path.join(traj,"1/two_pose.npy")
        # 加载npy文件
        all_pose = np.load(record_npy_path)
        # 计算两个pose的误差
        pose_error,Fall = calculate_error(all_pose)
        if not Fall:
            # 保存路径的创建
            save_path = os.path.join(traj,"pose_error.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 数据的保存
            np.save(save_path,pose_error)

def visulize_error_distance_feature(data_dir):
    root_dir = Path(data_dir)
    # 使用 pathlib 的 rglob 方法查找所有 .npy 文件
    traj_paths = list(root_dir.rglob('**/pose_error.npy'))
    paths = [str(file) for file in traj_paths]

    # 初始化列表以存储第一个元素的值
    first_elements = []

    # 遍历所有找到的 .npy 文件
    for file_path in paths:
        try:
            data = np.load(file_path)
            first_element = data.flat[0]  # 提取第一个元素
            first_elements.append(first_element)
        except Exception as e:
            print(f"无法加载文件 {file_path}: {e}")

    # 将列表转换为 NumPy 数组
    first_elements = np.array(first_elements)

    # 计算均值和方差
    mean_value = np.mean(first_elements)
    std_dev = np.std(first_elements)

    # 输出统计结果
    print(f"Distance data 提取了 {len(first_elements)} 个值")
    print(f"Distance data 均值: {mean_value:.4f}")
    print(f"Distance data 方差: {std_dev:.4f}")

    # 使用 DataFrame 的 plot 方法，默认以索引为横坐标
    df = pd.DataFrame({'data': first_elements})
    df.plot(y='data', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value of pose error')
    plt.title('All the pose error data of distance')
    plt.grid(True)
    plt.show()


def visulize_error_theta_feature(data_dir):
    root_dir = Path(data_dir)
    # 使用 pathlib 的 rglob 方法查找所有 .npy 文件
    traj_paths = list(root_dir.rglob('**/pose_error.npy'))
    paths = [str(file) for file in traj_paths]

    # 初始化列表以存储第一个元素的值
    first_elements = []

    # 遍历所有找到的 .npy 文件
    for file_path in paths:
        try:
            data = np.load(file_path)
            first_element = data.flat[1]  # 提取第一个元素
            first_elements.append(first_element)
        except Exception as e:
            print(f"无法加载文件 {file_path}: {e}")

    # 将列表转换为 NumPy 数组
    first_elements = np.array(first_elements)

    # 计算均值和方差
    mean_value = np.mean(first_elements)
    std_dev = np.std(first_elements)

    # 输出统计结果
    print(f"Theta diff 提取了 {len(first_elements)} 个值")
    print(f"Theta diff 均值: {mean_value:.4f}")
    print(f"Theta diff 方差: {std_dev:.4f}")

    # 使用 DataFrame 的 plot 方法，默认以索引为横坐标
    df = pd.DataFrame({'data': first_elements})
    df.plot(y='data', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value of pose error theta diff')
    plt.title('All the pose error data of theta diff')
    plt.grid(True)
    plt.show()


def delete_all_pose_error(data_dir):
    root_dir = Path(data_dir)
    # 使用 pathlib 的 rglob 方法查找所有 .npy 文件
    traj_paths = list(root_dir.rglob('**/pose_error.npy'))
    paths = [str(file) for file in traj_paths]

    print(len(paths))
    for path in paths:
        os.remove(path)


def main():
    data_dir = "pose_record/test10"

    # Delete all pose error
    delete_all_pose_error(data_dir)

    # 进行pose_error的计算与保存
    calculate_save_pose_error(data_dir)

    # 进行pose_error_distance相关特征的计算与可视化
    visulize_error_distance_feature(data_dir)

    # 进行pose_error_theta_diff相关特征的计算与可视化
    visulize_error_theta_feature(data_dir)





if __name__ == "__main__":
    main()

