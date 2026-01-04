import os
from pathlib import Path
import json
import numpy as np
import math
import torch
from transforms3d.euler import quat2euler,euler2quat
import trimesh
from transforms3d.quaternions import quat2mat
import yaml
from chamferdist import ChamferDistance
import re
import open3d as o3d



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


def calculate_chamfer_distance(args):
    current_file = Path(__file__).resolve()
    five_parent_dir = current_file.parent.parent.parent.parent

    with open(args['obj_config_path'], 'r', encoding='utf-8') as file:
        object_data = yaml.safe_load(file)

    object_index = str(args['object_index']) + '_object'
    object_relative_path = object_data[object_index]['relative_path']
    object_path = os.path.join(five_parent_dir,object_relative_path)

    # Get the object test scale
    object_scale = np.array(object_data[object_index]['test_info']['test_scale'])

    # Load mesh
    mesh = trimesh.load(object_path)
    mesh.apply_scale(object_scale)
    mesh_start = mesh.copy()
    transformation = np.eye(4)
    transformation[:3,3] = args['start_pose'][:3]
    quat_matrix = quat2mat(args['start_pose'][3:7])
    transformation[:3,:3] = quat_matrix
    mesh_start.apply_transform(transformation)

    samples = []
    for _ in range(30):
        points, _ = trimesh.sample.sample_surface(mesh_start, 1024)
        samples.append(points)
    ori_points = np.stack(samples, axis=0)  # shape: (30, 1024, 3)

    transformation = np.eye(4)
    transformation[:3,3] = args['end_pose'][:3]
    quat_matrix = quat2mat(args['end_pose'][3:7])
    transformation[:3,:3] = quat_matrix
    mesh.apply_transform(transformation)

    samples = []
    for _ in range(30):
        points, _ = trimesh.sample.sample_surface(mesh, 1024)
        samples.append(points)
    future_points = np.stack(samples, axis=0)  # shape: (30, 1024, 3)

    cd_loss = ChamferDistance()

    x = torch.tensor(ori_points, dtype=torch.float32)
    y = torch.tensor(future_points, dtype=torch.float32)
    dist = cd_loss(x, y,batch_reduction = 'mean',bidirectional = True,point_reduction = 'mean').item() # default batch reduction = mean

    mean_euclidean = np.sqrt(dist / 2)

    # all_pcd = np.concatenate([ori_points[0],future_points[0]],axis=0)
    # # Pointcloud
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(all_pcd)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # 创建坐标轴：轴长为 1，颜色为红（X轴）、绿（Y轴）、蓝（Z轴）
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # # 添加点云
    # vis.add_geometry(pcd1)
    # vis.add_geometry(axis)
    # # 获取视角控制器
    # view_control = vis.get_view_control()
    # # 设置相机参数
    # view_control.set_zoom(0.3)  # 缩放视角，值越大，点云显示越小
    # view_control.set_front([1, 0, 0])  # 设置相机的前方方向
    # view_control.set_lookat([0, 0, 0])  # 设置观察点，视角会集中在该位置
    # view_control.set_up([0, 0, 1])  # 设置相机的上方方向
    # # 开始可视化
    # vis.run()
    # # 关闭窗口
    # vis.destroy_window()

    return mean_euclidean




def main():
    # The dir nam7
    dir_path = 'difficult_model_100_seed_66_wd1_ds1_set_1_repeat_1_only_actor/test12'

    dir_name = os.path.basename(dir_path)
    # 分离
    match = re.match(r'([a-zA-Z]+)(\d+)', dir_name)
    if match:
        string_part = match.group(1)  # 文本部分
        number_part = int(match.group(2))  # 数字部分转换为整数
    else:
        print(f'dir_path-{dir_path} error!')

    # Find all recorded poses
    pose_record_dir = os.path.join('pose_record',dir_path)

    # 使用 pathlib 的 rglob 方法查找所有文件
    pose_dir = Path(pose_record_dir)
    traj_paths = list(pose_dir.rglob('**/traj_*/'))
    traj_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))


    # Calculate
    traj_number = 0
    success_count = 0
    all_steps = 0
    all_mean_euclidean = 0
    all_position_error = 0
    all_orientation_error = 0
    for path in traj_paths:
        # Load two pose
        file_path = os.path.join(path,'two_pose.npy')
        if os.path.exists(file_path):
            traj_number += 1
        else:
            print(f"文件夹 {file_path} 不存在")
        all_pose = np.load(file_path)
        all_pose[1][2] = all_pose[0][2]

        # 计算由chamfer-distance变来的欧式距离,并判断轨迹是否成功
        args = {}
        args['object_index'] = number_part
        args['start_pose'] = all_pose[0].copy()
        args['end_pose'] = all_pose[1].copy()
        if string_part == 'train':
            args['obj_config_path'] = f'config/{string_part}_object_config_difficult.yaml'
        else:
            args['obj_config_path'] = f'config/{string_part}_object_config.yaml'

        mean_euclidean = calculate_chamfer_distance(args)

        if mean_euclidean < 0.015:
            success_count += 1
            all_mean_euclidean += mean_euclidean

            # 计算两个pose的误差
            pose_error, _ = calculate_error(all_pose)
            if len(pose_error) !=2:
                print('Error in pose_error!')
            all_position_error += pose_error[0]
            all_orientation_error += pose_error[1]


            # Load the push record file
            path_traj = Path(path)
            relative_path = Path(*path_traj.parts[1:])
            push_record_traj = 'push_record'/relative_path / 'log.json'
            # Get the steps
            with open(push_record_traj, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_steps += int(data['push_steps'])


    # Calculate success rate
    success_rate = success_count / traj_number
    # Calculate average steps
    average_steps = all_steps / success_count
    # Calculate average mean rooted chamfer distance
    mean_rooted_cdis = all_mean_euclidean / success_count
    # Calculate average position error
    average_position_error = all_position_error / success_count
    # Calculate average orientation error
    average_orientation_error = all_orientation_error / success_count

    print(f'This is {dir_path} object.')
    print(f'The trajectory number is {traj_number}. Success rate is {success_rate}.')
    print(f'The average steps of the success trajectory is {average_steps}.')
    print(f'The average mean rooted chamfer distance of the success trajectory is {mean_rooted_cdis}.')
    print(f'The average position error of the success trajectory is {average_position_error}.')
    print(f'The average orientation error of the success trajectory is {average_orientation_error}.')


if __name__ == "__main__":
    main()