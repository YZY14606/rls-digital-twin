import trimesh
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat
import yaml
import os
from pathlib import Path
import open3d as o3d
from chamferdist import ChamferDistance
import torch
from scipy.spatial.distance import cdist

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

    ori_points, _ = trimesh.sample.sample_surface(mesh, 1024)

    transformation = np.eye(4)
    transformation[:3,3] = args['end_pose'][:3]
    quat_matrix = quat2mat(args['end_pose'][3:7])
    transformation[:3,:3] = quat_matrix
    mesh.apply_transform(transformation)

    future_points, _ = trimesh.sample.sample_surface(mesh, 1024)

    all_pcd = np.concatenate([ori_points,future_points],axis=0)

    cd_loss = ChamferDistance()

    x = torch.tensor(ori_points, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(future_points, dtype=torch.float32).unsqueeze(0)
    dist = cd_loss(x, y,bidirectional = True,point_reduction = 'mean').item()

    mean_euclidean = np.sqrt(dist / 2)

    # min_distance_bruteforce(A = ori_points,B= future_points)

    return mean_euclidean,all_pcd


def calculate_self_chamfer_distance():
    np.random.seed(42)
    # 生成点云数据
    point_cloud_1 = np.random.rand(1024, 3) * 0.0

    point_cloud_2 = np.random.rand(1024, 3) * 0.0
    point_cloud_2[:, 0] = point_cloud_2[:, 0] + 1.0

    all_pcd = np.concatenate([point_cloud_1,point_cloud_2],axis=0)

    cd_loss = ChamferDistance()

    x = torch.tensor(point_cloud_1, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(point_cloud_2, dtype=torch.float32).unsqueeze(0)
    dist = cd_loss(x, y, bidirectional = True, point_reduction = 'mean').item() / 2


    return dist,all_pcd

def min_distance_bruteforce(A, B):
    # A: (N,3), B: (M,3)
    diff = A[:, None, :] - B[None, :, :]   # (N,M,3)
    dist = np.linalg.norm(diff, axis=2)   # (N,M)
    print(np.min(dist))


def main(args):

    dist_list = []
    for i in range(100):
        single_dist , all_pcd = calculate_chamfer_distance(args)
        dist_list.append(single_dist)
    average_dist = np.mean(dist_list)
    print(average_dist)

    # dist,all_pcd = calculate_self_chamfer_distance()
    # print(dist)

    # Pointcloud
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(all_pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 创建坐标轴：轴长为 1，颜色为红（X轴）、绿（Y轴）、蓝（Z轴）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # 添加点云
    vis.add_geometry(pcd1)
    vis.add_geometry(axis)
    # 获取视角控制器
    view_control = vis.get_view_control()

    # 设置相机参数
    view_control.set_zoom(0.3)  # 缩放视角，值越大，点云显示越小
    view_control.set_front([1, 0, 0])  # 设置相机的前方方向
    view_control.set_lookat([0, 0, 0])  # 设置观察点，视角会集中在该位置
    view_control.set_up([0, 0, 1])  # 设置相机的上方方向

    # 开始可视化
    vis.run()

    # 关闭窗口
    vis.destroy_window()

if __name__ == "__main__":
    args = {}
    args['object_index'] = 5

    start_q = euler2quat(0,0,0)
    args['start_pose'] = np.concatenate([np.array([0,0,0]),start_q],axis=0)

    end_q = euler2quat(0,0,np.pi/6)
    args['end_pose'] = np.concatenate([np.array([0.014,0.014,0]),end_q],axis=0)

    args['obj_config_path'] = 'config/train_object_config.yaml'


    main(args)
