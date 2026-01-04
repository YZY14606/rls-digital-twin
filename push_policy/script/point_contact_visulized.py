#进行碰撞的点云接触可视化
import open3d as o3d
import trimesh
import numpy as np
import torch
import cv2,os
import threading
from transforms3d.euler import quat2euler,euler2quat ,euler2mat
from transforms3d.quaternions import quat2mat
import matplotlib.pyplot as plt
from pathlib import Path
import h5py


def vis_ply_pcd(path):
    extension = os.path.splitext(path)[1]
    if extension =='.ply':
    #读取第一个点云文件
        pcd1 = o3d.io.read_point_cloud(path)
    elif extension =='.npy':
        points1 = np.load(path)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
    elif extension =='.h5':
        with h5py.File(path, 'r') as h5f:
            points1 = np.array(h5f["local_data"][f"{1}_data/current_pointcloud_local_normalized"][:]).reshape(-1,3)
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1)


    print('point len:',len(pcd1.points))

    # 合并两个点云
    combined_pcd = pcd1

    pcd_tree = o3d.geometry.KDTreeFlann(pcd1)
    [k, idx, _] = pcd_tree.search_radius_vector_3d([0, 0, 0], 0.01)
    print(f"半径0.01m球内的点数: {k}")

    points_np = np.asarray(pcd1.points)      # (N, 3) numpy 数组
    z = points_np[:, 2]                      # 取所有点的 z 坐标
    count = np.sum(z < 0.01)                 # 统计 z < 0.01m 的点数
    print("高度 < 0.01 m 的点数量：", count)

    # 创建球形网格用于可视化
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1, 0, 0])  # 红色
    sphere.translate([0, 0, 0])  # 移动到原点


    # 创建坐标轴：轴长为 1，颜色为红（X轴）、绿（Y轴）、蓝（Z轴）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # 可视化点云
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云
    vis.add_geometry(combined_pcd)
    vis.add_geometry(axis)
    vis.add_geometry(sphere)

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




def main():

    path = 'visualizations/difficult_model_100_seed_66_wd1_ds1_set_1_repeat_1_only_actor/test7/20251208_131220.1/traj_3/10_push/pred_single_result.ply'
    vis_ply_pcd(path)


if __name__ == "__main__":

    main()