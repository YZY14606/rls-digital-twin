import random
import trimesh
import numpy as np
import os
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
import open3d as o3d
from predictor.contact_predictor import contact_predictor



def get_obj_frame_points(num_points =4096):
    # Mesh文件的路径
    object_path_dic = dict([(1, '/home/yzy/nus/database/shapenet/ShapeNetCore.v2/03636649/e5ff9311bee487f5ca4aaad7dc0e3a16/models/model_normalized.obj'), 
                        (2, '/home/yzy/nus/database/shapenet/ShapeNetCore.v2/03797390/43f94ba24d2f075c4d32a65fb7bf4ebc/models/model_normalized.obj'), 
                        (3, '/home/yzy/nus/database/shapenet/ShapeNetCore.v2/02876657/3d8444d8717cc0b7c18cdaa4851a3c95/models/model_normalized.obj'),
                        (4, '/home/yzy/nus/database/shapenet/ShapeNetCore.v2/02880940/4eefe941048189bdb8046e84ebdc62d2/models/model_normalized.obj'),
                        (5, "/home/yzy/nus/database/shapenet/ShapeNetCore.v2/02871439/cc38bc7db90f43d214b86d5282eb8301/models/model_normalized.obj"),
                        (6, "/home/yzy/nus/database/shapenet/ShapeNetCore.v2/02933112/2c1af98d2058a8056588620c25b809f9/models/model_normalized.obj"),
                        (7, "/home/yzy/nus/database/shapenet/ShapeNetCore.v2/03046257/e59e73bc340207bfe214891c68fa8e36/models/model_normalized.obj"),
                        (8, "/home/yzy/nus/database/shapenet/ShapeNetCore.v2/02828884/c50aa1c3da488573ba5342d638d0c267/models/model_normalized.obj"),
                        (9, "/home/yzy/nus/database/shapenet/ShapeNetCore.v2/03636649/86ae11f8d3079f0869e321f074c1ab85/models/model_normalized.obj"),
                        (10,"/home/yzy/nus/database/shapenet/ShapeNetCore.v2/02992529/3a6a3db4a0174fddd2789f496481c83e/models/model_normalized.obj"),
                        ])
    
    #给出每个物体合理的scale范围(利用长边计算)
    object_scale_range = dict([(1, [0.25, 0.42]), (2, [0.15, 0.28]), (3, [0.21, 0.32]), (4, [0.25, 0.4]), (5, [0.14, 0.38]),
                           (6, [0.1, 0.34]), (7, [0.24, 0.37]), (8, [0.18, 0.37]), (9, [0.21, 0.41]), (10, [0.1,0.16])
                           ])
    #计算出每个物体在scale范围内随机取的scale
    object_scale_dic = object_scale_range.copy()
    for key in object_scale_range.keys():
        object_scale_dic[key] = random.uniform(object_scale_range[key][0], object_scale_range[key][1])

    # 进行文件索引的随机选取
    object_index = random.randint(1, 10)
    object_index = 4
    
    # Mesh文件的路径
    mesh_path = object_path_dic[object_index]

    mesh = trimesh.load(mesh_path, process=True)
    # 如果仍是 Scene，尝试提取 Trimesh 对象
    if isinstance(mesh, trimesh.Scene):
        # 合并场景中的所有几何体为一个 Trimesh
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    scale = object_scale_dic[object_index]
    mesh.apply_scale(scale)

    current_pc, _ = trimesh.sample.sample_surface(mesh, num_points)

    return current_pc, object_index

def get_wld_frame_pose(object_index):
    # 初始化
    object_pose = np.array([0,0,0.1,1,0,0,0])
    future_pose = np.array([0,0,0.1,1,0,0,0])

    # 进行姿态的随机初始化(绕z轴旋转)
    if object_index == 10:
        pose_q = euler2quat(0, np.pi, 0)
    else:
        pose_q = euler2quat(np.pi*0.5+0.00001, 0, 0)

    # 获取初始姿态
    object_pose[3:7] = np.array(pose_q)

    # 进行future_pose的设置
    if object_index == 10:
        pose_q_fut = euler2quat(0, np.pi, random.uniform(0, 2 * np.pi))
    else:
        pose_q_fut = euler2quat(np.pi*0.5, 0, random.uniform(0, 2 * np.pi))
    
    future_pose[3:7] = np.array(pose_q_fut)
    future_xy = np.random.uniform(0, 0.1, size=(2,))
    future_pose[:2] = future_xy


    return object_pose, future_pose

def get_pose_matrix(position, quaternion):
    """
    构造 4x4 的变换矩阵，从局部坐标到世界坐标。
    四元数格式为 [x, y, z, w]（scalar-last）
    """
    quaternion = np.array([quaternion[1],quaternion[2],quaternion[3],quaternion[0]])
    rot = R.from_quat(quaternion).as_matrix()  # (3,3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = position
    return T

def transform_points_local_to_new_local(points_local, pose1, pose2):
    """
    将局部点云 points_local (N,3)，从 pose1 坐标系变换为 pose2 后，
    得到点云在 pose1 坐标系下的新坐标。
    
    pose1 和 pose2 分别为 (position, quaternion)
    """
    pos1 = pose1[:3]
    quat1  = pose1[3:7]
    pos2 = pose2[:3]
    quat2  = pose2[3:7]

    T1 = get_pose_matrix(pos1, quat1)
    T2 = get_pose_matrix(pos2, quat2)

    # 计算相对变换（从 pose1 到 pose2）
    T_rel = np.linalg.inv(T1) @ T2

    # 齐次化点云
    N = points_local.shape[0]
    points_hom = np.hstack([points_local, np.ones((N, 1))])  # (N,4)

    # 应用相对变换
    transformed_points = (T_rel @ points_hom.T).T[:, :3]  # (N,3)
    return transformed_points


def vis_npy_pointcloud(points1,points2):
    # 加载第一个点云数据
    points1 = points1.reshape(-1, 3)  # 调整形状为 (N, 3)
    # 创建第一个点云对象并设置颜色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.paint_uniform_color([1, 0, 0])  # 红色

    points2 = points2.reshape(-1, 3)  # 调整形状为 (N, 3)
    # 创建第二个点云对象并设置颜色
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 1, 0])  # 绿色

    # 创建坐标轴：轴长为 1，颜色为红（X轴）、绿（Y轴）、蓝（Z轴）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

    combined_pcd = pcd1 +pcd2

    # 可视化两个点云
    o3d.visualization.draw_geometries([combined_pcd,axis])

def main():
    # Mesh文件的读取与采样
    current_pc,object_index = get_obj_frame_points()

    # 进行current_pose和future_pose的获取
    object_pose, future_pose = get_wld_frame_pose(object_index)

    # 得到下一步点云在pose1局部坐标系的数据
    future_pc = transform_points_local_to_new_local(current_pc, object_pose, future_pose)

    # 可视化生成的点云
    vis_npy_pointcloud(points1 = current_pc,points2 =future_pc)

    # Define a contact point predictor
    point_predictor = contact_predictor()

    #获取最佳动作的参数
    point_predictor.vis = True
    contact_point, orientation, push_distance = point_predictor.predict(current_pc,future_pc)

    # 进行结果的可视化
    if point_predictor.vis:
        path = os.path.join(point_predictor.vis_dir,"pred_muti_result.ply")
        #读取第一个点云文件
        pcd1 = o3d.io.read_point_cloud(path)
        # 创建坐标轴：轴长为 1，颜色为红（X轴）、绿（Y轴）、蓝（Z轴）
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        # 可视化点云
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # 添加点云
        vis.add_geometry(pcd1)
        vis.add_geometry(axis)
        # 开始可视化
        vis.run()
        # 关闭窗口
        vis.destroy_window()





if __name__=='__main__':
    main()