import numpy as np
import torch
import torch.random

import os
from datetime import datetime
from transforms3d import quaternions
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
import shutil
import random
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import math
from PIL import Image
from scipy.spatial import ConvexHull
from transforms3d.euler import quat2euler,euler2quat
from transforms3d.quaternions import quat2mat, mat2quat

from path_planner.planner import Path_planner
import object_planner_py as opp
from shapely.geometry import Polygon
from segmentation.segmentation import Segmentation_tool, Points_fusion_tool
from chamferdist import ChamferDistance




# 对于观测的点云，构建局部坐标系
def build_local_frame(points):
    # Down sample
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    o3d_pcd = o3d_pcd.voxel_down_sample(0.05)
    points = np.asarray(o3d_pcd.points)

    # Step 1: 计算质心并中心化
    centroid = np.mean(points, axis=0)
    centroid[2] = 0
    centered = points - centroid
    
    # Step 2: 确定局部z轴为世界z轴 (0,0,1)
    z_local = np.array([0, 0, 1])
    
    # Step 3: 投影到xy平面（忽略z坐标）
    projected = centered[:, :2]
    
    # Step 4: 二维PCA
    cov_2d = np.cov(projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    v1, v2 = eigenvectors[:, sorted_indices].T
    
    # Step 5: 调整方向构建右手系
    # 将v1作为x轴候选
    x_candidate = np.append(v1, 0)  # 转为三维向量
    x_local = x_candidate / np.linalg.norm(x_candidate)
    
    # 通过叉乘计算y轴
    y_local = np.cross(z_local, x_local)
    y_local = y_local / np.linalg.norm(y_local)
    
    # 验证右手系
    if not np.allclose(np.cross(x_local, y_local), z_local):
        raise ValueError("坐标轴不符合右手法则")
    
    # Step 6: 构建旋转矩阵
    R = np.column_stack([x_local, y_local, z_local])

    local_frame_pose = transform_ori_to_pose(Rot = R, centroid = centroid)
    
    return local_frame_pose


# 将局部坐标的旋转矩阵与质心位置转化为四元数(wxyz)形式pose
def transform_ori_to_pose(Rot, centroid):
    """将旋转矩阵转换为四元数 (w, x, y, z) 格式"""
    quat_wxyz = mat2quat(Rot)
    # 提取位置 (x, y, z)
    position = np.array(centroid)
    # 合并为 (7,) 数组
    pose_7d = np.concatenate([position, quat_wxyz])
    return pose_7d

# 插值计算下一步的pose
def interpolatio_compute_next_obj_pose(object_pose,path_point,push_step_dict):
    #下一时刻的pose
    next_pose = torch.zeros( 7, dtype = float)

    # Get single push parameter 
    single_push_len = push_step_dict['single_push_len']
    single_push_theta = push_step_dict['single_push_theta']

    #对于xy方向上的next_pose进行计算
    distance = torch.norm(path_point[:2] - object_pose[:2])
    delta_x = path_point[0]-object_pose[0]
    delta_y = path_point[1]-object_pose[1]
    if distance > single_push_len:
        # 计算单位方向向量
        unit_x = delta_x / distance
        unit_y = delta_y / distance
        # 计算沿方向行走 0.07m 的 x, y 变化量
        step_len = single_push_len
        new_delta_x = unit_x * step_len
        new_delta_y = unit_y * step_len
        # 计算net_pose的xyz
        next_pose[0] = new_delta_x + object_pose[0]
        next_pose[1] = new_delta_y + object_pose[1]
        next_pose[2] = object_pose[2]
    else:
        # 计算net_pose的xyz
        next_pose[0] = delta_x + object_pose[0]
        next_pose[1] = delta_y + object_pose[1]
        next_pose[2] = object_pose[2]

    #对于旋转的角度进行计算
    #计算物体的欧拉角(绕z轴)
    euler_object = quat2euler(object_pose[3:7])[2]
    target_eule_z = path_point[2]
    #转换为正值
    if euler_object < 0:
        euler_object = math.pi*2 + euler_object
    if target_eule_z <0:
        target_eule_z = math.pi*2 + target_eule_z

    # 计算差值并判断是否小于 pi / 6
    threshold = single_push_theta
    theta_diff = torch.abs(euler_object - target_eule_z)
    # 将差值转换到0-pi之间
    processed_diff = min(theta_diff, 2 * np.pi - theta_diff)
    if processed_diff > threshold:
        delta_degree = threshold
    elif processed_diff <= threshold:
        delta_degree = theta_diff
    # 当delta_degree为负数时，说明物体需要顺时针旋转
    if (target_eule_z - euler_object) * (math.pi -theta_diff ) <0:
        delta_degree =  -1 * delta_degree
    
    # 绕z轴旋转的弧度
    rad_z = delta_degree
    euler_obj = list(quat2euler(object_pose[3:7]))
    euler_obj[2] = rad_z + euler_obj[2]
    next_pose[3:7] = torch.tensor(euler2quat(euler_obj[0],euler_obj[1],euler_obj[2]))

    return next_pose

# 获得环境中的mesh，并将其转化为少量的场景点云
def get_obstacel_pointcloud(obstacle_mesh, buff_len = 0.04):
    # 1. 获取mesh的所有边顶点（比单纯顶点更准确表示轮廓）
    hull = obstacle_mesh.convex_hull
    points = np.array(hull.vertices)
    points_xy =  points[:,:2]

    point_2d = points_xy[ConvexHull(points_xy).vertices]
    hull = Polygon(point_2d)
    # Expand the hull
    expanded_hull = hull.buffer(buff_len)
    boundary = expanded_hull.boundary
    
    # 在边界上按距离均匀采样
    distances = np.linspace(0, boundary.length, 25, endpoint=False)
    points_xy = np.array([boundary.interpolate(d).coords[0] for d in distances])

    # 第一层的点云
    pointcloud_1 = np.zeros((points_xy.shape[0], 3))
    pointcloud_1[:, :2] = points_xy
    pointcloud_1[:, 2] = 0.001
    # 第二层的点云
    pointcloud_2 = np.zeros((points_xy.shape[0], 3))
    pointcloud_2[:, :2] = points_xy
    pointcloud_2[:, 2] = 0.02

    center = np.array([-0.615,0,0])
    radius = 0.23 + buff_len
    all_obstacle_pointcloud = generate_circle_point_cloud(center, radius, num_points = 100)

    return np.vstack([pointcloud_1, pointcloud_2,all_obstacle_pointcloud])

def generate_circle_point_cloud(center, radius, num_points):
    """
    生成圆形点云
    
    参数:
    center: 圆心坐标 (x, y, z)
    radius: 圆半径
    num_points: 采样点数
    
    返回:
    points: 点云坐标数组 (num_points, 3)
    """
    # 生成均匀的角度采样
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # 计算圆上的点坐标（在xy平面）
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.full(num_points, center[2])  # z坐标保持不变
    
    # 组合成点云数组
    points = np.column_stack((x, y, z))
    
    return points

    

def get_object_pointcloud_for_plan(points,buff_len = 0.01):
    points_xy = points[:,:2]
    point_2d = points_xy[ConvexHull(points_xy).vertices]
    hull = Polygon(point_2d)
    # Expand the hull
    expanded_hull = hull.buffer(buff_len)
    boundary = expanded_hull.boundary
    # 在边界上按距离均匀采样
    distances = np.linspace(0, boundary.length, 25, endpoint=False)
    points_xy = np.array([boundary.interpolate(d).coords[0] for d in distances])

    # 第一层的点云
    pointcloud_1 = np.zeros((points_xy.shape[0], 3))
    pointcloud_1[:, :2] = points_xy
    pointcloud_1[:, 2] = 0.001
    # 第二层的点云
    pointcloud_2 = np.zeros((points_xy.shape[0], 3))
    pointcloud_2[:, :2] = points_xy
    pointcloud_2[:, 2] = 0.02

    return np.vstack([pointcloud_1, pointcloud_2])


def transform_to_object_frame(points, obj_position, obj_quaternion):
    """
    Transform points from world frame to object frame

    Args:
        points: Points in world frame (N, 3)
        obj_position: Object position in world frame [x, y, z]
        obj_quaternion: Object orientation in world frame [w, x, y, z]

    Returns:
        Points in object frame (N, 3)
    """
    #转换四元数为(x,y,z,w)
    obj_quaternion = np.array([obj_quaternion[1],obj_quaternion[2],obj_quaternion[3],obj_quaternion[0]])
    # 构造世界→局部坐标的旋转矩阵
    rot_matrix = R.from_quat(obj_quaternion).as_matrix()  # shape (3, 3)

    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = obj_position[:3]

    # 计算相对变换: T_final = inv(T)
    T_final = np.linalg.inv(T)
    points_obj_frame = trimesh.transform_points(points, T_final)  # 使用 trimesh 的变换函数

    return points_obj_frame

# 进行全局路径规划，得到可行的路径点
def First_path_plan(target_pose,object_pose,object_wrld_frame_pcd,
                    obstacle_buff_len_list = [0.04, 0.01, 0], object_buff_len = 0.01, visulize_result = False):
    # Get target information for path planning
    target_xy = target_pose[:2].copy()
    target_quat = target_pose[3:].copy()

    # Object pointcloud generation
    object_point = get_object_pointcloud_for_plan(object_wrld_frame_pcd,buff_len = object_buff_len)
    euler_z_object = quat2euler(object_pose[3:7])[2]
    quat_z_object = euler2quat(0,0,euler_z_object)
    object_point = transform_to_object_frame(points = object_point, obj_position = object_pose[:3], obj_quaternion = quat_z_object)
    object_point[:int(object_point.shape[0]/2),2] = 0.001
    object_point[int(object_point.shape[0]/2):,2] = 0.02

    # Set planner params
    planner_params = dict()
    planner_params["planner_iterations"] = 8000
    planner_params["planner_step_size"] = 0.05
    planner_params["goal_bias"] = 0.05
    planner_params["neighborhood_radius"] = 0.1
    path_planner = Path_planner(planner_params)

    # Get start configs
    x_now = object_pose[0]
    y_now = object_pose[1]
    # Calculate the theta in z-aix
    theta_now = quat2euler(object_pose[3:7])[2]
    # Transform to (0, 2pi)
    if theta_now < 0:
        theta_now = math.pi*2 + theta_now
    start_config = opp.Config(x_now, y_now, theta_now)

    # Get target configs
    x_target = target_xy[0]
    y_target = target_xy[1]
    # Calculate the theta in z-aix
    theta_target = quat2euler(target_quat)[2]
    # Transform to (0, 2pi)
    if theta_target < 0:
        theta_target = math.pi*2 + theta_target
    target_config = opp.Config(x_target, y_target, theta_target)

    # Change the obstacle's size, and replan
    for obstacle_buff_len in obstacle_buff_len_list:
        # Get obstacle information for path planning
        obstacle_index = env.obstacle_index
        obstacle_pointcloud = []
        for index in obstacle_index:
            obstacle_mesh = env.obstacle_dic[index].get_collision_meshes()[0]
            pointcloud = get_obstacel_pointcloud(obstacle_mesh, buff_len = obstacle_buff_len)
            obstacle_pointcloud.append(pointcloud)
        if len(obstacle_pointcloud) != 0:
            all_obstacle_pointcloud = np.concatenate(obstacle_pointcloud, axis=0)
        else:
            center = np.array([-0.615,0,0])
            radius = 0.23 + obstacle_buff_len
            all_obstacle_pointcloud = generate_circle_point_cloud(center, radius, num_points = 100)

        # Get scene data
        scene_dic = dict()
        scene_dic["object_points"] = object_point
        scene_dic["all_obstacle_points"] = all_obstacle_pointcloud
        scene_dic["map_x_bounds"] = (-0.6, 0.6)
        scene_dic["map_y_bounds"] = (-1, 1)
        scene_dic["map_theta_bound"] = (0, 2 * np.pi)
        path_planner.set_up_planner(scene_dic)

        # Plan path
        path = path_planner.plan_path(start_config = start_config, goal_config = target_config)
        # When plan fail, try again.
        if len(path) == 1:
            path = path_planner.plan_path(start_config = start_config, goal_config = target_config)

        if len(path) > 1:
            break

    if visulize_result:
        path_planner.visualize_path(
            object_points = object_point,
            obstacle_points = all_obstacle_pointcloud,
            path = path,
            start_config = start_config,
            goal_config = target_config,
            map_bounds_x = (-0.6, 0.6),
            map_bounds_y = (-1, 1),
            )

    path_point = [[c.x, c.y, c.theta] for c in path]

    return np.array(path_point),path_planner

def plan_from_waypoint(target_pose,object_pose,path_planner,obstacle_buff_len_list = [0.04, 0.01, 0],visulize_result=False):
    # Get target information for path planning
    target_xy = target_pose[:2].copy()
    target_quat = target_pose[3:].copy()

    # Get start configs
    x_now = object_pose[0]
    y_now = object_pose[1]
    # Calculate the theta in z-aix
    theta_now = quat2euler(object_pose[3:7])[2]
    # Transform to (0, 2pi)
    if theta_now < 0:
        theta_now = math.pi*2 + theta_now
    start_config = opp.Config(x_now, y_now, theta_now)

    # Get target configs
    x_target = target_xy[0]
    y_target = target_xy[1]
    # Calculate the theta in z-aix
    theta_target = quat2euler(target_quat)[2]
    # Transform to (0, 2pi)
    if theta_target < 0:
        theta_target = math.pi*2 + theta_target
    target_config = opp.Config(x_target, y_target, theta_target)

    # Change the obstacle's size, and replan
    for obstacle_buff_len in obstacle_buff_len_list:
        # Get obstacle information for path planning
        obstacle_index = env.obstacle_index
        obstacle_pointcloud = []
        for index in obstacle_index:
            obstacle_mesh = env.obstacle_dic[index].get_collision_meshes()[0]
            pointcloud = get_obstacel_pointcloud(obstacle_mesh, buff_len = obstacle_buff_len)
            obstacle_pointcloud.append(pointcloud)
        if len(obstacle_pointcloud) != 0:
            all_obstacle_pointcloud = np.concatenate(obstacle_pointcloud, axis=0)
        else:
            center = np.array([-0.615,0,0])
            radius = 0.23 + obstacle_buff_len
            all_obstacle_pointcloud = generate_circle_point_cloud(center, radius, num_points = 100)

        scene_dic = path_planner.scene_dic
        scene_dic["all_obstacle_points"] = all_obstacle_pointcloud
        path_planner.set_up_planner(scene_dic)

        # Plan path
        path = path_planner.plan_path(start_config = start_config, goal_config = target_config)
        # When plan fail, try again.
        if len(path) == 1:
            path = path_planner.plan_path(start_config = start_config, goal_config = target_config)

        if len(path) > 1:
            break



    if visulize_result ==True:
        path_planner.visualize_path(
            object_points = path_planner.object_point,
            obstacle_points = path_planner.all_obstacle_pointcloud,
            path = path,
            start_config = start_config,
            goal_config = target_config,
            map_bounds_x = (-0.6, 0.6),
            map_bounds_y = (-1, 1),
            )

    path_point = [[c.x, c.y, c.theta] for c in path]

    return np.array(path_point),path_planner


# 根据规划的路径点，计算下一个push的future pose
def get_future_pose(path_points,object_pose,path_point_index,error_radius,target_quat,push_step_dict):
    # 判断当前应该移动向哪个路径点
    point = path_points[path_point_index]
    current_xy = object_pose[:2].cpu().numpy().copy()

    distance = np.linalg.norm(point[:2]- current_xy)

    if distance < error_radius:
        path_point_index = path_point_index + 1
        print("Object has arrived at a path-point.")
    if path_point_index >= len(path_points):
        path_point_index = len(path_points)-1
    
    goal_point = path_points[path_point_index]
    # Calculate the theta in z-aix
    theta_target = quat2euler(target_quat)[2]
    theta_target = goal_point[2] # yzy
    # Transform to (0, 2pi)
    if theta_target < 0:
        theta_target = math.pi*2 + theta_target
    goal_point[2] = theta_target

    next_pose = interpolatio_compute_next_obj_pose(object_pose,torch.tensor(goal_point),push_step_dict)

    return next_pose , path_point_index


def generate_segmentation_mask(image_rgb,points,labels):
    """
    image_rgb: shape (H,W,3);
    points: prompt points (N,2);
    labels: the labels of prompt points, shape (N)
    """
    img_seg_tool = Segmentation_tool()
    img_seg_tool.set_image(image_rgb)
    masks = img_seg_tool.predict(points=points,labels = labels) # shape (1,H,W)

    return masks


def pcd_downsample(points, num_samples=1024):
    points_output = points.copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(pcd.points) > num_samples:
        downsampled_pcd = pcd.farthest_point_down_sample(num_samples)
        points_output = np.asarray(downsampled_pcd.points)
    if len(points_output) < num_samples:
        # 随机补齐
        idx = np.random.choice(len(points_output), num_samples - len(points_output), replace=True)
        points_output = np.concatenate([points_output, points_output[idx]], axis=0)

    return points_output

def pointcloud_segmentation_fusion(simulation_control,topic,camera_name_list):
    all_pointcloud = []
    pcd_fusion_tool =  Points_fusion_tool()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for camera_name in camera_name_list:
        # Get the rgb data of this camera
        rgb_topic = topic[camera_name][0]
        depth_topic = topic[camera_name][1]
        rgb, depth , cam2world, intrinsic_cv = simulation_control.get_rgb_depth(rgb_topic= rgb_topic,depth_topic= depth_topic,camera_name= camera_name)
        # Get the chair mask
        green_mask = (rgb[:, :, 1] > 130) & (rgb[:, :, 0] < 60) & (rgb[:, :, 2] < 60)
        green_points = np.argwhere(green_mask) # shape(N,2)
        chosen_indices = np.random.choice(len(green_points), size=3, replace=False)
        # 返回这两个点的位置 (row, col)
        position = green_points[chosen_indices]
        labels = np.array([1,1,1])
        # Use sam2 to get the mask of target object
        position[:,[0,1]] = position[:,[1,0]]
        mask_data = generate_segmentation_mask(rgb, position, labels)
        segmentation = torch.tensor(mask_data).squeeze().to(device)

        # Add parameters to Points_fusion_tool
        cam2world_para = torch.tensor(cam2world).view(4,4).to(device).to(torch.float)
        camera_intrinsic_para = torch.tensor(intrinsic_cv).view(3,3).to(device).to(torch.float)
        pcd_fusion_tool.cam2world.append(cam2world_para.clone())
        pcd_fusion_tool.intrinsic.append(camera_intrinsic_para.clone())
        pcd_fusion_tool.mask.append(segmentation.clone())

        # Acquire the pointcloud in world frame
        depth_tensor = torch.tensor(depth).squeeze().to(device).clone()
        points_wld_frame = calculate_object_pointcloud_wld(depth = depth_tensor,intrinsic_cv = camera_intrinsic_para, 
                                        cam2world = cam2world_para, mask = segmentation)
        # Filter the points below a hight
        mask = points_wld_frame[:, 2] >= 0.02
        filtered_wld_frame_points = points_wld_frame[mask]
        

        all_pointcloud.append(filtered_wld_frame_points)

    output_pointcloud = torch.cat(all_pointcloud,dim=0)
    output_pointcloud = pcd_fusion_tool.filter_point(output_pointcloud).cpu().numpy()


    if len(output_pointcloud)  > 1024:
        output_pointcloud = pcd_downsample(output_pointcloud,num_samples=1024)
    return output_pointcloud



def calculate_object_pointcloud_wld(depth,intrinsic_cv,cam2world,mask):
    assert depth.ndim == 2 and intrinsic_cv.shape == (3, 3)

    H, W = depth.shape
    device = depth.device

    # Create the pixel grid (u, v)
    u = torch.arange(W, device= device)
    v = torch.arange(H, device= device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # (H, W)

    # flat and construct pixel coordinate (3, N)
    u_flat = u_grid.reshape(-1)
    v_flat = v_grid.reshape(-1)
    mask_flat = mask.reshape(-1)
    pixels = torch.stack((u_flat, v_flat, mask_flat), dim=0)  # shape: (3, H*W)
    # flat the depth (N,)
    depth_flat = depth.reshape(-1)  # shape: (H*W,)

    # K^-1
    K_inv = torch.inverse(intrinsic_cv.to(torch.float32))
    # Calculate X_cam = d * (K^-1 @ [u, v, 1])
    cam_points = ((K_inv @ pixels) * depth_flat).T  # shape: (H*W,3)

    # Filter the segmented points
    mask = cam_points[..., 2] != 0  # 获取 w≠0 的掩码，形状 [1, N]
    filtered_xyz = cam_points[mask]  # 过滤后形状 [M, 3]，其中 M 是有效点数

    # Get the augmented points
    ones = torch.ones(filtered_xyz.shape[0],device=device).view(-1,1)
    augmented_cam_points = torch.cat([filtered_xyz,ones],dim=1)

    # Convert the points to the world frame
    points_wld_frame_augmented = (cam2world @ augmented_cam_points.T).T
    final_points = points_wld_frame_augmented[:,:3]


    return final_points


#得到物体坐标系下的表面点
def get_current_pc_future_pose(object_pcd_wld_frame, object_pose, future_pose):
    """
    Get current_pc and future_pc in object frame. They are uesd for model input.

    Args:
        object_pose: Object pose in world frame (7).
        future_pose: Object pose in world frame next time (7).
        points_num: Sample number from mesh.

    Returns:
        current_pc: Current point cloud in object frame (N, 3).
        local_frame_future_pose: Future pose of object in local frame (7).
    """
    # Transform current point cloud from world frame to object frame
    Rot = quat2mat(object_pose.clone().cpu().numpy()[3:7])
    centroid = object_pose.clone().cpu().numpy()[:3]
    current_pc = transform_to_local(object_pcd_wld_frame, R = Rot, centroid = centroid)

    self_frame_future_pose = get_target_pose_local_frame(object_pose.clone().cpu().numpy(),future_pose.clone().cpu().numpy())

    return current_pc,self_frame_future_pose

def transform_to_local(points, R, centroid,num_samples=1024):
    # 平移点云到质心
    centered = points - centroid
    # 应用旋转矩阵的逆（即转置）
    local_points = (R.T @ centered.T).T
    # 将点的数量下采样
    if len(local_points) > num_samples:
        # 进行点云下采样
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_points)
        downsampled_pcd = pcd.farthest_point_down_sample(num_samples)
        local_points = np.asarray(downsampled_pcd.points)

    return local_points

def get_target_pose_local_frame(object_pose,future_pose):
    """获得构建的局部坐标系下,下一步的目标pose"""
    # 获得世界坐标系下，物体当前pose的矩阵
    Rotation_current = quat2mat(object_pose[3:7])
    T_current = np.eye(4)
    T_current[:3, :3] = Rotation_current
    T_current[:3, 3] = object_pose[:3]

    # 获得世界坐标系下，物体下一步pose的矩阵
    Rotation_future = quat2mat(future_pose[3:7])
    T_future = np.eye(4)
    T_future[:3, :3] = Rotation_future
    T_future[:3, 3] = future_pose[:3]

    # 获得世界坐标系下，物体gt坐标系，按照自己坐标进行变换的矩阵
    T_self = np.linalg.inv(T_current) @ T_future

    # 当在局部坐标系下，T_self_transformation即为下一步的pose
    T_self_transformation = np.zeros(7)
    T_self_transformation[:3] = T_self[:3,3]
    T_self_transformation[3:7] = mat2quat(T_self[:3,:3])

    return T_self_transformation


# 得到两个垂直且距离为2的pose
def two_perp_poses_radius2(pose7):
    """
    pose7: [x, y, z, qx, qy, qz, qw]
    返回两个 7D pose：位于半径=2的圆上，方向与原 (x,y) 垂直
    """
    x, y, z = pose7[:3]
    quat = pose7[3:]
    v = np.array([x, y], dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("原 (x,y) 在原点，无法定义‘与原方向垂直’。")

    # 单位法向量（与 v 垂直）
    u_perp = np.array([-y, x], dtype=float) / n

    # 半径为 2 的两个点（xy 平面）
    p1_xy =  0.02 * u_perp
    p2_xy = -0.02 * u_perp

    pose1 = np.array([p1_xy[0], p1_xy[1], z, *quat], dtype=float)
    pose2 = np.array([p2_xy[0], p2_xy[1], z, *quat], dtype=float)
    return pose1, pose2



def generate_action(contact_point, orientation, push_distance,object_pose, push_gap = 0.05):
    """
    Transform action parameters to a complete action.

    Args:
        contact_point: The position of contact point (3).
        orientation: The direction of push (3).
        push_distance: The push distance (1).

    Returns:
        ps: The start position of ee (3).
        pe: The end position of ee (3).
    """
    # Normalize orientation
    orientation_normalized = orientation / np.linalg.norm(orientation)

    # Calculate the pe_position in object frame
    pe_position = contact_point + orientation_normalized * push_distance.item()
    # Calculate the ps_position in object frame
    ps_position = contact_point - orientation_normalized * push_gap
    points_obj_frame =np.stack([ps_position, pe_position], axis=0)

    points_world = transform_to_world_frame(points = points_obj_frame, obj_position = object_pose[:3], obj_quaternion = object_pose[3:7])

    # Get world frame ps and pe
    ps_position_wld = points_world[0]
    pe_position_wld = points_world[1]

    # Rondom q
    q = euler2quat(np.pi, 0, 0)

    # Combine p and q
    ps_pose = np.concatenate([ps_position_wld, q])
    pe_pose = np.concatenate([pe_position_wld, q])

    return ps_pose, pe_pose


def transform_to_world_frame(points, obj_position, obj_quaternion):
    """
    Transform points from object frame to world frame

    Args:
        points: Points in object frame (N, 3)
        obj_position: Object position in world frame [x, y, z]
        obj_quaternion: Object orientation in world frame [w, x, y, z]

    Returns:
        Points in world frame (N, 3)
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternions.quat2mat(obj_quaternion)

    # Construt the matrix, convert the points in object frame to world frame
    T_local_to_world = np.eye(4) 
    T_local_to_world[:3, :3] = rotation_matrix  # Rotation
    T_local_to_world[:3, 3] = obj_position      # Translation

    ones = np.ones((points.shape[0], 1))  # shape (N,1)
    points_homogeneous = np.hstack([points, ones])  # shape (N,4)
    points_wld_frame = points_homogeneous @ T_local_to_world.T # @ is matrix multiple

    return points_wld_frame[:,:3]


def convert_action_use(ps_pose,pe_pose,robot_theta_y):
    delta_pose = pe_pose - ps_pose

    angle = np.arctan2(delta_pose[1],delta_pose[0])
    theta_z = angle % (2 * np.pi)
    robot_ee_q = euler2quat(np.pi ,-robot_theta_y, theta_z)

    # Normalize orientation
    orientation_normalized = delta_pose[:3] / np.linalg.norm(delta_pose[:3])
    # Calculate the ps_position in object frame
    ps_xyz = ps_pose[:3]

    ps_pose[:3] = ps_xyz
    ps_pose[3:7] = np.array(robot_ee_q)

    # Calculate the pe_xy
    pe_xyz = pe_pose[:3] - orientation_normalized * 0.01
    pe_pose[:3] = pe_xyz
    pe_pose[3:7] = ps_pose[3:7]


    # 进行ps与pe高度的判断，防止与桌面过度碰撞
    if ps_pose[2]<0.02:
        ps_pose[2] = 0.02
    if pe_pose[2]<0.02:
        pe_pose[2] = 0.02

    return ps_pose ,pe_pose



def evaluate_complete_action(target_pose,source_obj_pcd,source_obj_pose,object_wrld_frame_pcd):

    # Set chamfer distance evaluation
    expected_pointcloud = get_pose2_wld_frame_points(points_wld_frame_pose1 = source_obj_pcd, object_pose = source_obj_pose,
                                                      future_pose = target_pose)

    cd_loss = ChamferDistance()

    x = torch.tensor(expected_pointcloud, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(object_wrld_frame_pcd, dtype=torch.float32).unsqueeze(0)
    cm_dist = cd_loss(x, y, bidirectional = True, point_reduction = 'mean').item()
    mean_euclidean = np.sqrt(cm_dist / 2)

    is_obj_placed = (mean_euclidean < 0.015)
    if is_obj_placed:
        all_pcd = np.concatenate([expected_pointcloud,object_wrld_frame_pcd],axis=0)
        np.save('chamfer_dis_pcd.npy',all_pcd)

    return is_obj_placed



def get_pose2_wld_frame_points(points_wld_frame_pose1, object_pose, future_pose):
    """
    Get future_pc in world frame.

    Args:
        points_wld_frame_pose1: The current_pc in world frame.
        object_pose: Object pose in world frame (7).
        future_pose: Object pose in world frame next time (7).

    Returns:
        points_world_pose2: Future point cloud in world frame (N, 3).
    """
    future_pose[2] = object_pose[2]

    # 获得世界坐标系下，物体当前pose的矩阵
    Rotation_current_gt = quat2mat(object_pose[3:7])
    T_current_gt = np.eye(4)
    T_current_gt[:3, :3] = Rotation_current_gt
    T_current_gt[:3, 3] = object_pose[:3]

    # 获得世界坐标系下，物体下一步pose的矩阵
    Rotation_future_gt = quat2mat(future_pose[3:7])
    T_future_gt = np.eye(4)
    T_future_gt[:3, :3] = Rotation_future_gt
    T_future_gt[:3, 3] = future_pose[:3]

    # 获得世界坐标系下，物体gt坐标系，按照基坐标进行变换的矩阵
    T_base_gt = T_future_gt @ np.linalg.inv(T_current_gt) 

    # 计算相对变换: T_final = T2 * inv(T1)
    points_world_pose2 = trimesh.transform_points(points_wld_frame_pose1, T_base_gt)  # 使用 trimesh 的变换函数
    
    return points_world_pose2


def test_fetch_motion_plan(fetch,current_state,target_pose):
    """
    Just used to test.
    """
    # Get current configuration
    current_joints = current_state['current_joints']
    if current_joints is None:
        print("In bi-level planning test, failed to get current joint positions")
        return None,None

    # Get current base position
    current_base = current_state['current_base']

    # Step 1: Solve whole-body IK
    ik_solution = fetch.solve_whole_body_ik(target_pose, max_attempts=100, manipulation_radius=1.0, normalized_arm_seed=None)

    if ik_solution is None:
        print("In bi-level planning test, failed to find IK solution")
        return None,None

    goal_base = ik_solution["base_config"]
    goal_joints = ik_solution["arm_config"]

    # Step 2: Plan whole-body motion
    plan_result = fetch.plan_whole_body_motion(
        current_joints, goal_joints, list(current_base), goal_base
    )

    if not plan_result or not plan_result["success"]:
        print("In bi-level planning test, failed to plan whole-body motion")
        return None,None
    
    # Step 3: Get the last pose
    last_state = dict()
    last_state['current_joints'] = plan_result['arm_path'][-1]
    last_state['current_base'] = plan_result['base_configs'][-1]

    return last_state, plan_result

def move_base_to_target(fetch,target_base_pose):
    # Get current joints configuration
    current_joints = fetch.get_current_planning_joints()
    if current_joints is None:
        print("Failed to get current joint positions in move base.")
        return False
    
    # Get current base position
    current_base = fetch.get_base_params()

    # Step 2: Plan whole-body motion
    print('Begin fetch base plan!')
    plan_result = fetch.plan_whole_body_motion(
        current_joints, current_joints, list(current_base), target_base_pose
    )

    if not plan_result or not plan_result["success"]:
        print("Failed to plan whole-body motion")
        return False

    # Step 3: Execute the planned motion
    print("Step 3: Executing whole-body motion...")
    execution_success = fetch.execute_whole_body_motion(
        plan_result["arm_path"], plan_result["base_configs"]
    )

    if not execution_success:
        print("Failed to execute whole-body motion")
        return False

    print("Successfully moved to target pose")
    return True



def collect_and_segmented_pcd(simulation_control,topic,fetch):
    """
    collect_and_segmented_pcd 的 Docstring:
        获得障碍物与物体的点云。
    """
    all_pointcloud = []
    pcd_fusion_tool =  Points_fusion_tool()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the camera psoe
    camera_pose = fetch.get_camera_pose()

    # Get the rgb data of this camera
    rgb_topic = topic[0]
    depth_topic = topic[1]
    intrinsic_topic = topic[2]
    rgb, depth , cam2world, intrinsic_cv = simulation_control.get_rgb_depth(rgb_topic= rgb_topic,depth_topic= depth_topic,
                                                                            intrinsic_topic = intrinsic_topic, camera_pose = camera_pose)
    # Get the chair mask
    green_mask = (rgb[:, :, 1] > 130) & (rgb[:, :, 0] < 60) & (rgb[:, :, 2] < 60)
    green_points = np.argwhere(green_mask) # shape(N,2)
    chosen_indices = np.random.choice(len(green_points), size=3, replace=False)
    # 返回这两个点的位置 (row, col)
    position = green_points[chosen_indices]
    labels = np.array([1,1,1])
    # Use sam2 to get the mask of target object
    position[:,[0,1]] = position[:,[1,0]]
    mask_data = generate_segmentation_mask(rgb, position, labels)
    segmentation = torch.tensor(mask_data).squeeze().to(device)

    import cv2
    # 假设你已有：
    # rgb: (H, W, 3), uint8, RGB 格式
    # segmenation: (H, W), bool 或 0/1 的 tensor/array
    # 1. 确保 segmentation 是 NumPy 数组且为布尔或 0/1
    if isinstance(segmentation, torch.Tensor):
        seg_mask = segmentation.cpu().numpy()
    else:
        seg_mask = segmentation
    # 转为布尔（如果还不是）
    seg_mask = seg_mask.astype(bool)
    # 2. 创建一个彩色 overlay（例如红色高亮）
    overlay = rgb.copy()  # 在副本上操作
    # 设置 mask 区域为红色（注意：rgb 是 RGB 格式！）
    # 所以红色 = [255, 0, 0]
    overlay[seg_mask] = [255, 0, 0]  # R=255, G=0, B=0 → 纯红
    # 3. （推荐）使用半透明融合，更美观
    alpha = 0.5  # 透明度：0~1，越小越透明
    output = rgb.copy()
    output[seg_mask] = (alpha * np.array([255, 0, 0]) + (1 - alpha) * output[seg_mask]).astype(np.uint8)
    # 4. 如果要用 OpenCV 显示，需转为 BGR
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # 5. 显示
    cv2.imshow("Segmentation Overlay", output_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Add parameters to Points_fusion_tool
    cam2world_para = torch.tensor(cam2world).view(4,4).to(device).to(torch.float)
    camera_intrinsic_para = torch.tensor(intrinsic_cv).view(3,3).to(device).to(torch.float)
    # pcd_fusion_tool.cam2world.append(cam2world_para.clone())
    # pcd_fusion_tool.intrinsic.append(camera_intrinsic_para.clone())
    # pcd_fusion_tool.mask.append(segmentation.clone())

    # Acquire the pointcloud in world frame
    depth_tensor = torch.tensor(depth).squeeze().to(device).clone()
    points_wld_frame = calculate_object_pointcloud_wld(depth = depth_tensor,intrinsic_cv = camera_intrinsic_para, 
                                    cam2world = cam2world_para, mask = segmentation)
    # # Filter the points below a hight; to be determined
    # mask = points_wld_frame[:, 2] >= 0.02
    # filtered_wld_frame_points = points_wld_frame[mask]

    return points_wld_frame.cpu().numpy()
