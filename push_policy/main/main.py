import numpy as np
from predictor.contact_predictor import contact_predictor
import os
from transforms3d.euler import euler2quat, mat2euler
from main_utils import (process_object_pointcloud,build_local_frame,First_path_plan,plan_from_waypoint,get_future_pose,get_current_pc_future_pose,
                        two_perp_poses_radius2,generate_action,convert_action_use,evaluate_complete_action,test_fetch_motion_plan,move_base_to_target,
                        collect_and_segmented_pcd,test_cartesian_interpolated_motion,pointcloud_segmentation_fusion_by_video,convert_source_pcd_to_current)
from pose_estimate.pose_estimator import Pose_Estimator
import torch
from sim_tool.Information_tool import SimController
from fetch.fetch import Fetch
from pathlib import Path
import rospy
import open3d as o3d
from segmentation.segmentation import Segmentation_tool,Video_sam_tool, Points_fusion_tool


def main():
    # Set target information
    target_xyz = np.array([3.0,2.0,0.75])
    target_quat = np.array(euler2quat(0,0,np.pi/2))
    target_pose = np.concatenate([target_xyz,target_quat],axis=0)

    # Set object original pose (It should be same with the real world pose.)
    original_pose = np.array([-3.0,1.5,0.732,1,0,0,0])

    # Set the scene parameters for object path planning
    scene_args = {}
    scene_args['robot_radius'] = 0.3 # It can be set larger than real robot.
    scene_args['table_height'] = 0.74 # It is the real height of the table.

    # Set the single push parameter
    push_step_dict = dict()
    push_step_dict['single_push_len'] = 0.07
    push_step_dict['single_push_theta'] = 0.5236  # pi/6

    # Set the robot_theta_y
    robot_theta_y = np.pi/6

    # Define a controller to manage simulation information
    simulation_control = SimController()

    # Define a contact point predictor
    point_predictor = contact_predictor()

    # Initialize the fetch controller
    current_file = Path(__file__).resolve()  # resolve() 确保是绝对路径并解析符号链接
    # 获取上三级目录
    parent_3 = current_file.parent.parent.parent
    urdf_path= parent_3 / "resources/fetch_ext/fetch ext.urdf"
    costmap_path = parent_3 / "resources/costmap.npz"
    fetch = Fetch(urdf_path = urdf_path,costmap_path=costmap_path)

    # define the image segmentation tool
    image_seg_tool = Segmentation_tool()

    # define the video segmentation tool
    video_seg_tool = Video_sam_tool()

    # Define a pointcloud fusion tool
    pcd_fusion_tool =  Points_fusion_tool()

    # Record the times of pushing
    itr_plan = 0
    traj_id = '0'
    file_name = 'test_yzy'

    # 进行机器人初始的信息采集(从三个视角获取物体与环境点云),以及设定初始操作pose
    robot_base_pose = {}
    robot_base_pose[0] = [-2,0.5,np.pi*2/3]
    robot_base_pose[1] = [-4,0.5,np.pi/3]
    robot_base_pose[2] = [-3,3,-np.pi/2]
    robot_base_pose[3] = [-3.0,0.5,np.pi/2] # The manipulation pose.


    # 注释掉，为了方便后续的调试
    object_pcd_list = []
    obstacle_pcd_list = []
    for key in range(3): 
        target_base_pose = robot_base_pose[key]
        position = [target_base_pose[0],target_base_pose[1],0]
        orientation_wxyz = euler2quat(0,0,target_base_pose[2])
        orientation_xyzw = [orientation_wxyz[1],orientation_wxyz[2],orientation_wxyz[3],orientation_wxyz[0]]
        fetch.send_target_position(position,orientation_xyzw)
        rospy.sleep(0.5)  # Wait between movements
        fetch.move_head(pan = 0.0, tilt = 0.0, duration=1.0)

        # If use whole body controller
        # move_base_to_target(fetch=fetch,target_base_pose=target_base_pose)

        rospy.sleep(0.5)  # Wait between movements
        topic = ['/head_camera/rgb/image_raw','/head_camera/depth_registered/image_raw','/head_camera/rgb/camera_info']
        obj_pcd_wld_frame, obstacle_pcd_wld_frame = collect_and_segmented_pcd(simulation_control = simulation_control,topic = topic,fetch = fetch,
                                                                              image_seg_tool = image_seg_tool, pcd_fusion_tool = pcd_fusion_tool)
        object_pcd_list.append(obj_pcd_wld_frame)
        obstacle_pcd_list.append(obstacle_pcd_wld_frame)

    object_pcds = np.concatenate(object_pcd_list,axis=0)
    obstacle_pcds = np.concatenate(obstacle_pcd_list,axis=0)

    # Visulize the objectpoints and obstacle pointcloud
    # print(object_pcds.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(object_pcds)
    # o3d.visualization.draw_geometries([pcd])
    # print("Point cloud bounds:")
    # print("Min:", object_pcds.min(axis=0))
    # print("Max:", object_pcds.max(axis=0))
    # print("Center:", object_pcds.mean(axis=0))

    # print(obstacle_pcds.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(obstacle_pcds)
    # o3d.visualization.draw_geometries([pcd])
    # print("Point cloud bounds:")
    # print("Min:", obstacle_pcds.min(axis=0))
    # print("Max:", obstacle_pcds.max(axis=0))
    # print("Center:", obstacle_pcds.mean(axis=0))

    # Move fetch to manipulation pose
    manipulation_pose = robot_base_pose[3]
    position = [manipulation_pose[0],manipulation_pose[1],0]
    orientation_wxyz = euler2quat(0,0,manipulation_pose[2])
    orientation_xyzw = [orientation_wxyz[1],orientation_wxyz[2],orientation_wxyz[3],orientation_wxyz[0]]
    fetch.send_target_position(position,orientation_xyzw)
    rospy.sleep(0.5)  # Wait between movements
    fetch.move_head(pan = 0.0, tilt = 0.0, duration=1.0)
    rospy.sleep(0.5)  # Wait between movem

    # # get object pointcloud for test: YZY
    # topic = ['/head_camera/rgb/image_raw','/head_camera/depth_registered/image_raw','/head_camera/rgb/camera_info']
    # object_pcds, obstacle_pcds = collect_and_segmented_pcd(simulation_control = simulation_control,topic = topic,fetch = fetch,
    #                                                                 image_seg_tool = image_seg_tool,pcd_fusion_tool = pcd_fusion_tool)
    
    object_wrld_frame_pcd = process_object_pointcloud(object_pcds,pcd_fusion_tool)
    obstacle_wrld_frame_pcd = process_object_pointcloud(obstacle_pcds,num_samples=4096)

    # Add obstacle pointcloud for fetch
    fetch.add_pointcloud(points = obstacle_wrld_frame_pcd)

    first_prompt = image_seg_tool.latest_prompt
    path_point_index = 1
    path_plan_state = 'Normal'
    # 进行25次循环，如果机器人在25个循环内完成目标，就算成功
    for i in range(25):
        itr_plan = itr_plan +1
        print(f"This is the {itr_plan} push.")

        # Record the original object's pointcloud and pose for final evaluation
        if itr_plan == 1:
            source_obj_pcd = object_wrld_frame_pcd.copy()
            source_obj_pose = original_pose.copy()


        # Construct local frame
        local_frame_pose = build_local_frame(object_wrld_frame_pcd.copy())

        if itr_plan == 1:
            # Define a object pose estimator
            pose_estimator = Pose_Estimator(source_pcd_wld = object_wrld_frame_pcd, ori_pose = original_pose.copy(),
                                            original_PCA_frame_pose = local_frame_pose)
            object_pose = torch.tensor(original_pose)
        else:
            object_pose = pose_estimator.estimate_object_pose(now_pointcloud = object_wrld_frame_pcd,local_frame_pose = local_frame_pose)
            object_pose = torch.tensor(object_pose)

            # Get the registered pcd
            object_wrld_frame_pcd = convert_source_pcd_to_current(source_pcd = source_obj_pcd,source_pose = source_obj_pose,current_pose = object_pose)


        loop_times = 0
        while True:
            loop_times += 1
            # Path plan
            if itr_plan == 1:
                # Get the robot base position
                robot_base_position = fetch.get_base_params()
                scene_args['base_pos'] = np.array([robot_base_position[0],robot_base_position[1]])

                path_points,path_planner = First_path_plan(target_pose,object_pose.clone(),object_wrld_frame_pcd.copy(),obstacle_wrld_frame_pcd,scene_args)
            if path_point_index == 2:
                # Get the robot base position
                robot_base_position = fetch.get_base_params()
                scene_args['base_pos'] = np.array([robot_base_position[0],robot_base_position[1]])

                path_points,path_planner = plan_from_waypoint(target_pose,object_pose.clone(),path_planner,obstacle_wrld_frame_pcd,scene_args)
                print("The object has arrived at the middle waypoint. Plan path again.")
                path_point_index = 1

            if len(path_points) <= 1 and loop_times <= 4:
                path_point_index = 2
                continue

            if  loop_times > 4:
                path_plan_state = 'Error'
                break

            # Test the path plan result
            future_pose, path_point_index = get_future_pose(path_points,object_pose,path_point_index,error_radius = 0.02,
                                                        target_quat = target_quat,push_step_dict = push_step_dict)
            # 得到物体坐标系下的当前点云和未来点云
            current_pc,self_frame_future_pose = get_current_pc_future_pose(object_wrld_frame_pcd, object_pose, future_pose)
            if loop_times == 3:
                pose1, pose2 = two_perp_poses_radius2(self_frame_future_pose)
                self_frame_future_pose = pose1
            if loop_times == 4:
                self_frame_future_pose = pose2
            #获取最佳动作的参数
            point_predictor.vis = True
            contact_point, orientation, push_distance = point_predictor.predict(current_pc,self_frame_future_pose,itr_push = itr_plan,traj_id = traj_id,time_name=file_name)
            #根据参数，将其转化为一个完整的action
            ps_pose, pe_pose = generate_action(contact_point, orientation, push_distance,object_pose)
            #对于不能仅考虑2d的push物体，进行pose的变换
            ps_pose, pe_pose = convert_action_use(ps_pose,pe_pose,robot_theta_y)

            # Test this path point with motion planning and path planning
            # 让机械臂移动到ps
            # Record bi-level plan results
            bi_level_plan_results = []
            # Get robot's current state
            current_joints = fetch.get_current_planning_joints()
            current_base = fetch.get_base_params()
            current_state = dict()
            current_state['current_joints'] = current_joints
            current_state['current_base'] = current_base
            now_current_state,result = test_fetch_motion_plan(fetch = fetch, current_state = current_state, target_pose = ps_pose)

            bi_level_plan_results.append(result)
            if result == None:
                path_point_index = 2
                continue
            # 让机械臂移动到pe
            now_current_state,result = test_cartesian_interpolated_motion(fetch= fetch, current_state = now_current_state,target_ee_pose = pe_pose)
            bi_level_plan_results.append(result)
            if result == None:
                path_point_index = 2
                continue
            #将机械臂返回ps处，便于下个循环
            now_current_state,result = test_fetch_motion_plan(fetch = fetch, current_state = now_current_state, target_pose = ps_pose)
            bi_level_plan_results.append(result)
            if result == None:
                path_point_index = 2
                continue
            else:
                break

        if path_plan_state == 'Error':
            print("Path plan fail. There is no path to psuh object successfully!")
            break


        # 首先让机械臂移动到ps
        fetch.collect_camera_data = True
        state = fetch.move_to_pose(target_pose = ps_pose)
        if not state:
            # If plan fails, execute the bi-level planning result.
            result_0 = bi_level_plan_results[0]
            state = fetch.execute_whole_body_motion(
            result_0["arm_path"], result_0["base_configs"]
        )

        # 让机械臂移动到pe
        state = fetch.send_cartesian_interpolated_motion(target_ee_pose = pe_pose)
        if not state:
            # If plan fails, execute the bi-level planning result.
            result_1 = bi_level_plan_results[1]
            joint_trajectory = result_1
            state = fetch.execute_joint_trajectory(joint_trajectory, duration = 3.0)


        # 让机械臂移动到ps
        state = fetch.move_to_pose(target_pose = ps_pose)
        if not state:
            # If plan fails, execute the bi-level planning result.
            result_2 = bi_level_plan_results[2]
            state = fetch.execute_whole_body_motion(
            result_2["arm_path"], result_2["base_configs"]
        )

        # Stop the automatically iamge collection
        fetch.collect_camera_data = False

        # 进行收集的图像的处理
        rgb_list = fetch.rgb_list
        depth_list = fetch.depth_list
        intrinsics_list = fetch.intrinsics_list
        camera_pose_list = fetch.camera_pose_list
        # Clear the image lists
        fetch.clear_images_info_list()

        # Get the object's pointcloud
        object_wrld_frame_pcd, obstacle_pcd_wld_frame = pointcloud_segmentation_fusion_by_video(rgb_list,depth_list,intrinsics_list,
                                                        camera_pose_list,video_seg_tool,itr_plan,first_prompt,source_obj_pcd)

        # Judge the success
        success_judge = evaluate_complete_action(target_pose,source_obj_pcd,source_obj_pose,object_wrld_frame_pcd.copy())
        # 判断是否成功
        if success_judge == True:
            print("Success!")
            break


    print(f"This trial uses {itr_plan} steps.")

    return 0



if __name__ == '__main__':
    main()