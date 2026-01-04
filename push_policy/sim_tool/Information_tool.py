import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import struct
import tf
import ros_numpy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from transforms3d.quaternions import quat2mat 
from transforms3d.euler import quat2euler,euler2mat
from cv_bridge import CvBridge
import cv2


class  SimController:
    def __init__(self,):

        self.control_frequency = 10


    def get_pointcloud(self,topic='/camera_1/depth/points', camera_name = 'camera_1'):
        """
        等待并获取一帧点云数据，并返回 numpy 数组 (N, 3)
        """
        msg = rospy.wait_for_message(topic, PointCloud2)

        # 获取带 RGB 的点
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        points = []
        for x, y, z, rgb in gen:
            # 先将 float 转换为 int
            rgb_uint32 = struct.unpack('I', struct.pack('f', rgb))[0]
            # 提取颜色通道
            b = (rgb_uint32 >> 16) & 0x0000ff
            g = (rgb_uint32 >> 8) & 0x0000ff
            r = (rgb_uint32) & 0x0000ff
            points.append([x, y, z, r, g, b])
        points_with_color = np.array(points)

        # Get the pointcloud in world frame
        # Get the pose of the camera_1 in world frame
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        response = get_model_state(camera_name, '')  # 第二个参数为空表示 world frame
        position = response.pose.position
        orientation = response.pose.orientation # qx,qy,qz,qw
        # Transform camrea_1 pose to normal
        quat_ori = np.array([orientation.w,orientation.x,orientation.y,orientation.z])
        mat_ori = quat2mat(quat_ori)
        T = np.eye(4)
        T[:3,:3] = mat_ori
        T[:3,3] = np.array([position.x,position.y,position.z])
        T_gazebo_to_cv = np.array([
                [0,  0, 1, 0],
                [-1,  0, 0, 0],
                [0, -1, 0, 0],
                [0,  0, 0, 1],])
        T = T @ T_gazebo_to_cv
        # Transform points from camera frame to world frame 
        points_xyz_camera = points_with_color[:,:3]
        ones_column = np.ones((points_xyz_camera.shape[0], 1))
        points_augmented = np.hstack([points_xyz_camera, ones_column])
        points_xyz_world = (T @ points_augmented.T).T[:,:3]
        all_pointcloud = np.hstack([points_xyz_world,points_with_color[:,3:6]])

        return all_pointcloud


    def get_rgb_depth(self,rgb_topic='/camera_1/rgb/image_raw',depth_topic='/camera_1/depth/image_raw',
                      intrinsic_topic = None, camera_pose = None):
        """
        同步等待一帧 RGB 和 Depth 图像，返回两个 numpy 数组：
        - rgb: (H, W, 3)，uint8
        - depth: (H, W)，float32 或 uint16，单位可能是米或毫米，取决于相机
        """
        # Wait for the message of image
        rgb_msg = rospy.wait_for_message(rgb_topic, Image)
        depth_msg = rospy.wait_for_message(depth_topic, Image)
        intrinsic_msg = rospy.wait_for_message(intrinsic_topic, CameraInfo)

        # 转换 RGB 图像（ros_numpy 输出为 RGB 格式，如需 BGR 需转换）
        rgb = ros_numpy.numpify(rgb_msg)  # shape: [H, W, 3]，rgb8

        # 如果你需要显示图像，可能需要将RGB转换为BGR，因为OpenCV默认使用BGR格式
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # 显示图像
        cv2.imshow('RGB Image', rgb_bgr)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()

        # 转换深度图像（'32FC1' 编码）
        depth = ros_numpy.numpify(depth_msg)  # shape: [H, W], dtpye=float32
        # # 归一化处理，以适应显示
        # depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # # 显示图像
        # cv2.imshow('Depth Image', depth_display)
        # cv2.waitKey(0)  # 按任意键关闭窗口
        # cv2.destroyAllWindows()

        # Get the extrinsic_cv of the camera_1 in world frame
        position = camera_pose[:3,3]
        orientation_mat = camera_pose[:3,:3]

        cam2world = np.eye(4)
        cam2world[:3,:3] = orientation_mat
        cam2world[:3,3] = position
        # T_gazebo_to_cv = np.array([
        #         [0,  0, 1, 0],
        #         [-1,  0, 0, 0],
        #         [0, -1, 0, 0],
        #         [0,  0, 0, 1],])
        # cam2world = cam2world @ T_gazebo_to_cv

        # Get the intrinsic_cv 
        intrinsic_cv = np.array(intrinsic_msg.K, dtype=np.float64).reshape(3, 3)
        # print('intrinsic_cv:',intrinsic_cv)
        

        return rgb, depth , cam2world, intrinsic_cv



    def save_object_pose(self,object_name,path):
        state_now = self.get_state(object_name, "world")
        position = state_now.pose.position
        orientation = state_now.pose.orientation
        pose = np.array([position.x,position.y,position.z,orientation.w,orientation.x,orientation.y,orientation.z])
        np.save(path,pose)




if __name__ == "__main__":
    controller = SimController()  # 修改成你的小球模型名

