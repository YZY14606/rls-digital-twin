import os
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.spatial import KDTree
import glob
from pathlib import Path
import random
from transforms3d import quaternions
from utils.logging_utils import Logger, LogLevel, log_function
from utils.visualization_utils import VisualizationUtils
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
from tqdm import tqdm
from multiprocessing import Pool


class ObjectCentricPushDataset(Dataset):
    """
    Object-centric dataset for push sampling training.
    Loads object meshes and trajectories, and dynamically samples point clouds
    with contact points and push directions.
    """

    @log_function(level=LogLevel.INFO)
    def __init__(self, data_dir,num_process, logger_level= LogLevel.INFO, debug_dir="debug"):
        """
        Args:
            data_dir: Directory containing trajectory data
            mesh_path: Path to the object mesh file
            num_points: Number of points to sample from the mesh
            debug_dir: Directory to save debug visualizations
        """
        self.logger = Logger.get_instance()
        self.logger.set_level(logger_level)
        # If data_dir is string, convert it to list
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dir = data_dir
        self.debug_dir = debug_dir

        # Create debug directory if it doesn't exist and we're in debug mode
        if self.logger.logger.level <= LogLevel.DEBUG.value:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Find all trajectory data
        self.trajectories = self._load_trajectories(num_process)
        self.logger.info(f"Loaded {len(self.trajectories)} trajectories")

        # Create flattened list of (trajectory_idx, step_idx) pairs for indexing
        self.samples = len(self.trajectories)

    @log_function(level=LogLevel.DEBUG)
    def _load_trajectories(self,num_process):
        """Load all trajectory data from data directory"""
        # Find all trajectory folders
        if len(self.data_dir) ==1:
            scene_dir = Path(self.data_dir[0])
        else:
            print("The data is wrong!")

        # traj_paths = []
        # # 使用 glob 递归匹配所有 traj_* 目录
        # for scene in scene_dir.iterdir():
        #     if scene.is_dir() and "_" in scene.name and scene.name.replace("_", "").replace(".", "").isdigit():
        #         for traj in scene.iterdir():
        #             if traj.is_dir() and traj.name.startswith("traj_"):
        #                 traj_paths.append(traj)
        # # Filter traj_i（i is number）
        # traj_dirs1 = [p for p in traj_paths if p.name[5:].isdigit()]
        traj_dirs1 = [f for f in scene_dir.glob("traj_*") if f.is_dir()] #yzy
        # sort by traj_i
        traj_dirs1.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))
        all_trajectory_folders = traj_dirs1

        self.logger.info(
            f"Found {len(all_trajectory_folders)} trajectory folders in {self.data_dir}"
        )

        if len(all_trajectory_folders) == 0:
            self.logger.error(f"No trajectory folders found in {self.data_dir}")
            # You might want to check if the path exists
            if not os.path.exists(self.data_dir[0]):
                self.logger.error(f"Data directory {self.data_dir} does not exist!")
            return []

        trajectories = []

        with Pool(processes = num_process) as pool:
            with tqdm(total=len(all_trajectory_folders), desc="Load folders") as pbar:
                for result in pool.imap(self._load_single_trajectory, all_trajectory_folders):
                    trajectories.append(result)
                    pbar.update(1)


        return trajectories
    
    @log_function(level=LogLevel.DEBUG)
    def _load_single_trajectory(self,folder):
        try:
            # Define paths for required data files
            local_data_path = os.path.join(
                folder, "local_data"
            )

            # Check if required files exist
            if not os.path.exists(local_data_path):
                self.logger.warning(
                    f"Object trajectory file not found at {local_data_path}, skipping..."
                )
                return {}


            # Load the local_frame_pose and pointcloud
            for local_folder_name in os.listdir(local_data_path):
                local_folder_path = os.path.join(local_data_path, local_folder_name)

                push_distance_path = os.path.join(local_folder_path, "push_distance_normalized.npy")
                # Load push_distance
                push_distance = np.load(push_distance_path)
                if isinstance(push_distance, np.ndarray) and push_distance > 0:
                    push_distance_factor = push_distance
                else:
                    self.logger.warning(
                        f"Could not load push_distance properly from {push_distance_path}, using [1]"
                    )
                    push_distance_factor = np.ones(1)

                #Get the local_pose path
                local_frame_pose_path = os.path.join(local_folder_path, "local_frame_pose.npy")
                # Get the future_local_pose path
                future_local_pose_path = os.path.join(local_folder_path,"local_frame_future_pose_normalized.npy")

                # Load local frame pose
                local_frame_pose = np.load(local_frame_pose_path)
                # Load future_local_pose
                future_local_pose = np.load(future_local_pose_path)

                # Get the current_point_cloud path
                current_pcd_path = os.path.join(local_folder_path,"current_pointcloud_local_normalized.npy")
                # Load object current and future pointcloud
                current_pointcloud = np.load(current_pcd_path)

                # Get the contact quality
                contact_quality_path = os.path.join(local_folder_path,"contact_quality.npy")
                contact_quality = np.load(contact_quality_path)

                # Get the push direction
                push_direction_path = os.path.join(local_folder_path,"push_direction_local_frame.npy")
                direction = np.load(push_direction_path)

                # Add trajectory data
                trajectory = {
                    "folder" :folder,
                    "current_point_cloud": current_pointcloud,
                    "future_local_pose": future_local_pose,
                    "local_frame_pose":local_frame_pose,
                    "contact_quality": contact_quality,
                    "direction": direction,
                    "push_distance":push_distance_factor,
                }

                # Only add if we have at least one contact frame
                return trajectory

        except Exception as e:
            self.logger.error(f"Error loading trajectory from {folder}: {e}")


    @log_function(level=LogLevel.DEBUG)
    def __len__(self):
        """Return the total number of samples"""
        return self.samples

    @log_function(level=LogLevel.DEBUG)
    def transform_to_object_frame(self,points, obj_position, obj_quaternion):
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
        local_quaternion = np.array([obj_quaternion[1],obj_quaternion[2],obj_quaternion[3],obj_quaternion[0]])
        # 构造世界→局部坐标的旋转矩阵
        rot_matrix = R.from_quat(local_quaternion).as_matrix()  # shape (3, 3)

        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = obj_position[:3]

        # 计算相对变换: T_final = inv(T)
        T_final = np.linalg.inv(T)
        points_obj_frame = trimesh.transform_points(points, T_final)  # 使用 trimesh 的变换函数

        return points_obj_frame

    @log_function(level=LogLevel.DEBUG)
    def transform_direction_to_object_frame(self, direction, obj_quaternion):
        """
        Transform a direction vector from world frame to object frame

        Args:
            direction: Direction vector in world frame (3,)
            obj_quaternion: Object orientation in world frame [w, x, y, z]

        Returns:
            Direction vector in object frame (3,)
        """
        # Convert quaternion to rotation matrix
        R_obj_to_wld = quaternions.quat2mat(obj_quaternion)

        # Rotate direction vector
        direction_obj_frame = R_obj_to_wld.T @ direction

        return direction_obj_frame

    @log_function(level=LogLevel.DEBUG)
    def find_nearest_points(self, object_points, contact_points, k=10):
        """
        Find k nearest points on object to each contact point

        Args:
            object_points: Points on the object (N, 3)
            contact_points: Contact points (M, 3)
            k: Number of nearest neighbors to find

        Returns:
            indices: Indices of the nearest points for each contact point (M*k,)
        """
        # Build KD-tree for fast nearest neighbor lookup
        tree = KDTree(object_points)

        # Find k nearest neighbors for each contact point
        _, indices = tree.query(contact_points, k=k)

        # Flatten and get unique indices
        return np.unique(indices.flatten())

    @log_function(level=LogLevel.DEBUG)
    def __getitem__(self, idx):
        """Get a single training sample"""
        # Get trajectory index
        traj_idx = idx
        trajectory = self.trajectories[traj_idx]

        #Get the push_distance from this trajectory
        push_distance = trajectory["push_distance"]

        # 1. Get the contact quality
        quality = trajectory["contact_quality"]
        contact_indices = np.where(quality == 1)[0]

        # 3.Get the current and future point cloud
        current_pc = trajectory["current_point_cloud"]
        future_local_pose = trajectory["future_local_pose"]

        # 7. Get push direction to object frame
        push_direction_obj_frame = trajectory["direction"]


        # Normalize push direction
        push_dir_norm = np.linalg.norm(push_direction_obj_frame)
        if push_dir_norm > 1e-6:
            push_direction_obj_frame = push_direction_obj_frame / push_dir_norm
        else:
            # If near-zero direction, use a default direction
            push_direction_obj_frame = np.array([0, 0, 1])
            self.logger.warning(
                f"Zero-length push direction in sample traj {traj_idx}"
            )

        # 8. Create orientation tensor for push direction
        orientation = torch.zeros((len(current_pc), 3), dtype=torch.float)
        all_directions = np.zeros((len(current_pc), 3), dtype=np.float32)
        for idx in contact_indices:
            if idx < len(current_pc):  # Safety check
                orientation[idx] = torch.tensor(
                    push_direction_obj_frame, dtype=torch.float
                )
                all_directions[idx] = push_direction_obj_frame

        # Normalize orientations
        orientation = F.normalize(orientation, p=2, dim=1)

        # Set push_distance for every point
        push_distances = torch.zeros((len(current_pc),1), dtype=torch.float)
        if np.ndim(push_distance) == 0:
            push_distances[:] = push_distance.item()
        else:
            self.logger.warning(
                "Push_distance shape isn't (), data load error!"
            )

        # 12. Create visualizations if in debug mode
        if self.logger.logger.level <= LogLevel.DEBUG.value:

            folder = trajectory["folder"]

            # Get the world frame current pointcloud
            current_pc_wprld_path = os.path.join(folder,"world_obj_pcd","obj_pcd_before.npy")
            world_current_pc = np.load(current_pc_wprld_path)

            # 获得future_pc
            fut_pose = future_local_pose.copy()
            fut_pose[3:7] = np.array([fut_pose[4],fut_pose[5],fut_pose[6],fut_pose[3]])
            rot_matrix = R.from_quat(fut_pose[3:7]).as_matrix()  # 3x3
            T2 = np.eye(4)
            T2[:3, :3] = rot_matrix
            T2[:3, 3] = fut_pose[:3]
            # 计算相对变换: T_final = T2 * inv(T1)
            T_final = T2
            future_pc = trimesh.transform_points(current_pc, T_final)  # 使用 trimesh 的变换函数

            # Load the contact points in world frame
            contact_points_dir = os.path.join(folder,"contact_points")
            contact_files = sorted(glob.glob(os.path.join(contact_points_dir, "*contact_points.ply")),
                    key=lambda path: int("".join(filter(str.isdigit, os.path.basename(path).split("contact_points")[0])))
            )
            # Load contact points
            contact_points_mesh = trimesh.load(contact_files[0])
            world_contact_points = np.asarray(contact_points_mesh.vertices)


            # Create one-hot mask for contact points (1 for contact, 0 for non-contact)
            contact_mask = np.zeros(len(current_pc), dtype=bool)
            for idx in contact_indices:
                if idx < len(current_pc):  # Safety check
                    contact_mask[idx] = True


            # Create comprehensive visualizations
            VisualizationUtils.visualize_object_centric_sample(
                current_pc=current_pc,
                contact_mask=contact_mask,
                directions=all_directions,
                future_pc=future_pc,
                world_pc=world_current_pc,
                world_contacts=world_contact_points,
                traj_idx=traj_idx,
                frame_idx=0,
                debug_dir=self.debug_dir,
                arrow_length=np.array(push_distances),
            )

        return {
            "current_pc": torch.tensor(current_pc, dtype=torch.float),
            "future_local_pose": torch.tensor(future_local_pose, dtype=torch.float),
            "quality": torch.tensor(quality),
            "orientation": orientation,
            "push_distance":push_distances,
        }


def main():
    dataset = ObjectCentricPushDataset(data_dir= ['/home/yzy/nus/git_rop/Object-centric-Manipulation-Policy/Data_generation/scene_data_100/object_6/20251030_230823.0'
                                                    ], 
                                        num_process= 1,
                                       logger_level= LogLevel.DEBUG,
                                       debug_dir= "debug")

    #查看数据数量
    print("数据集中最大采样索引：", len(dataset))

    for i in range(len(dataset)):
        results = dataset[i]


if __name__ == "__main__":

    main()