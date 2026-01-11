import argparse
import os
from predictor import config as args
from predictor.predictor_utils import visulize_pred_results
import torch
import open3d as o3d
import numpy as np
from utils.logging_utils import Logger, LogLevel, log_function
from network.new_NN_model import PointCloudEncoderDecoder
from utils.visualization_utils import VisualizationUtils
from scipy.spatial.transform import Rotation as R #(x,y,z,w)
import trimesh
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from shapely.geometry import Point, LineString, Polygon
from scipy.special import expit


class contact_predictor:

    def __init__(self):
        # Setup logging
        self.logger = Logger.get_instance()
        self.logger.set_level(LogLevel.INFO)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        model_path = 'checkpoints/model_100_seed_66_wd1_ds1_set_1_repeat_1'
        model_name = "checkpoint_epoch_149.pth"
        self.model_path = os.path.join(model_path, model_name)
        # Set model
        self.model = self.set_model()

        # Visualize
        self.vis = args.visualization

    @log_function(level=LogLevel.DEBUG)
    def set_model(self):
        """Initialize model, loss function, optimizer, and scheduler"""
        self.logger.info("Initializing model")

        # Initialize model
        model = PointCloudEncoderDecoder(
            point_dim=args.point_dim,
            condition_dim=args.condition_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            ).to(self.device)

        # Load model state
        path = self.model_path
        if os.path.isfile(path):
            self.logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path,weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.logger.warning(f"No checkpoint found at {path}")

        return model.eval()
    

    @log_function(level=LogLevel.DEBUG)
    def predict(self,current_pc, local_frame_future_pose, time_name, traj_id,itr_push):
        # Normalize the current pointcloud
        furthest_distance = np.max(np.linalg.norm(current_pc, axis=1))
        current_pc_nrd = current_pc.copy() / furthest_distance.item()

        local_future_pose_nrd = local_frame_future_pose.copy()
        local_future_pose_nrd[:3] = local_frame_future_pose[:3] / furthest_distance.item()

        with torch.no_grad():
            # Convert pointcloud to input shape (B,N,3)
            input_current_pc = torch.tensor(current_pc_nrd,dtype=torch.float).view(1,-1,args.point_dim).to(self.device)
            input_local_frame_future_pose = torch.tensor(local_future_pose_nrd,dtype=torch.float).view(1,args.transformation_dim).to(self.device)

            # Forward pass
            pred_contact, pred_orientation, pred_push_distance = self.model(input_current_pc, input_local_frame_future_pose)

        # Convert tensors to numpy 
        normalized_orientation = F.normalize(pred_orientation, p=2, dim=-1)
        pred_orientation_np = normalized_orientation[0].cpu().detach().numpy()
        pred_push_distance_np = pred_push_distance[0].cpu().detach().numpy()

        # Predicted quality and orientation
        # Convert sigmoid probabilities to mask using threshold
        pred_contact_np = torch.sigmoid(pred_contact[0]).cpu().detach().numpy()
        # Get the max value index
        max_index = self.select_contact_point(current_pc_nrd.copy(),pred_contact_np.squeeze(),pred_orientation_np.copy())

        # Get the output shape
        output_contact = current_pc[max_index]
        output_orientation = pred_orientation_np[max_index]
        output_push_distance = pred_push_distance_np[max_index] * furthest_distance.item()

        # Visualize
        if self.vis:
            # 获得future_pc
            fut_pose = input_local_frame_future_pose.view(args.transformation_dim).clone().cpu().numpy()
            fut_pose[3:7] = np.array([fut_pose[4],fut_pose[5],fut_pose[6],fut_pose[3]])
            rot_matrix = R.from_quat(fut_pose[3:7]).as_matrix()  # 3x3
            T2 = np.eye(4)
            T2[:3, :3] = rot_matrix
            T2[:3, 3] = fut_pose[:3]
            # 计算相对变换: T_final = T2 * inv(T1)
            T_final = T2
            future_pc = trimesh.transform_points(current_pc_nrd, T_final)  # 使用 trimesh 的变换函数
            # Save path
            save_path = os.path.join(args.vis_dir,time_name,traj_id,str(itr_push)+"_push")
            print(save_path)
            self.vis_dir = visulize_pred_results(current_pc_np = current_pc_nrd,future_pc_np =future_pc,
                                  pred_contact_np = pred_contact_np, pred_orientation_np =pred_orientation_np, push_idx = max_index,
                                  pred_push_distance_np = pred_push_distance_np,save_path = save_path, logger = self.logger)
            

        return output_contact, output_orientation, output_push_distance
    

    def _self_modified_sigmoid(self,original_score):
        input_score = 5 * original_score
        final_score = (expit(input_score) - 0.5 ) * 2
        return final_score
 

    def select_contact_point(self,current_pc_nrd,pred_contact_np,pred_orientation_np):

        valid_indices = np.arange(current_pc_nrd.shape[0])
        valid_probs = pred_contact_np.copy()

        # 筛选概率 > 0.5 的点
        pos_mask = valid_probs > 0.5
        pos_idx = valid_indices[pos_mask]
        # print("The number of prob > 0.5:",len(pos_idx))

        if len(pos_idx) >= 10:
            result_idx = pos_idx
        else:
            # 概率从大到小排序，取前 10
            top10_rel = np.argsort(valid_probs)[-10:][::-1]    # 在 valid_probs 内部排序
            result_idx = valid_indices[top10_rel]

        z_all = current_pc_nrd[:, 2]
        mask = z_all < z_all.min() + 0.25 * (z_all.max() - z_all.min())
        xy_points = current_pc_nrd[mask, :2]
        # 2. 求凸包
        hull = ConvexHull(xy_points)
        hull_polygon = Polygon(xy_points[hull.vertices])
        # 3. 对每个 result_idx 计算 d/z
        final_scores = []
        for idx in result_idx:
            x, y, z = current_pc_nrd[idx]
            orientation_nrd = pred_orientation_np[idx][:2] / np.linalg.norm(pred_orientation_np[idx][:2])

            d_score = self.calculate_projection(hull_polygon,orientation_nrd)

            # 计算 score = d / z
            if z > z_all.min() + 0.02 * (z_all.max() - z_all.min()):
                score = d_score / z
                score = self._self_modified_sigmoid(original_score = score)
            else:
                score = 0 # 避免除零或负高
            actor_score = pred_contact_np[idx]
            final_score = actor_score * score

            final_scores.append(final_score)

            # print(f'final_score: {final_score}, actor_score: {actor_score}, score: {score}, z: {z}')

        # 4. 找到最大 score 对应的索引
        max_idx = result_idx[np.argmax(final_scores)]

        return max_idx
    
    def calculate_projection(self,hull_polygon,orientation_nrd):

        scores = []

        # 直接从 polygon.exterior.coords 获取边顶点序列（自动闭合）
        coords = np.array(hull_polygon.exterior.coords)
        for p1, p2 in zip(coords[:-1], coords[1:]):
            edge = p2 - p1
            n = np.array([edge[1], -edge[0]])  # 旋转90°得到法向量
            n /= np.linalg.norm(n)

            # 统一法向量方向（指向外侧）
            if np.dot(n, p1) < 0:
                n = -n

            proj = np.dot(orientation_nrd, n)
            if proj > 0:  # 只保留可见边
                dist = np.dot(n, p1)
                scores.append(dist/proj)

        if scores:
            score = min(scores)
            return score


    def noly_critic_select_point(self,current_pc_nrd,pred_orientation_np,local_frame_future_pose):
        valid_indices = np.arange(current_pc_nrd.shape[0])
        z = current_pc_nrd[:, 2]
        threshold = z.max() * 0.2     # 最大高度的 1/5
        valid_indices = np.where(z > threshold)[0]
        result_idx = valid_indices

        # Calculate the move direction of the object
        direction = local_frame_future_pose[:3] / np.linalg.norm(local_frame_future_pose[:3])

        z_all = current_pc_nrd[:, 2]
        mask = z_all < z_all.min() + 0.25 * (z_all.max() - z_all.min())
        xy_points = current_pc_nrd[mask, :2]
        # 2. 求凸包
        hull = ConvexHull(xy_points)
        hull_polygon = Polygon(xy_points[hull.vertices])
        # 3. 对每个 result_idx 计算 d/z
        final_scores = []
        for idx in result_idx:
            x, y, z = current_pc_nrd[idx]
            orientation_nrd = pred_orientation_np[idx][:2] / np.linalg.norm(pred_orientation_np[idx][:2])

            vector_dot = np.dot(direction,orientation_nrd)
            if vector_dot <=0 :
                # 夹角大于等于90度
                final_scores.append(0)
                continue

            d_score = self.calculate_projection(hull_polygon,orientation_nrd)

            # 计算 score = d / z
            if z > z_all.min() + 0.02 * (z_all.max() - z_all.min()):
                score = d_score / z
                score = self._self_modified_sigmoid(original_score = score)
            else:
                score = 0 # 避免除零或负高
            final_score = score

            final_scores.append(final_score)

        # 4. 找到最大 score 对应的索引
        max_idx = result_idx[np.argmax(final_scores)]

        return max_idx




if __name__=='__main__':
    import h5py

    folder = '/home/yzy/nus/git_rop/Object-centric-Manipulation-Policy/Data_generation/scene_data_100/object_6/20251030_230823.0/traj_66/local_data/3_data'

    current_pc_path = os.path.join(folder,'current_pointcloud_local_normalized.npy')
    current_pc = np.load(current_pc_path)

    local_frame_future_pose_path = os.path.join(folder,'local_frame_future_pose_normalized.npy')
    local_frame_future_pose = np.load(local_frame_future_pose_path)
    print(local_frame_future_pose)
    
    normalized_push_distance_path = os.path.join(folder,'push_distance_normalized.npy')
    normalized_push_distance = np.load(normalized_push_distance_path)
    print('gt_push_dist_normalized:',normalized_push_distance)

    # with h5py.File(folder, 'r') as h5f:
    #     current_pc = h5f["local_data"][f"{0}_data/current_pointcloud_local_normalized"][:]
    #     local_frame_future_pose = h5f["local_data"][f"{0}_data/local_frame_future_pose_normalized"][:]
    #     normalized_push_distance  = h5f["local_data"][f"{0}_data/push_distance_normalized"][()]
    #     print('gt_normalized_push_distance:',normalized_push_distance)

    our_predictor = contact_predictor()
    our_predictor.vis = True
    output_contact, output_orientation, output_push_distance = our_predictor.predict(current_pc,local_frame_future_pose,time_name = 'model_100', traj_id ='12',itr_push = 0)

    print('output_push_distance:',output_push_distance)
