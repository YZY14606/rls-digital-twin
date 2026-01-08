import cv2
import os
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
import matplotlib.pyplot as plt
from segmentation.seg_utils import PointSelector


class Segmentation_tool():
    def __init__(self,mask_generator_type = "image"):

        self.latest_prompt = {}

        self.sam2_model = self._load_predictor()
        if mask_generator_type == "image":
            self.predictor = SAM2ImagePredictor(self.sam2_model)
        else: 
            print(f"Error: cannot find the {mask_generator_type} type mask_generator!")

    def _load_predictor(self):
        # Get the path of trained model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        checkpoint = os.path.join(parent_dir,"third_part/sam2/checkpoints/sam2.1_hiera_large.pt")

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        sam2_model = build_sam2(model_cfg, checkpoint,device,apply_postprocessing=False)
        return sam2_model
    
    def set_image(self,image):
        self.predictor.set_image(image)

    def predict(self,points,labels):
        """
        points: the prompt points, shape (N,2)
        labels: the labels of points, shape (N)
        """
        masks, _, _ = self.predictor.predict(points,labels,multimask_output=False)
        return masks
    
    def click_set_image_prompt(self,rgb):

        point_selector = PointSelector(rgb = rgb)
        points_pos, point_labels = point_selector.select_points()

        points = np.array(points_pos)
        labels = np.array(point_labels)

        self.latest_prompt['points'] = points
        self.latest_prompt['labels'] = labels

    def segment_image_by_click(self,image_rgb):

        self.set_image(image_rgb)
        self.click_set_image_prompt(image_rgb)
        masks = self.predict(points=self.latest_prompt['points'],labels = self.latest_prompt['labels']) # shape (1,H,W)

        return masks



class Video_sam_tool():
    def __init__(self,mask_generator_type = "video"):

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        self.video_dir = None
        self.inference_state = None
        self.ann_obj_id = 1

        self.latest_prompt = None

        if mask_generator_type == "video":
            self.predictor = self._load_predictor()
        else: 
            print(f"Error: cannot find the {mask_generator_type} type mask_generator!")

    def _load_predictor(self):
        # Get the path of trained model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sam2_checkpoint = os.path.join(parent_dir,"third_part/sam2/checkpoints/sam2.1_hiera_large.pt")

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device = self.device)
        return predictor
    
    def init_model_state(self,video_dir):
        self.video_dir = video_dir
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)

    def click_set_video_prompt(self,frame_id = 0):

        if self.video_dir == None:
            print('Error in video dir!')
            return None

        # Get the image path 
        image_path = os.path.join(self.video_dir,str(frame_id))
        point_selector = PointSelector(image_path = image_path)
        points_pos, point_labels = point_selector.select_points()
        points = np.array(points_pos)
        labels = np.array(point_labels)
        _, object_ids, masks = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_id,
            obj_id=self.ann_obj_id,
            points=points,
            labels=labels,
            )
            
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments
    

    def get_mask_from_video(self,prompt,frame_id =0):

        points = prompt['points']
        labels = prompt['labels']
        _, object_ids, masks = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_id,
            obj_id=self.ann_obj_id,
            points=points,
            labels=labels,
            )
            
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Get the latest mask
        last_frame_idx = max(video_segments.keys())
        last_frame_obj_masks = video_segments[last_frame_idx][self.ann_obj_id]

        # Sample from the latest mask for next prompt
        self._sample_from_mask_as_prompt(last_frame_obj_masks)


        return last_frame_obj_masks

    def _sample_from_mask_as_prompt(self,last_frame_obj_masks):
        # last_frame_obj_masks 可能是 (1,H,W) 或 (H,W)
        mask = last_frame_obj_masks.copy()
        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.astype(bool)  # (H,W)

        # 正点：mask内
        ys_pos, xs_pos = np.where(mask)
        # 负点：mask外
        ys_neg, xs_neg = np.where(~mask)

        # 简单随机抽样（不足则允许重复）
        rng = np.random.default_rng()

        pos_n = min(3, xs_pos.size)
        neg_n = min(3, xs_neg.size)

        if xs_pos.size == 0:
            raise ValueError("mask 为空，无法选正点")
        if xs_neg.size == 0:
            raise ValueError("mask 覆盖全图，无法选负点")

        pos_idx = rng.choice(xs_pos.size, size=3, replace=(xs_pos.size < 3))
        neg_idx = rng.choice(xs_neg.size, size=3, replace=(xs_neg.size < 3))

        points_pos = [(int(xs_pos[i]), int(ys_pos[i])) for i in pos_idx]
        points_neg = [(int(xs_neg[i]), int(ys_neg[i])) for i in neg_idx]

        points_pos = points_pos + points_neg
        point_labels = [1, 1, 1, 0, 0, 0]

        points = np.array(points_pos, dtype=np.float32)   # shape (6,2), (x,y)
        labels = np.array(point_labels, dtype=np.int32)   # shape (6,)

        self.latest_prompt['points'] = points
        self.latest_prompt['labels'] = labels




class Points_fusion_tool():

    def __init__(self):
        self.extrinsic = []
        self.intrinsic = []
        self.mask = []


    def batch_project_to_2d(self, points_3d, intrinsic, world2cam):
        """
        Batch projects 3D points onto the 2D camera plane.
        """
        # Transform points using camera pose
        points_transformed = (world2cam[:3, :3] @ points_3d.T).T + world2cam[:3, 3]

        # Project points using the camera intrinsic matrix
        points_projected = (intrinsic[:3, :3] @ points_transformed.T).T
        points_2d = points_projected[:, :2] / points_projected[:, 2:]

        return points_2d
    
    def batch_filter_points(self, points_2d, mask):
        """
        Filters the points based on the mask in a batch operation.
        """
        x, y = points_2d.T
        valid_indices = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])

        # Create an array to hold the final results
        final_validity = torch.zeros_like(valid_indices, dtype=torch.bool)

        # Ensure indices are integers for mask indexing
        x_int = x[valid_indices].long()
        y_int = y[valid_indices].long()

        # Get mask values at the projected 2D points
        mask_values = mask[y_int, x_int]

        # Update the final_validity array for valid indices
        final_validity[valid_indices] = mask_values != 0

        return final_validity

    def filter_point(self, points,threshold=0.2):

        valid_points = torch.zeros(points.shape[0], dtype=int).to("cuda")

        # Process each camera pose and update the valid_points array
        for i, extrinsic in enumerate(self.extrinsic):
            # Project all points onto the 2D camera plane
            points_2d = self.batch_project_to_2d(
                points, self.intrinsic[i], extrinsic
            )

            # Update the valid_points array
            valid_points += self.batch_filter_points(points_2d, self.mask[i])

        # Filter the points based on the valid_points array
        valid_points[valid_points <= len(self.extrinsic) // 2] = 0
        valid_points = valid_points.bool()
        filtered_points = points[valid_points]

        # Filter the outlier points for our object
        centroid_xy = filtered_points[:, :2].mean(dim=0)
        distances = torch.norm(filtered_points[:, :2] - centroid_xy, dim=1)
        mask = distances <= threshold
        second_filtered_points = filtered_points[mask]

        return second_filtered_points




def main():

    # Define a image segmenter
    img_seg_tool = Segmentation_tool()
    # 机器人捕获的图像
    image = cv2.imread("/home/yzy/图片/object/2025-04-29 11-05-57 的屏幕截图.png")  # 替换为实际摄像头输入
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image_rgb.shape)

    # Generate mask
    img_seg_tool.set_image(image_rgb)
    masks = img_seg_tool.predict(points=np.array([[128,128],[256,256]]),labels = np.array([1,1]))

    # Save image
    arr = masks[0]  # 移除第一个维度，变为 (N, N)
    # 将0/1转换为0/255（黑白图像）
    binary_image = (arr * 255).astype(np.uint8)
    # 保存图像
    cv2.imwrite('binary_image.png', binary_image)



if __name__ == "__main__":
    main()

