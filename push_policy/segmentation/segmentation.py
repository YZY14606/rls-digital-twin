import cv2
import os
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
import matplotlib.pyplot as plt



class Segmentation_tool():
    def __init__(self,mask_generator_type = "image"):

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

