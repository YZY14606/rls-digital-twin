import torch
import numpy as np
from utils.logging_utils import Logger, LogLevel, log_function


@log_function(level=LogLevel.DEBUG)
def collate_fn(batch):
    """Custom collate function for variable-sized point clouds"""
    logger = Logger.get_instance()

    current_pcs = [item["current_pc"] for item in batch]
    future_local_poses = [item["future_local_pose"] for item in batch]
    qualities = [item["quality"] for item in batch]
    orientations = [item["orientation"] for item in batch]
    push_distances = [item["push_distance"] for item in batch]


    # Pad point clouds to the same size if necessary
    max_points = max(pc.shape[0] for pc in current_pcs)
    logger.debug(f"Batch max points: {max_points}")

    padded_current_pcs = []
    padded_qualities = []
    padded_orientations = []
    padded_push_distances = []

    for current_pc,  quality, orientation, push_distance in zip(
        current_pcs,  qualities, orientations, push_distances
    ):
        if current_pc.shape[0] < max_points:
            # Pad with zeros
            padding = torch.zeros(
                (max_points - current_pc.shape[0], current_pc.shape[1]),
                dtype=torch.float,
            )
            padded_current_pc = torch.cat([current_pc, padding], dim=0)

            # Pad quality
            quality_padding = torch.zeros(
                max_points - quality.shape[0], dtype=torch.long
            )
            padded_quality = torch.cat([quality, quality_padding], dim=0)

            # Pad orientation
            orientation_padding = torch.zeros(
                (max_points - orientation.shape[0], orientation.shape[1]),
                dtype=torch.float,
            )
            padded_orientation = torch.cat([orientation, orientation_padding], dim=0)

            # Pad push_distance
            push_distance_padding = torch.zeros(
                (max_points - push_distance.shape[0], push_distance.shape[1]),
                dtype=torch.float,
            )
            padded_push_distance = torch.cat([push_distance, push_distance_padding], dim=0)

            logger.debug(
                f"Padded point cloud from {current_pc.shape[0]} to {max_points} points"
            )
        else:
            padded_current_pc = current_pc
            padded_quality = quality
            padded_orientation = orientation
            padded_push_distance = push_distance


        padded_current_pcs.append(padded_current_pc)
        padded_qualities.append(padded_quality)
        padded_orientations.append(padded_orientation)
        padded_push_distances.append(padded_push_distance)

    return {
        "current_pc": torch.stack(padded_current_pcs),
        "future_local_pose": torch.stack(future_local_poses),
        "quality": torch.stack(padded_qualities),
        "orientation": torch.stack(padded_orientations),
        "push_distance":torch.stack(padded_push_distances),
    }
