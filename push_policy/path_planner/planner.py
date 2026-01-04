import numpy as np
import object_planner_py as opp
import time
import os
import open3d as o3d
import random
import typing
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull




class Path_planner():
    """This planner is used to plan a path of the pushed object.
    The pipline of the ptah planner is: 
    1. __init__()
    2. set_up_planner()
    3. plan_path()
    """
    def __init__(
        self,
        planner_params: typing.Dict
    ):
        planner_iterations = planner_params["planner_iterations"]
        planner_step_size = planner_params["planner_step_size"]
        goal_bias = planner_params["goal_bias"]
        neighborhood_radius = planner_params["neighborhood_radius"]

        self.plan_params = self.set_up_planner_params(planner_iterations,planner_step_size,goal_bias,neighborhood_radius)

        self.path_planner = None

        self.path = None
        self.obstacle_points = None
        self.object_points = None


    def _trans_object_points_to_sphere_tree_file(self,object_points):
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)

        fk_file = os.path.join(current_dir,"planner","object_model.fk")
        os.makedirs(os.path.dirname(fk_file), exist_ok=True)
        opp.create_sphere_tree_file(object_points, fk_file)

        return fk_file
    
    def set_up_planner_params(self,planner_iterations,planner_step_size,goal_bias,neighborhood_radius):

        plan_params = opp.PlanParams()
        plan_params.max_iterations = planner_iterations
        plan_params.step_size = planner_step_size
        plan_params.goal_bias = goal_bias
        plan_params.neighborhood_radius = neighborhood_radius

        return plan_params
    
    def set_up_planner(self,scene_dic: typing.Dict):
        """scene_dic includes:
        object_points: np.ndarray, # shepe(N,3)
        all_obstacle_points: typing.List[np.ndarray], # All obstacle object pointcloud list
        map_x_bounds: typing.Tuple[float, float] = (-1.5, 1.5),  
        map_y_bounds: typing.Tuple[float, float] = (-1.5, 1.5),  
        map_theta_bound: typing.Tuple[float, float] = (-np.pi, np.pi)
        """
        self.scene_dic = scene_dic

        object_points = scene_dic["object_points"]
        all_obstacle_points = scene_dic["all_obstacle_points"]
        map_x_bounds = scene_dic["map_x_bounds"]
        map_y_bounds = scene_dic["map_y_bounds"]
        map_theta_bound = scene_dic["map_theta_bound"]

        self.sphere_tree_file = self._trans_object_points_to_sphere_tree_file(object_points)

        obstacle_points = np.vstack(all_obstacle_points)

        self.path_planner = opp.Planner(
                                        sphere_tree_file= self.sphere_tree_file,
                                        obstacle_points=obstacle_points,
                                        x_bounds=map_x_bounds,
                                        y_bounds=map_y_bounds,
                                        theta_bounds=map_theta_bound,
                                        )
        
        self.obstacle_points = all_obstacle_points
        self.object_points = object_points
        
    def plan_path(self,start_config,goal_config,smoothing_iterations = 50):

        path =  self.path_planner.plan(
                                    start= start_config,
                                    goal= goal_config,
                                    plan_params= self.plan_params,
                                    smoothing_iterations= smoothing_iterations,
                                    )
        self.path = path
        return path
    

    def visualize_path(self,
        object_points,
        obstacle_points,
        path,
        start_config,
        goal_config,
        map_bounds_x,
        map_bounds_y,
    ):
        """
        Visualizes the sparse planning path directly from the planner.
        """
        if not path:
            print("No path to visualize.")
            return
        def create_object_at_config(p, c, clr):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p)
            T = np.eye(4)
            T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, c.theta))
            T[0, 3] = c.x
            T[1, 3] = c.y
            pcd.transform(T)
            pcd.paint_uniform_color(clr)
            return pcd

        geometries = []
        obstacle_pcd = o3d.geometry.PointCloud()
        obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
        obstacle_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(obstacle_pcd)
        geometries.append(
            create_object_at_config(object_points, start_config, [0.0, 0.8, 0.2])
        )
        geometries.append(
            create_object_at_config(object_points, goal_config, [0.0, 0.2, 0.8])
        )
        path_points_3d = [[c.x, c.y, 0.01] for c in path]
        path_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(path_points_3d),
            lines=o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(path_points_3d) - 1)]
            ),
        )
        path_lineset.colors = o3d.utility.Vector3dVector(
            [[1, 0, 0] for _ in range(len(path_lineset.lines))]
        )
        geometries.append(path_lineset)

        num_ghosts = 4
        if len(path) > num_ghosts + 1:
            path_without_ends = path[1:-1]
            step = len(path_without_ends) // num_ghosts
            indices_in_subpath = range(0, len(path_without_ends), step if step > 0 else 1)[
                :num_ghosts
            ]
            for i in indices_in_subpath:
                geometries.append(
                    create_object_at_config(
                        object_points, path_without_ends[i], [0.9, 0.7, 0.1]
                    )
                )

        for point in path_points_3d:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.015
            )  # Made spheres slightly larger for visibility
            sphere.translate(point)
            sphere.paint_uniform_color([0.6, 0.2, 0.8])
            geometries.append(sphere)

        map_w = map_bounds_x[1] - map_bounds_x[0]
        map_h = map_bounds_y[1] - map_bounds_y[0]
        table = o3d.geometry.TriangleMesh.create_box(width=map_w, height=map_h, depth=0.005)
        table.translate([map_bounds_x[0], map_bounds_y[0], -0.005])
        table.paint_uniform_color([0.8, 0.7, 0.6])
        geometries.append(table)
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        )
        print("This Step Displays Path Plan Visualization")
        o3d.visualization.draw_geometries(geometries)



