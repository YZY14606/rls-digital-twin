import trimesh
import open3d as o3d

# 用 trimesh 读取
mesh = trimesh.load("rls_fetch_ws/src/apps/low_level_planning/models/bottle/meshes/bottle.dae")

# 转为 Open3D 格式
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

# 可视化
o3d.visualization.draw_geometries([o3d_mesh])