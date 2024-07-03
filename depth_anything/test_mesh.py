
# import open3d as o3d
# import numpy as np

# Load the .ply file
ply_file_path = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_clouds/driving_test/Sequence 01_frame_0000.ply"  # Replace with the path to your .ply file
# point_cloud = o3d.io.read_point_cloud(ply_file_path)


# # Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])

import open3d as o3d
import numpy as np
import cv2

circle_idx = 0
N_circle = 360
x = np.cos(np.linspace(0, 2 * np.pi, N_circle))
y = np.sin(np.linspace(0, 2 * np.pi, N_circle))

# def render_point_cloud_as_image(ply_file, angle):
#     global circle_idx
    
#     circle_idx = (circle_idx + 1) % N_circle
    
#     # Load the point cloud
#     pcd = o3d.io.read_point_cloud(ply_file)
    
#     # Rotate the point cloud
#     R = pcd.get_rotation_matrix_from_xyz((-np.deg2rad(5), 0, 0))
#     pcd.rotate(R, center=(0, 0, 0))

#     # Estimate normals with a faster method
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
#     # Perform Poisson surface reconstruction
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
#     # Crop the mesh to remove low-density areas
#     bbox = pcd.get_axis_aligned_bounding_box()
#     mesh = mesh.crop(bbox)
    
#     # Create an offscreen renderer
#     width, height = 800, 600
#     render = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     material = o3d.visualization.rendering.MaterialRecord()
#     material.shader = "defaultLit"
    
#     render.scene.add_geometry("mesh", mesh, material)
    
#     # Set up camera parameters
#     center = pcd.get_center()
#     eye = center + np.array([0 + x[circle_idx], 40 + y[circle_idx], 200])  # Adjust this based on your scene
#     print(x[circle_idx], y[circle_idx])
#     up = np.array([0, -1, 0])
#     render.setup_camera(50, center, eye, up)

#     # Render the scene
#     image = render.render_to_image()
    
#     # Convert to uint8 and return the image
#     image = np.asarray(image)
#     image = cv2.flip(image, 1)
#     return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def render_point_cloud_as_image(ply_file, angle):
    global circle_idx
    
    circle_idx = (circle_idx + 1) % N_circle
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Rotate the point cloud
    R = pcd.get_rotation_matrix_from_xyz((-np.deg2rad(5), 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # Downsample the point cloud
    voxel_size = 0.05  # Adjust this value to control the level of downsampling
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals with a faster method
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=10))
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15)
    
    # Crop the mesh to remove low-density areas
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    
    # Create an offscreen renderer
    width, height = 800, 600
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    
    render.scene.add_geometry("mesh", mesh, material)
    
    # Set up camera parameters
    center = pcd.get_center()
    eye = center + np.array([0 + x[circle_idx], 40 + y[circle_idx], 200])  # Adjust this based on your scene
    print(x[circle_idx], y[circle_idx])
    up = np.array([0, -1, 0])
    render.setup_camera(50, center, eye, up)

    # Render the scene
    image = render.render_to_image()
    
    # Convert to uint8 and return the image
    image = np.asarray(image)
    image = cv2.flip(image, 1)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Example usage
angle = 0  # or any other angle you want to start with
image = render_point_cloud_as_image(ply_file_path, angle)
print(image)
cv2.imwrite('/accounts/projects/binyu/timothygao/Depth-Anything/rendered_image.jpg', image)
print('/accounts/projects/binyu/timothygao/Depth-Anything/rendered_image.jpg')
