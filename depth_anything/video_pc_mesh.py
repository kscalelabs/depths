import open3d as o3d
import numpy as np
import cv2
import os
from tqdm import tqdm

# Define the directory containing the .ply files and the output directory for the frames
ply_dir = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_clouds/house_tour"
output_frames_dir = "/accounts/projects/binyu/timothygao/Depth-Anything/tour"
os.makedirs(output_frames_dir, exist_ok=True)
output_video_file = "/accounts/projects/binyu/timothygao/Depth-Anything/tour/output_tour_video.mp4"

circle_idx = 0
N_circle = 360
x = np.cos(np.linspace(0, 2 * np.pi, N_circle))
y = np.sin(np.linspace(0, 2 * np.pi, N_circle))

def render_point_cloud_as_image(ply_file, angle):
    global circle_idx
    
    circle_idx = (circle_idx + 1) % N_circle
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Rotate the point cloud
    R = pcd.get_rotation_matrix_from_xyz((-np.deg2rad(5), 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # Estimate normals with a faster method
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
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

# Iterate through all .ply files, rotate, render, and save frames
ply_files = sorted([os.path.join(ply_dir, f) for f in os.listdir(ply_dir) if f.endswith(".ply")])
angle = 0
frame_files = []
for idx in tqdm(range(len(ply_files)), desc="Processing PLY files"):
    ply_file = ply_files[idx]
    frame = render_point_cloud_as_image(ply_file, angle)
    frame_file = os.path.join(output_frames_dir, f"frame_{idx:04d}.png")
    print(frame_file)
    cv2.imwrite(frame_file, frame)
    frame_files.append(frame_file)

# Compile the frames into a video using OpenCV
frame = cv2.imread(frame_files[0])
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(output_video_file, fourcc, 20, (width, height))

for frame_file in frame_files:
    video.write(cv2.imread(frame_file))

video.release()

print(f"Video saved as {output_video_file}")