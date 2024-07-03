import open3d as o3d
import numpy as np
import cv2
import os

# Define the directory containing the .ply files and the output directory for the frames
# ply_dir = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_clouds/driving_test"
ply_dir = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_clouds/house_tour"
# output_frames_dir = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_clouds/point_cloud_frames"

output_frames_dir = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_clouds/tour_pc_frames"

os.makedirs(output_frames_dir, exist_ok=True)

# /Sequence 01_frame_0000.ply
# Define the output video file
output_video_file = "/accounts/projects/binyu/timothygao/Depth-Anything/my_point_cloud_videos/output_video_tour.mp4"

# Define the rotation angle and the rotation axis
rotation_angle = 0  # Degrees per frame
rotation_axis = [0, 1, 0]  # Rotate around the y-axis

# Function to render a point cloud as an image
def render_point_cloud_as_image(ply_file, angle):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Rotate the point cloud
    R = pcd.get_rotation_matrix_from_xyz((-np.deg2rad(5), 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # Create an offscreen renderer
    width, height = 800, 600
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.point_size = 6
    
    render.scene.add_geometry("pcd", pcd, material)
    
    # Set up camera parameters
    center = pcd.get_center()
    eye = center + np.array([0, 0, 200])  # Adjust this based on your scene
    up = np.array([0, -1, 0])
    render.setup_camera(60, center, eye, up)

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
for idx, ply_file in enumerate(ply_files):
    print(ply_file)
    frame = render_point_cloud_as_image(ply_file, angle)
    frame_file = os.path.join(output_frames_dir, f"test_frame_{idx:04d}.png")
    print(frame_file)
    cv2.imwrite(frame_file, frame)
    frame_files.append(frame_file)
    angle += rotation_angle
    break

# # Compile the frames into a video using OpenCV
# frame = cv2.imread(frame_files[0])
# height, width, layers = frame.shape
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# video = cv2.VideoWriter(output_video_file, fourcc, 20, (width, height))

# for frame_file in frame_files:
#     video.write(cv2.imread(frame_file))

# video.release()

# print(f"Video saved as {output_video_file}")