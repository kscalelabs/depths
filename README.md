# Humanoid at Home Depth Estimation

## Demos

<video width="320" height="240" controls>
  <source src="https://github.com/kscalelabs/depths/blob/master/House_video_depth.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Depth Estimation

<video width="320" height="240" controls>
  <source src="https://github.com/kscalelabs/depths/blob/master/3d_scene.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

3D Scene Reconstruction

## Instructions

1. run_video_depth_only.py: Get depth estimation inference frames from video

2. depth_to_pc.py: Get point clouds for each frame in video, by cross referencing original (*_OG.png) and generated depth frames. Change camera intrinsics appropriately to generate accurate point clouds.

3. video_pc_testing.py: To test camera view ports, angles, and positions to generate a point cloud video

4. video_pc_actual.py: generate point cloud video, camera rotates on x-y plane in a circle for 3d effects. Utilizes parameters found after video_pc_testing.py.

5. video_pc_mesh.py: generates mesh from pointcloud, using KDTree for normal estimation and Poisson Surface Reconstruction for creating smooth, detailed surfaces

## Requirements

environment.yml
