# Humanoid at Home Depth Estimation

## Demos




https://github.com/kscalelabs/depths/assets/35588167/b0321dd0-14ba-4014-8b70-482e8c7bc855



Depth Estimation


https://github.com/kscalelabs/depths/assets/35588167/14ba11ce-9ab1-4e66-ad3c-7b88a147d692


3D Scene Reconstruction




https://github.com/kscalelabs/depths/assets/35588167/c89eb4b1-59f9-44cb-854f-511049e8f317




https://github.com/kscalelabs/depths/assets/35588167/eabd46d2-7343-45d2-8430-a6a2157cad1b



## Instructions

1. run_video_depth_only.py: Get depth estimation inference frames from video

2. depth_to_pc.py: Get point clouds for each frame in video, by cross referencing original (*_OG.png) and generated depth frames. Change camera intrinsics appropriately to generate accurate point clouds.

3. video_pc_testing.py: To test camera view ports, angles, and positions to generate a point cloud video

4. video_pc_actual.py: generate point cloud video, camera rotates on x-y plane in a circle for 3d effects. Utilizes parameters found after video_pc_testing.py.

5. video_pc_mesh.py: generates mesh from pointcloud, using KDTree for normal estimation and Poisson Surface Reconstruction for creating smooth, detailed surfaces

## Requirements

environment.yml
