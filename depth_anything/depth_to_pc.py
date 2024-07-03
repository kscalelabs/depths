import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm

# Global settings
FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 256
FINAL_WIDTH = 256

# INPUT_DEPTH_DIR = './150_raw_img_depth.png'
# INPUT_COLOR_DIR = './150.jpg'
# DATASET = 'nyu'

OUTPUT_DIR = './my_point_clouds/house_tour'
folder_path = '/accounts/projects/binyu/timothygao/Depth-Anything/house_depth_frames'

def process_images(INPUT_DEPTH_DIR, INPUT_COLOR_DIR):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    color_image = Image.open(INPUT_COLOR_DIR).convert('RGB')
    depth_image = Image.open(INPUT_DEPTH_DIR)

    original_width, original_height = color_image.size
    color_image_tensor = transforms.ToTensor()(color_image).unsqueeze(0)
    depth_image_np = np.array(depth_image)

    # Ensure depth image is resized
    depth_image_resized = Image.fromarray(depth_image_np).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)
    depth_image_resized_np = np.array(depth_image_resized)

    resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)

    focal_length_x, focal_length_y = FL, FL
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    x = (x - FINAL_WIDTH / 2) / focal_length_x
    y = (y - FINAL_HEIGHT / 2) / focal_length_y
    z = depth_image_resized_np

    # Ensure z is 2D
    if len(z.shape) == 3:
        z = z[:, :, 0]

    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(INPUT_DEPTH_DIR))[0] + ".ply"), pcd)
        
# process_images('./150_raw_img_depth.png', './150.jpg')


for filename in tqdm(os.listdir(folder_path)):
    print(filename)
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.gif') or filename.endswith('.bmp'):
        if '_OG' in filename:
            continue
        
        dep_img_in = folder_path + "/" + filename
        og_img_in = folder_path + '/' + filename.split('.')[0] + '_OG.' + filename.split('.')[1]
        
        process_images(dep_img_in, og_img_in)
