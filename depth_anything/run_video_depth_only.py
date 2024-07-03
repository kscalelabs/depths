import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    print("SKLFDJ")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()
    
    print(args.outdir)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.video_path):
        filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    os.makedirs(args.outdir, exist_ok=True)

    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_count = 0

        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(frame)

            depth = F.interpolate(depth[None], (raw_frame.shape[0], raw_frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            frame_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_frame_{frame_count}.png"
            frame_path = os.path.join(args.outdir, frame_filename)
            cv2.imwrite(frame_path, depth_color)
            
            frame_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_frame_{frame_count}_OG.png"
            frame_path = os.path.join(args.outdir, frame_filename)
            cv2.imwrite(frame_path, raw_frame)

            frame_count += 1

        raw_video.release()

    print("Processing complete.")