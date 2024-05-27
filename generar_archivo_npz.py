import os
import numpy as np
import cv2

depth_images_path = "./monodepth2/datasets/rgbd_photos/"
depth_images = sorted([f for f in os.listdir(depth_images_path) if f.endswith(".png")])
depth_maps = []

for image_name in depth_images:
    image_path = os.path.join(depth_images_path, image_name)
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Depth map not found at path: {image_path}")
    depth_maps.append(depth_image)

depth_maps = np.array(depth_maps)
np.savez("./monodepth2/splits/eigen/gt_depths.npz", data=depth_maps)
