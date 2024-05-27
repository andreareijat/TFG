from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RealSenseDataset(Dataset):
    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, is_train=False):
        super(RealSenseDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        
        inputs = {}

        filename = self.filenames[index].strip().split()[1]
        image_path = os.path.join(self.data_path, "monocular_photos",'{}.png'.format(filename))
        image = cv2.imread(image_path)

        image = cv2.resize(image, (self.width, self.height))
        image = image.astype(np.float32) / 255.0
        inputs[("color", 0, 0)] = torch.from_numpy(image).permute(2,0,1)
        depth_path = os.path.join(self.data_path, "rgbd_photos", '{}_map_depth.png'.format(filename))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = cv2.resize(depth, (self.width, self.height))
        depth = depth / 1000.0 #Profundidad en milimetros
        inputs["depth"] = torch.from_numpy(depth).unsqueeze(0)

        return inputs