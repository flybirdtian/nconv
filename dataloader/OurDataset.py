
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import numpy as np
import glob
import os
from dataloader.KittiDepthDataset import KittiDepthDataset


class OurDataset(KittiDepthDataset):
    def __init__(self, base_dir, setname='train', transform=None, norm_factor=256):
        super().__init__(base_dir, setname, transform, norm_factor)

    def get_paths(self, base_dir, setname):
        lidar_dir = os.path.join(base_dir, 'lidar')
        depth_dir = os.path.join(base_dir, 'depth')
        rgb_dir = os.path.join(base_dir, 'image')

        return lidar_dir, depth_dir, rgb_dir

    def get_depth_path(self, lidar_path):
        file_names = lidar_path.split('/')
        depth_path = os.path.join(self.depth_dir, *file_names[-1:])
        return depth_path

    def get_rgb_path(self, lidar_path):
        file_names = lidar_path.split('/')
        rgb_path = os.path.join(self.rgb_dir, *file_names[-1:])
        return rgb_path
