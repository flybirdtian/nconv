########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import numpy as np
import glob
import os


def get_dataset_path(base_dir, setname='train'):
    """
    get dataset path according to setname
    :param base_dir: basic data dir
    :param setname: train, val, seval, test
    :param use_compose_depth: use left-right composed depth as ground-truth or not
    :return: lidar_dir, depth_dir, rgb_dir
    """
    if setname == 'train':
        lidar_dir = os.path.join(base_dir, 'data_depth_velodyne', 'train')
        depth_dir = os.path.join(base_dir, 'data_depth_annotated', 'train')
        rgb_dir = os.path.join(base_dir, 'raw')
    elif setname == 'val':
        lidar_dir = os.path.join(base_dir, 'data_depth_velodyne', 'val')
        depth_dir = os.path.join(base_dir, 'data_depth_annotated', 'val')
        rgb_dir = os.path.join(base_dir, 'raw')
    elif setname == 'selval':
        lidar_dir = os.path.join(base_dir, 'val_selection_cropped', 'velodyne_raw')
        depth_dir = os.path.join(base_dir, 'val_selection_cropped', 'groundtruth_depth')
        rgb_dir = os.path.join(base_dir, 'val_selection_cropped', 'image')
    elif setname == 'test':
        lidar_dir = os.path.join(base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
        depth_dir = os.path.join(base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
        rgb_dir = os.path.join(base_dir, 'test_depth_completion_anonymous', 'image')

    return lidar_dir, depth_dir, rgb_dir


class KittiDepthDataset(Dataset):
    def __init__(self, base_dir, setname='train', transform=None, norm_factor=256):
        lidar_dir, depth_dir, rgb_dir = self.get_paths(base_dir, setname=setname)
        self.lidar_dir = lidar_dir
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir

        self.use_sparsity = False
        self.sparsity_ratio = 0.2

        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        
        self.data = list(sorted(glob.iglob(self.lidar_dir + "/**/*.png", recursive=True)))

    def get_paths(self, base_dir, setname):
        return get_dataset_path(base_dir, setname)

    def split_selval_filename(self, filename):
        mid = 'velodyne_raw_'
        name_split = filename.split(mid)
        pre, pos = name_split[0], name_split[1]
        return pre, mid, pos

    def get_rgb_path(self, lidar_path):
        rgb_path = None
        if self.setname == 'train' or self.setname == 'val':
            file_names = lidar_path.split('/')
            rgb_path = os.path.join(self.rgb_dir, file_names[-5].split('_drive')[0], file_names[-5],
                                    file_names[-2], 'data', file_names[-1])
        elif self.setname == 'test':
            file_names = lidar_path.split('/')
            rgb_path = os.path.join(self.rgb_dir, file_names[-1])
        elif self.setname == 'selval':
            file_names = lidar_path.split('/')
            file_name = file_names[-1]
            pre, mid, pos = self.split_selval_filename(file_name)
            rgb_file_name = pre + "image_" + pos
            rgb_path = os.path.join(self.rgb_dir, rgb_file_name)
        return rgb_path

    def get_depth_path(self, lidar_path):
        depth_path = None
        if self.setname == 'train' or self.setname == 'val':
            file_names = lidar_path.split('/')
            depth_path = os.path.join(self.depth_dir, *file_names[-5:-3], 'groundtruth', *file_names[-2:])
        elif self.setname == 'test':
            file_names = lidar_path.split('/')
            depth_path = os.path.join(self.depth_dir, file_names[-1])
        elif self.setname == 'selval':
            file_names = lidar_path.split('/')
            file_name = file_names[-1]
            pre, mid, pos = self.split_selval_filename(file_name)
            depth_file_name = pre + "groundtruth_depth_" + pos
            depth_path = os.path.join(self.depth_dir, depth_file_name)
        return depth_path

    def __len__(self):
        return len(self.data)

    def get_file_name(self, item):
        lidar_path = self.data[item]
        file_names = lidar_path.split('/')
        return file_names[-1]

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        lidar_path = self.data[item]
        gt_path = self.get_depth_path(lidar_path)
        rgb_path = self.get_rgb_path(lidar_path)

        # Read images and convert them to 4D floats
        data = Image.open(lidar_path)
        gt = Image.open(gt_path)

        # Read RGB images
        rgb = Image.open(rgb_path)

        # Apply transformations if given
        if self.transform is not None:
            data = self.transform(data)
            gt = self.transform(gt)
            rgb = self.transform(rgb)

        # Convert to numpy
        data = np.array(data, dtype=np.float16)      
        gt = np.array(gt, dtype=np.float16)

        if self.use_sparsity:
            rand_value = np.random.rand(*data.shape)
            rand_mask = rand_value <= self.sparsity_ratio
            data = data * rand_mask

        # define the certainty 
        C = (data > 0).astype(float)
    
        # Normalize the data
        data = data / self.norm_factor  #[0,1]
        gt = gt / self.norm_factor           
        
        # Expand dims into Pytorch format 
        data = np.expand_dims(data, 0)      
        gt = np.expand_dims(gt, 0)
        C = np.expand_dims(C, 0) 

        # Convert RGB image to tensor
        rgb = np.array(rgb, dtype=np.float16)
        rgb /= 255
        rgb = np.transpose(rgb, (2, 0, 1))

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)
        rgb = torch.tensor(rgb, dtype=torch.float)

        return data, C, gt, item, rgb
