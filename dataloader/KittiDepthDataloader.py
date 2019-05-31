########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.KittiDepthDataset import KittiDepthDataset
from dataloader.VkittiDepthDataset import VkittiDepthDataset
from dataloader.NuScenesDataset import NuScenesDataset
from dataloader.OurDataset import OurDataset


def get_dataloaders(params, dataset_name="kitti"):
    if dataset_name == "kitti":
        params["dataset_dir"] = "/home/bird/data2/dataset/kitti_depth_completion"
        return get_kitti_dataloader(params)
    elif dataset_name == "nuscenes":
        params["dataset_dir"] = "/home/bird/data2/dataset/nuscenes/projected"
        return get_nuscenes_dataloader(params)
    elif dataset_name == "ours":
        params["dataset_dir"] = "/home/bird/data2/dataset/our_lidar/20190315/f_c_1216_352"
        return get_our_dataloader(params)
    elif dataset_name == "ours_20190318":
        params["dataset_dir"] = "/home/bird/data2/dataset/our_lidar/20190318/f_c_1216_352"
        return get_our_dataloader(params)
    elif dataset_name == "vkitti":
        params["dataset_dir"] = "/home/bird/data2/dataset/vkitti"
        params['data_normalize_factor'] = 100.
        return get_vkitti_dataloader(params)


def get_kitti_dataloader(params):
    # Input images are 16-bit, but only 15-bits are utilized
    # so we normalized the data to [0:1] using a normalization factor
    norm_factor = params['data_normalize_factor']
    ds_dir = params['dataset_dir']

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    ###### Training Set ######
    train_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])
    image_datasets['train'] = KittiDepthDataset(ds_dir, setname='train', transform=train_transform,
                                                norm_factor=norm_factor)
    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=4)
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ###### Validation Set ######
    val_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])
    image_datasets['val'] = KittiDepthDataset(ds_dir, setname='val', transform=val_transform, norm_factor=norm_factor)
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'],
                                    num_workers=4)
    dataset_sizes['val'] = {len(image_datasets['val'])}

    ###### Selected Validation set ######
    image_datasets['selval'] = KittiDepthDataset(ds_dir, setname='selval', transform=None, norm_factor=norm_factor)
    dataloaders['selval'] = DataLoader(image_datasets['selval'], shuffle=False, batch_size=params['test_batch_sz'],
                                       num_workers=4)
    dataset_sizes['selval'] = {len(image_datasets['selval'])}

    ###### Selected test set ######
    image_datasets['test'] = KittiDepthDataset(ds_dir, setname='test', transform=None, norm_factor=norm_factor)
    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=False, batch_size=params['test_batch_sz'],
                                     num_workers=4)
    dataset_sizes['test'] = {len(image_datasets['test'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes, image_datasets


def get_vkitti_dataloader(params):
    norm_factor = params['data_normalize_factor']
    ds_dir = params['dataset_dir']
    train_subset = 'clone'
    test_subset = 'clone'

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    ###### Train Set ######
    train_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])
    image_datasets['train'] = VkittiDepthDataset(ds_dir, setname='train', subset=train_subset, transform=train_transform,
                                                norm_factor=norm_factor)
    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=4)
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ###### test set ######
    test_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])
    image_datasets['test'] = VkittiDepthDataset(ds_dir, setname='test', subset=test_subset, transform=test_transform, norm_factor=norm_factor)
    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=False, batch_size=params['test_batch_sz'],
                                     num_workers=4)
    dataset_sizes['test'] = {len(image_datasets['test'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes, image_datasets


def get_nuscenes_dataloader(params):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    norm_factor = params['data_normalize_factor']
    ds_dir = params['dataset_dir']

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    dataset = NuScenesDataset(ds_dir, setname='train', transform=None, norm_factor=norm_factor)
    dataloader = DataLoader(dataset,shuffle=False, batch_size=params['train_batch_sz'], num_workers=4)


    ###### Training Set ######
    train_transform = None
    image_datasets['train'] = dataset
    dataloaders['train'] = dataloader
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ###### Validation Set ######
    val_transform = None
    image_datasets['val'] = dataset
    dataloaders['val'] = dataloader
    dataset_sizes['val'] = {len(image_datasets['val'])}

    ###### Selected Validation set ######
    image_datasets['selval'] = dataset
    dataloaders['selval'] = dataloader
    dataset_sizes['selval'] = {len(image_datasets['selval'])}

    ###### Selected test set ######
    image_datasets['test'] = dataset
    dataloaders['test'] = dataloader
    dataset_sizes['test'] = {len(image_datasets['test'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes, image_datasets


def get_our_dataloader(params):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    norm_factor = params['data_normalize_factor']
    ds_dir = params['dataset_dir']

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    dataset = OurDataset(ds_dir, setname='train', transform=None, norm_factor=norm_factor)
    dataloader = DataLoader(dataset,shuffle=False, batch_size=params['train_batch_sz'], num_workers=4)

    ###### Training Set ######
    train_transform = None
    image_datasets['train'] = dataset
    dataloaders['train'] = dataloader
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ###### Validation Set ######
    val_transform = None
    image_datasets['val'] = dataset
    dataloaders['val'] = dataloader
    dataset_sizes['val'] = {len(image_datasets['val'])}

    ###### Selected Validation set ######
    image_datasets['selval'] = dataset
    dataloaders['selval'] = dataloader
    dataset_sizes['selval'] = {len(image_datasets['selval'])}

    ###### Selected test set ######
    image_datasets['test'] = dataset
    dataloaders['test'] = dataloader
    dataset_sizes['test'] = {len(image_datasets['test'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes, image_datasets
