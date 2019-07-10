from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np

from torchvision import transforms

class NYUv2(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, shrinkage_factor=1):
        self.train = train
        self.root = os.path.expanduser(root)

        # R\read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

        self.shrinkage_factor = shrinkage_factor

        self.image_transforms = None
        self.label_transforms_semantic = None
        self.label_transforms_depth_normal = None
        if self.shrinkage_factor != 1:
            self.image_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(288 // self.shrinkage_factor)),
                transforms.ToTensor()
            ])

            self.label_transforms_semantic = transforms.Compose([
                transforms.ToPILImage('I'),
                transforms.Resize(int(288 // self.shrinkage_factor), 0),
                transforms.ToTensor()
            ])

            self.label_transforms_depth_normal = transforms.Compose([
                transforms.ToPILImage('F'),
                transforms.Resize(int(288 // self.shrinkage_factor), 0),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))

        if self.label_transforms_semantic:
            semantic = np.load(self.data_path + '/label/{:d}.npy'.format(index)).astype(np.int32)
        else:
            semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))

        if self.label_transforms_depth_normal:
            depth = np.load(self.data_path + '/depth/{:d}.npy'.format(index)).astype(np.float32)
            normal = np.load(self.data_path + '/normal/{:d}.npy'.format(index)).astype(np.float32)
        else:
            depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
            normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        image = image.type(torch.FloatTensor)
        if not self.label_transforms_depth_normal:
            depth = depth.type(torch.FloatTensor)
            normal = normal.type(torch.FloatTensor)

        if self.image_transforms and self.label_transforms_semantic and self.label_transforms_depth_normal:
            image = self.image_transforms(image)
            semantic = self.label_transforms_semantic(semantic)
            depth = self.label_transforms_depth_normal(depth)
            normal_x = self.label_transforms_depth_normal(normal[:, :, 0])
            normal_y = self.label_transforms_depth_normal(normal[:, :, 1])
            normal_z = self.label_transforms_depth_normal(normal[:, :, 2])
            normal = torch.stack((normal_x, normal_y, normal_z), dim=0).squeeze()

        return image, semantic.squeeze().type(torch.FloatTensor), depth, normal

    def __len__(self):
        return self.data_len

