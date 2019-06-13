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
        self.label_transforms = None
        if self.shrinkage_factor != 1:
            self.image_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(288 // self.shrinkage_factor)),
                transforms.ToTensor()
            ])

            self.label_transforms = transforms.Compose([
                transforms.ToPILImage('I'),
                transforms.Resize(int(288 // self.shrinkage_factor), 0),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        image = image.type(torch.FloatTensor)
        semantic = semantic.type(torch.IntTensor)
        depth = depth.type(torch.FloatTensor)
        normal = normal.type(torch.FloatTensor)

        if self.image_transforms is not None and self.label_transforms is not None:
            image = self.image_transforms(image)
            semantic = self.label_transforms(semantic)
            depth = self.image_transforms(depth)
            normal = self.image_transforms(normal)

        return image, semantic.squeeze().type(torch.FloatTensor), depth, normal

    def __len__(self):
        return self.data_len

