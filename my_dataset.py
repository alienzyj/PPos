import os
import random

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.nn import functional as F
from torch.utils.data import Dataset

from my_utils import normalize


class UNetDataset(Dataset):
    def __init__(self, data_dir, shape, train, transform):
        self.shape = shape
        self.transform = transform
        self.sample_path = os.path.join(data_dir, 'samples')
        self.sample_files = os.listdir(self.sample_path)
        self.sample_files.sort()
        self.mask_path = os.path.join(data_dir, 'masks')
        self.train = train

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.sample_path, self.sample_files[index]))
        mask = np.load(os.path.join(self.mask_path, self.sample_files[index]))
        mask = mask.astype(np.float32)

        if int(self.sample_files[index][-6:-4]) == 0:
            rand = random.randrange(3, len(sample) - 3)
            sample = sample[rand - 3:rand + 4]
            mask = mask[rand]

        if self.transform is not None:
            sample = self.transform(sample)
            sample = np.concatenate((sample[0], sample[1]))

        if self.train:
            htranslation = random.randint(-10, 10)
            vtranslation = random.randint(-10, 10)
            angle = random.randint(-10, 10)

            sample = ndi.shift(sample, (0, htranslation, vtranslation), mode='nearest')
            sample = ndi.rotate(sample, angle, (-1, -2), mode='nearest', reshape=False)
            mask = ndi.shift(mask, (htranslation, vtranslation), mode='nearest')
            mask = ndi.rotate(mask, angle, (-1, -2), mode='nearest', reshape=False)

            if random.randint(0, 1) == 1:
                sample = np.flip(sample, -1)
                mask = np.flip(mask, -1)

        sample = torch.from_numpy(sample[np.newaxis, ...].copy())
        sample = F.interpolate(sample, self.shape, mode='bilinear', align_corners=False)
        mask = torch.from_numpy(mask[np.newaxis, np.newaxis, ...].copy())
        mask = F.interpolate(mask, self.shape, mode='nearest')
        mask2 = F.interpolate(mask, scale_factor=0.5, mode='nearest', recompute_scale_factor=False)
        mask3 = F.interpolate(mask, scale_factor=0.25, mode='nearest', recompute_scale_factor=False)

        return sample[0], mask[0], mask2[0], mask3[0]


class GenesisDataset2D(Dataset):
    def __init__(self, data_dir, shape, transform, flip_rate):
        self.shape = shape
        self.transform = transform
        self.flip_rate = flip_rate
        self.sample_path = os.path.join(data_dir, 'samples')
        self.sample_files = os.listdir(self.sample_path)
        self.sample_files.sort()

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        x = np.load(os.path.join(self.sample_path, self.sample_files[index]))

        rand = random.randrange(3, len(x) - 3)
        x = x[rand - 3:rand + 4]

        if random.random() < self.flip_rate:
            x = np.flip(x, -1)

        x = normalize(x)
        x = np.concatenate((x[0], x[1]))
        x = ndi.zoom(x, (1, self.shape[0] / x.shape[1], self.shape[1] / x.shape[2]), order=2, mode="nearest")

        y = self.transform(x)

        return torch.from_numpy(y.copy().astype(np.float32)), torch.from_numpy(x.copy().astype(np.float32))


class PPos2DDataset(Dataset):
    def __init__(self, data_dir, shape, num_classes, transform):
        self.shape = shape
        self.transform = transform
        self.sample_path = os.path.join(data_dir, 'samples')
        self.sample_files = os.listdir(self.sample_path)
        self.sample_files.sort()
        self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.sample_path, self.sample_files[index]))

        rand = random.randrange(3, len(sample) - 3)
        target = (rand - 3) / (len(sample) - 6)
        sample = sample[rand - 3:rand + 4]

        if self.transform is not None:
            sample = self.transform(sample)
            sample = np.concatenate((sample[0], sample[1]))

        return torch.from_numpy(sample), torch.tensor([target])


class UNetClassifierDataset(Dataset):
    def __init__(self, data_dir, train, transform):
        self.transform = transform
        self.sample_path = os.path.join(data_dir, 'samples')
        self.sample_files = os.listdir(self.sample_path)
        self.sample_files.sort()
        self.mask_path = os.path.join(data_dir, 'masks')
        self.train = train

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.sample_path, self.sample_files[index]))
        mask = np.load(os.path.join(self.mask_path, self.sample_files[index]))
        mask = mask.astype(np.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.train:
            htranslation = random.randint(-10, 10)
            vtranslation = random.randint(-10, 10)
            dtranslation = random.randint(-2, 2)
            angle = random.randint(-10, 10)

            sample = ndi.shift(sample, (0, dtranslation, htranslation, vtranslation), mode='nearest')
            sample = ndi.rotate(sample, angle, (-1, -2), mode='nearest', reshape=False)
            mask = ndi.shift(mask, (dtranslation, htranslation, vtranslation), mode='nearest')
            mask = ndi.rotate(mask, angle, (-1, -2), mode='nearest', reshape=False)

            if random.randint(0, 1) == 1:
                sample = np.flip(sample, -1)
                mask = np.flip(mask, -1)

        mask2 = ndi.zoom(mask, 0.5, order=0, mode='nearest')
        mask3 = ndi.zoom(mask, 0.25, order=0, mode='nearest')

        return torch.from_numpy(sample.copy()), torch.from_numpy(mask[np.newaxis, ...].copy()), torch.from_numpy(
            mask2[np.newaxis, ...].copy()), torch.from_numpy(mask3[np.newaxis, ...].copy()), torch.tensor(
            [self.sample_files[index][:5].isdigit()], dtype=torch.float)

        # return torch.from_numpy(sample.copy()), torch.from_numpy(mask[np.newaxis, ...].copy()), torch.from_numpy(
        #     mask2[np.newaxis, ...].copy()), torch.from_numpy(mask3[np.newaxis, ...].copy()), torch.tensor(
        #     [self.sample_files[index][6:11].isdigit()], dtype=torch.float)


class ClassifierDataset(Dataset):
    def __init__(self, data_dir, shape, train, transform=None):
        self.shape = shape
        self.train = train
        self.transform = transform
        self.sample_path = os.path.join(data_dir, 'samples')
        self.sample_files = os.listdir(self.sample_path)
        self.sample_files.sort()

        if train:
            self.mask_path = os.path.join(data_dir, 'masks')

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.sample_path, self.sample_files[index]))
        if self.train and self.sample_files[index][-5] == '1':
            mask = np.load(os.path.join(self.mask_path, self.sample_files[index]))
            indices = mask.nonzero()
            nodule_length = [0, 0, 0]
            scale_length = [0, 0, 0]
            for i in range(3):
                start = np.min(indices[i])
                end = np.max(indices[i]) + 1
                nodule_length[i] = end - start
            while True:
                for i in range(3):
                    while True:
                        scale_length[i] = round(nodule_length[i] * random.uniform(1, 3))
                        if scale_length[i] < sample.shape[i]:
                            break
                depth = random.randint(0, sample.shape[0] - scale_length[0])
                height = random.randint(0, sample.shape[1] - scale_length[1])
                width = random.randint(0, sample.shape[2] - scale_length[2])
                if depth > np.max(indices[0]) or depth + scale_length[0] < np.min(indices[0]) or height > np.max(
                        indices[1]) or height + \
                        scale_length[1] < np.min(indices[1]) or width > np.max(indices[2]) or width + scale_length[2] < \
                        np.min(indices[2]):
                    sample = sample[depth:depth + scale_length[0], height:height + scale_length[1],
                             width:width + scale_length[2]]
                    break

        if self.transform is not None:
            sample = self.transform(sample)

        sample = torch.from_numpy(sample[np.newaxis, ...].copy())
        sample = F.interpolate(sample, self.shape, mode='trilinear', align_corners=True)

        return sample[0], torch.tensor([self.sample_files[index][-5] == '0'], dtype=torch.float)


class PCL2DDataset(Dataset):
    def __init__(self, data_dir, shape, transform):
        self.shape = shape
        self.transform = transform
        self.sample_path = os.path.join(data_dir, 'samples')
        self.sample_files = os.listdir(self.sample_path)
        self.sample_files.sort()

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.sample_path, self.sample_files[index]))

        rand = random.randrange(3, len(sample) - 3)
        slice_position = (rand - 3) / (len(sample) - 6)
        partition = int((rand - 3) / (len(sample) - 6) * 4) + 1
        sample = sample[rand - 3:rand + 4]

        img1 = self.transform(sample)
        img2 = self.transform(sample)
        img1 = np.concatenate((img1[0], img1[1]))
        img2 = np.concatenate((img2[0], img2[1]))

        return torch.from_numpy(img1), torch.from_numpy(img2), torch.tensor(slice_position), torch.tensor(partition)
