import random
import os
import glob
import numpy as np
np.random.seed(0)

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsat.transforms import transforms_cls

from skimage import io
from skimage.transform import rescale


class RandomApply(object):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(object, self).__init__(self,transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img


def resize_bands(img, size=120):
    return np.array(rescale(img, size/img.shape[0], anti_aliasing=False))

def load_patch(patch_dir):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    patch_name = os.path.basename(patch_dir)
    patch = [io.imread(os.path.join(patch_dir, f'{patch_name}_{band}.tif')) for band in bands]
    patch = np.stack([resize_bands(xx) for xx in patch], axis=2)
    return patch


class TileDataloader(Dataset):
    def __init__(self, tile_dir, transform):
        self.tile_dir = tile_dir
        self.tile_files = glob.glob(os.path.join(self.tile_dir, '*'))
        self.transform = transform
        self.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    def __len__(self):
        return len(self.tile_files)

    def __getitem__(self, idx):
        sample = load_patch(os.path.join(self.tile_dir, str(os.path.basename(self.tile_files[idx]))))
        sample = self.transform(sample)
        return sample


class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, input_dir):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        # train_dataset = datasets.STL10('./data',
        #                                split='train+unlabeled',
        #                                download=True,
        #                                transform=SimCLRDataTransform(data_augment))

        train_dataset = TileDataloader(tile_dir  = self.input_dir,
                                       transform = SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)

        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):

        gray = transforms_cls.ToGray()
        # Mising Color Jitter (and others check paper, probably need to be handcrafted).
        data_transforms = transforms_cls.Compose([transforms_cls.RandomResizedCrop(crop_size=int(self.input_shape[0]*.8), target_size=self.input_shape[0]),
                                                  transforms_cls.RandomHorizontalFlip(), # Missing Color Jitter
                                                  #RandomApply([gray], p=0.8), #Should be random w some probability
                                                  transforms_cls.GaussianBlur(kernel_size=13),
                                                  transforms_cls.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  sampler=train_sampler,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  shuffle=False)

        valid_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  sampler=valid_sampler,
                                  num_workers=self.num_workers,
                                  drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
