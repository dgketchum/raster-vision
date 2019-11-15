import glob
from os.path import join, basename

import numpy as np
from PIL import Image
from torch import as_tensor, float32, int64, ones, zeros, cat
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

from rastervision.backend.torch_utils.data import DataBunch


# class ToTensor(object):
#     def __init__(self):
#         self.to_tensor = torchvision.transforms.ToTensor()
#
#     def __call__(self, x, y):
#         return (self.to_tensor(x), (255 * self.to_tensor(y)).squeeze().long())


class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class InstanceSegmentationDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_paths = glob.glob(join(data_dir, 'img', '*.png'))
        self.transforms = transforms

    def __getitem__(self, ind):

        img_path = self.img_paths[ind]
        label_path = join(self.data_dir, 'labels', basename(img_path))

        img = Image.open(img_path).convert('RGB')
        x = np.array(img)

        y = Image.open(label_path)
        mask = np.array(y)
        features = np.unique(mask)
        features = features[1:]
        masks = mask == features[:, None, None]

        max_features = 15
        nb_features = len(features)

        boxes = []
        for i in range(max_features):
            if i < nb_features:
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                boxes.append([0, 0, 0, 0])
        boxes = as_tensor(boxes, dtype=float32)

        # the following is a hack to generate labels and masks of consistent shape,
        # regardless of the number of features in the training data
        # to get around exception:
        # RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0.
        # this is probably not a real problem
        # this will need to loop over nb_classes, nb_features

        nb_empty = max_features - nb_features

        labels = ones((nb_features,), dtype=int64)
        null_labels = zeros((nb_empty,), dtype=int64)
        labels = cat([labels, null_labels])

        masks = as_tensor(masks, dtype=int64)
        null_masks = zeros((nb_empty, mask.shape[0], mask.shape[1]), dtype=int64)
        masks = cat([masks, null_masks])

        # consider reinstating id_
        # consider reinstating area
        # consider reinstating iscrowd

        if self.transforms is not None:
            x = self.transforms(x)

        target = {'boxes': boxes, 'labels': labels, 'masks': masks}

        return x, target

    def __len__(self):
        return len(self.img_paths)


def build_databunch(data_dir, img_sz, batch_sz, class_names):

    num_workers = 0

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    aug_transforms = Compose([ToTensor()])
    transforms = Compose([ToTensor()])

    train_ds = InstanceSegmentationDataset(train_dir, transforms=aug_transforms)
    valid_ds = InstanceSegmentationDataset(valid_dir, transforms=transforms)

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn)

    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)


def collate_fn(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    return x, y
