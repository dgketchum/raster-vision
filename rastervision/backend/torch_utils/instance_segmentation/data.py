import glob
from os.path import join, basename

import numpy as np
from PIL import Image
from torch import as_tensor, float32, int64, ones, zeros
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip

from rastervision.backend.torch_utils.data import DataBunch


class InstanceSegmentationDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_paths = glob.glob(join(data_dir, 'img', '*.png'))
        self.transforms = transforms

    def __getitem__(self, ind):

        img_path = self.img_paths[ind]
        label_path = join(self.data_dir, 'labels', basename(img_path))

        img = Image.open(img_path).convert('RGB')

        y = Image.open(label_path)
        mask = np.array(y)
        features = np.unique(mask)
        features = features[1:]
        masks = mask == features[:, None, None]

        nb_features = len(features)

        if nb_features == 0:
            x, target = self.make_background_feature(img, img_path)
            return x, target

        boxes = []
        for i in range(nb_features):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = as_tensor(boxes, dtype=float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = ones((nb_features,), dtype=int64)
        masks = as_tensor(masks, dtype=int64)

        try:
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]
            masks = masks[keep]
            area = area[keep]

        except IndexError:
            pass

        # consider reinstating id_
        # consider reinstating iscrowd

        if self.transforms is not None:
            x = self.transforms(img)

        if area.sum() < 1.0:
            x, target = self.make_background_feature(img, img_path)
            return x, target

        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'area': area, 'path': img_path}
        return x, target

    def __len__(self):
        return len(self.img_paths)

    def make_background_feature(self, img, path):
        area = as_tensor([img.size[0] * img.size[1]], dtype=int64)
        masks = ones((1, img.size[0], img.size[1]), dtype=int64)
        labels = zeros((1,), dtype=int64)
        boxes = [[0, 0, img.size[0], img.size[1]]]
        boxes = as_tensor(boxes, dtype=float32)
        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'area': area, 'path': path}
        x = self.transforms(img)
        return x, target


def build_databunch(data_dir, img_sz, batch_sz, class_names):
    num_workers = 0

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    data_transforms = {
        'train': Compose([
            RandomResizedCrop(img_sz),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_ds = InstanceSegmentationDataset(train_dir, transforms=data_transforms['train'])
    valid_ds = InstanceSegmentationDataset(valid_dir, transforms=data_transforms['val'])

    train_dl = DataLoader(
        train_ds,
        shuffle=False,
        batch_size=batch_sz,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn)

    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)


def collate_fn(data):
    x, y = [], []
    for d in data:
        if d:
            x.append(d[0]), y.append(d[1])
    return x, y
