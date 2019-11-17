import glob
from os.path import join, basename

import numpy as np
from PIL import Image
from torch import as_tensor, float32, int64, ones, zeros, cat
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip

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
        # this is probably not a real problem
        # TODO need to loop over nb_classes, nb_features

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
            x = self.transforms(img)

        target = {'boxes': boxes, 'labels': labels, 'masks': masks}

        return x, target

    def __len__(self):
        return len(self.img_paths)


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
            Resize(img_sz),
            CenterCrop(img_sz),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_ds = InstanceSegmentationDataset(train_dir, transforms=data_transforms['train'])
    valid_ds = InstanceSegmentationDataset(valid_dir, transforms=data_transforms['val'])

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
