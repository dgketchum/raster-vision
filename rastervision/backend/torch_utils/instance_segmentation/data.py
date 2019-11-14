from os.path import join, basename
import glob

from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch import as_tensor, tensor, float32, int64, uint8, ones, zeros

from rastervision.backend.torch_utils.data import DataBunch


class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, x, y):
        return (self.to_tensor(x), (255 * self.to_tensor(y)).squeeze().long())


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
        x = Image.open(img_path)
        y = Image.open(label_path)
        mask = np.array(y)
        features = np.unique(mask)
        features = features[1:]
        masks = mask == features[:, None, None]
        boxes = []
        for i in range(len(features)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = as_tensor(boxes, dtype=float32)
        # following assumes one class
        labels = ones(tuple(features), dtype=int64)
        masks = as_tensor(masks, dtype=uint8)
        image_id = tensor([ind])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = zeros(tuple(features), dtype=int64)

        if self.transforms is not None:
            x, y = self.transforms(x, y)

        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'label_arr': y,
                  'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        return x, target

    def __len__(self):
        return len(self.img_paths)


def build_databunch(data_dir, img_sz, batch_sz, class_names):
    # set to zero to prevent "dataloader is killed by signal"
    # TODO fix this
    num_workers = 0

    train_dir = join(data_dir, 'train')
    valid_dir = join(data_dir, 'valid')

    aug_transforms = ComposeTransforms([ToTensor()])
    transforms = ComposeTransforms([ToTensor()])

    train_ds = InstanceSegmentationDataset(train_dir, transforms=aug_transforms)
    valid_ds = InstanceSegmentationDataset(valid_dir, transforms=transforms)

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_sz,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True)

    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, class_names)
