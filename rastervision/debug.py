import os
from random import shuffle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.utils import load_state_dict_from_url
from torchvision.transforms import ToTensor
from viz import display_instances

model_urls = {'maskrcnn_resnet50_fpn_coco':
                  'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'}

dir_ = '/home/dgketchum/field_extraction/training_data/example/COCO/image_train'
out_dir_ = '/home/dgketchum/field_extraction/training_data/example/COCO/pred'
images = [os.path.join(dir_, x) for x in os.listdir(dir_)]
shuffle(images)


def load_dataset():
    training_dataset = ImageFolder(root=dir_, transform=ToTensor())
    train_loader = DataLoader(
        training_dataset,
        batch_size=6,
        num_workers=0,
        shuffle=True)
    return train_loader


def get_model():
    model_arch = 'resnet50'
    num_labels = 91
    device = torch.device("cuda:0")
    backbone = resnet_fpn_backbone(model_arch, pretrained=True)
    model = MaskRCNN(backbone, num_labels,
                     image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
                     rpn_pre_nms_top_n_test=5, rpn_nms_thresh=0.5, box_score_thresh=0.5,
                     box_nms_thresh=0.5)
    state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'])
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def train(model, load_dataset):
    model = model.train()
    for batch_idx, (data, target) in enumerate(load_dataset()):
        x = [_.to(device) for _ in x]
        target = [{k: v.to(device) for (k, v) in dict_.items()} for dict_ in target]
        out = model(x, target)
        opt.zero_grad()

        # TODO figure out why loss_rpn_box_reg gos to inf
        # removed 'loss_rpn_box_reg' because it goes to inf
        loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_mask'] + \
               out['loss_objectness'] + torch.log(out['loss_rpn_box_reg'])

        loss.backward()
        print(loss)


def visualize(model, images, max_images):
    ct = 0
    model = model.eval()
    for image in images:
        im = np.array(Image.open(image).convert('RGB'))
        arr = torch.from_numpy(im).to(device) / 255.
        arr = [arr.permute(2, 0, 1)]
        with torch.no_grad():
            out = model(arr)
        boxes = out[0]['boxes'].cpu().numpy()
        box_permute = [1, 0, 3, 2]
        boxes = boxes[:, np.argsort(box_permute)]
        masks = out[0]['masks'].squeeze(1).permute(1, 2, 0).cpu().numpy()
        labels = out[0]['labels'].cpu().numpy()
        img = arr[0].permute(1, 2, 0).cpu().numpy()
        class_ids = np.array([x for x in range(len(labels))])
        class_names = [str(x) for x in labels]
        display_instances(im, boxes=boxes, masks=masks,
                          class_ids=class_ids,
                          class_names=class_names,
                          show_mask=True, show_bbox=True)
        ct += 1
        if ct >= max_images:
            break


if __name__ == '__main__':
    model = get_model()
    train(model, load_dataset)
# ========================= EOF ====================================================================
