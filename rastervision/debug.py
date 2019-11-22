# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os
import numpy as np
import torch
from random import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.transforms import Normalize

from viz import display_instances


model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}

dir_ = '/home/dgketchum/field_extraction/training_data/example/WA/image_train'
out_dir_ = '/home/dgketchum/field_extraction/training_data/example/COCO/pred'

images = [os.path.join(dir_, x) for x in os.listdir(dir_)]
shuffle(images)

model_arch = 'resnet50'
num_labels = 91
batch_size = 8

device = torch.device("cuda:0")

backbone = resnet_fpn_backbone(model_arch, pretrained=True)
model = MaskRCNN(backbone, num_labels,
                 image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
                 rpn_pre_nms_top_n_test=5, rpn_nms_thresh=0.5, box_score_thresh=0.5,
                 box_nms_thresh=0.5)
state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'])
model.load_state_dict(state_dict)
model.to(device)
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
    if ct > 10:
        break


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
