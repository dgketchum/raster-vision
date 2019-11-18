from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


def get_model(model_arch, num_labels, pretrained=False, progress=True):

    if pretrained:
        pretrained_backbone = False
    else:
        pretrained_backbone = True

    backbone = resnet_fpn_backbone(model_arch, pretrained_backbone)
    model = MaskRCNN(backbone, num_labels)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        # noinspection PyUnresolvedReferences
        model.load_state_dict(state_dict)

    return model
