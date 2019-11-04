from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN, load_state_dict_from_url


def get_model(model_arch, num_labels, pretrained=False):
    model_urls = {
        'maskrcnn_resnet50_fpn_coco':
            'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
    }
    progress = True

    pretrained_backbone = False

    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_labels)
