from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.utils import load_state_dict_from_url


def get_model(num_classes=91, pretrained=True):

    if pretrained:
        assert num_classes == 91
        model = maskrcnn_resnet50_fpn(pretrained=True)

    else:
        # TODO: specify user options for hyper-parameters
        backbone = maskrcnn_resnet50_fpn(pretrained=False).backbone
        model = MaskRCNN(backbone, num_classes,
                         image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225),
                         rpn_pre_nms_top_n_test=15, rpn_nms_thresh=0.5, box_score_thresh=0.5,
                         box_nms_thresh=0.5)

    return model
