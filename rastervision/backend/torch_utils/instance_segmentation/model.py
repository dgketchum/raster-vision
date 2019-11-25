from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(num_classes=91, pretrained=True):

    if pretrained or num_classes == 91:
        assert num_classes == 91
        model = maskrcnn_resnet50_fpn(pretrained=True)

    elif num_classes != 91:
        # TODO: specify user options for hyper-parameters
        model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                           dim_reduced=hidden_layer,
                                                           num_classes=num_classes)
    else:
        raise NotImplementedError

    return model
