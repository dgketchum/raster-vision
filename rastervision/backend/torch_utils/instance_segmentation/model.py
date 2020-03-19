import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from rastervision.backend.torch_utils.object_detection.boxlist import BoxList


def get_out_channels(model):
    out = {}

    def make_save_output(layer_name):
        def save_output(layer, input, output):
            out[layer_name] = output.shape[1]

        return save_output

    model.layer1.register_forward_hook(make_save_output('layer1'))
    model.layer2.register_forward_hook(make_save_output('layer2'))
    model.layer3.register_forward_hook(make_save_output('layer3'))
    model.layer4.register_forward_hook(make_save_output('layer4'))

    model(torch.empty((1, 3, 128, 128)))
    return [out['layer1'], out['layer2'], out['layer3'], out['layer4']]


def get_model(num_classes=91, pretrained=False):

    if pretrained or num_classes == 91:
        assert num_classes == 91
        model = maskrcnn_resnet50_fpn(pretrained=True)

    elif num_classes != 91:

        # TODO: specify user options for hyper-parameters
        model = maskrcnn_resnet50_fpn(pretrained=True)

        # freeze/unfreeze layers
        unfreeze_heads = ['cls_score', 'bbox_pred', 'mask_fcn_logits']
        unfreeze_layers = ['layer2', 'layer3', 'layer4', 'fpn']

        for name, parameter in model.named_parameters():
            if any(unf in name for unf in unfreeze_layers):
                parameter.requires_grad_(True)
                # print('grad', name)
            elif any(unf in name for unf in unfreeze_heads):
                parameter.requires_grad_(True)
                # print('grad', name)
            else:
                parameter.requires_grad_(False)
                # print('no grad', name)

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


if __name__ == '__main__':
    get_model(num_classes=2, pretrained=False)
