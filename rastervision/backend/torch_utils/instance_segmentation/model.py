from torchvision import models


def get_model(model_arch, num_labels, pretrained=True):
    model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=num_labels, pretrained_backbone=True)
    return model
