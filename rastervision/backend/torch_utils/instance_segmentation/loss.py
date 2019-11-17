"""
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable


############################################################
#  Loss Functions -
#  From https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/model.py
############################################################


class MaskRCNNLoss(object):

    def __init__(self, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                 mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask):
        self.rpn_match = rpn_match
        self.rpn_bbox = rpn_bbox
        self.rpn_class_logits = rpn_class_logits
        self.rpn_pred_bbox = rpn_pred_bbox
        self.target_class_ids = target_class_ids
        self.mrcnn_class_logits = mrcnn_class_logits
        self.target_deltas = target_deltas
        self.mrcnn_bbox = mrcnn_bbox
        self.target_mask = target_mask
        self.mrcnn_mask = mrcnn_mask

    def compute_rpn_class_loss(self):
        """RPN anchor classifier loss.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
        """

        # Squeeze last dim to simplify
        rpn_match = self.rpn_match.squeeze(2)

        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = (rpn_match == 1).long()

        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = torch.nonzero(rpn_match != 0)

        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = self.rpn_class_logits[indices.data[:, 0], indices.data[:, 1], :]
        anchor_class = anchor_class[indices.data[:, 0], indices.data[:, 1]]

        # Crossentropy loss
        loss = F.cross_entropy(rpn_class_logits, anchor_class)

        return loss

    def compute_rpn_bbox_loss(self):
        """Return the RPN bounding box loss graph.
        target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                   -1=negative, 0=neutral anchor.
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        """

        # Squeeze last dim to simplify
        rpn_match = self.rpn_match.squeeze(2)

        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        indices = torch.nonzero(rpn_match == 1)

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = self.rpn_bbox[indices.data[:, 0], indices.data[:, 1]]

        # Trim target bounding box deltas to the same length as rpn_bbox.
        target_bbox = self.target_bbox[0, :rpn_bbox.size()[0], :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

        return loss

    def compute_mrcnn_class_loss(self):
        """Loss for the classifier head of Mask RCNN.
        target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        pred_class_logits: [batch, num_rois, num_classes]
        """

        # Loss
        if self.target_class_ids.size():
            loss = F.cross_entropy(self.pred_class_logits, self.target_class_ids.long())
        else:
            loss = Variable(torch.FloatTensor([0]), requires_grad=False)
            if self.target_class_ids.is_cuda:
                loss = loss.cuda()

        return loss

    def compute_mrcnn_bbox_loss(self):
        """Loss for Mask R-CNN bounding box refinement.
        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """

        if self.target_class_ids.size():
            # Only positive ROIs contribute to the loss. And only
            # the right class_id of each ROI. Get their indicies.
            positive_roi_ix = torch.nonzero(self.target_class_ids > 0)[:, 0]
            positive_roi_class_ids = self.target_class_ids[positive_roi_ix.data].long()
            indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

            # Gather the deltas (predicted and true) that contribute to loss
            target_bbox = self.target_bbox[indices[:, 0].data, :]
            pred_bbox = self.pred_bbox[indices[:, 0].data, indices[:, 1].data, :]

            # Smooth L1 loss
            loss = F.smooth_l1_loss(pred_bbox, target_bbox)
        else:
            loss = Variable(torch.FloatTensor([0]), requires_grad=False)
            if self.target_class_ids.is_cuda:
                loss = loss.cuda()

        return loss

    def compute_mrcnn_mask_loss(self):
        """Mask binary cross-entropy loss for the masks head.
        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
        if self.target_class_ids.size():
            # Only positive ROIs contribute to the loss. And only
            # the class specific mask of each ROI.
            positive_ix = torch.nonzero(self.target_class_ids > 0)[:, 0]
            positive_class_ids = self.target_class_ids[positive_ix.data].long()
            indices = torch.stack((positive_ix, positive_class_ids), dim=1)

            # Gather the masks (predicted and true) that contribute to loss
            y_true = self.target_masks[indices[:, 0].data, :, :]
            y_pred = self.pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]

            # Binary cross entropy
            loss = F.binary_cross_entropy(y_pred, y_true)
        else:
            loss = Variable(torch.FloatTensor([0]), requires_grad=False)
            if self.target_class_ids.is_cuda:
                loss = loss.cuda()

        return loss

    def compute_losses(self):
        rpn_class_loss = self.compute_rpn_class_loss()
        rpn_bbox_loss = self.compute_rpn_bbox_loss()
        mrcnn_class_loss = self.compute_mrcnn_class_loss()
        mrcnn_bbox_loss = self.compute_mrcnn_bbox_loss()
        mrcnn_mask_loss = self.compute_mrcnn_mask_loss()

        return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]