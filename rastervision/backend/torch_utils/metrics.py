import torch


def compute_conf_mat(out, y, num_labels):
    labels = torch.arange(0, num_labels)
    return ((out == labels[:, None]) & (y == labels[:, None, None])).sum(
        dim=2, dtype=torch.float32)


def compute_conf_mat_metrics(conf_mat, eps=1e-6):
    # eps is to avoid dividing by zero.
    eps = torch.tensor(eps)
    gt_count = conf_mat.sum(dim=1)
    pred_count = conf_mat.sum(dim=0)
    total = conf_mat.sum()
    true_pos = torch.diag(conf_mat)
    precision = true_pos / torch.max(pred_count, eps)
    recall = true_pos / torch.max(gt_count, eps)

    weights = gt_count / total
    weighted_precision = (weights * precision).sum()
    weighted_recall = (weights * recall).sum()
    weighted_f1 = ((2 * weighted_precision * weighted_recall) / torch.max(
        weighted_precision + weighted_recall, eps))
    metrics = {
        'precision': weighted_precision.item(),
        'recall': weighted_recall.item(),
        'f1': weighted_f1.item()
    }
    return metrics


def intersection_over_union_bbox(a_box, b_box):

    if a_box.shape[-1] == 2 or len(a_box.shape) == 1:
        a_box = a_box.reshape(1, 4) if len(a_box.shape) <= 2 else a_box.reshape(a_box.shape[0], 4)
    if b_box.shape[-1] == 2 or len(b_box.shape) == 1:
        b_box = b_box.reshape(1, 4) if len(b_box.shape) <= 2 else b_box.reshape(b_box.shape[0], 4)
    point_num = max(a_box.shape[0], b_box.shape[0])
    b1p1, b1p2, b2p1, b2p2 = a_box[:, :2], a_box[:, 2:], b_box[:, :2], b_box[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = torch.ones((point_num,))
    base_mat *= torch.all(torch.gt(b1p2 - b2p1, 0), 1)
    base_mat *= torch.all(torch.gt(b2p2 - b1p1, 0), 1)

    # I area
    intersect_area = torch.prod(torch.min(b2p2, b1p2) - torch.max(b1p1, b2p1), 1)
    # U area
    union_area = torch.prod(b1p2 - b1p1, 1) + torch.prod(b2p2 - b2p1, 1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio