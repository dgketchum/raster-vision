import click
import torch
from torch.autograd import detect_anomaly
from rastervision.backend.torch_utils.metrics import (compute_conf_mat,
                                                      compute_conf_mat_metrics)


def train_epoch(model, device, data_loader, opt, loss_fn, step_scheduler=None):
    model.train()
    total_loss = 0
    num_samples = 0

    with click.progressbar(data_loader, label='Training') as bar:
        for batch_ind, (x, target) in enumerate(bar):

            # check for empty boxes, which corrupt the model
            # should this be in data.py?
            not_empty = [True if t['boxes'].size()[0] > 0 else False for t in target]
            target = [{k: v.to(device) for (k, v) in dict_.items()} for dict_, c in zip(target, not_empty) if c]

            x = [t.to(device) for t, c in zip(x, not_empty) if c]

            out = model(x, target)
            opt.zero_grad()

            loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_mask'] + \
                   out['loss_objectness'] + out['loss_rpn_box_reg']

            loss.backward()
            opt.step()
            if step_scheduler:
                step_scheduler.step()

            loss_floats = {k: v.item() for (k, v) in out.items()}
            if torch.isnan(loss):
                num_samples += len(x)

            print(len(x))
        return loss_floats


def validate_epoch(model, device, data_loader, num_labels):
    model.eval()

    conf_mat = torch.zeros((num_labels, num_labels))
    with torch.no_grad():
        with click.progressbar(data_loader, label='Validating') as bar:
            for batch_ind, (x, target) in enumerate(bar):
                pass
                # x = [_.to(device) for _ in x]
                # out = model(x)

    # Ignore index zero.
    conf_mat = conf_mat[1:, 1:]
    return compute_conf_mat_metrics(conf_mat)
