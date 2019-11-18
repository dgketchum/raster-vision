import click
import torch

from rastervision.backend.torch_utils.metrics import (compute_conf_mat,
                                                      compute_conf_mat_metrics)


def train_epoch(model, device, data_loader, opt, loss_fn, step_scheduler=None):
    model.train()
    total_loss = 0
    num_samples = 0

    with click.progressbar(data_loader, label='Training') as bar:
        for batch_ind, (x, target) in enumerate(bar):
            x = [_.to(device) for _ in x]
            target = [{k: v.to(device) for (k, v) in dict_.items()} for dict_ in target]
            out = model(x, target)
            opt.zero_grad()

            loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_mask'] + \
                   out['loss_objectness'] + 0.0625 * out['loss_rpn_box_reg']

            loss.backward()

            total_loss += torch.tensor([v for (k, v) in out.items()]).sum()

            opt.step()

        if step_scheduler:
            step_scheduler.step()

        num_samples += len(x)

        loss_floats = {k: v.item() for (k, v) in out.items()}
        loss_floats['total_loss'] = total_loss

    return loss_floats


def validate_epoch(model, device, data_loader, num_labels):
    model.eval()

    conf_mat = torch.zeros((num_labels, num_labels))
    with torch.no_grad():
        with click.progressbar(data_loader, label='Validating') as bar:
            for batch_ind, (x, target) in enumerate(bar):
                x = [_.to(device) for _ in x]
                out = model(x)

    # Ignore index zero.
    conf_mat = conf_mat[1:, 1:]
    return compute_conf_mat_metrics(conf_mat)
