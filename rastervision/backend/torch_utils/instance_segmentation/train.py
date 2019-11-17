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
                   out['loss_objectness'] + 0.1 * out['loss_rpn_box_reg']

            total_loss += torch.tensor([v for (k, v) in out.items()]).sum()

            loss.backward()
            opt.step()

        if step_scheduler:
            step_scheduler.step()

        num_samples += len(x)

    return out


def validate_epoch(model, device, data_loader, num_labels):
    model.eval()

    conf_mat = torch.zeros((num_labels, num_labels))
    with torch.no_grad():
        with click.progressbar(data_loader, label='Validating') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = x.to(device)
                out = model(x)['out']

                out = out.argmax(1).view(-1).cpu()
                y = y.view(-1).cpu()
                conf_mat += compute_conf_mat(out, y, num_labels)

    # Ignore index zero.
    conf_mat = conf_mat[1:, 1:]
    return compute_conf_mat_metrics(conf_mat)
