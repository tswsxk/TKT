# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["fit_f", "eval_f"]

import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

from longling.ML.PytorchHelper import set_device, tensor2list, pick


def _fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    data, data_mask, label, pick_index, label_mask = _data
    output, _ = _net(data, data_mask)
    bp_loss = None
    for name, func in loss_function.items():
        loss = func(output, pick_index, label, label_mask)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = torch.mean(loss).item()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


def eval_f(_net, test_data, ctx=None):
    ground_truth = []
    prediction = []
    pred_labels = []

    if ctx is not None:
        _net = set_device(_net, ctx)

    for (data, data_mask, label, pick_index, label_mask) in tqdm(test_data, "evaluating"):
        with torch.no_grad():
            output, _ = _net(data, data_mask)
            output = output[:, :-1]
            output = pick(output, pick_index.to(output.device))
            pred = tensor2list(output)
            label = tensor2list(label)
        for i, length in enumerate(label_mask.numpy().tolist()):
            length = int(length)
            ground_truth.extend(label[i][:length])
            prediction.extend(pred[i][:length])
            pred_labels.extend([0 if p < 0.5 else 1 for p in pred[i][:length]])

    auc = roc_auc_score(ground_truth, prediction)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, pred_labels)

    evaluation_result = {}

    evaluation_result.update(
        {"precision_%d" % i: precision[i] for i in range(len(precision))}
    )
    evaluation_result.update(
        {"recall_%d" % i: recall[i] for i in range(len(recall))}
    )
    evaluation_result.update(
        {"f1_%d" % i: f1[i] for i in range(len(f1))}
    )

    evaluation_result.update({"auc": auc})
    return evaluation_result


# #####################################################################
# ###     The following codes usually do not need modification      ###
# #####################################################################

def fit_f(net, batch_data,
          trainer, bp_loss_f, loss_function, loss_monitor=None
          ):
    """
    Defined how each step of batch train goes

    Parameters
    ----------
    net: HybridBlock
        The network which has been initialized
        or loaded from the existed model
    batch_data: Iterable
        The batch data for train
    trainer:
        The trainer used to update the parameters of the net
    bp_loss_f: dict with only one value and one key
        The function to compute the loss for the procession
        of back propagation
    loss_function: dict of function
        Some other measurement in addition to bp_loss_f
    loss_monitor: LossMonitor
        Default to ``None``
    ctx: Context or list of Context
        Defaults to ``mx.cpu()``.

    Returns
    -------

    """
    bp_loss = _fit_f(
        net, batch_data, bp_loss_f, loss_function, loss_monitor
    )
    assert bp_loss is not None
    trainer.zero_grad()
    bp_loss.backward()
    trainer.step()
