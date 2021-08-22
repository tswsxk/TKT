# coding: utf-8
# 2021/8/18 @ tongshiwei

from tqdm import tqdm
from baize import path_append, get_epoch_params_filepath, get_params_filepath
from baize.const import CFG_JSON
from baize.metrics import classification_report
from baize.torch import (
    light_module as lm, Configuration, ConfigurationParser, fit_wrapper, eval_wrapper
)
from baize.torch.functional import pick, tensor2list
from TKT import KTM
from EduKTM.DKTPlus.DKTPlus import DKTNet
from EduKTM.DKTPlus import etl as _etl
from EduKTM.utils import SLMLoss

__all__ = ["etl", "Configuration", "ConfigurationParser", "DKT"]


def etl(data_src, cfg: Configuration = None, batch_size=None):
    batch_size = batch_size if batch_size is not None else cfg.batch_size
    return _etl(data_src, batch_size)


@fit_wrapper
def fit(net, batch_data, loss_function, *args, **kwargs):
    response, data_mask, label, next_item_id, label_mask = batch_data
    predicted_response, _ = net(response, data_mask)
    loss = []
    loss.append(loss_function(predicted_response, next_item_id, label, label_mask))
    return sum(loss)


@eval_wrapper
def evaluation(net, test_data, *args, **kwargs):
    y_true = []
    y_pred_score = []
    y_pred_label = []
    for (data, data_mask, label, pick_index, label_mask) in tqdm(test_data, "evaluating"):
        output, _ = net(data, data_mask)
        output = output[:, :-1]
        output = pick(output, pick_index.to(output.device))
        pred = tensor2list(output)
        label = tensor2list(label)
        for i, length in enumerate(label_mask.numpy().tolist()):
            length = int(length)
            y_true.extend(label[i][:length])
            y_pred_score.extend(pred[i][:length])
            y_pred_label.extend([1 if s >= 0.5 else 0 for s in pred[i][:length]])
    return classification_report(y_true, y_pred_label, y_pred_score)


def get_net(**kwargs):
    return DKTNet(**kwargs)


class DKT(KTM):
    def __init__(self, init_net=True, cfg_path=None, *args, **kwargs):
        super(DKT, self).__init__(Configuration(params_path=cfg_path, *args, **kwargs))
        if init_net:
            self.net = DKTNet(**self.cfg.hyper_params)

    def train(self, train_data, valid_data=None, re_init_net=False, enable_hyper_search=False,
              save=False, *args, **kwargs) -> ...:
        self.cfg.update(**kwargs)

        print(self.cfg)

        lm.train(
            net=self.net,
            cfg=self.cfg,
            get_net=get_net if re_init_net is True else None,
            fit_f=fit,
            eval_f=evaluation,
            trainer=None,
            loss_function=SLMLoss(**self.cfg.loss_params),
            train_data=train_data,
            test_data=valid_data,
            enable_hyper_search=enable_hyper_search,
            dump_result=save,
            params_save=save,
            primary_key="macro_auc"
        )

    def eval(self, test_data, *args, **kwargs) -> ...:
        return evaluation(self.net, test_data, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_dir, best_epoch=None, *args, **kwargs):
        cfg_path = path_append(model_dir, CFG_JSON)
        model = DKT(init_net=True, cfg_path=cfg_path, model_dir=model_dir)
        cfg = model.cfg
        model.load(
            get_epoch_params_filepath(cfg.model_name, best_epoch, cfg.model_dir)
            if best_epoch is not None else get_params_filepath(cfg.model_name, cfg.model_dir)
        )
        return model

    @classmethod
    def benchmark_train(cls, train_path, valid_path=None, enable_hyper_search=False,
                        save=False, *args, **kwargs):
        dkt = DKT(init_net=not enable_hyper_search, *args, **kwargs)
        train_data = etl(train_path, dkt.cfg)
        valid_data = etl(valid_path, dkt.cfg) if valid_path is not None else None
        dkt.train(train_data, valid_data, re_init_net=enable_hyper_search, enable_hyper_search=enable_hyper_search,
                  save=save)

    @classmethod
    def benchmark_eval(cls, test_path, model_path, best_epoch, *args, **kwargs):
        dkt = DKT.from_pretrained(model_path, best_epoch)
        test_data = etl(test_path, dkt.cfg)
        return dkt.eval(test_data)
