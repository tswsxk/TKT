# coding: utf-8
# 2021/8/18 @ tongshiwei

from EduKTM import KTM as _KTM
from baize import path_append, get_params_filepath
from baize.const import CFG_JSON
from baize.torch import save_params, load_net, Configuration


class KTM(_KTM):
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.net = None

    def train(self, train_data, valid_data=None, re_init_net=False, enable_hyper_search=False,
              save=False, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, test_data, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, model_dir=None, *args, **kwargs) -> ...:
        model_dir = model_dir if model_dir is not None else self.cfg.model_dir
        select = kwargs.get("select", self.cfg.save_select)
        save_params(get_params_filepath(self.cfg.model_name, model_dir), self.net, select)
        self.cfg.dump(path_append(model_dir, CFG_JSON, to_str=True))
        return model_dir

    def load(self, model_path, *args, **kwargs) -> ...:
        load_net(model_path, self.net)

    @classmethod
    def from_pretrained(cls, model_dir, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def benchmark_train(cls, train_path, valid_path=None, enable_hyper_search=False,
                        save=False, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def benchmark_eval(cls, test_path, model_path, best_epoch, *args, **kwargs):
        raise NotImplementedError
