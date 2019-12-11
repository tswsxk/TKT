# coding: utf-8
# 2019/9/28 @ tongshiwei

import torch
from torch.nn import DataParallel


def set_device(_net, ctx, **kwargs):
    if ctx == "cpu":
        return _net
    elif any(map(lambda x: x in ctx, ["cuda", "gpu"])):
        if not torch.cuda.is_available():
            try:
                torch.ones((1,), device=torch.device("cuda: 0"))
            except AssertionError as e:
                raise TypeError("no cuda detected, noly cpu is supported, the detailed error msg:%s" % str(e))
        if torch.cuda.device_count() > 1:
            if ":" in ctx:
                ctx_name, device_ids = ctx.split(":")
                assert ctx_name in ["cuda", "gpu"], "the equipment should be 'cpu', 'cuda' or 'gpu', now is %s" % ctx
                device_ids = [int(i) for i in device_ids.strip().split(",")]
                return DataParallel(_net, device_ids)
            elif ctx in ["cuda", "gpu"]:
                return DataParallel(_net).cuda()
            else:
                raise TypeError("the equipment should be 'cpu', 'cuda' or 'gpu', now is %s" % ctx)
        else:
            raise TypeError("0 gpu can be used, use cpu")
    else:
        return DataParallel(_net, device_ids=ctx).cuda()
