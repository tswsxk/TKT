# coding: utf-8
# 2021/5/26 @ tongshiwei

from TKT.DKT import etl
from TKT import DKT

import torch

torch.manual_seed(0)

batch_size = 64
train = etl("../../data/a0910c/train.json", batch_size=batch_size)
valid = etl("../../data/a0910c/valid.json", batch_size=batch_size)
test = etl("../../data/a0910c/test.json", batch_size=batch_size)

model = DKT(hyper_params=dict(ku_num=146, hidden_num=100))
model.train(train, valid, end_epoch=2)
model.save("dkt")

model = DKT.from_pretrained("dkt")
print(model.eval(test))
