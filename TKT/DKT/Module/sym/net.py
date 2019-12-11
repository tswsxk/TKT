# coding: utf-8
# create by tongshiwei on 2019-9-1

__all__ = ["get_net", "get_bp_loss"]

import torch
from torch import nn

from TKT.shared import SLMLoss
from TKT.shared import set_device

from longling.ML.PytorchHelper import sequence_mask
# from longling.ML.PytorchHelper import set_device, sequence_mask


def get_net(ku_num, hidden_num, nettype="DKT", dropout=0.0, **kwargs):
    if nettype in {"EmbedDKT", "DKT"}:
        return DKTNet(ku_num, hidden_num, nettype, dropout, **kwargs)
    else:
        raise TypeError("Unknown nettype: %s" % nettype)


def get_bp_loss(ctx, **kwargs):
    return {"SMLoss": set_device(SLMLoss(**kwargs), ctx)}


class DKTNet(nn.Module):
    def __init__(self, ku_num, hidden_num, nettype="DKT", dropout=0.0, **kwargs):
        super(DKTNet, self).__init__()
        self.ku_num = ku_num
        self.hidden_dim = hidden_num
        self.output_dim = ku_num
        if nettype == "EmbedDKT":
            self.embeddings = nn.Sequential(
                nn.Embedding(ku_num * 2, kwargs["latent_dim"]),
                nn.Dropout(kwargs.get("embedding_dropout", 0.2))
            )
            rnn_input_dim = kwargs["latent_dim"]
        else:
            self.embeddings = lambda x: torch.nn.functional.one_hot(x, num_classes=self.output_dim * 2).float()
            rnn_input_dim = ku_num * 2

        self.rnn = nn.RNN(rnn_input_dim, hidden_num, 1, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, responses, mask=None, begin_state=None):
        responses = self.embeddings(responses)
        output, hn = self.rnn(responses)
        output = self.sig(self.fc(self.dropout(output)))
        if mask is not None:
            output = sequence_mask(output, mask)
        return output, hn
