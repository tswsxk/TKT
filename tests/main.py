import torch
import torch.nn as nn

from TKT.shared import set_device


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Linear(100, 1000)

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    # _net = set_device(FC(), "cuda")
    # _net(torch.ones((16, 100)))

    print(FC().__doc__)
