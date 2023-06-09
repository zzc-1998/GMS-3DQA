import torch.nn as nn
# import torch
# from torchvision.ops import roi_pool, roi_align
# from torch.nn import functional as F
import numpy as np
import math


class IQAHead(nn.Module):
    """MLP Regression Head for IQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc_last = nn.Linear(self.hidden_channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score
