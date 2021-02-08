import torch.nn as nn
from .sequence_wise import SequenceWise
import torch.nn.functional as F


class BatchConv(nn.Module):
    # lookahead: https://arxiv.org/pdf/1609.03193.pdf
    # wav2letter: http://research.baidu.com/Public/uploads/5ac0504e3ed84.pdf
    def __init__(self, input_size, kernel_size, algorithm):
        super(BatchConv, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = 1
        self.algorithm = algorithm

        self.norm = nn.Sequential(
            SequenceWise(nn.BatchNorm1d(self.input_size)),
        )

        if self.algorithm == 'lookahead':
            self.pad = (0, self.kernel_size - 1)
        elif self.algorithm == 'wav2letter':
            if self.kernel_size // 2 == 0:
                raise AttributeError('kernel_size must be odd')
            _pad = (self.kernel_size - 1) // 2
            self.pad = (_pad, _pad)
        else:
            raise AttributeError('model.algorithm must be lookahead or wav2letter')
        self.conv = nn.Conv1d(
            self.input_size,
            self.input_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.input_size,
            padding=0,
            bias=False
        )

        self.activation = nn.Hardtanh(0, 20, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        x = self.activation(x)
        return x