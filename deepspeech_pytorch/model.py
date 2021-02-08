import math
from typing import List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import CTCLoss

from deepspeech_pytorch.configs.train_config import SpectConfig, BiDirectionalConfig, OptimConfig, AdamConfig, \
    SGDConfig, UniDirectionalConfig, ConvolutionConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.validation import CharErrorRate, WordErrorRate

from deepspeech_pytorch.modules import SequenceWise, MaskConv, InferenceBatchSoftmax, \
    BatchRNN, Lookahead, BatchConv


class DeepSpeech(pl.LightningModule):
    def __init__(self,
                 labels: List,
                 model_cfg: Union[UniDirectionalConfig, BiDirectionalConfig, ConvolutionConfig],
                 precision: int,
                 optim_cfg: Union[AdamConfig, SGDConfig],
                 spect_cfg: SpectConfig
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.convolutional = True if OmegaConf.get_type(model_cfg) is ConvolutionConfig else False
        self.bidirectional = True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False

        self.labels = labels

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        if self.convolutional is False:
            self.rnns, self.lookahead, self.fc = self._rnn_construct(rnn_input_size)

        else:
            self.deep_conv, self.fc = self._conv_construct(rnn_input_size)

        self.inference_softmax = InferenceBatchSoftmax()
        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )

    def _conv_construct(self, input_size):
        deep_conv = nn.Sequential(*(
            BatchConv(input_size,
                      self.model_cfg.kernel_size,
                      self.model_cfg.algorithm)
            for _ in range(self.model_cfg.depth)
        ))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, len(self.labels), bias=False)
        )

        fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        return deep_conv, fc

    def _rnn_construct(self, input_size):
        rnns = nn.Sequential(
            BatchRNN(
                input_size=input_size,
                hidden_size=self.model_cfg.hidden_size,
                rnn_type=self.model_cfg.rnn_type.value,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.model_cfg.hidden_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=self.model_cfg.rnn_type.value,
                    bidirectional=self.bidirectional
                ) for x in range(self.model_cfg.hidden_layers - 1)
            )
        )

        lookahead = nn.Sequential(
            Lookahead(self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.model_cfg.hidden_size),
            nn.Linear(self.model_cfg.hidden_size, len(self.labels), bias=False)
        )
        fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        return rnns, lookahead, fc

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        if self.convolutional:
            # x = self.deep_conv(x)
            for conv in self.deep_conv:
                x = conv(x)
        else:
            for rnn in self.rnns:
                x = rnn(x, output_lengths)

            if not self.bidirectional:  # no need for lookahead layer in bidirectional
                x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        out, output_sizes = self(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)

        loss = self.criterion(out, targets, output_sizes, target_sizes)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_sizes)
        self.wer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.cer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        if OmegaConf.get_type(self.optim_cfg) is SGDConfig:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay
            )
        elif OmegaConf.get_type(self.optim_cfg) is AdamConfig:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()
