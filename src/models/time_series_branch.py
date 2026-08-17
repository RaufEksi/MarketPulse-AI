"""
PyTorch Time-Series Branch: Bi-LSTM & Temporal Convolutional Network (TCN) Encoders.
Input: [Batch, Sequence_Length, Num_Features] -> Output: [Batch, Sequence_Length, Hidden_Dim]
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Slices causal padding off the end of conv output."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual causal dilated convolution block for TCN."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TimeSeriesEncoder(nn.Module):
    """
    Time-Series encoder supporting Bidirectional LSTM and TCN.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: str = "bilstm",
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.hidden_dim = hidden_dim

        if self.model_type == "bilstm":
            lstm_hidden = hidden_dim // 2
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)
        elif self.model_type == "tcn":
            layers = []
            num_channels = [64, hidden_dim]
            for i in range(len(num_channels)):
                dilation_size = 2**i
                in_ch = input_dim if i == 0 else num_channels[i - 1]
                out_ch = num_channels[i]
                layers.append(
                    TemporalBlock(
                        in_ch,
                        out_ch,
                        kernel_size=3,
                        stride=1,
                        dilation=dilation_size,
                        padding=(3 - 1) * dilation_size,
                        dropout=dropout,
                    )
                )
            self.tcn = nn.Sequential(*layers)
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown time series encoder type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, Seq_Len, Input_Dim]
        Output: [Batch, Seq_Len, Hidden_Dim]
        """
        if self.model_type == "bilstm":
            out, _ = self.lstm(x)
            return self.layer_norm(out)
        else:
            # TCN expects [Batch, In_Channels, Seq_Len]
            x_perm = x.transpose(1, 2)
            out = self.tcn(x_perm)
            out_perm = out.transpose(1, 2)  # [Batch, Seq_Len, Hidden_Dim]
            return self.layer_norm(out_perm)
