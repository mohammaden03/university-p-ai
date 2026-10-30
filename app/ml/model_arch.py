"""
PyTorch Convolutional Recurrent Neural Network (CRNN) for Text Line Recognition.
Architecture:
  - CNN Feature Extractor (Convolution + BatchNorm + MaxPooling)
  - Map-to-Sequence transformation
  - Bidirectional LSTM / GRU sequence modeling
  - Fully Connected projection to CTC vocabulary logits
"""


import torch
from torch import nn


class CRNN(nn.Module):
    """
    CRNN architecture tailored for handwritten manuscript text-line recognition.
    Supports right-to-left scripts like Persian and Arabic via bidirectional recurrent context.
    """

    def __init__(
        self,
        img_height: int = 64,
        in_channels: int = 1,
        num_classes: int = 120,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.img_height = img_height
        self.in_channels = in_channels
        self.num_classes = num_classes

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # 16 -> 8
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),  # 8 -> 4
            # Block 5
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0),  # 4 -> 1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Sequence modeling: Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )

        # Output projection to CTC character classes
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor [Batch, Channels, Height, Width]
        Returns:
            Logits: [Time, Batch, NumClasses] suitable for torch.nn.CTCLoss
        """
        # CNN feature extraction: [B, C, H, W] -> [B, 512, 1, W']
        features = self.cnn(x)

        # Squeeze height dimension: [B, 512, W']
        features = features.squeeze(2)

        # Permute to [W' (Time), B (Batch), 512 (Features)] for RNN
        seq_features = features.permute(2, 0, 1)

        # Bidirectional RNN forward pass: [Time, Batch, 2 * Hidden]
        rnn_out, _ = self.rnn(seq_features)

        # Linear projection to class vocabulary: [Time, Batch, NumClasses]
        logits = self.fc(rnn_out)
        return logits
