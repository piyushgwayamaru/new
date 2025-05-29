import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Decoder(nn.Module):
    def __init__(self, in_channels: int = 192, out_channels: int = 80):
        super().__init__()
        try:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(in_channels // 2),
                nn.Conv1d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            )
        except Exception as e:
            logger.error(f"Failed to initialize Decoder: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.conv(x)
        except Exception as e:
            logger.error(f"Decoder forward pass failed: {str(e)}")
            raise