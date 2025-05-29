# # Placeholder file
# # models/hifigan.py
# import torch
# import torch.nn as nn
# import logging

# logger = logging.getLogger(__name__)

# class HiFiGANGenerator(nn.Module):
#     def __init__(self, in_channels: int = 80, out_channels: int = 1, upsample_rates: list = [8, 8, 2, 2]):
#         super().__init__()
#         try:
#             self.conv_pre = nn.Conv1d(in_channels, 128, 7, padding=3)
#             self.upsample = nn.ModuleList([
#                 nn.ConvTranspose1d(128 // (2**i), 128 // (2**(i+1)), r*2, stride=r)
#                 for i, r in enumerate(upsample_rates)
#             ])
#             self.conv_post = nn.Conv1d(128 // (2**len(upsample_rates)), out_channels, 7, padding=3)
#         except Exception as e:
#             logger.error(f"Failed to initialize HiFiGANGenerator: {str(e)}")
#             raise

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         try:
#             x = self.conv_pre(x)
#             for up in self.upsample:
#                 x = up(x)
#                 x = nn.functional.leaky_relu(x, 0.2)
#             x = self.conv_post(x)
#             return torch.tanh(x)
#         except Exception as e:
#             logger.error(f"HiFiGANGenerator forward pass failed: {str(e)}")
#             raise

# Placeholder file
# models/hifigan.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class HiFiGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 80, out_channels: int = 1, upsample_rates: list = [8, 8, 2, 2]):
        super().__init__()
        try:
            self.conv_pre = nn.Conv1d(in_channels, 128, 7, padding=3)
            self.upsample = nn.ModuleList([
                nn.ConvTranspose1d(128 // (2**i), 128 // (2**(i+1)), r*2, stride=r)
                for i, r in enumerate(upsample_rates)
            ])
            self.conv_post = nn.Conv1d(128 // (2**len(upsample_rates)), out_channels, 7, padding=3)
        except Exception as e:
            logger.error(f"Failed to initialize HiFiGANGenerator: {str(e)}")
            raise

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     try:
    #         print(f"Input shape to HiFiGANGenerator: {x.shape}")  # Debug: Should be [batch, 80, time]
    #         x = self.conv_pre(x)
    #         print(f"After conv_pre shape: {x.shape}")  # Debug: Should be [batch, 128, time]
    #         for i, up in enumerate(self.upsample):
    #             x = up(x)
    #             x = nn.functional.leaky_relu(x, 0.2)
    #             print(f"After upsample {i} shape: {x.shape}")  # Debug
    #         x = self.conv_post(x)
    #         print(f"Output shape before tanh: {x.shape}")  # Debug: Should be [batch, 1, time]
    #         return torch.tanh(x)
    #     except Exception as e:
    #         logger.error(f"HiFiGANGenerator forward pass failed: {str(e)}")
    #         raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            print(f"Input shape to HiFiGANGenerator: {x.shape}")
            x = self.conv_pre(x)
            print(f"After conv_pre shape: {x.shape}")
            for i, up in enumerate(self.upsample):
                x = up(x)
                x = nn.functional.leaky_relu(x, 0.2)
                print(f"After upsample {i} shape: {x.shape}")
            x = self.conv_post(x)
            print(f"Output shape before tanh: {x.shape}")
            return torch.tanh(x)
        except Exception as e:
            logger.error(f"HiFiGANGenerator forward pass failed: {str(e)}")
            raise 