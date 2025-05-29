# Placeholder file
# models/discriminator.py
# import torch
# import torch.nn as nn
# import logging

# logger = logging.getLogger(__name__)

# class Discriminator(nn.Module):
#     def __init__(self, periods: list = [2, 3, 5, 7, 11]):
#         super().__init__()
#         try:
#             self.periods = periods
#             self.convs = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv1d(1, 128, 15, padding=7),
#                     nn.LeakyReLU(0.2),
#                     nn.Conv1d(128, 128, 41, stride=2, padding=20, groups=4),
#                     nn.LeakyReLU(0.2)
#                 ) for _ in periods
#             ])
#         except Exception as e:
#             logger.error(f"Failed to initialize Discriminator: {str(e)}")
#             raise

#     def forward(self, x: torch.Tensor) -> list:
#         try:
#             outputs = []
#             for i, conv in enumerate(self.convs):
#                 period = self.periods[i]
#                 if x.size(-1) % period != 0:
#                     x_padded = nn.functional.pad(x, (0, period - x.size(-1) % period))
#                 else:
#                     x_padded = x
#                 x_reshaped = x_padded.view(x.size(0), 1, -1, period).reshape(x.size(0), 1, -1)
#                 out = conv(x_reshaped)
#                 outputs.append(out)
#             return outputs
#         except Exception as e:
#             logger.error(f"Discriminator forward pass failed: {str(e)}")
#             raise
        
# ##Optimized Discriminator
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Discriminator(nn.Module):
#     def __init__(self, periods=[2, 3, 5, 7, 11]):
#         super().__init__()
#         self.periods = periods
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),  # Reduced from 128
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#             ) for _ in periods
#         ])
#         self.final_conv = nn.Conv1d(256 * len(periods), 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         outs = []
#         for period, conv in zip(self.periods, self.convs):
#             batch, channels, length = x.shape
#             if length % period != 0:
#                 n_pad = period - (length % period)
#                 x_padded = F.pad(x, (0, n_pad), "reflect")
#             else:
#                 x_padded = x
#             x_reshaped = x_padded.view(batch, channels, length // period, period)
#             x_reshaped = x_reshaped.permute(0, 1, 3, 2).reshape(batch, period, -1)
#             out = conv(x_reshaped)
#             outs.append(out)
#         out = torch.cat(outs, dim=1)
#         out = self.final_conv(out)
#         return outs + [out]

# ## Optimized Discriminator
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging

# logger = logging.getLogger(__name__)

# class Discriminator(nn.Module):
#     def __init__(self, periods=[2, 3, 5, 7, 11]):
#         super().__init__()
#         self.periods = periods
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),  # Input channels=1
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#             ) for _ in periods
#         ])
#         self.final_conv = nn.Conv1d(256 * len(periods), 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         try:
#             print(f"Discriminator input shape: {x.shape}")  # Debug: Should be [batch, 1, time]
#             outs = []
#             for period, conv in zip(self.periods, self.convs):
#                 batch, channels, length = x.shape
#                 if length % period != 0:
#                     n_pad = period - (length % period)
#                     x_padded = F.pad(x, (0, n_pad), "reflect")
#                 else:
#                     x_padded = x
#                 # Reshape to keep channels=1
#                 x_reshaped = x_padded.view(batch, channels, -1, period)  # [batch, 1, time//period, period]
#                 x_reshaped = x_reshaped.permute(0, 1, 3, 2)  # [batch, 1, period, time//period]
#                 x_reshaped = x_reshaped.reshape(batch, 1, -1)  # [batch, 1, period * (time//period)]
#                 print(f"After reshape for period {period}: {x_reshaped.shape}")  # Debug
#                 out = conv(x_reshaped)
#                 outs.append(out)
#             out = torch.cat(outs, dim=1)
#             print(f"After concat shape: {out.shape}")  # Debug
#             out = self.final_conv(out)
#             print(f"Final output shape: {out.shape}")  # Debug
#             return outs + [out]
#         except Exception as e:
#             logger.error(f"Discriminator forward pass failed: {str(e)}")
#             raise    

# ## Optimized Discriminator
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging

# logger = logging.getLogger(__name__)

# class Discriminator(nn.Module):
#     def __init__(self, periods=[2, 3, 5, 7, 11]):
#         super().__init__()
#         self.periods = periods
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),  # Input channels=1
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#             ) for _ in periods
#         ])
#         self.final_conv = nn.Conv1d(256 * len(periods), 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         try:
#             print(f"Discriminator input shape: {x.shape}")  # Debug: Should be [batch, 1, time]
#             outs = []
#             for period, conv in zip(self.periods, self.convs):
#                 batch, channels, length = x.shape
#                 if length % period != 0:
#                     n_pad = period - (length % period)
#                     x_padded = F.pad(x, (0, n_pad), "reflect")
#                 else:
#                     x_padded = x
#                 # Reshape to keep channels=1
#                 x_reshaped = x_padded.view(batch, channels, -1, period)  # [batch, 1, time//period, period]
#                 x_reshaped = x_reshaped.permute(0, 1, 3, 2)  # [batch, 1, period, time//period]
#                 x_reshaped = x_reshaped.reshape(batch, 1, -1)  # [batch, 1, period * (time//period)]
#                 print(f"After reshape for period {period}: {x_reshaped.shape}")  # Debug
#                 out = conv(x_reshaped)
#                 print(f"Conv output for period {period}: {out.shape}")  # Debug
#                 outs.append(out)
#             # Truncate to minimum time dimension
#             min_length = min(out.shape[2] for out in outs)
#             outs = [out[:, :, :min_length] for out in outs]
#             print(f"After truncation, min length: {min_length}")  # Debug
#             out = torch.cat(outs, dim=1)
#             print(f"After concat shape: {out.shape}")  # Debug
#             out = self.final_conv(out)
#             print(f"Final output shape: {out.shape}")  # Debug
#             return outs + [out]
#         except Exception as e:
#             logger.error(f"Discriminator forward pass failed: {str(e)}")
#             raise

## Optimized Discriminator
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging

# logger = logging.getLogger(__name__)

# class Discriminator(nn.Module):
#     def __init__(self, periods=[2, 3, 5, 7, 11]):
#         super().__init__()
#         self.periods = periods
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),  # Input channels=1
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#                 nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
#                 nn.LeakyReLU(0.1),
#             ) for _ in periods
#         ])
#         self.final_conv = nn.Conv1d(256 * len(periods), 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         try:
#             print(f"Discriminator input shape: {x.shape}")  # Debug
#             outs = []
#             for period, conv in zip(self.periods, self.convs):
#                 print(f"Processing period {period}")  # Debug
#                 batch, channels, length = x.shape
#                 if length % period != 0:
#                     n_pad = period - (length % period)
#                     x_padded = F.pad(x, (0, n_pad), "reflect")
#                     print(f"Padded length for period {period}: {x_padded.shape[2]}")  # Debug
#                 else:
#                     x_padded = x
#                 x_reshaped = x_padded.view(batch, channels, -1, period)  # [batch, 1, time//period, period]
#                 print(f"View shape for period {period}: {x_reshaped.shape}")  # Debug
#                 x_reshaped = x_reshaped.permute(0, 1, 3, 2)  # [batch, 1, period, time//period]
#                 print(f"Permute shape for period {period}: {x_reshaped.shape}")  # Debug
#                 x_reshaped = x_reshaped.reshape(batch, 1, -1)  # [batch, 1, period * (time//period)]
#                 print(f"After reshape for period {period}: {x_reshaped.shape}")  # Debug
#                 out = conv(x_reshaped)
#                 print(f"Conv output for period {period}: {out.shape}")  # Debug
#                 outs.append(out)
#             min_length = min(out.shape[2] for out in outs)
#             print(f"After truncation, min length: {min_length}")  # Debug
#             outs = [out[:, :, :min_length] for out in outs]
#             print(f"Truncated shapes: {[out.shape for out in outs]}")  # Debug
#             out = torch.cat(outs, dim=1)
#             print(f"After concat shape: {out.shape}")  # Debug
#             out = self.final_conv(out)
#             print(f"Final output shape: {out.shape}")  # Debug
#             return outs + [out]
#         except Exception as e:
#             logger.error(f"Discriminator forward pass failed: {str(e)}")
#             raise

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, periods=[2, 3]):
        super(Discriminator, self).__init__()
        self.periods = periods
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.1),
                nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.1),
                nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            ) for _ in periods
        ])
        self.final_conv = nn.Conv2d(256 * len(periods), 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

    def forward(self, x):
        try:
            print(f"Discriminator input shape: {x.shape}")
            outs = []
            for i, period in enumerate(self.periods):
                print(f"Processing period {period}")
                length = x.size(-1)
                pad_length = (period - length % period) % period
                x_padded = F.pad(x, (0, pad_length))
                print(f"Padded length for period {period}: {x_padded.size(-1)}")
                x_view = x_padded.view(x.size(0), x.size(1), -1, period)
                print(f"View shape for period {period}: {x_view.shape}")
                x_permute = x_view.permute(0, 1, 3, 2)
                print(f"Permute shape for period {period}: {x_permute.shape}")
                x_reshaped = x_permute.reshape(x.size(0), 1, period, -1)
                print(f"After reshape for period {period}: {x_reshaped.shape}")
                out = self.convs[i](x_reshaped)
                print(f"Conv output for period {period}: {out.shape}")
                outs.append(out)
            min_length = min(out.size(-1) for out in outs)
            print(f"After truncation, min length: {min_length}")
            outs = [out[..., :min_length] for out in outs]
            print(f"Truncated shapes: {[o.shape for o in outs]}")
            out_concat = torch.cat(outs, dim=1)
            print(f"After concat shape: {out_concat.shape}")
            final_out = self.final_conv(out_concat)
            print(f"Final output shape: {final_out.shape}")
            return [final_out.squeeze(1)]
        except Exception as e:
            print(f"Discriminator forward pass failed: {str(e)}")
            raise

if __name__ == "__main__":
    periods = [2, 3]
    model = Discriminator(periods)
    x = torch.randn(2, 1, 128294)
    outs = model(x)
    for out in outs:
        print(out.shape)