# # # Placeholder file
# # # pipeline/training_pipeline.py
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# # from data.dataset import VITSDataset, get_dataloader
# # from utils.logging import Logger
# # import yaml
# # import os
# # from typing import Dict

# # class TrainingPipeline:
# #     def __init__(self, config_path: str, manifest_file: str = None):
# #         try:
# #             self.config = self.load_config(config_path)
# #             if manifest_file is not None:
# #                 self.config['manifest_file'] = manifest_file
# #             self.logger = Logger(self.config['log_dir'])
# #             self.models = self.initialize_models()
# #             self.optimizers = self.initialize_optimizers()
# #             self.train_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['manifest_file'],
# #                 self.config['sample_rate'],
# #                 self.config['n_mels']
# #             )
# #             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
# #             self.val_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['val_manifest_file'],
# #                 self.config['sample_rate'],
# #                 self.config['n_mels']
# #             )
# #             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
# #             raise

# #     def load_config(self, config_path: str) -> Dict:
# #         try:
# #             with open(config_path, 'r') as f:
# #                 return yaml.safe_load(f)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
# #             raise

# #     def initialize_models(self) -> Dict[str, nn.Module]:
# #         return {
# #             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
# #             'posterior_encoder': PosteriorEncoder().cuda(),
# #             'flow': Flow().cuda(),
# #             'duration_predictor': DurationPredictor().cuda(),
# #             'generator': HiFiGANGenerator().cuda(),
# #             'discriminator': Discriminator(self.config['periods']).cuda()
# #         }

# #     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
# #         return {
# #             'gen': torch.optim.Adam(
# #                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
# #                 lr=self.config['lr']
# #             ),
# #             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
# #         }

# #     def save_checkpoint(self, epoch: int, path: str):
# #         try:
# #             state = {
# #                 'epoch': epoch,
# #                 'models': {k: v.state_dict() for k, v in self.models.items()},
# #                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
# #             }
# #             torch.save(state, path)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
# #             raise

# #     def validate(self):
# #         """Validate the model on the validation set."""
# #         try:
# #             total_loss = 0
# #             for phonemes, mels in self.val_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(self.val_loader)
# #             self.logger.log({'val_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Validation failed: {str(e)}")
# #             raise

# #     def run(self):
# #         try:
# #             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
# #             for epoch in range(self.config['epochs']):
# #                 for phonemes, mels in self.train_loader:
# #                     phonemes, mels = phonemes.cuda(), mels.cuda()

# #                     # Generator forward
# #                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                     z_flow, log_det = self.models['flow'](z)
# #                     text_embed = self.models['text_encoder'](phonemes)
# #                     durations = self.models['duration_predictor'](text_embed)
# #                     mel_pred = self.models['generator'](z_flow)
# #                     audio = self.models['generator'](mel_pred)

# #                     # Losses
# #                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
# #                     recon_loss = nn.MSELoss()(mel_pred, mels)
# #                     adv_loss = 0
# #                     d_loss = 0
# #                     real_out = self.models['discriminator'](mels.unsqueeze(1))
# #                     fake_out = self.models['discriminator'](audio)
# #                     for r, f in zip(real_out, fake_out):
# #                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
# #                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

# #                     # Optimize generator
# #                     self.optimizers['gen'].zero_grad()
# #                     total_gen_loss = recon_loss + kl_loss + adv_loss
# #                     total_gen_loss.backward()
# #                     self.optimizers['gen'].step()

# #                     # Optimize discriminator
# #                     self.optimizers['disc'].zero_grad()
# #                     d_loss.backward()
# #                     self.optimizers['disc'].step()

# #                     # Log metrics
# #                     self.logger.log({
# #                         'recon_loss': recon_loss.item(),
# #                         'kl_loss': kl_loss.item(),
# #                         'adv_loss': adv_loss.item(),
# #                         'd_loss': d_loss.item()
# #                     })

# #                 # Validate after each epoch
# #                 val_loss = self.validate()
# #                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

# #                 # Save checkpoint
# #                 if (epoch + 1) % 10 == 0:
# #                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
# #         except Exception as e:
# #             self.logger.logger.error(f"Training failed: {str(e)}")
# #             raise

# #     def evaluate(self, test_manifest: str = "data/csv/test.csv"):
# #         """Evaluate the model on the test set."""
# #         try:
# #             test_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 test_manifest,
# #                 self.config['sample_rate'],
# #                 self.config['n_mels']
# #             )
# #             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
# #             total_loss = 0
# #             for phonemes, mels in test_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(test_loader)
# #             self.logger.log({'test_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Evaluation failed: {str(e)}")
# #             raise

# # if __name__ == "__main__":
# #     pipeline = TrainingPipeline("configs/config.yaml")
# #     pipeline.run()
# ############################################################################33
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# # from data.dataset import VITSDataset, get_dataloader
# # from utils.logging import Logger
# # import yaml
# # import os
# # from typing import Dict

# # class TrainingPipeline:
# #     def __init__(self, config_path: str, manifest_file: str = None):
# #         try:
# #             self.config = self.load_config(config_path)
# #             if manifest_file is not None:
# #                 self.config['manifest_file'] = manifest_file
# #             self.logger = Logger(self.config['log_dir'])
# #             self.models = self.initialize_models()
# #             self.optimizers = self.initialize_optimizers()
# #             # Initialize train dataset
# #             self.train_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['manifest_file']
# #             )
# #             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
# #             # Initialize validation dataset
# #             self.val_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['val_manifest_file']
# #             )
# #             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
# #             raise

# #     def load_config(self, config_path: str) -> Dict:
# #         try:
# #             with open(config_path, 'r') as f:
# #                 config = yaml.safe_load(f)
# #             # Validate required config fields
# #             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
# #             for field in required_fields:
# #                 if field not in config:
# #                     raise ValueError(f"Missing required config field: {field}")
# #             return config
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
# #             raise

# #     def initialize_models(self) -> Dict[str, nn.Module]:
# #         return {
# #             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
# #             'posterior_encoder': PosteriorEncoder().cuda(),
# #             'flow': Flow().cuda(),
# #             'duration_predictor': DurationPredictor().cuda(),
# #             'generator': HiFiGANGenerator().cuda(),
# #             'discriminator': Discriminator(self.config['periods']).cuda()
# #         }

# #     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
# #         return {
# #             'gen': torch.optim.Adam(
# #                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
# #                 lr=self.config['lr']
# #             ),
# #             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
# #         }

# #     def save_checkpoint(self, epoch: int, path: str):
# #         try:
# #             state = {
# #                 'epoch': epoch,
# #                 'models': {k: v.state_dict() for k, v in self.models.items()},
# #                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
# #             }
# #             torch.save(state, path)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
# #             raise

# #     def validate(self):
# #         """Validate the model on the validation set."""
# #         try:
# #             total_loss = 0
# #             for phonemes, mels in self.val_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(self.val_loader)
# #             self.logger.log({'val_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Validation failed: {str(e)}")
# #             raise

# #     def run(self):
# #         try:
# #             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
# #             for epoch in range(self.config['epochs']):
# #                 for phonemes, mels in self.train_loader:
# #                     phonemes, mels = phonemes.cuda(), mels.cuda()

# #                     # Generator forward
# #                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                     z_flow, log_det = self.models['flow'](z)
# #                     text_embed = self.models['text_encoder'](phonemes)
# #                     durations = self.models['duration_predictor'](text_embed)
# #                     mel_pred = self.models['generator'](z_flow)
# #                     audio = self.models['generator'](mel_pred)

# #                     # Losses
# #                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
# #                     recon_loss = nn.MSELoss()(mel_pred, mels)
# #                     adv_loss = 0
# #                     d_loss = 0
# #                     real_out = self.models['discriminator'](mels.unsqueeze(1))
# #                     fake_out = self.models['discriminator'](audio)
# #                     for r, f in zip(real_out, fake_out):
# #                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
# #                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

# #                     # Optimize generator
# #                     self.optimizers['gen'].zero_grad()
# #                     total_gen_loss = recon_loss + kl_loss + adv_loss
# #                     total_gen_loss.backward()
# #                     self.optimizers['gen'].step()

# #                     # Optimize discriminator
# #                     self.optimizers['disc'].zero_grad()
# #                     d_loss.backward()
# #                     self.optimizers['disc'].step()

# #                     # Log metrics
# #                     self.logger.log({
# #                         'recon_loss': recon_loss.item(),
# #                         'kl_loss': kl_loss.item(),
# #                         'adv_loss': adv_loss.item(),
# #                         'd_loss': d_loss.item()
# #                     })

# #                 # Validate after each epoch
# #                 val_loss = self.validate()
# #                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

# #                 # Save checkpoint
# #                 if (epoch + 1) % 10 == 0:
# #                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
# #         except Exception as e:
# #             self.logger.logger.error(f"Training failed: {str(e)}")
# #             raise

# #     def evaluate(self, test_manifest: str = "data/csv/test.csv"):
# #         """Evaluate the model on the test set."""
# #         try:
# #             test_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 test_manifest
# #             )
# #             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
# #             total_loss = 0
# #             for phonemes, mels in test_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(test_loader)
# #             self.logger.log({'test_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Evaluation failed: {str(e)}")
# #             raise

# # if __name__ == "__main__":
# #     pipeline = TrainingPipeline("configs/config.yaml")
# #     pipeline.run()
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# import yaml
# import os
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             # Initialize train dataset
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file']
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             # Initialize validation dataset
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file']
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             # Validate required config fields
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         return {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
#             'posterior_encoder': PosteriorEncoder().cuda(),
#             'flow': Flow().cuda(),
#             'duration_predictor': DurationPredictor().cuda(),
#             'generator': HiFiGANGenerator().cuda(),
#             'discriminator': Discriminator(self.config['periods']).cuda()
#         }

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         """Validate the model on the validation set."""
#         try:
#             total_loss = 0
#             for phonemes, mels in self.val_loader:
#                 phonemes, mels = phonemes.cuda(), mels.cuda()
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for phonemes, mels in self.train_loader:
#                     phonemes, mels = phonemes.cuda(), mels.cuda()

#                     # Generator forward
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes)
#                     durations = self.models['duration_predictor'](text_embed)
#                     mel_pred = self.models['generator'](z_flow)
#                     audio = self.models['generator'](mel_pred)

#                     # Losses
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred, mels)
#                     adv_loss = 0
#                     d_loss = 0
#                     real_out = self.models['discriminator'](mels.unsqueeze(1))
#                     fake_out = self.models['discriminator'](audio)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

#                     # Optimize generator
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()

#                     # Optimize discriminator
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()

#                     # Log metrics
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })

#                 # Validate after each epoch
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

#                 # Save checkpoint
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         """Evaluate the model on the test set."""
#         try:
#             test_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 test_manifest
#             )
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for phonemes, mels in test_loader:
#                 phonemes, mels = phonemes.cuda(), mels.cuda()
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# import yaml


# from typing import Dict

# import torch
# from torch.nn.utils.rnn import pad_sequence

# def custom_collate_fn(batch):
#     """
#     Custom collate function to pad variable-length phoneme sequences and mel-spectrograms.
#     Args:
#         batch: List of samples from VITSDataset, where each sample is a tuple (phonemes, mels).
#     Returns:
#         Dict with padded phonemes, mel-spectrograms, and their lengths.
#     """
#     # Separate phonemes and mels from the batch
#     phonemes = [item[0] for item in batch]  # Assuming item[0] is phonemes tensor
#     mels = [item[1] for item in batch]      # Assuming item[1] is mels tensor

#     # Pad sequences to the longest in the batch
#     phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)
#     mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)

#     # Compute lengths of original sequences
#     phoneme_lengths = torch.tensor([len(p) for p in phonemes], dtype=torch.long)
#     mel_lengths = torch.tensor([len(m) for m in mels], dtype=torch.long)

#     return {
#         'phonemes': phonemes_padded,
#         'mels': mels_padded,
#         'phoneme_lengths': phoneme_lengths,
#         'mel_lengths': mel_lengths
#     }

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             # Initialize train dataset
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file']
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             # Initialize validation dataset
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file']
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             # Validate required config fields
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         """Validate the model on the validation set."""
#         try:
#             total_loss = 0
#             for phonemes, mels in self.val_loader:
#                 phonemes, mels = phonemes.to(self.device), mels.to(self.device)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for phonemes, mels in self.train_loader:
#                     phonemes, mels = phonemes.to(self.device), mels.to(self.device)

#                     # Generator forward
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes)
#                     durations = self.models['duration_predictor'](text_embed)
#                     mel_pred = self.models['generator'](z_flow)
#                     audio = self.models['generator'](mel_pred)

#                     # Losses
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred, mels)
#                     adv_loss = 0
#                     d_loss = 0
#                     real_out = self.models['discriminator'](mels.unsqueeze(1))
#                     fake_out = self.models['discriminator'](audio)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

#                     # Optimize generator
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()

#                     # Optimize discriminator
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()

#                     # Log metrics
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })

#                 # Validate after each epoch
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

#                 # Save checkpoint
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         """Evaluate the model on the test set."""
#         try:
#             test_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 test_manifest
#             )
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for phonemes, mels in test_loader:
#                 phonemes, mels = phonemes.to(self.device), mels.to(self.device)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

#NEWER update
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             # Initialize train dataset
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file']
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             # Initialize validation dataset
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file']
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             # Validate required config fields
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         """Validate the model on the validation set."""
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device)
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)

#                     # Generator forward
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes)  # Update if lengths are needed
#                     durations = self.models['duration_predictor'](text_embed)
#                     mel_pred = self.models['generator'](z_flow)
#                     audio = self.models['generator'](mel_pred)

#                     # Losses
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred, mels)
#                     adv_loss = 0
#                     d_loss = 0
#                     real_out = self.models['discriminator'](mels.unsqueeze(1))
#                     fake_out = self.models['discriminator'](audio)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

#                     # Optimize generator
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()

#                     # Optimize discriminator
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()

#                     # Log metrics
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })

#                 # Validate after each epoch
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

#                 # Save checkpoint
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         """Evaluate the model on the test set."""
#         try:
#             test_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 test_manifest
#             )
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in test_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()



#UPDATE 1
# This includes the usage of duration_prediction which for now is avoided

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             # Initialize train dataset
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file']
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             # Initialize validation dataset
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file']
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),  # New decoder for mel-spectrogram reconstruction
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         """Validate the model on the validation set."""
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)  # Shape: (batch_size, max_mel_frames, 80)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 # Transpose mels to (batch_size, 80, max_mel_frames) for PosteriorEncoder
#                 mels_pe = mels.transpose(1, 2)  # (batch_size, 80, max_mel_frames)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)  # Mel-spectrogram output
#                 # Use original mels for loss (batch_size, max_mel_frames, 80)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)  # Shape: (batch_size, max_phoneme_len)
#                     mels = batch['mels'].to(self.device)         # Shape: (batch_size, max_mel_frames, 80)
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)

#                     # Generate padding mask for TextEncoder (True for padding positions)
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]

#                     # Transpose mels to (batch_size, 80, max_mel_frames) for PosteriorEncoder
#                     mels_pe = mels.transpose(1, 2)  # (batch_size, 80, max_mel_frames)

#                     # Generator forward
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     # Transpose text_embed for DurationPredictor
#                     text_embed = text_embed.transpose(1, 2)  # (batch_size, embed_dim, max_phoneme_len)
#                     durations = self.models['duration_predictor'](text_embed)
#                     mel_pred = self.models['decoder'](z_flow)  # Mel-spectrogram output
#                     audio = self.models['generator'](mel_pred)  # Audio output

#                     # Losses
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     duration_loss = nn.MSELoss()(durations, mel_lengths.float() / phoneme_lengths.float().clamp(min=1))  # Simplified
#                     adv_loss = 0
#                     d_loss = 0
#                     # Use audio for Discriminator
#                     real_out = self.models['discriminator'](mels_pe.unsqueeze(1))  # Approximate real audio
#                     fake_out = self.models['discriminator'](audio)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

#                     # Optimize generator
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + adv_loss + duration_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()

#                     # Optimize discriminator
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()

#                     # Log metrics
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item(),
#                         'duration_loss': duration_loss.item()
#                     })

#                 # Validate after each epoch
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

#                 # Save checkpoint
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         """Evaluate the model on the test set."""
#         try:
#             test_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 test_manifest
#             )
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in test_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)  # Shape: (batch_size, max_mel_frames, 80)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 # Transpose mels to (batch_size, 80, max_mel_frames) for PosteriorEncoder
#                 mels_pe = mels.transpose(1, 2)  # (batch_size, 80, max_mel_frames)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)  # Mel-spectrogram output
#                 # Use original mels for loss
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

#UPDATE 2:
# THis is for the MAS usage for duation prediction
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# from vits_nepali.utils.mas import monotonic_alignment_search
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             # Initialize train dataset
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file']
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             # Initialize validation dataset
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file']
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         """Validate the model on the validation set."""
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)  # Shape: (batch_size, max_mel_frames, 80)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 # Transpose mels to (batch_size, 80, max_mel_frames) for PosteriorEncoder
#                 mels_pe = mels.transpose(1, 2)  # (batch_size, 80, max_mel_frames)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)  # Mel-spectrogram output
#                 # Use original mels for loss (batch_size, max_mel_frames, 80)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)  # Shape: (batch_size, max_phoneme_len)
#                     mels = batch['mels'].to(self.device)         # Shape: (batch_size, max_mel_frames, 80)
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)

#                     # Generate padding mask for TextEncoder (True for padding positions)
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]

#                     # Transpose mels to (batch_size, 80, max_mel_frames) for PosteriorEncoder and Generator
#                     mels_pe = mels.transpose(1, 2)  # (batch_size, 80, max_mel_frames)

#                     # Generator forward
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     # Transpose text_embed for DurationPredictor
#                     text_embed_dp = text_embed.transpose(1, 2)  # (batch_size, embed_dim, max_phoneme_len)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     mel_pred = self.models['decoder'](z_flow)  # Mel-spectrogram output
#                     audio = self.models['generator'](mels_pe)  # Real audio from ground-truth mels
#                     audio_fake = self.models['generator'](mel_pred)  # Fake audio from predicted mels

#                     # Compute ground-truth durations using MAS
#                     durations_gt = monotonic_alignment_search(
#                         text_embed, z_mu, phoneme_lengths, mel_lengths
#                     )

#                     # Losses
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     duration_loss = nn.MSELoss()(durations_pred, durations_gt.float())
#                     adv_loss = 0
#                     d_loss = 0
#                     real_out = self.models['discriminator'](audio)  # Real audio
#                     fake_out = self.models['discriminator'](audio_fake)  # Fake audio
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

#                     # Optimize generator
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + duration_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()

#                     # Optimize discriminator
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()

#                     # Log metrics
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'duration_loss': duration_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })

#                 # Validate after each epoch
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

#                 # Save checkpoint
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         """Evaluate the model on the test set."""
#         try:
#             test_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 test_manifest
#             )
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in test_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)  # Shape: (batch_size, max_mel_frames, 80)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 # Transpose mels to (batch_size, 80, max_mel_frames) for PosteriorEncoder
#                 mels_pe = mels.transpose(1, 2)  # (batch_size, 80, max_mel_frames)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)  # Mel-spectrogram output
#                 # Use original mels for loss
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

#UPDATE 3
# reusage of MAS algo
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# from vits_nepali.utils.mas import monotonic_alignment_search
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             self.train_dataset = VITSDataset(self.config['data_dir'], self.config['manifest_file'])
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             self.val_dataset = VITSDataset(self.config['data_dir'], self.config['val_manifest_file'])
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device)
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
#                     mels_pe = mels.transpose(1, 2)
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     text_embed_dp = text_embed.transpose(1, 2)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     # Compute ground-truth durations using MAS
#                     durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths)
#                     mel_pred = self.models['decoder'](z_flow)
#                     audio = self.models['generator'](mels_pe)
#                     audio_fake = self.models['generator'](mel_pred)
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     duration_loss = nn.MSELoss()(durations_pred, durations_gt.float()) * 0.1  # Weighted
#                     adv_loss = 0
#                     d_loss = 0
#                     real_out = self.models['discriminator'](audio)
#                     fake_out = self.models['discriminator'](audio_fake)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + duration_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'duration_loss': duration_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         try:
#             test_dataset = VITSDataset(self.config['data_dir'], test_manifest)
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

# UPDATE 3
# reusage of MAS algo
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# from vits_nepali.utils.mas import monotonic_alignment_search
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             self.train_dataset = VITSDataset(self.config['data_dir'], self.config['manifest_file'])
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             self.val_dataset = VITSDataset(self.config['data_dir'], self.config['val_manifest_file'])
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device)
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
#                     mels_pe = mels.transpose(1, 2)
#                     print(f"mels_pe shape: {mels_pe.shape}")  # Debug: Should be [batch_size, n_mels, max_mel_frames]
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     text_embed_dp = text_embed.transpose(1, 2)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     # Compute ground-truth durations using MAS
#                     durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths)
#                     mel_pred = self.models['decoder'](z_flow)
#                     audio = self.models['generator'](mels_pe)
#                     print(f"audio shape: {audio.shape}")  # Debug: Should be [batch_size, 1, time]
#                     audio_fake = self.models['generator'](mel_pred)
#                     print(f"audio_fake shape: {audio_fake.shape}")  # Debug: Should be [batch_size, 1, time]
#                     real_out = self.models['discriminator'](audio)
#                     fake_out = self.models['discriminator'](audio_fake)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + duration_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'duration_loss': duration_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         try:
#             test_dataset = VITSDataset(self.config['data_dir'], test_manifest)
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device)
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

# # Optimized TrainingPipeline
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# from vits_nepali.utils.mas import monotonic_alignment_search
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             self.config['batch_size'] = 2  # Reduced batch size
#             self.train_dataset = VITSDataset(self.config['data_dir'], self.config['manifest_file'])
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             self.val_dataset = VITSDataset(self.config['data_dir'], self.config['val_manifest_file'])
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device).float()  # Ensure float
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()  # Ensure float
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     print(f"phonemes dtype: {phonemes.dtype}, mels dtype: {mels.dtype}")  # Debug
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
#                     mels_pe = mels.transpose(1, 2)
#                     print(f"mels_pe shape: {mels_pe.shape}")  # Debug
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     text_embed_dp = text_embed.transpose(1, 2)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths).float()  # Cast to float
#                     print(f"durations_pred shape: {durations_pred.shape}, dtype: {durations_pred.dtype}")  # Debug
#                     print(f"durations_gt shape: {durations_gt.shape}, dtype: {durations_gt.dtype}")  # Debug
#                     mel_pred = self.models['decoder'](z_flow)
#                     audio = self.models['generator'](mels_pe)
#                     print(f"audio shape: {audio.shape}")  # Debug
#                     audio_fake = self.models['generator'](mel_pred)
#                     print(f"audio_fake shape: {audio_fake.shape}")  # Debug
#                     # Compute discriminator outputs for generator loss
#                     real_out_gen = self.models['discriminator'](audio)
#                     fake_out_gen = self.models['discriminator'](audio_fake)
#                     print(f"real_out_gen shapes: {[o.shape for o in real_out_gen]}")  # Debug
#                     print(f"fake_out_gen shapes: {[o.shape for o in fake_out_gen]}")  # Debug
#                     # Compute losses for generator
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     print(f"recon_loss: {recon_loss.item()}")  # Debug
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     print(f"kl_loss: {kl_loss.item()}")  # Debug
#                     duration_loss = nn.MSELoss()(durations_pred, durations_gt)
#                     print(f"duration_loss: {duration_loss.item()}")  # Debug
#                     adv_loss = 0
#                     for f in fake_out_gen:
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                     print(f"adv_loss: {adv_loss.item()}")  # Debug
#                     total_gen_loss = recon_loss + kl_loss + duration_loss + adv_loss
#                     print(f"total_gen_loss: {total_gen_loss.item()}")  # Debug
#                     # Backprop generator loss
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()
#                     # Recompute discriminator outputs for discriminator loss
#                     audio = self.models['generator'](mels_pe.detach())  # Detach to avoid graph reuse
#                     audio_fake = self.models['generator'](mel_pred.detach())  # Detach to avoid graph reuse
#                     real_out_disc = self.models['discriminator'](audio)
#                     fake_out_disc = self.models['discriminator'](audio_fake)
#                     print(f"real_out_disc shapes: {[o.shape for o in real_out_disc]}")  # Debug
#                     print(f"fake_out_disc shapes: {[o.shape for o in fake_out_disc]}")  # Debug
#                     # Compute discriminator loss
#                     d_loss = 0
#                     for r, f in zip(real_out_disc, fake_out_disc):
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))
#                     print(f"d_loss: {d_loss.item()}")  # Debug
#                     # Backprop discriminator loss
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'duration_loss': duration_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         try:
#             test_dataset = VITSDataset(self.config['data_dir'], test_manifest)
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device).float()  # Ensure float
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

# # Optimized TrainingPipeline for GPU
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# from vits_nepali.utils.mas import monotonic_alignment_search
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.logger.logger.info(f"Using device: {self.device}")
#             if self.device.type == 'cuda':
#                 self.logger.logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             self.config['batch_size'] = 4  # Increased for GPU
#             self.train_dataset = VITSDataset(self.config['data_dir'], self.config['manifest_file'])
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             self.val_dataset = VITSDataset(self.config['data_dir'], self.config['val_manifest_file'])
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         try:
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device).float()  # Ensure float
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 torch.cuda.empty_cache()  # Clear GPU memory
#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()  # Ensure float
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     print(f"phonemes dtype: {phonemes.dtype}, mels dtype: {mels.dtype}, device: {phonemes.device}")  # Debug
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
#                     mels_pe = mels.transpose(1, 2)
#                     print(f"mels_pe shape: {mels_pe.shape}, device: {mels_pe.device}")  # Debug
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     text_embed_dp = text_embed.transpose(1, 2)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths).float().to(self.device)  # Ensure float and GPU
#                     print(f"durations_pred shape: {durations_pred.shape}, dtype: {durations_pred.dtype}, device: {durations_pred.device}")  # Debug
#                     print(f"durations_gt shape: {durations_gt.shape}, dtype: {durations_gt.dtype}, device: {durations_gt.device}")  # Debug
#                     mel_pred = self.models['decoder'](z_flow)
#                     audio = self.models['generator'](mels_pe)
#                     print(f"audio shape: {audio.shape}, device: {audio.device}")  # Debug
#                     audio_fake = self.models['generator'](mel_pred)
#                     print(f"audio_fake shape: {audio_fake.shape}, device: {audio_fake.device}")  # Debug
#                     # Compute discriminator outputs for generator loss
#                     real_out_gen = self.models['discriminator'](audio)
#                     fake_out_gen = self.models['discriminator'](audio_fake)
#                     print(f"real_out_gen shapes: {[o.shape for o in real_out_gen]}, device: {real_out_gen[0].device}")  # Debug
#                     print(f"fake_out_gen shapes: {[o.shape for o in fake_out_gen]}, device: {fake_out_gen[0].device}")  # Debug
#                     # Compute losses for generator
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     print(f"recon_loss: {recon_loss.item()}")  # Debug
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     print(f"kl_loss: {kl_loss.item()}")  # Debug
#                     duration_loss = nn.MSELoss()(durations_pred, durations_gt)
#                     print(f"duration_loss: {duration_loss.item()}")  # Debug
#                     adv_loss = 0
#                     for f in fake_out_gen:
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                     print(f"adv_loss: {adv_loss.item()}")  # Debug
#                     total_gen_loss = recon_loss + kl_loss + duration_loss + adv_loss
#                     print(f"total_gen_loss: {total_gen_loss.item()}")  # Debug
#                     # Backprop generator loss
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()
#                     # Recompute discriminator outputs for discriminator loss
#                     audio = self.models['generator'](mels_pe.detach())  # Detach to avoid graph reuse
#                     audio_fake = self.models['generator'](mel_pred.detach())  # Detach to avoid graph reuse
#                     real_out_disc = self.models['discriminator'](audio)
#                     fake_out_disc = self.models['discriminator'](audio_fake)
#                     print(f"real_out_disc shapes: {[o.shape for o in real_out_disc]}, device: {real_out_disc[0].device}")  # Debug
#                     print(f"fake_out_disc shapes: {[o.shape for o in fake_out_disc]}, device: {fake_out_disc[0].device}")  # Debug
#                     # Compute discriminator loss
#                     d_loss = 0
#                     for r, f in zip(real_out_disc, fake_out_disc):
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))
#                     print(f"d_loss: {d_loss.item()}")  # Debug
#                     # Backprop discriminator loss
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'duration_loss': duration_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         try:
#             test_dataset = VITSDataset(self.config['data_dir'], test_manifest)
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for batch in self.val_loader:
#                 phonemes = batch['phonemes'].to(self.device)
#                 mels = batch['mels'].to(self.device).float()  # Ensure float
#                 phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                 mel_lengths = batch['mel_lengths'].to(self.device)
#                 mels_pe = mels.transpose(1, 2)
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['decoder'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

# Optimized TrainingPipeline for GPU last worked
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator, Decoder
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# from vits_nepali.utils.mas import monotonic_alignment_search
# import yaml
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.logger.logger.info(f"Using device: {self.device}")
#             if self.device.type == 'cuda':
#                 self.logger.logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             self.config['batch_size'] = 4  # For GTX 1650 (4GB VRAM)
#             self.config['max_mel_length'] = 300  # Reduce memory usage
#             self.config['grad_accum_steps'] = 2  # Effective batch size = 2
#             self.train_dataset = VITSDataset(self.config['data_dir'], self.config['manifest_file'], max_mel_length=self.config['max_mel_length'])
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             self.val_dataset = VITSDataset(self.config['data_dir'], self.config['val_manifest_file'], max_mel_length=self.config['max_mel_length'])
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods'])
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         try:
#             total_loss = 0
#             with torch.no_grad():  # Save memory
#                 for batch in self.val_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     mels_pe = mels.transpose(1, 2)
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, _ = self.models['flow'](z)
#                     mel_pred = self.models['decoder'](z_flow)
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise


#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 torch.cuda.empty_cache()  # Clear GPU memory at start of epoch
#                 if torch.cuda.is_available():
#                     self.logger.logger.info(f"After empty_cache(): Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

#                 gen_loss_accum = 0
#                 disc_loss_accum = 0
#                 accum_count = 0

#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     self.logger.logger.info(f"phonemes dtype: {phonemes.dtype}, mels dtype: {mels.dtype}, device: {phonemes.device}")
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
#                     mels_pe = mels.transpose(1, 2)
#                     self.logger.logger.info(f"mels_pe shape: {mels_pe.shape}, device: {mels_pe.device}")
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     text_embed_dp = text_embed.transpose(1, 2)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths).float().to(self.device)
#                     self.logger.logger.info(f"durations_pred shape: {durations_pred.shape}, dtype: {durations_pred.dtype}, device: {durations_pred.device}")
#                     self.logger.logger.info(f"durations_gt shape: {durations_gt.shape}, dtype: {durations_gt.dtype}, device: {durations_gt.device}")
#                     mel_pred = self.models['decoder'](z_flow)
#                     audio = self.models['generator'](mels_pe)
#                     self.logger.logger.info(f"audio shape: {audio.shape}, device: {audio.device}")
#                     audio_fake = self.models['generator'](mel_pred)
#                     self.logger.logger.info(f"audio_fake shape: {audio_fake.shape}, device: {audio_fake.device}")
#                     real_out_gen = self.models['discriminator'](audio)
#                     fake_out_gen = self.models['discriminator'](audio_fake)
#                     self.logger.logger.info(f"real_out_gen shapes: {[o.shape for o in real_out_gen]}, device: {real_out_gen[0].device}")
#                     self.logger.logger.info(f"fake_out_gen shapes: {[o.shape for o in fake_out_gen]}, device: {fake_out_gen[0].device}")
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     duration_loss = nn.MSELoss()(durations_pred, durations_gt)
#                     adv_loss = 0
#                     for f in fake_out_gen:
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                     total_gen_loss = (recon_loss + kl_loss + duration_loss + adv_loss) / self.config['grad_accum_steps']
#                     self.logger.logger.info(f"recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}, duration_loss: {duration_loss.item()}, adv_loss: {adv_loss.item()}, total_gen_loss: {total_gen_loss.item()}")
#                     total_gen_loss.backward()
#                     gen_loss_accum += total_gen_loss.item()
#                     audio = self.models['generator'](mels_pe.detach())
#                     audio_fake = self.models['generator'](mel_pred.detach())
#                     real_out_disc = self.models['discriminator'](audio)
#                     fake_out_disc = self.models['discriminator'](audio_fake)
#                     self.logger.logger.info(f"real_out_disc shapes: {[o.shape for o in real_out_disc]}, device: {real_out_disc[0].device}")
#                     self.logger.logger.info(f"fake_out_disc shapes: {[o.shape for o in fake_out_disc]}, device: {fake_out_disc[0].device}")
#                     d_loss = 0
#                     for r, f in zip(real_out_disc, fake_out_disc):
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))
#                     d_loss = d_loss / self.config['grad_accum_steps']
#                     d_loss.backward()
#                     disc_loss_accum += d_loss.item()
#                     accum_count += 1
#                     if accum_count == self.config['grad_accum_steps']:
#                         self.optimizers['gen'].step()
#                         self.optimizers['disc'].step()
#                         self.optimizers['gen'].zero_grad()
#                         self.optimizers['disc'].zero_grad()
#                         self.logger.log({
#                             'recon_loss': recon_loss.item(),
#                             'kl_loss': kl_loss.item(),
#                             'duration_loss': duration_loss.item(),
#                             'adv_loss': adv_loss.item(),
#                             'd_loss': disc_loss_accum / accum_count,
#                             'total_gen_loss': gen_loss_accum / accum_count
#                         })
#                         gen_loss_accum = 0
#                         disc_loss_accum = 0
#                         accum_count = 0
#                 if accum_count > 0:  # Apply remaining gradients
#                     self.optimizers['gen'].step()
#                     self.optimizers['disc'].step()
#                     self.optimizers['gen'].zero_grad()
#                     self.optimizers['disc'].zero_grad()
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")

#                 # 🔁 Optionally clear cache again after val (less useful, but safe)
#                 torch.cuda.empty_cache()
#                 if torch.cuda.is_available():
#                     self.logger.logger.info(f"End of epoch {epoch+1} GPU: Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv"):
#         try:
#             test_dataset = VITSDataset(self.config['data_dir'], test_manifest, max_mel_length=self.config['max_mel_length'])
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             with torch.no_grad():
#                 for batch in test_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     mels_pe = mels.transpose(1, 2)
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, _ = self.models['flow'](z)
#                     mel_pred = self.models['decoder'](z_flow)
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml")
#     pipeline.run()


# import os
# import sys
# from typing import Dict, Optional
# from pathlib import Path

# # Ensure project root is in sys.path for module imports
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import yaml

# # Suppress IDE warnings for custom module imports
# # If your IDE flags these as unresolved, ensure vits_nepali is in your project directory
# from vits_nepali.models import (  # type: ignore
#     TextEncoder,
#     PosteriorEncoder,
#     Flow,
#     DurationPredictor,
#     HiFiGANGenerator,
#     Discriminator,
#     Decoder,
# )
# from vits_nepali.data.dataset import VITSDataset, get_dataloader  # type: ignore
# from vits_nepali.utils.logging import Logger  # type: ignore
# from vits_nepali.utils.mas import monotonic_alignment_search  # type: ignore


# class TrainingPipeline:
#     def __init__(
#         self,
#         config_path: str,
#         manifest_file: Optional[str] = None,
#         checkpoint_path: Optional[str] = None,
#     ) -> None:
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.logger.logger.info(f"Using device: {self.device}")
#             if self.device.type == 'cuda':
#                 cuda_version = getattr(torch.version, 'cuda', 'N/A')  # Fallback if CUDA is unavailable
#                 self.logger.logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {cuda_version}")
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             self.config['batch_size'] = 4  # For GTX 1650 (4GB VRAM)
#             self.config['max_mel_length'] = 300  # Reduce memory usage
#             self.config['grad_accum_steps'] = 2  # Effective batch size = 2
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file'],
#                 max_mel_length=self.config['max_mel_length'],
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file'],
#                 max_mel_length=self.config['max_mel_length'],
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#             self.start_epoch: int = 1  # Default start epoch
#             if checkpoint_path is not None:
#                 self.start_epoch = self.load_checkpoint(checkpoint_path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             required_fields = [
#                 'data_dir',
#                 'manifest_file',
#                 'val_manifest_file',
#                 'batch_size',
#                 'log_dir',
#                 'checkpoint_dir',
#                 'epochs',
#                 'lr',
#                 'n_vocab',
#                 'embed_dim',
#                 'periods',
#             ]
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         models = {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
#             'posterior_encoder': PosteriorEncoder(),
#             'flow': Flow(),
#             'duration_predictor': DurationPredictor(),
#             'decoder': Decoder(),
#             'generator': HiFiGANGenerator(),
#             'discriminator': Discriminator(self.config['periods']),
#         }
#         return {k: v.to(self.device) for k, v in models.items()}

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr'],
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr']),
#         }

#     def save_checkpoint(self, epoch: int, path: str) -> None:
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
#             }
#             torch.save(state, path)
#             self.logger.logger.info(f"Saved checkpoint to {path}")
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def load_checkpoint(self, checkpoint_path: str) -> int:
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             for model_name, state_dict in checkpoint['models'].items():
#                 if model_name in self.models:
#                     self.models[model_name].load_state_dict(state_dict)
#                     self.logger.logger.info(f"Loaded {model_name} from checkpoint")
#                 else:
#                     self.logger.logger.warning(f"Model {model_name} in checkpoint not found in current models")
#             for opt_name, state_dict in checkpoint['optimizers'].items():
#                 if opt_name in self.optimizers:
#                     self.optimizers[opt_name].load_state_dict(state_dict)
#                     self.logger.logger.info(f"Loaded {opt_name} optimizer from checkpoint")
#                 else:
#                     self.logger.logger.warning(f"Optimizer {opt_name} in checkpoint not found in current optimizers")
#             start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
#             self.logger.logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
#             return start_epoch
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
#             raise

#     def validate(self) -> float:
#         try:
#             total_loss = 0.0
#             with torch.no_grad():
#                 for batch in self.val_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     mels_pe = mels.transpose(1, 2)
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, _ = self.models['flow'](z)
#                     mel_pred = self.models['decoder'](z_flow)
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self) -> None:
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.start_epoch, self.config['epochs'] + 1):
#                 torch.cuda.empty_cache()
#                 if torch.cuda.is_available():
#                     self.logger.logger.info(
#                         f"After empty_cache(): Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
#                         f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB"
#                     )

#                 gen_loss_accum = 0.0
#                 disc_loss_accum = 0.0
#                 accum_count = 0

#                 for batch in self.train_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     self.logger.logger.info(f"phonemes dtype: {phonemes.dtype}, mels dtype: {mels.dtype}, device: {phonemes.device}")
#                     phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
#                     mels_pe = mels.transpose(1, 2)
#                     self.logger.logger.info(f"mels_pe shape: {mels_pe.shape}, device: {mels_pe.device}")
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
#                     text_embed_dp = text_embed.transpose(1, 2)
#                     durations_pred = self.models['duration_predictor'](text_embed_dp)
#                     durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths).float().to(self.device)
#                     self.logger.logger.info(
#                         f"durations_pred shape: {durations_pred.shape}, dtype: {durations_pred.dtype}, device: {durations_pred.device}"
#                     )
#                     self.logger.logger.info(
#                         f"durations_gt shape: {durations_gt.shape}, dtype: {durations_gt.dtype}, device: {durations_gt.device}"
#                     )
#                     mel_pred = self.models['decoder'](z_flow)
#                     audio = self.models['generator'](mels_pe)
#                     self.logger.logger.info(f"audio shape: {audio.shape}, device: {audio.device}")
#                     audio_fake = self.models['generator'](mel_pred)
#                     self.logger.logger.info(f"audio_fake shape: {audio_fake.shape}, device: {audio_fake.device}")
#                     real_out_gen = self.models['discriminator'](audio)
#                     fake_out_gen = self.models['discriminator'](audio_fake)
#                     self.logger.logger.info(f"real_out_gen shapes: {[o.shape for o in real_out_gen]}, device: {real_out_gen[0].device}")
#                     self.logger.logger.info(f"fake_out_gen shapes: {[o.shape for o in fake_out_gen]}, device: {fake_out_gen[0].device}")
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     duration_loss = nn.MSELoss()(durations_pred, durations_gt)
#                     # Initialize adv_loss as a tensor to avoid type issues
#                     adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
#                     for f in fake_out_gen:
#                         loss = nn.MSELoss()(f, torch.ones_like(f))
#                         adv_loss = adv_loss + loss  # Accumulate tensor losses
#                     total_gen_loss = (recon_loss + kl_loss + duration_loss + adv_loss) / self.config['grad_accum_steps']
#                     self.logger.logger.info(
#                         f"recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}, duration_loss: {duration_loss.item()}, "
#                         f"adv_loss: {adv_loss.item()}, total_gen_loss: {total_gen_loss.item()}"
#                     )
#                     total_gen_loss.backward()
#                     gen_loss_accum += total_gen_loss.item()
#                     audio = self.models['generator'](mels_pe.detach())
#                     audio_fake = self.models['generator'](mel_pred.detach())
#                     real_out_disc = self.models['discriminator'](audio)
#                     fake_out_disc = self.models['discriminator'](audio_fake)
#                     self.logger.logger.info(f"real_out_disc shapes: {[o.shape for o in real_out_disc]}, device: {real_out_disc[0].device}")
#                     self.logger.logger.info(f"fake_out_disc shapes: {[o.shape for o in fake_out_disc]}, device: {fake_out_disc[0].device}")
#                     d_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
#                     for r, f in zip(real_out_disc, fake_out_disc):
#                         real_loss = nn.MSELoss()(r, torch.ones_like(r))
#                         fake_loss = nn.MSELoss()(f, torch.zeros_like(f))
#                         d_loss = d_loss + (real_loss + fake_loss)
#                     d_loss = d_loss / self.config['grad_accum_steps']
#                     d_loss.backward()
#                     disc_loss_accum += d_loss.item()
#                     accum_count += 1
#                     if accum_count == self.config['grad_accum_steps']:
#                         self.optimizers['gen'].step()
#                         self.optimizers['disc'].step()
#                         self.optimizers['gen'].zero_grad()
#                         self.optimizers['disc'].zero_grad()
#                         self.logger.log({
#                             'recon_loss': recon_loss.item(),
#                             'kl_loss': kl_loss.item(),
#                             'duration_loss': duration_loss.item(),
#                             'adv_loss': adv_loss.item(),
#                             'd_loss': disc_loss_accum / accum_count,
#                             'total_gen_loss': gen_loss_accum / accum_count,
#                         })
#                         gen_loss_accum = 0.0
#                         disc_loss_accum = 0.0
#                         accum_count = 0
#                 if accum_count > 0:  # Apply remaining gradients
#                     self.optimizers['gen'].step()
#                     self.optimizers['disc'].step()
#                     self.optimizers['gen'].zero_grad()
#                     self.optimizers['disc'].zero_grad()
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch}: Validation Loss = {val_loss}")
#                 if epoch % 10 == 0:  # Save checkpoint every 10 epochs
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch}.pt")
#                 torch.cuda.empty_cache()
#                 if torch.cuda.is_available():
#                     self.logger.logger.info(
#                         f"End of epoch {epoch} GPU: Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
#                         f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB"
#                     )
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv") -> float:
#         try:
#             test_dataset = VITSDataset(self.config['data_dir'], test_manifest, max_mel_length=self.config['max_mel_length'])
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0.0
#             with torch.no_grad():
#                 for batch in self.val_loader:
#                     phonemes = batch['phonemes'].to(self.device)
#                     mels = batch['mels'].to(self.device).float()
#                     phoneme_lengths = batch['phoneme_lengths'].to(self.device)
#                     mel_lengths = batch['mel_lengths'].to(self.device)
#                     mels_pe = mels.transpose(1, 2)
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
#                     z_flow, _ = self.models['flow'](z)
#                     mel_pred = self.models['decoder'](z_flow)
#                     recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
#                     total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise


# if __name__ == "__main__":
#     pipeline = TrainingPipeline(
#         config_path="/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml",
#         checkpoint_path="/teamspace/studios/this_studio/old/checkpoints/epoch_50.pt",  # Replace with path to checkpoint if resuming
#     )
#     pipeline.run()


import os
import sys
from typing import Dict, Optional
from pathlib import Path
import time

# Ensure project root is in sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Suppress IDE warnings for custom module imports
# If your IDE flags these as unresolved, ensure vits_nepali is in your project directory
from vits_nepali.models import (  # type: ignore
    TextEncoder,
    PosteriorEncoder,
    Flow,
    DurationPredictor,
    HiFiGANGenerator,
    Discriminator,
    Decoder,
)
from vits_nepali.data.dataset import VITSDataset, get_dataloader  # type: ignore
from vits_nepali.utils.logging import Logger  # type: ignore
from vits_nepali.utils.mas import monotonic_alignment_search  # type: ignore


class TrainingPipeline:
    def __init__(
        self,
        config_path: str,
        manifest_file: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        try:
            self.config = self.load_config(config_path)
            if manifest_file is not None:
                self.config['manifest_file'] = manifest_file
            self.logger = Logger(self.config['log_dir'])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.logger.info(f"Using device: {self.device}")
            if self.device.type == 'cuda':
                cuda_version = getattr(torch.version, 'cuda', 'N/A')  # Fallback if CUDA is unavailable
                self.logger.logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {cuda_version}")
            
            # Verify checkpoint directory
            checkpoint_dir = self.config['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            if not os.access(checkpoint_dir, os.W_OK):
                raise PermissionError(f"Checkpoint directory {checkpoint_dir} is not writable")
            self.logger.logger.info(f"Checkpoint directory verified: {checkpoint_dir}")

            self.models = self.initialize_models()
            self.optimizers = self.initialize_optimizers()
            self.config['batch_size'] = 4  # For GTX 1650 (4GB VRAM)
            self.config['max_mel_length'] = 300  # Reduce memory usage
            self.config['grad_accum_steps'] = 2  # Effective batch size = 2
            self.train_dataset = VITSDataset(
                self.config['data_dir'],
                self.config['manifest_file'],
                max_mel_length=self.config['max_mel_length'],
            )
            self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
            self.val_dataset = VITSDataset(
                self.config['data_dir'],
                self.config['val_manifest_file'],
                max_mel_length=self.config['max_mel_length'],
            )
            self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
            self.start_epoch: int = 1  # Default start epoch
            if checkpoint_path is not None:
                self.start_epoch = self.load_checkpoint(checkpoint_path)
        except Exception as e:
            self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
            raise

    def load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            required_fields = [
                'data_dir',
                'manifest_file',
                'val_manifest_file',
                'batch_size',
                'log_dir',
                'checkpoint_dir',
                'epochs',
                'lr',
                'n_vocab',
                'embed_dim',
                'periods',
            ]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
            return config
        except Exception as e:
            self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def initialize_models(self) -> Dict[str, nn.Module]:
        models = {
            'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
            'posterior_encoder': PosteriorEncoder(),
            'flow': Flow(),
            'duration_predictor': DurationPredictor(),
            'decoder': Decoder(),
            'generator': HiFiGANGenerator(),
            'discriminator': Discriminator(self.config['periods']),
        }
        return {k: v.to(self.device) for k, v in models.items()}

    def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            'gen': torch.optim.Adam(
                sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
                lr=self.config['lr'],
            ),
            'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr']),
        }

    def save_checkpoint(self, epoch: int, path: str) -> None:
        try:
            state = {
                'epoch': epoch,
                'models': {k: v.state_dict() for k, v in self.models.items()},
                'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            }
            torch.save(state, path)
            self.logger.logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> int:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            for model_name, state_dict in checkpoint['models'].items():
                if model_name in self.models:
                    self.models[model_name].load_state_dict(state_dict)
                    self.logger.logger.info(f"Loaded {model_name} from checkpoint")
                else:
                    self.logger.logger.warning(f"Model {model_name} in checkpoint not found in current models")
            for opt_name, state_dict in checkpoint['optimizers'].items():
                if opt_name in self.optimizers:
                    self.optimizers[opt_name].load_state_dict(state_dict)
                    self.logger.logger.info(f"Loaded {opt_name} optimizer from checkpoint")
                else:
                    self.logger.logger.warning(f"Optimizer {opt_name} in checkpoint not found in current optimizers")
            start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
            self.logger.logger.info(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
            return start_epoch
        except Exception as e:
            self.logger.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise

    def validate(self) -> float:
        try:
            total_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    phonemes = batch['phonemes'].to(self.device)
                    mels = batch['mels'].to(self.device).float()
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                    mel_lengths = batch['mel_lengths'].to(self.device)
                    mels_pe = mels.transpose(1, 2)
                    z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
                    z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
                    z_flow, _ = self.models['flow'](z)
                    mel_pred = self.models['decoder'](z_flow)
                    recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
                    total_loss += recon_loss.item()
            avg_loss = total_loss / len(self.val_loader)
            self.logger.log({'val_loss': avg_loss})
            return avg_loss
        except Exception as e:
            self.logger.logger.error(f"Validation failed: {str(e)}")
            raise

    def run(self) -> None:
        try:
            os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
            for epoch in range(self.start_epoch, self.config['epochs'] + 1):
                self.logger.logger.info(f"Starting epoch {epoch} with {len(self.train_loader)} batches")
                start_time = time.time()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    self.logger.logger.info(
                        f"After empty_cache(): Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
                        f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB"
                    )

                gen_loss_accum = 0.0
                disc_loss_accum = 0.0
                accum_count = 0

                for batch_idx, batch in enumerate(self.train_loader):
                    self.logger.logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_loader)}")
                    phonemes = batch['phonemes'].to(self.device)
                    mels = batch['mels'].to(self.device).float()
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                    mel_lengths = batch['mel_lengths'].to(self.device)
                    self.logger.logger.info(f"phonemes dtype: {phonemes.dtype}, mels dtype: {mels.dtype}, device: {phonemes.device}")
                    phoneme_mask = torch.arange(phonemes.size(1), device=self.device)[None, :] >= phoneme_lengths[:, None]
                    mels_pe = mels.transpose(1, 2)
                    self.logger.logger.info(f"mels_pe shape: {mels_pe.shape}, device: {mels_pe.device}")
                    z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
                    z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
                    z_flow, log_det = self.models['flow'](z)
                    text_embed = self.models['text_encoder'](phonemes, mask=phoneme_mask)
                    text_embed_dp = text_embed.transpose(1, 2)
                    durations_pred = self.models['duration_predictor'](text_embed_dp)
                    durations_gt = monotonic_alignment_search(text_embed, z_mu, phoneme_lengths, mel_lengths).float().to(self.device)
                    self.logger.logger.info(
                        f"durations_pred shape: {durations_pred.shape}, dtype: {durations_pred.dtype}, device: {durations_pred.device}"
                    )
                    self.logger.logger.info(
                        f"durations_gt shape: {durations_gt.shape}, dtype: {durations_gt.dtype}, device: {durations_gt.device}"
                    )
                    mel_pred = self.models['decoder'](z_flow)
                    audio = self.models['generator'](mels_pe)
                    self.logger.logger.info(f"audio shape: {audio.shape}, device: {audio.device}")
                    audio_fake = self.models['generator'](mel_pred)
                    self.logger.logger.info(f"audio_fake shape: {audio_fake.shape}, device: {audio_fake.device}")
                    real_out_gen = self.models['discriminator'](audio)
                    fake_out_gen = self.models['discriminator'](audio_fake)
                    self.logger.logger.info(f"real_out_gen shapes: {[o.shape for o in real_out_gen]}, device: {real_out_gen[0].device}")
                    self.logger.logger.info(f"fake_out_gen shapes: {[o.shape for o in fake_out_gen]}, device: {fake_out_gen[0].device}")
                    recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
                    kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                    duration_loss = nn.MSELoss()(durations_pred, durations_gt)
                    # Initialize adv_loss as a tensor to avoid type issues
                    adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    for f in fake_out_gen:
                        loss = nn.MSELoss()(f, torch.ones_like(f))
                        adv_loss = adv_loss + loss  # Accumulate tensor losses
                    total_gen_loss = (recon_loss + kl_loss + duration_loss + adv_loss) / self.config['grad_accum_steps']
                    self.logger.logger.info(
                        f"recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}, duration_loss: {duration_loss.item()}, "
                        f"adv_loss: {adv_loss.item()}, total_gen_loss: {total_gen_loss.item()}"
                    )
                    total_gen_loss.backward()
                    gen_loss_accum += total_gen_loss.item()
                    audio = self.models['generator'](mels_pe.detach())
                    audio_fake = self.models['generator'](mel_pred.detach())
                    real_out_disc = self.models['discriminator'](audio)
                    fake_out_disc = self.models['discriminator'](audio_fake)
                    self.logger.logger.info(f"real_out_disc shapes: {[o.shape for o in real_out_disc]}, device: {real_out_disc[0].device}")
                    self.logger.logger.info(f"fake_out_disc shapes: {[o.shape for o in fake_out_disc]}, device: {fake_out_disc[0].device}")
                    d_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    for r, f in zip(real_out_disc, fake_out_disc):
                        real_loss = nn.MSELoss()(r, torch.ones_like(r))
                        fake_loss = nn.MSELoss()(f, torch.zeros_like(f))
                        d_loss = d_loss + (real_loss + fake_loss)
                    d_loss = d_loss / self.config['grad_accum_steps']
                    d_loss.backward()
                    disc_loss_accum += d_loss.item()
                    accum_count += 1
                    if accum_count == self.config['grad_accum_steps']:
                        self.optimizers['gen'].step()
                        self.optimizers['disc'].step()
                        self.optimizers['gen'].zero_grad()
                        self.optimizers['disc'].zero_grad()
                        self.logger.log({
                            'recon_loss': recon_loss.item(),
                            'kl_loss': kl_loss.item(),
                            'duration_loss': duration_loss.item(),
                            'adv_loss': adv_loss.item(),
                            'd_loss': disc_loss_accum / accum_count,
                            'total_gen_loss': gen_loss_accum / accum_count,
                        })
                        gen_loss_accum = 0.0
                        disc_loss_accum = 0.0
                        accum_count = 0
                if accum_count > 0:  # Apply remaining gradients
                    self.optimizers['gen'].step()
                    self.optimizers['disc'].step()
                    self.optimizers['gen'].zero_grad()
                    self.optimizers['disc'].zero_grad()
                
                elapsed_time = time.time() - start_time
                self.logger.logger.info(f"Epoch {epoch} completed in {elapsed_time:.2f} seconds")
                
                val_loss = self.validate()
                self.logger.logger.info(f"Epoch {epoch}: Validation Loss = {val_loss}")
                
                # Save checkpoint every 10 epochs
                self.logger.logger.info(f"Checking if checkpoint should be saved for epoch {epoch}")
                if epoch % 1 == 0:
                    checkpoint_path = f"{self.config['checkpoint_dir']}/epoch_{epoch}.pt"
                    self.logger.logger.info(f"Attempting to save checkpoint to {checkpoint_path}")
                    self.save_checkpoint(epoch, checkpoint_path)
                
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    self.logger.logger.info(
                        f"End of epoch {epoch} GPU: Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
                        f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB"
                    )
        except Exception as e:
            self.logger.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(self, test_manifest: str = "/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv") -> float:
        try:
            test_dataset = VITSDataset(self.config['data_dir'], test_manifest, max_mel_length=self.config['max_mel_length'])
            test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
            total_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    phonemes = batch['phonemes'].to(self.device)
                    mels = batch['mels'].to(self.device).float()
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                    mel_lengths = batch['mel_lengths'].to(self.device)
                    mels_pe = mels.transpose(1, 2)
                    z_mu, z_logvar = self.models['posterior_encoder'](mels_pe)
                    z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
                    z_flow, _ = self.models['flow'](z)
                    mel_pred = self.models['decoder'](z_flow)
                    recon_loss = nn.MSELoss()(mel_pred.transpose(1, 2), mels)
                    total_loss += recon_loss.item()
            avg_loss = total_loss / len(self.val_loader)
            self.logger.log({'test_loss': avg_loss})
            return avg_loss
        except Exception as e:
            self.logger.logger.error(f"Evaluation failed: {str(e)}")
            raise


if __name__ == "__main__":
    pipeline = TrainingPipeline(
        config_path="/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml",
        checkpoint_path="/teamspace/studios/this_studio/old/checkpoints/epoch_40.pt",  # Corrected to specific checkpoint file
    )
    pipeline.run()