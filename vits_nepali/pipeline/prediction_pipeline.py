# # Placeholder file
# # pipeline/prediction_pipeline.py
# import torch
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator
# from vits_nepali.utils.audio import save_audio
# from vits_nepali.utils.text import text_to_phonemes
# import yaml
# import logging
# import os

# logger = logging.getLogger(__name__)

# class PredictionPipeline:
#     def __init__(self, config_path: str, checkpoint_path: str):
#         try:
#             self.config = self.load_config(config_path)
#             self.models = self.load_model(checkpoint_path)
#         except Exception as e:
#             logger.error(f"Failed to initialize PredictionPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> dict:
#         try:
#             with open(config_path, 'r') as f:
#                 return yaml.safe_load(f)
#         except Exception as e:
#             logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def load_model(self, checkpoint_path: str) -> dict:
#         try:
#             if not os.path.exists(checkpoint_path):
#                 raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location='cuda')
#             models = {
#                 'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
#                 'posterior_encoder': PosteriorEncoder().cuda(),
#                 'flow': Flow().cuda(),
#                 'duration_predictor': DurationPredictor().cuda(),
#                 'generator': HiFiGANGenerator().cuda()
#             }
#             for name, model in models.items():
#                 model.load_state_dict(checkpoint['models'][name])
#                 model.eval()
#             return models
#         except Exception as e:
#             logger.error(f"Failed to load model from {checkpoint_path}: {str(e)}")
#             raise

#     def predict(self, text: str, output_path: str) -> None:
#         try:
#             phonemes = torch.tensor([text_to_phonemes(text)], dtype=torch.long).cuda()
#             with torch.no_grad():
#                 text_embed = self.models['text_encoder'](phonemes)
#                 durations = self.models['duration_predictor'](text_embed)
#                 z = torch.randn(1, self.config['embed_dim'], int(durations.sum())).cuda()
#                 z_flow, _ = self.models['flow'](z)
#                 mel = self.models['generator'](z_flow)
#                 audio = self.models['generator'](mel)
#             save_audio(audio.cpu(), output_path, self.config['sample_rate'])
#             logger.info(f"Synthesized audio saved to {output_path}")
#         except Exception as e:
#             logger.error(f"Prediction failed for text '{text}': {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = PredictionPipeline("configs/config.yaml", "checkpoints/epoch_100.pt")
#     pipeline.predict("नमस्ते", "output.wav")

# pipeline/prediction_pipeline.py
# import torch
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator
# from vits_nepali.utils.audio import save_audio
# from vits_nepali.utils.text import text_to_phonemes
# import yaml
# import logging
# import os

# logger = logging.getLogger(__name__)

# class PredictionPipeline:
#     def __init__(self, config_path: str, checkpoint_path: str):
#         try:
#             self.config = self.load_config(config_path)
#             self.models = self.load_model(checkpoint_path)
#         except Exception as e:
#             logger.error(f"Failed to initialize PredictionPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             # Validate required config keys
#             required_keys = ['n_vocab', 'embed_dim', 'sample_rate']
#             for key in required_keys:
#                 if key not in config:
#                     raise ValueError(f"Missing required config key: {key}")
#             return config
#         except Exception as e:
#             logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def load_model(self, checkpoint_path: str) -> dict:
#         try:
#             if not os.path.exists(checkpoint_path):
#                 raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location='cuda')
#             models = {
#                 'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
#                 'posterior_encoder': PosteriorEncoder().cuda(),
#                 'flow': Flow().cuda(),
#                 'duration_predictor': DurationPredictor().cuda(),
#                 'generator': HiFiGANGenerator().cuda()
#             }
#             for name, model in models.items():
#                 model.load_state_dict(checkpoint['models'][name])
#                 model.eval()
#             return models
#         except Exception as e:
#             logger.error(f"Failed to load model from {checkpoint_path}: {str(e)}")
#             raise

#     def predict(self, text: str, output_path: str) -> None:
#         try:
#             phonemes = torch.tensor([text_to_phonemes(text)], dtype=torch.long).cuda()
#             with torch.no_grad():
#                 text_embed = self.models['text_encoder'](phonemes)
#                 # Transpose text_embed to [batch, embed_dim, sequence_length]
#                 text_embed = text_embed.transpose(1, 2)  # From [1, seq_len, embed_dim] to [1, embed_dim, seq_len]
#                 durations = self.models['duration_predictor'](text_embed)
#                 z = torch.randn(1, self.config['embed_dim'], int(durations.sum())).cuda()
#                 z_flow, _ = self.models['flow'](z)
#                 # Generate audio directly from z_flow
#                 audio = self.models['generator'](z_flow)
#             save_audio(audio.cpu(), output_path, self.config['sample_rate'])
#             logger.info(f"Synthesized audio saved to {output_path}")
#         except Exception as e:
#             logger.error(f"Prediction failed for text '{text}': {str(e)}")
#             raise

# pipeline/prediction_pipeline.py
import torch
from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Decoder
from vits_nepali.utils.audio import save_audio
from vits_nepali.utils.text import text_to_phonemes
import yaml
import logging
import os

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device = torch.device("cpu")):
        self.device = device
        try:
            self.config = self.load_config(config_path)
            self.models = self.load_model(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to initialize PredictionPipeline: {str(e)}")
            raise

    def load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Validate required config keys
            required_keys = ['n_vocab', 'embed_dim', 'sample_rate']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            logger.info(f"Loaded config: {config}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def load_model(self, checkpoint_path: str) -> dict:
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info(f"Checkpoint keys: {checkpoint.keys()}")
            if 'models' not in checkpoint:
                raise KeyError("Checkpoint missing 'models' key")
            logger.info(f"Model keys: {checkpoint['models'].keys()}")
            models = {
                'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).to(self.device),
                'posterior_encoder': PosteriorEncoder().to(self.device),
                'flow': Flow().to(self.device),
                'duration_predictor': DurationPredictor().to(self.device),
                'decoder': Decoder().to(self.device),
                'generator': HiFiGANGenerator().to(self.device)
            }
            for name, model in models.items():
                if name not in checkpoint['models']:
                    raise KeyError(f"Checkpoint missing model state for {name}")
                model.load_state_dict(checkpoint['models'][name])
                model.eval()
            return models
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {str(e)}")
            raise

    def predict(self, text: str, output_path: str) -> None:
        try:
            phonemes = torch.tensor([text_to_phonemes(text)], dtype=torch.long).to(self.device)
            logger.info(f"Phoneme indices: {phonemes}")
            with torch.no_grad():
                text_embed = self.models['text_encoder'](phonemes)
                logger.info(f"Text embed shape: {text_embed.shape}")
                # Transpose text_embed to [batch, embed_dim, sequence_length]
                text_embed = text_embed.transpose(1, 2)  # From [1, seq_len, embed_dim] to [1, embed_dim, seq_len]
                logger.info(f"Text embed transposed shape: {text_embed.shape}")
                durations = self.models['duration_predictor'](text_embed)
                logger.info(f"Durations shape: {durations.shape}, sum: {durations.sum()}")
                z = torch.randn(1, self.config['embed_dim'], int(durations.sum())).to(self.device)
                logger.info(f"z shape: {z.shape}")
                z_flow, _ = self.models['flow'](z)
                logger.info(f"z_flow shape: {z_flow.shape}")
                mel = self.models['decoder'](z_flow)
                logger.info(f"Mel spectrogram shape: {mel.shape}")
                audio = self.models['generator'](mel)
                logger.info(f"Audio shape: {audio.shape}")
            # Squeeze audio to [1, time] if necessary for torchaudio.save
            audio = audio.squeeze(1) if audio.dim() == 3 else audio
            logger.info(f"Audio shape after squeeze: {audio.shape}")
            save_audio(audio.cpu(), output_path, self.config['sample_rate'])
            logger.info(f"Synthesized audio saved to {output_path}")
        except Exception as e:
            logger.error(f"Prediction failed for text '{text}': {str(e)}")
            raise