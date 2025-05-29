# TEST Set 1

# #####################
# # Text to phonemes module test
# from vits_nepali.utils.text import text_to_phonemes
# text = "नमस्ते"
# indices = text_to_phonemes(text)
# print(f"Text: {text}, Indices: {indices}")  # Expected: [29, 34, 41, 25, 6]

# #####################
# # VITS module test
# from vits_nepali.data.dataset import VITSDataset
# dataset = VITSDataset("vits_nepali/data/dataset/", "vits_nepali/data/csv/train_phonemes.csv")
# phonemes, mel = dataset[0]
# print(f"Phonemes: {phonemes}")  # Expected: tensor([29, 34, 41, 25, 6], dtype=torch.long)
# print(f"Mel shape: {mel.shape}")  # Expected: torch.Size([80, T]) for 80 mel bins

#####################
# Pipeline init test
# from vits_nepali.pipeline.training_pipeline import TrainingPipeline
# pipeline = TrainingPipeline("vits_nepali/configs/config.yaml")
# print("Pipeline initialized successfully")

# #####################
# # Audio path verification
# import csv
# import os
# from pathlib import Path

# def check_audio_paths(manifest_file: str, data_dir: str = "vits_nepali/data/dataset/"):
#     data_dir = Path(data_dir)
#     missing_files = []
#     with open(manifest_file, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             audio_path = data_dir / row['path']
#             if not audio_path.exists():
#                 missing_files.append(str(audio_path))
#     return missing_files

# #####################
# # Test all CSVs
# csvs = ["data/csv/train_phonemes.csv", "data/csv/val_phonemes.csv", "data/csv/test_phonemes.csv"]
# for csv_file in csvs:
#     missing = check_audio_paths("vits_nepali/" + csv_file)
#     if missing:
#         print(f"Missing files in {csv_file}: {missing}")
#     else:
#         print(f"All audio files found for {csv_file}")


# TEST Set 2
#####################
# TextEncoder Forward Pass
# import torch
# from vits_nepali.models.text_encoder import TextEncoder

# def test_text_encoder():
#     try:
#         # Define dummy input
#         batch_size = 2
#         seq_len = 5
#         vocab_size = 100
#         embed_dim = 192
#         dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # e.g., [[3, 12, 45, 23, 7], ...]

#         # Create model
#         model = TextEncoder(n_vocab=vocab_size, embed_dim=embed_dim)
        
#         # Forward pass
#         output = model(dummy_input)

#         # Assertions
#         assert output.shape == (batch_size, seq_len, embed_dim), \
#             f"Expected output shape {(batch_size, seq_len, embed_dim)}, but got {output.shape}"
        
#         print("✅ TextEncoder test passed.")
#         print(f"Input shape: {dummy_input.shape}")
#         print(f"Output shape: {output.shape}")
#     except Exception as e:
#         print(f"❌ TextEncoder test failed: {e}")

# # Run the test
# test_text_encoder()

#########################
# Test Dataset Loading
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from vits_nepali.data.dataset import VITSDataset

# try:
#     dataset = VITSDataset("vits_nepali/data/dataset", "vits_nepali/data/csv/train.csv")
#     print(f"Dataset size: {len(dataset)}")
#     for i in range(min(5, len(dataset))):
#         try:
#             item = dataset[i]
#             print(f"Item {i}: phonemes={item['phonemes'].shape}, mels={item['mels'].shape}, "
#                   f"phoneme_lengths={item['phoneme_lengths']}, mel_lengths={item['mel_lengths']}")
#         except Exception as e:
#             print(f"Error loading item {i}: {str(e)}")
# except Exception as e:
#     print(f"Failed to load dataset: {str(e)}")


#######################
## Audio length test

# import os
# import torchaudio
# import csv
# from pathlib import Path

# manifest_file = "vits_nepali/data/csv/train_phonemes.csv"
# data_dir = "vits_nepali/data/dataset"
# with open(manifest_file, 'r', encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         audio_path = Path(data_dir) / row['path']
#         try:
#             waveform, sr = torchaudio.load(audio_path)
#             duration = waveform.size(1) / sr
#             print(f"{audio_path}: {duration:.2f} seconds")
#         except Exception as e:
#             print(f"{audio_path}: Failed ({str(e)})")

###################
# Audio shape test

# import torchaudio
# import os

# audio_dir = "vits_nepali/data/dataset/train"  # Adjust to your audio directory
# for audio_file in os.listdir(audio_dir):
#     if audio_file.endswith(".wav"):
#         waveform, sample_rate = torchaudio.load(os.path.join(audio_dir, audio_file))
#         print(f"{audio_file}: Channels = {waveform.shape[0]}")

#############################
#checkpoint test
import torch

# Load checkpoint with CPU mapping
checkpoint_path = "/teamspace/studios/this_studio/old/checkpoints/epoch_50.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
print("Checkpoint keys:", checkpoint.keys())
if 'models' in checkpoint:
    print("Model keys:", checkpoint['models'].keys())