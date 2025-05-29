# # # Placeholder file
# # # data/dataset.py
# # from torch.utils.data import Dataset, DataLoader
# # from typing import Tuple, List
# # import torch
# # from .preprocess import audio_to_mel, preprocess_data
# # from utils.text import text_to_phonemes
# # import logging

# # logger = logging.getLogger(__name__)

# # class VITSDataset(Dataset):
# #     def __init__(self, data_dir: str, manifest_file: str, sample_rate: int = 16000, n_mels: int = 80):
# #         self.data = preprocess_data(data_dir, manifest_file)
# #         self.sample_rate = sample_rate
# #         self.n_mels = n_mels

# #     def __len__(self) -> int:
# #         return len(self.data)

# #     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
# #         audio_path, text = self.data[idx]
# #         try:
# #             phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)
# #             mel = audio_to_mel(audio_path, self.sample_rate, self.n_mels)
# #             return phonemes, mel
# #         except Exception as e:
# #             logger.error(f"Error loading item {idx} ({audio_path}): {str(e)}")
# #             raise

# # def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
# #     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from typing import Tuple
# from utils.text import text_to_phonemes

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         self.data_dir = Path(data_dir)
#         self.audio_files = []
#         self.texts = []
#         self.phonemes = []  # Store precomputed phoneme indices
        
#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             if 'path' not in reader.fieldnames or 'labels' not in reader.fieldnames:
#                 raise ValueError(f"CSV {manifest_file} must have 'path' and 'labels' columns")
#             for row in reader:
#                 self.audio_files.append(row['path'])
#                 self.texts.append(row['labels'])
#                 # Check for phonemes column
#                 if 'phonemes' in row and row['phonemes']:
#                     self.phonemes.append(row['phonemes'])  # Store phoneme string
#                 else:
#                     self.phonemes.append(None)
        
#         self.transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_mels=80, hop_length=256
#         )
    
#     def __len__(self) -> int:
#         return len(self.audio_files)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         audio_path = self.data_dir / self.audio_files[idx]
#         text = self.texts[idx]
#         phoneme_str = self.phonemes[idx]
        
#         # Load audio and compute mel-spectrogram
#         waveform, sample_rate = torchaudio.load(audio_path)
#         if sample_rate != 16000:
#             waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#         mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)
        
#         # Use precomputed phonemes if available, else compute on-the-fly
#         phonemes = torch.tensor(text_to_phonemes(phoneme_str or text), dtype=torch.long)
        
#         return phonemes, mel
#########################################33
# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from typing import Tuple
# from vits_nepali.utils.text import text_to_phonemes, phoneme_vocab  # Make sure phoneme_vocab is defined in text.py

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         self.data_dir = Path(data_dir)
#         self.audio_files = []
#         self.texts = []
#         self.phonemes = []

#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             if 'path' not in reader.fieldnames or 'labels' not in reader.fieldnames:
#                 raise ValueError(f"CSV {manifest_file} must have 'path' and 'labels' columns")

#             for row in reader:
#                 self.audio_files.append(row['path'])
#                 self.texts.append(row['labels'])

#                 # Check if phonemes column exists and is valid
#                 if 'phonemes' in row and row['phonemes']:
#                     phoneme_str = row['phonemes']
#                     phoneme_list = phoneme_str.split()
#                     phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
#                     self.phonemes.append(phoneme_indices)
#                 else:
#                     self.phonemes.append(None)

#         self.transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_mels=80, hop_length=256
#         )

#     def __len__(self) -> int:
#         return len(self.audio_files)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         audio_path = self.data_dir / self.audio_files[idx]
#         text = self.texts[idx]
#         precomputed_phonemes = self.phonemes[idx]

#         # Load audio and compute mel-spectrogram
#         waveform, sample_rate = torchaudio.load(audio_path)
#         if sample_rate != 16000:
#             waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#         mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)

#         # Convert phonemes to tensor
#         if precomputed_phonemes is not None:
#             phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
#         else:
#             phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

#         return phonemes, mel
#####################
#New change 

# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from torch.utils.data import DataLoader
# from typing import Tuple
# from vits_nepali.utils.text import text_to_phonemes, phoneme_vocab  # Ensure phoneme_vocab is defined

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         self.data_dir = Path(data_dir)
#         self.audio_files = []
#         self.texts = []
#         self.phonemes = []

#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             if 'path' not in reader.fieldnames or 'labels' not in reader.fieldnames:
#                 raise ValueError(f"CSV {manifest_file} must have 'path' and 'labels' columns")

#             for row in reader:
#                 self.audio_files.append(row['path'])
#                 self.texts.append(row['labels'])

#                 # Check if phonemes column exists and is valid
#                 if 'phonemes' in row and row['phonemes']:
#                     phoneme_str = row['phonemes']
#                     phoneme_list = phoneme_str.split()
#                     phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
#                     self.phonemes.append(phoneme_indices)
#                 else:
#                     self.phonemes.append(None)

#         self.transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_mels=80, hop_length=256
#         )

#     def __len__(self) -> int:
#         return len(self.audio_files)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         audio_path = self.data_dir / self.audio_files[idx]
#         text = self.texts[idx]
#         precomputed_phonemes = self.phonemes[idx]

#         # Load audio and compute mel-spectrogram
#         waveform, sample_rate = torchaudio.load(audio_path)
#         if sample_rate != 16000:
#             waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#         mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)

#         # Convert phonemes to tensor
#         if precomputed_phonemes is not None:
#             phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
#         else:
#             phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

#         return phonemes, mel

# def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

##PREVIOUS ONE

# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from typing import Tuple
# from vits_nepali.utils.text import text_to_phonemes, phoneme_vocab  # Ensure phoneme_vocab is defined

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         self.data_dir = Path(data_dir)
#         self.audio_files = []
#         self.texts = []
#         self.phonemes = []

#         with open(manifest_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             if 'path' not in reader.fieldnames or 'labels' not in reader.fieldnames:
#                 raise ValueError(f"CSV {manifest_file} must have 'path' and 'labels' columns")

#             for row in reader:
#                 self.audio_files.append(row['path'])
#                 self.texts.append(row['labels'])

#                 # Check if phonemes column exists and is valid
#                 if 'phonemes' in row and row['phonemes']:
#                     phoneme_str = row['phonemes']
#                     phoneme_list = phoneme_str.split()
#                     phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
#                     self.phonemes.append(phoneme_indices)
#                 else:
#                     self.phonemes.append(None)

#         self.transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_mels=80, hop_length=256
#         )

#     def __len__(self) -> int:
#         return len(self.audio_files)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         audio_path = self.data_dir / self.audio_files[idx]
#         text = self.texts[idx]
#         precomputed_phonemes = self.phonemes[idx]

#         # Load audio and compute mel-spectrogram
#         waveform, sample_rate = torchaudio.load(audio_path)
#         if sample_rate != 16000:
#             waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#         mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)

#         # Convert phonemes to tensor
#         if precomputed_phonemes is not None:
#             phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
#         else:
#             phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

#         return phonemes, mel

# def custom_collate_fn(batch):
#     """
#     Custom collate function to pad variable-length phoneme sequences and mel-spectrograms.
#     Args:
#         batch: List of tuples (phonemes, mel) from VITSDataset.
#     Returns:
#         Dict with padded phonemes, mel-spectrograms, and their lengths.
#     """
#     phonemes = [item[0] for item in batch]  # 1D tensor of phoneme indices
#     mels = [item[1] for item in batch]      # 2D tensor of mel-spectrograms (T, n_mels)

#     # Pad phonemes (1D tensors) to the longest sequence
#     phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)

#     # Pad mel-spectrograms (2D tensors) to the longest time dimension
#     mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)

#     # Compute lengths of original sequences
#     phoneme_lengths = torch.tensor([len(p) for p in phonemes], dtype=torch.long)
#     mel_lengths = torch.tensor([len(m) for m in mels], dtype=torch.long)

#     return {
#         'phonemes': phonemes_padded,      # Shape: (batch_size, max_phoneme_len)
#         'mels': mels_padded,              # Shape: (batch_size, max_mel_frames, n_mels)
#         'phoneme_lengths': phoneme_lengths,  # Shape: (batch_size,)
#         'mel_lengths': mel_lengths           # Shape: (batch_size,)
#     }

# def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=0,        # Set to 0 for Windows compatibility
#         pin_memory=False,     # Disable since no GPU is detected
#         collate_fn=custom_collate_fn
#     )

# import logging
# import os
# import pandas as pd
# import torch
# import torchaudio
# import csv
# from torch.utils.data import DataLoader, Dataset
# from typing import Dict, List, Tuple
# import numpy as np
# from vits_nepali.utils.text import phoneme_to_sequence

# logger = logging.getLogger(__name__)

# class VITSDataset(Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         try:
#             self.data_dir = data_dir
#             logger.info(f"Loading manifest from {manifest_file}")
#             self.manifest = pd.read_csv(manifest_file, quoting=csv.QUOTE_ALL, encoding='utf-8')
#             logger.info(f"Manifest loaded with {len(self.manifest)} rows")
#             if not isinstance(self.manifest, pd.DataFrame):
#                 raise ValueError("Manifest is not a Pandas DataFrame")
#             required_columns = ['path', 'phonemes']
#             if not all(col in self.manifest.columns for col in required_columns):
#                 raise ValueError(f"Manifest missing required columns: {required_columns}")
#             self.phoneme_vocab = phoneme_to_sequence
#             self.mel_transform = torchaudio.transforms.MelSpectrogram(
#                 sample_rate=16000,
#                 n_mels=80,
#                 n_fft=1024,
#                 hop_length=256,
#                 f_min=0,
#                 f_max=8000
#             )
#             self.max_audio_samples = 32000  # 2 seconds at 16,000 Hz
#         except Exception as e:
#             logger.error(f"Failed to initialize VITSDataset: {str(e)}")
#             raise

#     def __len__(self) -> int:
#         return len(self.manifest)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         try:
#             row = self.manifest.iloc[idx]
#             logger.debug(f"Processing item {idx}: {row.to_dict()}")
#             if not isinstance(row, pd.Series):
#                 raise ValueError(f"Row {idx} is not a Pandas Series: {type(row)}")
#             audio_path = os.path.join(self.data_dir, row['path'])
#             phonemes = row['phonemes'].split() if isinstance(row['phonemes'], str) else []
            
#             # Load audio with error handling
#             try:
#                 waveform, sample_rate = torchaudio.load(audio_path)
#                 if sample_rate != 16000:
#                     waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#                 # Truncate audio to max duration
#                 if waveform.size(1) > self.max_audio_samples:
#                     waveform = waveform[:, :self.max_audio_samples]
#             except Exception as e:
#                 logger.error(f"Failed to load audio {audio_path}: {str(e)}")
#                 return {
#                     'phonemes': torch.zeros(1, dtype=torch.long),
#                     'mels': torch.zeros(1, 80),
#                     'phoneme_lengths': torch.tensor(0, dtype=torch.long),
#                     'mel_lengths': torch.tensor(0, dtype=torch.long)
#                 }

#             # Process audio
#             mel = self.mel_transform(waveform)
#             mel = mel.squeeze(0).transpose(0, 1)  # Shape: (time, n_mels)
            
#             # Process phonemes
#             phoneme_ids = [self.phoneme_vocab.get(p, 0) for p in phonemes]
#             phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
            
#             return {
#                 'phonemes': phoneme_tensor,
#                 'mels': mel,
#                 'phoneme_lengths': torch.tensor(len(phoneme_ids), dtype=torch.long),
#                 'mel_lengths': torch.tensor(mel.size(0), dtype=torch.long)
#             }
#         except Exception as e:
#             logger.error(f"Failed to process item {idx}: {str(e)}")
#             raise

# def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     try:
#         phonemes = [item['phonemes'] for item in batch]
#         mels = [item['mels'] for item in batch]
#         phoneme_lengths = torch.stack([item['phoneme_lengths'] for item in batch])
#         mel_lengths = torch.stack([item['mel_lengths'] for item in batch])
        
#         valid_indices = [i for i, (p, m) in enumerate(zip(phonemes, mels)) if p.size(0) > 0 and m.size(0) > 0]
#         if not valid_indices:
#             logger.warning("Batch contains no valid samples")
#             return {
#                 'phonemes': torch.zeros(1, 1, dtype=torch.long),
#                 'mels': torch.zeros(1, 1, 80),
#                 'phoneme_lengths': torch.zeros(1, dtype=torch.long),
#                 'mel_lengths': torch.zeros(1, dtype=torch.long)
#             }
        
#         phonemes = [phonemes[i] for i in valid_indices]
#         mels = [mels[i] for i in valid_indices]
#         phoneme_lengths = phoneme_lengths[valid_indices]
#         mel_lengths = mel_lengths[valid_indices]
        
#         max_phoneme_len = max(p.size(0) for p in phonemes)
#         max_mel_frames = max(m.size(0) for m in mels)
        
#         phoneme_padded = torch.zeros(len(phonemes), max_phoneme_len, dtype=torch.long)
#         mel_padded = torch.zeros(len(mels), max_mel_frames, 80)
        
#         for i, p in enumerate(phonemes):
#             phoneme_padded[i, :p.size(0)] = p
#         for i, m in enumerate(mels):
#             mel_padded[i, :m.size(0), :] = m
        
#         return {
#             'phonemes': phoneme_padded,
#             'mels': mel_padded,
#             'phoneme_lengths': phoneme_lengths,
#             'mel_lengths': mel_lengths
#         }
#     except Exception as e:
#         logger.error(f"Failed in custom_collate_fn: {str(e)}")
#         raise

# def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
#     try:
#         return DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             collate_fn=custom_collate_fn
#         )
#     except Exception as e:
#         logger.error(f"Failed to create DataLoader: {str(e)}")
#         raise


# # NEW ONE 
# import logging
# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from typing import Tuple
# from vits_nepali.utils.text import text_to_phonemes, phoneme_vocab  # Ensure these are defined

# logger = logging.getLogger(__name__)

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         try:
#             self.data_dir = Path(data_dir)
#             self.audio_files = []
#             self.texts = []
#             self.phonemes = []

#             logger.info(f"Loading manifest from {manifest_file}")
#             with open(manifest_file, 'r', encoding='utf-8') as f:
#                 reader = csv.DictReader(f)
#                 required_fields = ['path', 'labels', 'phonemes']
#                 if not all(field in reader.fieldnames for field in required_fields):
#                     raise ValueError(f"CSV {manifest_file} must have {required_fields} columns")

#                 for row in reader:
#                     self.audio_files.append(row['path'])
#                     self.texts.append(row['labels'])
#                     phoneme_str = row['phonemes']
#                     if phoneme_str:
#                         phoneme_list = phoneme_str.split()
#                         phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
#                         self.phonemes.append(phoneme_indices)
#                     else:
#                         logger.warning(f"No phonemes for {row['path']}; using text-to-phonemes")
#                         self.phonemes.append(None)

#             logger.info(f"Loaded {len(self.audio_files)} items from manifest")
#             self.transform = torchaudio.transforms.MelSpectrogram(
#                 sample_rate=16000,  # Match config.yaml
#                 n_mels=80,
#                 n_fft=1024,
#                 hop_length=256,
#                 f_min=0,
#                 f_max=8000
#             )
#         except Exception as e:
#             logger.error(f"Failed to initialize VITSDataset: {str(e)}")
#             raise

#     def __len__(self) -> int:
#         return len(self.audio_files)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         try:
#             audio_path = self.data_dir / self.audio_files[idx]
#             text = self.texts[idx]
#             precomputed_phonemes = self.phonemes[idx]

#             # Load audio and compute mel-spectrogram
#             try:
#                 waveform, sample_rate = torchaudio.load(audio_path)
#                 if sample_rate != 16000:
#                     waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#             except Exception as e:
#                 logger.error(f"Failed to load audio {audio_path}: {str(e)}")
#                 return (
#                     torch.zeros(1, dtype=torch.long),  # Dummy phonemes
#                     torch.zeros(1, 80),               # Dummy mel
#                     torch.tensor(0, dtype=torch.long),  # Phoneme length
#                     torch.tensor(0, dtype=torch.long)   # Mel length
#                 )

#             mel = self.transform(waveform).squeeze(0).transpose(0, 1)  # (T, n_mels)

#             # Convert phonemes to tensor
#             if precomputed_phonemes is not None:
#                 phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
#             else:
#                 phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

#             return {
#                 'phonemes': phonemes,
#                 'mels': mel,
#                 'phoneme_lengths': torch.tensor(len(phonemes), dtype=torch.long),
#                 'mel_lengths': torch.tensor(mel.size(0), dtype=torch.long)
#             }
#         except Exception as e:
#             logger.error(f"Failed to process item {idx}: {str(e)}")
#             raise

# def custom_collate_fn(batch):
#     try:
#         phonemes = [item['phonemes'] for item in batch]
#         mels = [item['mels'] for item in batch]
#         phoneme_lengths = torch.tensor([item['phoneme_lengths'] for item in batch], dtype=torch.long)
#         mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], dtype=torch.long)

#         # Pad phonemes (1D tensors) to the longest sequence
#         phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)

#         # Pad mel-spectrograms (2D tensors) to the longest time dimension
#         mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)

#         return {
#             'phonemes': phonemes_padded,      # Shape: (batch_size, max_phoneme_len)
#             'mels': mels_padded,              # Shape: (batch_size, max_mel_frames, n_mels)
#             'phoneme_lengths': phoneme_lengths,  # Shape: (batch_size,)
#             'mel_lengths': mel_lengths           # Shape: (batch_size,)
#         }
#     except Exception as e:
#         logger.error(f"Failed in custom_collate_fn: {str(e)}")
#         raise

# def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
#     try:
#         return DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=0,        # Windows compatibility
#             pin_memory=False,     # No GPU
#             collate_fn=custom_collate_fn
#         )
#     except Exception as e:
#         logger.error(f"Failed to create DataLoader: {str(e)}")
#         raise

# import logging
# import torch
# import torchaudio
# import csv
# from pathlib import Path
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from typing import Tuple
# from vits_nepali.utils.text import text_to_phonemes, phoneme_vocab  # Ensure these are defined

# logger = logging.getLogger(__name__)

# class VITSDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir: str, manifest_file: str):
#         try:
#             self.data_dir = Path(data_dir)
#             self.audio_files = []
#             self.texts = []
#             self.phonemes = []

#             logger.info(f"Loading manifest from {manifest_file}")
#             with open(manifest_file, 'r', encoding='utf-8') as f:
#                 reader = csv.DictReader(f)
#                 required_fields = ['path', 'labels', 'phonemes']
#                 if not all(field in reader.fieldnames for field in required_fields):
#                     raise ValueError(f"CSV {manifest_file} must have {required_fields} columns")

#                 for row in reader:
#                     self.audio_files.append(row['path'])
#                     self.texts.append(row['labels'])
#                     phoneme_str = row['phonemes']
#                     if phoneme_str:
#                         phoneme_list = phoneme_str.split()
#                         phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
#                         self.phonemes.append(phoneme_indices)
#                     else:
#                         logger.warning(f"No phonemes for {row['path']}; using text-to-phonemes")
#                         self.phonemes.append(None)

#             logger.info(f"Loaded {len(self.audio_files)} items from manifest")
#             self.transform = torchaudio.transforms.MelSpectrogram(
#                 sample_rate=16000,  # Match config.yaml
#                 n_mels=80,
#                 n_fft=1024,
#                 hop_length=256,
#                 f_min=0,
#                 f_max=8000
#             )
#         except Exception as e:
#             logger.error(f"Failed to initialize VITSDataset: {str(e)}")
#             raise

#     def __len__(self) -> int:
#         return len(self.audio_files)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         try:
#             audio_path = self.data_dir / self.audio_files[idx]
#             text = self.texts[idx]
#             precomputed_phonemes = self.phonemes[idx]

#             # Load audio and compute mel-spectrogram
#             try:
#                 waveform, sample_rate = torchaudio.load(audio_path)
#                 print(f"Waveform shape for {audio_path}: {waveform.shape}")  # Debug: Should be [1, time]
#                 if sample_rate != 16000:
#                     waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
#                 if waveform.shape[0] > 1:  # Ensure mono
#                     waveform = waveform.mean(dim=0, keepdim=True)
#                     print(f"Converted to mono: {waveform.shape}")  # Debug
#             except Exception as e:
#                 logger.error(f"Failed to load audio {audio_path}: {str(e)}")
#                 return (
#                     torch.zeros(1, dtype=torch.long),  # Dummy phonemes
#                     torch.zeros(1, 80),               # Dummy mel
#                     torch.tensor(0, dtype=torch.long),  # Phoneme length
#                     torch.tensor(0, dtype=torch.long)   # Mel length
#                 )

#             mel = self.transform(waveform)  # Shape: [1, n_mels, time]
#             print(f"Mel shape before squeeze for {audio_path}: {mel.shape}")  # Debug: Should be [1, 80, time]
#             mel = mel.squeeze(0).transpose(0, 1)  # Shape: [time, n_mels]
#             print(f"Mel shape after squeeze for {audio_path}: {mel.shape}")  # Debug: Should be [time, 80]
#             if mel.dim() != 2 or mel.size(1) != 80:
#                 logger.error(f"Unexpected mel shape for {audio_path}: {mel.shape}")
#                 raise ValueError(f"Mel-spectrogram has unexpected shape: {mel.shape}")

#             # Convert phonemes to tensor
#             if precomputed_phonemes is not None:
#                 phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
#             else:
#                 phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

#             return {
#                 'phonemes': phonemes,
#                 'mels': mel,
#                 'phoneme_lengths': torch.tensor(len(phonemes), dtype=torch.long),
#                 'mel_lengths': torch.tensor(mel.size(0), dtype=torch.long)
#             }
#         except Exception as e:
#             logger.error(f"Failed to process item {idx}: {str(e)}")
#             raise

# def custom_collate_fn(batch):
#     try:
#         phonemes = [item['phonemes'] for item in batch]
#         mels = [item['mels'] for item in batch]
#         phoneme_lengths = torch.tensor([item['phoneme_lengths'] for item in batch], dtype=torch.long)
#         mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], dtype=torch.long)

#         # Pad phonemes (1D tensors) to the longest sequence
#         phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)

#         # Pad mel-spectrograms (2D tensors) to the longest time dimension
#         mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)
#         print(f"mels_padded shape: {mels_padded.shape}")  # Debug: Should be [batch_size, max_mel_frames, 80]

#         return {
#             'phonemes': phonemes_padded,      # Shape: (batch_size, max_phoneme_len)
#             'mels': mels_padded,              # Shape: (batch_size, max_mel_frames, n_mels)
#             'phoneme_lengths': phoneme_lengths,  # Shape: (batch_size,)
#             'mel_lengths': mel_lengths           # Shape: (batch_size,)
#         }
#     except Exception as e:
#         logger.error(f"Failed in custom_collate_fn: {str(e)}")
#         raise

# def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
#     try:
#         return DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=0,        # Windows compatibility
#             pin_memory=False,     # No GPU
#             collate_fn=custom_collate_fn
#         )
#     except Exception as e:
#         logger.error(f"Failed to create DataLoader: {str(e)}")
#         raise

import logging
import torch
import torchaudio
import csv
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional
from vits_nepali.utils.text import text_to_phonemes, phoneme_vocab

logger = logging.getLogger(__name__)

class VITSDataset(Dataset):
    def __init__(self, data_dir: str, manifest_file: str, max_mel_length: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.max_mel_length = max_mel_length
        self.audio_files = []
        self.texts = []
        self.phonemes = []

        try:
            logger.info(f"Loading manifest from {manifest_file}")
            with open(manifest_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                required_fields = ['path', 'labels', 'phonemes']
                if not all(field in reader.fieldnames for field in required_fields):
                    raise ValueError(f"CSV {manifest_file} must have {required_fields} columns")

                for row in reader:
                    self.audio_files.append(row['path'])
                    self.texts.append(row['labels'])
                    phoneme_str = row['phonemes']
                    if phoneme_str:
                        phoneme_list = phoneme_str.split()
                        phoneme_indices = [phoneme_vocab.get(p, 0) for p in phoneme_list]
                        self.phonemes.append(phoneme_indices)
                    else:
                        logger.warning(f"No phonemes for {row['path']}; using text-to-phonemes")
                        self.phonemes.append(None)

            logger.info(f"Loaded {len(self.audio_files)} items from manifest")
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=80,
                n_fft=1024,
                hop_length=256,
                f_min=0,
                f_max=8000
            )
        except Exception as e:
            logger.error(f"Failed to initialize VITSDataset: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict:
        audio_path = self.data_dir / self.audio_files[idx]
        text = self.texts[idx]
        precomputed_phonemes = self.phonemes[idx]

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.info(f"Waveform shape for {audio_path}: {waveform.shape}")
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                logger.info(f"Converted to mono: {waveform.shape}")

            mel = self.transform(waveform)  # [1, n_mels, time]
            mel = mel.squeeze(0).transpose(0, 1)  # [time, n_mels]
            logger.info(f"Mel shape for {audio_path}: {mel.shape}")
            if mel.dim() != 2 or mel.size(1) != 80:
                raise ValueError(f"Unexpected mel shape: {mel.shape}")
            if self.max_mel_length is not None and mel.shape[0] > self.max_mel_length:
                mel = mel[:self.max_mel_length, :]

            if precomputed_phonemes is not None:
                phonemes = torch.tensor(precomputed_phonemes, dtype=torch.long)
            else:
                phonemes = torch.tensor(text_to_phonemes(text), dtype=torch.long)

            return {
                'phonemes': phonemes,
                'mels': mel,
                'phoneme_lengths': torch.tensor(len(phonemes), dtype=torch.long),
                'mel_lengths': torch.tensor(mel.size(0), dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {str(e)}")
            raise  # Re-raise to debug; replace with `return None` after testing

def custom_collate_fn(batch: List[Optional[Dict]]) -> Dict:
    try:
        # Filter out None or invalid items
        batch = [item for item in batch if item is not None and isinstance(item, dict)]
        if not batch:
            raise ValueError("Empty batch after filtering")

        phonemes = [item['phonemes'] for item in batch]
        mels = [item['mels'] for item in batch]
        phoneme_lengths = torch.tensor([item['phoneme_lengths'] for item in batch], dtype=torch.long)
        mel_lengths = torch.tensor([item['mel_lengths'] for item in batch], dtype=torch.long)

        phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)
        mels_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)
        logger.info(f"mels_padded shape: {mels_padded.shape}")

        return {
            'phonemes': phonemes_padded,
            'mels': mels_padded,
            'phoneme_lengths': phoneme_lengths,
            'mel_lengths': mel_lengths
        }
    except Exception as e:
        logger.error(f"Failed in custom_collate_fn: {str(e)}")
        raise

def get_dataloader(dataset: VITSDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    try:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Windows compatibility
            pin_memory=True,  # GPU optimization
            collate_fn=custom_collate_fn
        )
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {str(e)}")
        raise