# import torch
# import logging

# logger = logging.getLogger(__name__)

# def monotonic_alignment_search(
#     text_embed: torch.Tensor,  # Shape: (batch_size, max_phoneme_len, embed_dim)
#     z_mu: torch.Tensor,       # Shape: (batch_size, embed_dim, max_mel_frames)
#     phoneme_lengths: torch.Tensor,  # Shape: (batch_size,)
#     mel_lengths: torch.Tensor,      # Shape: (batch_size,)
#     sigma: float = 1.0
# ) -> torch.Tensor:
#     """
#     Compute ground-truth durations using Monotonic Alignment Search.
#     Returns durations of shape (batch_size, max_phoneme_len).
#     """
#     try:
#         batch_size, max_phoneme_len, embed_dim = text_embed.size()
#         _, _, max_mel_frames = z_mu.size()
        
#         # Transpose z_mu to (batch_size, max_mel_frames, embed_dim) for similarity
#         z_mu = z_mu.transpose(1, 2)  # (batch_size, max_mel_frames, embed_dim)
        
#         # Initialize log probability matrix
#         log_probs = torch.zeros(batch_size, max_phoneme_len, max_mel_frames, device=text_embed.device)
        
#         # Compute log probabilities based on normal distribution
#         for b in range(batch_size):
#             phoneme_len = phoneme_lengths[b].item()
#             mel_len = mel_lengths[b].item()
#             for i in range(phoneme_len):
#                 # Compute similarity between text_embed[b, i] and z_mu[b, j]
#                 text_vec = text_embed[b, i].unsqueeze(0)  # (1, embed_dim)
#                 mel_vecs = z_mu[b, :mel_len]              # (mel_len, embed_dim)
#                 # Log probability of normal distribution
#                 diff = text_vec - mel_vecs                # (mel_len, embed_dim)
#                 log_prob = -0.5 * (diff ** 2).sum(dim=-1) / (sigma ** 2)  # (mel_len,)
#                 log_probs[b, i, :mel_len] = log_prob
        
#         # Dynamic programming for monotonic alignment
#         durations = torch.zeros(batch_size, max_phoneme_len, device=text_embed.device, dtype=torch.long)
#         for b in range(batch_size):
#             phoneme_len = phoneme_lengths[b].item()
#             mel_len = mel_lengths[b].item()
            
#             # Initialize DP table
#             Q = torch.full((phoneme_len, mel_len), float('-inf'), device=text_embed.device)
#             Q[0, 0] = log_probs[b, 0, 0]
#             for j in range(1, mel_len):
#                 Q[0, j] = Q[0, j-1] + log_probs[b, 0, j]
            
#             # Fill DP table
#             for i in range(1, phoneme_len):
#                 Q[i, 0] = float('-inf')  # No skipping phonemes
#                 for j in range(1, mel_len):
#                     Q[i, j] = torch.max(Q[i-1, j-1], Q[i, j-1]) + log_probs[b, i, j]
            
#             # Backtrack to find durations
#             i, j = phoneme_len - 1, mel_len - 1
#             while i >= 0 and j >= 0:
#                 if i == 0:
#                     durations[b, i] += j + 1
#                     break
#                 if j == 0:
#                     break
#                 if Q[i-1, j-1] >= Q[i, j-1]:
#                     durations[b, i] += 1
#                     i -= 1
#                     j -= 1
#                 else:
#                     j -= 1
        
#         return durations
#     except Exception as e:
#         logger.error(f"Monotonic Alignment Search failed: {str(e)}")
#         raise
import torch
import logging

logger = logging.getLogger(__name__)

def monotonic_alignment_search(
    text_embed: torch.Tensor,  # Shape: (batch_size, max_phoneme_len, embed_dim)
    z_mu: torch.Tensor,       # Shape: (batch_size, embed_dim, max_mel_frames)
    phoneme_lengths: torch.Tensor,  # Shape: (batch_size,)
    mel_lengths: torch.Tensor,      # Shape: (batch_size,)
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute ground-truth durations using vectorized Monotonic Alignment Search.
    Returns durations of shape (batch_size, max_phoneme_len).
    """
    try:
        batch_size, max_phoneme_len, embed_dim = text_embed.size()
        _, _, max_mel_frames = z_mu.size()
        
        # Transpose z_mu to (batch_size, max_mel_frames, embed_dim)
        z_mu = z_mu.transpose(1, 2)  # (batch_size, max_mel_frames, embed_dim)
        
        # Compute log probabilities in a vectorized manner
        # Expand dimensions for broadcasting: text_embed (batch_size, max_phoneme_len, 1, embed_dim)
        text_embed_exp = text_embed.unsqueeze(2)  # (batch_size, max_phoneme_len, 1, embed_dim)
        z_mu_exp = z_mu.unsqueeze(1)  # (batch_size, 1, max_mel_frames, embed_dim)
        diff = text_embed_exp - z_mu_exp  # (batch_size, max_phoneme_len, max_mel_frames, embed_dim)
        log_probs = -0.5 * (diff ** 2).sum(dim=-1) / (sigma ** 2)  # (batch_size, max_phoneme_len, max_mel_frames)
        
        # Mask invalid positions based on lengths
        phoneme_mask = torch.arange(max_phoneme_len, device=text_embed.device)[None, :] >= phoneme_lengths[:, None]
        mel_mask = torch.arange(max_mel_frames, device=text_embed.device)[None, :] >= mel_lengths[:, None]
        mask = phoneme_mask.unsqueeze(2) | mel_mask.unsqueeze(1)  # (batch_size, max_phoneme_len, max_mel_frames)
        log_probs = log_probs.masked_fill(mask, float('-inf'))
        
        # Dynamic programming (still requires loop over batch due to variable lengths)
        durations = torch.zeros(batch_size, max_phoneme_len, device=text_embed.device, dtype=torch.long)
        for b in range(batch_size):
            phoneme_len = phoneme_lengths[b].item()
            mel_len = mel_lengths[b].item()
            if phoneme_len == 0 or mel_len == 0:
                continue
            
            # Initialize DP table
            Q = torch.full((phoneme_len, mel_len), float('-inf'), device=text_embed.device)
            Q[0, 0] = log_probs[b, 0, 0]
            for j in range(1, mel_len):
                Q[0, j] = Q[0, j-1] + log_probs[b, 0, j]
            
            # Fill DP table
            for i in range(1, phoneme_len):
                Q[i, 0] = float('-inf')
                for j in range(1, mel_len):
                    Q[i, j] = torch.max(Q[i-1, j-1], Q[i, j-1]) + log_probs[b, i, j]
            
            # Backtrack
            i, j = phoneme_len - 1, mel_len - 1
            while i >= 0 and j >= 0:
                if i == 0:
                    durations[b, i] += j + 1
                    break
                if j == 0:
                    break
                if Q[i-1, j-1] >= Q[i, j-1]:
                    durations[b, i] += 1
                    i -= 1
                    j -= 1
                else:
                    j -= 1
        
        return durations
    except Exception as e:
        logger.error(f"Monotonic Alignment Search failed: {str(e)}")
        raise