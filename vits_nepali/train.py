# # Placeholder file
# # train.py
# from pipeline.training_pipeline import TrainingPipeline
# import logging

# logger = logging.getLogger(__name__)

# if __name__ == "__main__":
#     try:
#         pipeline = TrainingPipeline("vits_nepali/configs/config.yaml", manifest_file="vits_nepali/data/csv/train.csv")
#         pipeline.run()
#         test_loss = pipeline.evaluate("data/csv/test.csv")
#         logger.info(f"Test Loss: {test_loss}")
#     except Exception as e:
#         logger.error(f"Training script failed: {str(e)}")
#         raise
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from vits_nepali.pipeline.training_pipeline import TrainingPipeline
# import logging

# # Configure logging for console and file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('/teamspace/studios/this_studio/old/logs/train.log', mode='a')
#     ]
# )
# logger = logging.getLogger(__name__)

# if __name__ == "__main__":
#     print("Starting VITS Nepali training script...")
#     try:
#         print("Initializing TrainingPipeline...")
#         pipeline = TrainingPipeline("/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml", manifest_file="/teamspace/studios/this_studio/old/vits_nepali/data/csv/train_phonemes.csv")
#         logger.info("TrainingPipeline initialized successfully.")
#         print("Starting training...")
#         pipeline.run()
#         logger.info("Training completed. Starting evaluation...")
#         print("Running evaluation...")
#         test_loss = pipeline.evaluate("/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv")
#         logger.info(f"Test Loss: {test_loss}")
#         print(f"Evaluation completed. Test Loss: {test_loss}")
#     except Exception as e:
#         logger.error(f"Training script failed: {str(e)}")
#         print(f"Error in training script: {str(e)}")
#         raise   


###########################
# import sys
# import os
# import gc
# import torch
# import logging

# # Add project root to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from vits_nepali.pipeline.training_pipeline import TrainingPipeline

# # Configure logging for console and file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('/teamspace/studios/this_studio/old/logs/train.log', mode='a')
#     ]
# )
# logger = logging.getLogger(__name__)

# def clear_cuda_cache():
#     """Free up unused GPU memory."""
#     gc.collect()
#     torch.cuda.empty_cache()
#     print("🧹 CUDA memory cleared.")

# if __name__ == "__main__":
#     print("🚀 Starting VITS Nepali training script...")
#     try:
#         print("🔧 Initializing TrainingPipeline...")
#         pipeline = TrainingPipeline(
#             "/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml",
#             manifest_file="/teamspace/studios/this_studio/old/vits_nepali/data/csv/train_phonemes.csv"
#         )
#         logger.info("✅ TrainingPipeline initialized successfully.")

#         print("🏋️ Starting training...")
#         pipeline.run()
#         logger.info("✅ Training completed.")

#         clear_cuda_cache()  # Clean memory after training

#         print("🧪 Running evaluation...")
#         test_loss = pipeline.evaluate("/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv")
#         logger.info(f"📉 Test Loss: {test_loss}")
#         print(f"✅ Evaluation completed. Test Loss: {test_loss}")

#         clear_cuda_cache()  # Clean memory after evaluation

#     except Exception as e:
#         logger.error(f"❌ Training script failed: {str(e)}")
#         print(f"❌ Error in training script: {str(e)}")
#         clear_cuda_cache()  # Clean memory on failure
#         raise


# import sys
# import os
# import gc
# import torch
# import logging

# # Add project root to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from vits_nepali.pipeline.training_pipeline import TrainingPipeline

# # Configure logging for console and file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('/teamspace/studios/this_studio/old/logs/train.log', mode='a')
#     ]
# )
# logger = logging.getLogger(__name__)

# def clear_cuda_cache():
#     """Free up unused GPU memory."""
#     gc.collect()
#     torch.cuda.empty_cache()
#     print("🧹 CUDA memory cleared.")

# if __name__ == "__main__":
#     print("🚀 Starting VITS Nepali training script...")
#     try:
#         print("🔧 Initializing TrainingPipeline...")
#         # Add checkpoint_path to resume from a specific checkpoint, set to None if starting fresh
#         pipeline = TrainingPipeline(
#             config_path="/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml",
#             manifest_file="/teamspace/studios/this_studio/old/vits_nepali/data/csv/train_phonemes.csv",
#             checkpoint_path="/teamspace/studios/this_studio/old/checkpoints/epoch_50.pt"  # Replace with path to checkpoint, e.g., "/teamspace/studios/this_studio/old/vits_nepali/checkpoints/epoch_10.pt"
#         )
#         logger.info("✅ TrainingPipeline initialized successfully.")

#         print("🏋️ Starting training...")
#         pipeline.run()
#         logger.info("✅ Training completed.")

#         clear_cuda_cache()  # Clean memory after training

#         print("🧪 Running evaluation...")
#         test_loss = pipeline.evaluate("/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv")
#         logger.info(f"📉 Test Loss: {test_loss}")
#         print(f"✅ Evaluation completed. Test Loss: {test_loss}")

#         clear_cuda_cache()  # Clean memory after evaluation

#     except Exception as e:
#         logger.error(f"❌ Training script failed: {str(e)}")
#         print(f"❌ Error in training script: {str(e)}")
#         clear_cuda_cache()  # Clean memory on failure
#         raise

import sys
import os
import gc
import torch
import logging

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vits_nepali.pipeline.training_pipeline import TrainingPipeline

# Configure logging for console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/teamspace/studios/this_studio/old/logs/train.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def clear_cuda_cache():
    """Free up unused GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("🧹 CUDA memory cleared.")

if __name__ == "__main__":
    logger.info("🚀 Starting VITS Nepali training script...")
    try:
        logger.info("🔧 Initializing TrainingPipeline...")
        # Add checkpoint_path to resume from a specific checkpoint, set to None if starting fresh
        pipeline = TrainingPipeline(
            config_path="/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml",
            manifest_file="/teamspace/studios/this_studio/old/vits_nepali/data/csv/train_phonemes.csv",
            checkpoint_path="/teamspace/studios/this_studio/old/checkpoints/epoch_40.pt"  # Replace with path to checkpoint, e.g., "/teamspace/studios/this_studio/old/checkpoints/epoch_50.pt"
        )
        logger.info("✅ TrainingPipeline initialized successfully.")

        logger.info("🏋️ Starting training...")
        pipeline.run()
        logger.info("✅ Training completed.")

        clear_cuda_cache()  # Clean memory after training

        logger.info("🧪 Running evaluation...")
        test_loss = pipeline.evaluate("/teamspace/studios/this_studio/old/vits_nepali/data/csv/test_phonemes.csv")
        logger.info(f"📉 Test Loss: {test_loss}")
        logger.info(f"✅ Evaluation completed. Test Loss: {test_loss}")

        clear_cuda_cache()  # Clean memory after evaluation

    except Exception as e:
        logger.error(f"❌ Training script failed: {str(e)}")
        clear_cuda_cache()  # Clean memory on failure
        raise