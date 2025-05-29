# inference.py
import torch
import os
from pipeline.prediction_pipeline import PredictionPipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"Current working directory: {os.getcwd()}")
        pipeline = PredictionPipeline(
            "/teamspace/studios/this_studio/old/vits_nepali/configs/config.yaml",
            "/teamspace/studios/this_studio/old/checkpoints/epoch_50.pt",
            device=device
        )
        output_path = "/teamspace/studios/this_studio/old/prediction/output1.wav"
        logger.info(f"Output audio will be saved to: {os.path.abspath(output_path)}")
        pipeline.predict("नेपालमा पाइने एक किसिमको झार हो", output_path)
    except Exception as e:
        logger.error(f"Inference script failed: {str(e)}")
        raise