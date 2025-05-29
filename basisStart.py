import os
import sys

# Ensure vits_nepali is importable
sys.path.append("/teamspace/studios/this_studio/old")
from vits_nepali.data.preprocess import split_manifest

def main():
    # Get the absolute path of the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to manifest.csv
    manifest_path = os.path.join(script_dir, "vits_nepali", "data", "csv", "manifest.csv")
    
    try:
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
        
        # Call the split_manifest function
        split_manifest(manifest_path, train_ratio=0.8, val_ratio=0.1)
        print("Successfully split the manifest and moved audio files.")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
