import pandas as pd
import os

def select_random_rows_and_clean_audio(csv_path, audio_folder, n, path_column="path", audio_ext=".wav"):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Ensure n is valid
    if n > len(df):
        print(f"CSV has only {len(df)} rows; can't select {n}.")
        return

    # Sample n rows
    df_sampled = df.sample(n, random_state=42)

    # Create set of audio filenames to keep
    keep_files = set(df_sampled[path_column].astype(str) + audio_ext)

    # Delete audio files not in keep list
    deleted = 0
    for fname in os.listdir(audio_folder):
        if fname.endswith(audio_ext) and fname not in keep_files:
            os.remove(os.path.join(audio_folder, fname))
            deleted += 1

    # Save the sampled CSV
    df_sampled.to_csv(csv_path, index=False)

    print(f"✅ Kept {n} entries in CSV.")
    print(f"🗑️ Deleted {deleted} audio files not in selection.")

# Example usage:
select_random_rows_and_clean_audio("vits_nepali/data/csv/manifest.csv", "vits_nepali/data/dataset/audio", 30)
