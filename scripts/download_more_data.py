import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
DEST_DIR = Path("data/archive/test")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Optional: Read token from env if repository is gated
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")

print(f"Connecting to Hugging Face repository: {REPO_ID}...")
try:
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=hf_token)
    
    # Filter for HDF5 splits (e.g., test or train splits)
    h5_files = [f for f in all_files if f.endswith(".h5")]
    print(f"Discovered {len(h5_files)} HDF5 files in repository.")

    for remote_file in h5_files:
        filename = Path(remote_file).name
        target_path = DEST_DIR / filename
        
        if target_path.exists():
            print(f"• Skipping {filename} (already exists).")
            continue
            
        print(f"• Downloading {filename}...")
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_file,
            repo_type="dataset",
            local_dir="data/download_temp",
            token=hf_token
        )
        
        # Move directly into target directory
        Path(downloaded_path).replace(target_path)
        print(f"  ✓ Saved to {target_path}")

    print("\nDownload complete. New splits are available in data/archive/test/")

except Exception as e:
    print(f"\nError accessing repository: {e}")
    print("If this repository requires authentication, export your Hugging Face token:")
    print("  export HF_TOKEN='hf_your_token_here'")