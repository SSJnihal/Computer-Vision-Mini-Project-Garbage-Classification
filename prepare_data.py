import os
import shutil
import random
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor

def download_single_file(api_args):
    repo_id, filename, local_dir = api_args
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def main():
    print("Preparing dataset folders...")
    output_dir = 'Garbage_Augmented'
    os.makedirs(output_dir, exist_ok=True)
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for c in classes:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

    # 1. Copy original local images
    print("Copying existing local original images...")
    local_dir = os.path.join('Garbage', 'original_images')
    for c in classes:
        local_class_dir = os.path.join(local_dir, c)
        if os.path.isdir(local_class_dir):
            for filename in os.listdir(local_class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                     src = os.path.join(local_class_dir, filename)
                     dst = os.path.join(output_dir, c, f"orig_{filename}")
                     shutil.copy2(src, dst)
    print("Local images copied successfully.")

    # 2. Get file list from Hugging Face
    repo_id = "omasteam/waste-garbage-management-dataset"
    print(f"Fetching repository file listing from Hugging Face ({repo_id})...")
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"Total files in remote repository: {len(all_files)}")

    # Map class to Hugging Face folder name
    CLASS_MAPPING = {
        'cardboard': 'cardboard',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'paper',
        'plastic': 'plastic',
        'trash': 'trash'
    }

    # Filter target files to download (limit 350 per class)
    LIMIT = 350
    download_list = []
    
    for c in classes:
        hf_folder = CLASS_MAPPING[c]
        # Find all image files starting with this folder name
        class_files = [
            f for f in all_files 
            if f.startswith(f"{hf_folder}/") and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        class_files = sorted(class_files)
        # Select first LIMIT files
        selected_files = class_files[:LIMIT]
        for f in selected_files:
            download_list.append((repo_id, f, "omasteam_dataset"))
        print(f"Queued {len(selected_files)} files to download for class '{c}'")

    # 3. Download in parallel using ThreadPoolExecutor
    print(f"\nDownloading {len(download_list)} files in parallel (using 32 threads)...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(download_single_file, download_list)
        for r in results:
            if r:
                success_count += 1
                if success_count % 100 == 0:
                    print(f"Downloaded {success_count}/{len(download_list)} files...")

    print(f"Finished download. Successfully downloaded {success_count}/{len(download_list)} files.")

    # 4. Copy downloaded HF images to Garbage_Augmented
    print("Copying downloaded HF images to Garbage_Augmented folders...")
    for c in classes:
        hf_folder = CLASS_MAPPING[c]
        hf_class_dir = os.path.join('omasteam_dataset', hf_folder)
        if os.path.isdir(hf_class_dir):
            files = [f for f in os.listdir(hf_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for filename in files:
                src = os.path.join(hf_class_dir, filename)
                dst = os.path.join(output_dir, c, f"hf_{filename}")
                shutil.copy2(src, dst)
            print(f"- Copied {len(files)} HF images for '{c}'")

    # 5. Balance the classes
    print("\nEvaluating final class sizes:")
    final_counts = {}
    for c in classes:
        class_path = os.path.join(output_dir, c)
        files = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        final_counts[c] = len(files)
        print(f"- {c}: {len(files)} images")
        
    min_size = min(final_counts.values())
    print(f"\nMinimum class size is {min_size}. Balancing all classes to match this count...")
    
    for c in classes:
        class_path = os.path.join(output_dir, c)
        files = sorted([f for f in os.listdir(class_path) if f.lower().endswith('.jpg')])
        if len(files) > min_size:
            # Randomly select samples to remove to enforce balance
            excess_count = len(files) - min_size
            remove_files = random.sample(files, excess_count)
            for f in remove_files:
                os.remove(os.path.join(class_path, f))
            print(f"- Balanced class '{c}' down to {min_size} images.")
            
    print("\nDataset preparation and class balancing complete! Folder structure is ready at 'Garbage_Augmented/'")

if __name__ == '__main__':
    main()
