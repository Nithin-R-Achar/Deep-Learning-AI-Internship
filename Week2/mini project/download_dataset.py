"""
download_dataset.py
Automated script to fetch and prepare an open dataset of plant leaf images
from the PlantVillage benchmark dataset (https://github.com/spMohanty/PlantVillage-Dataset).

Downloads balanced train/val sets and extracts curated sample test images for the web app UI.
"""

import os
import sys
import time
import json
import shutil
import random
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

REPO_API_URL = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
RAW_BASE_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"

def fetch_class_file_list(class_name, retries=3):
    """Fetch list of image filenames for a class from GitHub API with retry logic."""
    url = f"{REPO_API_URL}/{urllib.parse.quote(class_name)}"
    headers = {"User-Agent": "PlantDiseaseDetector/1.0"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode("utf-8")
                items = json.loads(content)
                images = [item["name"] for item in items if item["type"] == "file" and item["name"].lower().endswith(('.jpg', '.jpeg', '.png'))]
                return class_name, images
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"[-] Error fetching file list for {class_name}: {e}")
            return class_name, []

def download_single_image(url, target_path):
    """Download single image and verify it is not corrupt."""
    headers = {"User-Agent": "PlantDiseaseDetector/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            content = response.read()
        
        # Save temp then verify with PIL
        with open(target_path, "wb") as f:
            f.write(content)
        
        # Verify image integrity
        with Image.open(target_path) as img:
            img.verify()
        return True
    except Exception as e:
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        return False

def prepare_dataset(classes, samples_per_class=40, val_ratio=0.2, data_dir="dataset", samples_dir="static/samples"):
    """Download and split dataset into train, val, and static/samples."""
    print("=" * 65)
    print("  PLANT DISEASE DATASET DOWNLOADER (PlantVillage Open Dataset)")
    print("=" * 65)
    print(f"Target classes count: {len(classes)}")
    print(f"Images per class:     {samples_per_class}")
    print(f"Validation ratio:     {val_ratio:.0%}")
    print(f"Destination root:     {os.path.abspath(data_dir)}")
    print("=" * 65)

    os.makedirs(data_dir, exist_ok=True)
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print("\n[1/3] Querying dataset repository file index...")
    class_files = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_class_file_list, c): c for c in classes}
        for future in as_completed(futures):
            c_name, file_list = future.result()
            if file_list:
                class_files[c_name] = file_list
                print(f"  [+] Found {len(file_list):>4} images for {c_name}")
            else:
                print(f"  [!] Failed or zero images for {c_name}")

    if not class_files:
        print("[-] Could not retrieve any class file listings. Check network connection.")
        sys.exit(1)

    print("\n[2/3] Scheduling parallel image downloads...")
    tasks = []
    sample_images_selected = {}

    for c_name, file_list in class_files.items():
        # Shuffle for random sampling
        shuffled = list(file_list)
        random.seed(42)
        random.shuffle(shuffled)

        selected = shuffled[:samples_per_class]
        val_count = max(1, int(len(selected) * val_ratio))
        val_items = selected[:val_count]
        train_items = selected[val_count:]

        # Select 1 nice sample image for the web demo gallery
        if len(selected) > 0:
            sample_images_selected[c_name] = selected[0]

        c_train_dir = os.path.join(train_root, c_name)
        c_val_dir = os.path.join(val_root, c_name)
        os.makedirs(c_train_dir, exist_ok=True)
        os.makedirs(c_val_dir, exist_ok=True)

        for img_name in train_items:
            dest = os.path.join(c_train_dir, img_name)
            if not os.path.exists(dest):
                url = f"{RAW_BASE_URL}/{urllib.parse.quote(c_name)}/{urllib.parse.quote(img_name)}"
                tasks.append((url, dest))

        for img_name in val_items:
            dest = os.path.join(c_val_dir, img_name)
            if not os.path.exists(dest):
                url = f"{RAW_BASE_URL}/{urllib.parse.quote(c_name)}/{urllib.parse.quote(img_name)}"
                tasks.append((url, dest))

    print(f"  Total images to download: {len(tasks)}")

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_image, url, path): path for url, path in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                downloaded += 1
            else:
                failed += 1
            if (downloaded + failed) % 25 == 0 or (downloaded + failed) == len(tasks):
                print(f"  -> Progress: {downloaded + failed}/{len(tasks)} (Success: {downloaded}, Errors: {failed})")

    print("\n[3/3] Setting up sample demo images for web UI...")
    # Clean label mapping for sample filenames
    sample_manifest = []
    for c_name, sample_img in sample_images_selected.items():
        clean_name = c_name.replace("___", "_").replace("_(maize)", "").replace(" ", "_").lower()
        sample_dest = os.path.join(samples_dir, f"{clean_name}.jpg")
        
        # Check if already downloaded in train or val
        src_train = os.path.join(train_root, c_name, sample_img)
        src_val = os.path.join(val_root, c_name, sample_img)
        
        if os.path.exists(src_train):
            shutil.copy2(src_train, sample_dest)
        elif os.path.exists(src_val):
            shutil.copy2(src_val, sample_dest)
        else:
            url = f"{RAW_BASE_URL}/{urllib.parse.quote(c_name)}/{urllib.parse.quote(sample_img)}"
            download_single_image(url, sample_dest)

        if os.path.exists(sample_dest):
            sample_manifest.append({
                "filename": f"{clean_name}.jpg",
                "class_name": c_name,
                "label": c_name.replace("___", " - ").replace("_", " ")
            })

    # Save sample manifest json
    with open(os.path.join(samples_dir, "samples.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("  DATASET DOWNLOAD & PREPARATION COMPLETE!")
    print(f"  Train directory:   {train_root}")
    print(f"  Val directory:     {val_root}")
    print(f"  Demo samples:      {samples_dir} ({len(sample_manifest)} sample images)")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PlantVillage leaf disease dataset")
    parser.add_argument("--samples-per-class", type=int, default=40, help="Images per class (default: 40)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Target dataset directory")
    parser.add_argument("--samples-dir", type=str, default="static/samples", help="UI demo samples directory")
    args = parser.parse_args()

    prepare_dataset(
        classes=DEFAULT_CLASSES,
        samples_per_class=args.samples_per_class,
        val_ratio=args.val_ratio,
        data_dir=args.data_dir,
        samples_dir=args.samples_dir
    )
