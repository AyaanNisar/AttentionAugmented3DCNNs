import os
import numpy as np
import pandas as pd
from glob import glob
from utils.load_scan import load_scan
from tqdm import tqdm
import argparse

PATCH_SIZE = 32

def world_to_voxel(world_coord, origin, spacing):
    return np.round((np.array(world_coord) - origin) / spacing).astype(int)

def extract_patch(volume, center, size=PATCH_SIZE):
    z, y, x = center
    half = size // 2
    z_min, z_max = max(z - half, 0), min(z + half, volume.shape[0])
    y_min, y_max = max(y - half, 0), min(y + half, volume.shape[1])
    x_min, x_max = max(x - half, 0), min(x + half, volume.shape[2])
    patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    pad_width = [
        (0, size - patch.shape[0]),
        (0, size - patch.shape[1]),
        (0, size - patch.shape[2])
    ]
    pad_width = [(max(0, p[0]), max(0, p[1])) for p in pad_width]
    patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
    return patch

def dedup_entries(entries, tol=2):
    # Deduplicate by rounded voxel coordinates (tol=2 voxels)
    seen = set()
    deduped = []
    for e in entries:
        key = tuple(np.round(e['center_voxel'] / tol).astype(int)) + (e['seriesuid'],)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped

def extract_detection_patches(annotations_path, candidates_path, scans_dirs, out_dir, meta_out):
    # Clear out_dir if it exists
    if os.path.exists(out_dir):
        print(f"Clearing output directory: {out_dir}")
        for f in os.listdir(out_dir):
            if f.endswith('.npy'):
                os.remove(os.path.join(out_dir, f))
    else:
        os.makedirs(out_dir, exist_ok=True)
    # Remove old metadata file if it exists
    if os.path.exists(meta_out):
        print(f"Removing old metadata file: {meta_out}")
        os.remove(meta_out)
    
    all_entries = []
    scans_dirs = [d.strip() for d in scans_dirs]
    print("Scan directories to process:", scans_dirs)
    
    # Load all scan files from all subsets (20% for detection fine-tuning)
    scan_files = {}
    total_scans = 0
    for scans_dir in scans_dirs:
        mhd_files = glob(os.path.join(scans_dir, '*.mhd'))
        print(f"  {scans_dir}: {len(mhd_files)} total scans")
        
        # Use only the remaining 40% of each subset for detection
        detection_count = len(mhd_files) - int(len(mhd_files) * 0.6)  # Remaining 40%
        mhd_files = mhd_files[int(len(mhd_files) * 0.6):]  # Take last 40%
        print(f"    Using {len(mhd_files)} scans for detection (40%)")
        
        total_scans += len(mhd_files)
        for scan_path in mhd_files:
            seriesuid = os.path.basename(scan_path).replace('.mhd', '')
            scan_files[seriesuid] = scan_path
    print(f"Found {total_scans} scans in total for detection.")
    
    # Load annotations and candidates
    ann = pd.read_csv(annotations_path)
    cand = pd.read_csv(candidates_path) if candidates_path else None
    
    # 1. Extract all positives from annotations.csv across all subsets
    print(f"Extracting {len(ann)} positive (nodule) patches...")
    n_pos_actual = 0
    skipped_positives = 0
    
    for i, row in tqdm(ann.iterrows(), total=len(ann), desc="Nodule patches"):
        seriesuid = row['seriesuid']
        if seriesuid not in scan_files:
            print(f"Warning: Scan {seriesuid} not found for positive at index {i}.")
            skipped_positives += 1
            continue
            
        # Find which subset contains this scan
        scan_path = scan_files[seriesuid]
        volume, origin, spacing = load_scan(scan_path)
        center_world = [row['coordZ'], row['coordY'], row['coordX']]
        center_voxel = world_to_voxel(center_world, origin, spacing)
        patch = extract_patch(volume, center_voxel)
        fname = f'{seriesuid}_ann_{i}.npy'
        np.save(os.path.join(out_dir, fname), patch)
        all_entries.append({'filename': fname, 'label': 1, 'seriesuid': seriesuid, 'center_voxel': center_voxel})
        n_pos_actual += 1
    
    print(f"Successfully extracted {n_pos_actual} positive patches (skipped {skipped_positives})")
    
    # 2. Extract balanced negatives from candidates.csv
    if cand is not None:
        neg_cand = cand[cand['class'] == 0]
        n_neg_target = n_pos_actual * 6  # 15/85 split based on actual positives for better learning
        print(f"Sampling {n_neg_target} negatives from {len(neg_cand)} available candidates...")
        
        neg_sample = neg_cand.sample(n=min(n_neg_target, len(neg_cand)), random_state=42) if n_neg_target < len(neg_cand) else neg_cand
        n_neg_actual = 0
        skipped_negatives = 0
        
        for i, row in tqdm(neg_sample.iterrows(), total=len(neg_sample), desc="Negative patches"):
            seriesuid = row['seriesuid']
            if seriesuid not in scan_files:
                print(f"Warning: Scan {seriesuid} not found for negative at index {i}.")
                skipped_negatives += 1
                continue
                
            scan_path = scan_files[seriesuid]
            volume, origin, spacing = load_scan(scan_path)
            center_world = [row['coordZ'], row['coordY'], row['coordX']]
            center_voxel = world_to_voxel(center_world, origin, spacing)
            patch = extract_patch(volume, center_voxel)
            fname = f'{seriesuid}_cand_{i}.npy'
            np.save(os.path.join(out_dir, fname), patch)
            all_entries.append({'filename': fname, 'label': int(row['class']), 'seriesuid': seriesuid, 'center_voxel': center_voxel})
            n_neg_actual += 1
        
        print(f"Successfully extracted {n_neg_actual} negative patches (skipped {skipped_negatives})")
    
    # Deduplicate
    print("Deduplicating entries...")
    all_entries = dedup_entries(all_entries)
    
    # Save metadata
    meta_df = pd.DataFrame([{k: v for k, v in e.items() if k != 'center_voxel'} for e in all_entries])
    meta_df.to_csv(meta_out, index=False)
    print(f"Final dataset: {len(all_entries)} unique detection patches")
    print(f"Saved metadata to {meta_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, default='data/annotations.csv')
    parser.add_argument('--candidates', type=str, default='data/candidates.csv')
    parser.add_argument('--scans_dirs', type=str, nargs='+', default=['data/subset0', 'data/subset1', 'data/subset2', 'data/subset3', 'data/subset4', 'data/subset5', 'data/subset6', 'data/subset7', 'data/subset8', 'data/subset9'])
    parser.add_argument('--out_dir', type=str, default='data/patches_detection')
    parser.add_argument('--meta_out', type=str, default='data/patches_detection_meta.csv')
    args = parser.parse_args()
    print("Scan directories to process:", args.scans_dirs)
    extract_detection_patches(args.annotations, args.candidates, args.scans_dirs, args.out_dir, args.meta_out) 