import os
import numpy as np
import pandas as pd
from glob import glob
from utils.load_scan import load_scan
from tqdm import tqdm
import random

PATCH_SIZE = 32
PATCH_SHAPE = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)


def world_to_voxel(world_coord, origin, spacing):
    return np.round((np.array(world_coord) - origin) / spacing).astype(int)


def extract_patch(volume, center, size=PATCH_SIZE):
    """
    Extract a patch of specified size centered at the given voxel coordinates.
    Returns None if the patch cannot be extracted to the exact target size.
    """
    z, y, x = center
    half = size // 2
    
    # Calculate patch boundaries
    z_min, z_max = z - half, z + half
    y_min, y_max = y - half, y + half
    x_min, x_max = x - half, x + half
    
    # Check if patch would be completely outside volume bounds
    if (z_max < 0 or z_min >= volume.shape[0] or 
        y_max < 0 or y_min >= volume.shape[1] or 
        x_max < 0 or x_min >= volume.shape[2]):
        return None
    
    # Clamp boundaries to volume dimensions
    z_min = max(0, z_min)
    z_max = min(volume.shape[0], z_max)
    y_min = max(0, y_min)
    y_max = min(volume.shape[1], y_max)
    x_min = max(0, x_min)
    x_max = min(volume.shape[2], x_max)
    
    # Extract the patch
    patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Pad to target size (allow any patch that can be padded)
    pad_width = [
        (0, size - patch.shape[0]),
        (0, size - patch.shape[1]),
        (0, size - patch.shape[2])
    ]
    
    # Ensure all padding values are non-negative
    pad_width = [(max(0, p[0]), max(0, p[1])) for p in pad_width]
    patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
    
    # Final validation: ensure patch is exactly the target size
    if patch.shape != (size, size, size):
        return None
    
    return patch


def extract_balanced_ssl_patches(annotations_path, candidates_path, scans_dirs, out_dir, meta_out):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    scans_dirs = [d.strip() for d in scans_dirs]
    print("Scan directories to process:", scans_dirs)
    
    # Load all scan files from all subsets (80% for SSL pretraining)
    scan_files = []
    total_scans = 0
    for scans_dir in scans_dirs:
        mhd_files = glob(os.path.join(scans_dir, '*.mhd'))
        print(f"  {scans_dir}: {len(mhd_files)} total scans")
        
        # Use 60% of each subset for SSL pretraining
        ssl_count = int(len(mhd_files) * 0.6)
        mhd_files = mhd_files[:ssl_count]  # Take first 60%
        print(f"    Using {len(mhd_files)} scans for SSL pretraining (60%)")
        
        total_scans += len(mhd_files)
        scan_files.extend(mhd_files)
    print(f"Found {total_scans} scans in total for SSL pretraining.")
    
    # Extract random patches from each scan (no annotations needed for SSL)
    patch_count = 0
    skipped_scans = 0
    
    for scan_path in tqdm(scan_files, desc="Processing scans"):
        seriesuid = os.path.basename(scan_path).replace('.mhd', '')
        
        try:
            volume, origin, spacing = load_scan(scan_path)
            
            # Extract multiple random patches from this scan
            scan_patches = 0
            attempts = 0
            max_attempts = 10 * 2  # 10 patches per scan, allow some failed attempts
            
            while scan_patches < 10 and attempts < max_attempts:
                # Extract random patch (no annotation coordinates needed)
                patch = extract_random_patch(volume)
                attempts += 1
                
                if patch is not None:
                    fname = f'{seriesuid}_random_{scan_patches}.npy'
                    np.save(os.path.join(out_dir, fname), patch)
                    entries.append({
                        'filename': fname, 
                        'seriesuid': seriesuid,
                        'patch_type': 'random'
                    })
                    scan_patches += 1
                    patch_count += 1
            
            if scan_patches == 0:
                print(f"Warning: Could not extract any patches from {seriesuid}")
                skipped_scans += 1
                
        except Exception as e:
            print(f"Error processing scan {seriesuid}: {e}")
            skipped_scans += 1
            continue
    
    print(f"Successfully extracted {patch_count} random patches from {total_scans - skipped_scans} scans")
    print(f"Skipped {skipped_scans} scans due to errors or insufficient size")
    
    # Save metadata
    meta_df = pd.DataFrame(entries)
    meta_df.to_csv(meta_out, index=False)
    print(f"Saved {len(entries)} SSL patches and metadata to {meta_out}")


def extract_random_patch(volume, size=PATCH_SIZE):
    """
    Extract a random patch from anywhere in the volume.
    Returns None if the volume is too small.
    """
    if (volume.shape[0] < size or volume.shape[1] < size or volume.shape[2] < size):
        return None
    
    # Random starting position
    z_start = random.randint(0, volume.shape[0] - size)
    y_start = random.randint(0, volume.shape[1] - size)
    x_start = random.randint(0, volume.shape[2] - size)
    
    # Extract patch
    patch = volume[z_start:z_start+size, y_start:y_start+size, x_start:x_start+size]
    
    return patch


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, default='data/annotations.csv')
    parser.add_argument('--candidates', type=str, default='data/candidates.csv')
    parser.add_argument('--scans_dirs', type=str, nargs='+', default=['data/subset0', 'data/subset1', 'data/subset2', 'data/subset3', 'data/subset4', 'data/subset5', 'data/subset6', 'data/subset7', 'data/subset8', 'data/subset9'])
    parser.add_argument('--out_dir', type=str, default='data/patches_ssl')
    parser.add_argument('--meta_out', type=str, default='data/patches_ssl_meta.csv')
    args = parser.parse_args()

    ann = pd.read_csv('data/annotations.csv')
    scan_files = set(os.path.basename(f).replace('.mhd', '') for f in glob('data/subset*/*.mhd'))
    used_ann = ann[ann['seriesuid'].isin(scan_files)]
    print(f"Total annotations: {len(ann)}")
    print(f"Annotations with available scans: {len(used_ann)}")
    print("Scans with nodules:", used_ann['seriesuid'].unique())

    extract_balanced_ssl_patches(args.annotations, args.candidates, args.scans_dirs, args.out_dir, args.meta_out) 