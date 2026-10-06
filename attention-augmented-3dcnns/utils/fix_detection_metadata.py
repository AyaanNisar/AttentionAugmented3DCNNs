import os
import pandas as pd
from glob import glob

def fix_detection_metadata(meta_csv, patches_dir, output_csv):
    """Fix detection metadata by only including entries for files that actually exist."""
    
    # Read the original metadata
    print(f"Reading metadata from {meta_csv}...")
    meta_df = pd.read_csv(meta_csv)
    print(f"Original metadata has {len(meta_df)} entries")
    
    # Get list of actual files
    print(f"Checking for actual files in {patches_dir}...")
    actual_files = set()
    for file_path in glob(os.path.join(patches_dir, '*.npy')):
        filename = os.path.basename(file_path)
        actual_files.add(filename)
    
    print(f"Found {len(actual_files)} actual patch files")
    
    # Filter metadata to only include existing files
    filtered_df = meta_df[meta_df['filename'].isin(actual_files)]
    print(f"Filtered metadata has {len(filtered_df)} entries")
    
    # Save the fixed metadata
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved fixed metadata to {output_csv}")
    
    # Print some statistics
    print(f"\nLabel distribution:")
    print(filtered_df['label'].value_counts().sort_index())
    
    return filtered_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_csv', type=str, default='data/patches_detection_meta.csv')
    parser.add_argument('--patches_dir', type=str, default='data/patches_detection')
    parser.add_argument('--output_csv', type=str, default='data/patches_detection_meta_fixed.csv')
    args = parser.parse_args()
    
    fix_detection_metadata(args.meta_csv, args.patches_dir, args.output_csv) 