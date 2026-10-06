#!/usr/bin/env python3
"""
LUNA16 CPM (Competition Performance Metric) Calculator

This script calculates the LUNA16 CPM score by:
1. Running the trained model on the entire LUNA16 dataset
2. Generating an FROC curve (sensitivity vs average FPs per scan)
3. Interpolating sensitivity at the 7 specific FP rates
4. Computing the average sensitivity (CPM score)

The 7 FP rates are: 0.125, 0.25, 0.5, 1, 2, 4, 8 FPs per scan
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
import json
from scipy.interpolate import interp1d

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.load_scan import load_scan
from utils.extract_patches import world_to_voxel, extract_patch
from detection.model import NoduleDetector


def load_entire_luna16_dataset(annotations_path, candidates_path, scans_dirs):
    """
    Load the entire LUNA16 dataset for CPM calculation.
    Returns: list of (seriesuid, scan_path, candidates_df, annotations_df)
    """
    print("Loading entire LUNA16 dataset...")
    
    # Load annotations and candidates
    annotations = pd.read_csv(annotations_path)
    candidates = pd.read_csv(candidates_path)
    
    # Group by seriesuid
    annotations_by_series = annotations.groupby('seriesuid')
    candidates_by_series = candidates.groupby('seriesuid')
    
    # Find all scan files
    all_scan_files = {}
    total_scans = 0
    
    for scans_dir in scans_dirs:
        if os.path.exists(scans_dir):
            mhd_files = glob(os.path.join(scans_dir, '*.mhd'))
            print(f"  {scans_dir}: {len(mhd_files)} scans")
            for scan_path in mhd_files:
                seriesuid = os.path.basename(scan_path).replace('.mhd', '')
                all_scan_files[seriesuid] = scan_path
                total_scans += 1
    
    print(f"Found {total_scans} total scans across all subsets")
    
    # Create dataset
    dataset = []
    for seriesuid, scan_path in all_scan_files.items():
        # Get annotations and candidates for this scan
        scan_annotations = annotations_by_series.get_group(seriesuid) if seriesuid in annotations_by_series.groups else pd.DataFrame()
        scan_candidates = candidates_by_series.get_group(seriesuid) if seriesuid in candidates_by_series.groups else pd.DataFrame()
        
        dataset.append((seriesuid, scan_path, scan_candidates, scan_annotations))
    
    print(f"Dataset contains {len(dataset)} scans with annotations/candidates")
    return dataset


def calculate_distance_3d(coord1, coord2):
    """Calculate 3D Euclidean distance between two coordinates."""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)


def evaluate_detections_at_threshold(detections, confidence_threshold, distance_threshold=5.0):
    """
    Evaluate detections at a specific confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        confidence_threshold: Confidence threshold for filtering detections
        distance_threshold: Distance threshold for matching detections to ground truth (mm)
    
    Returns:
        total_tps: Total true positives across all scans
        total_fps: Total false positives across all scans
        total_gt: Total ground truth nodules across all scans
        num_scans: Number of scans
    """
    # Filter detections by confidence threshold
    filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
    
    # Group by seriesuid
    detections_by_scan = {}
    for det in filtered_detections:
        seriesuid = det['seriesuid']
        if seriesuid not in detections_by_scan:
            detections_by_scan[seriesuid] = []
        detections_by_scan[seriesuid].append(det)
    
    total_tps = 0
    total_fps = 0
    total_gt = 0
    
    for seriesuid, scan_detections in detections_by_scan.items():
        # Get ground truth nodules for this scan
        gt_nodules = scan_detections[0]['gt_nodules'] if scan_detections else []
        total_gt += len(gt_nodules)
        
        # Reset matched status
        for gt in gt_nodules:
            gt['matched'] = False
        
        # Check each detection
        for det in scan_detections:
            det_coord = det['coord']
            is_tp = False
            
            # Check if detection matches any ground truth nodule
            for gt in gt_nodules:
                if not gt['matched']:
                    gt_coord = gt['coord']
                    distance = calculate_distance_3d(det_coord, gt_coord)
                    
                    if distance <= distance_threshold:
                        gt['matched'] = True
                        is_tp = True
                        break
            
            if is_tp:
                total_tps += 1
            else:
                total_fps += 1
    
    num_scans = len(set(d['seriesuid'] for d in detections))
    return total_tps, total_fps, total_gt, num_scans


def calculate_froc_curve(detections, confidence_thresholds):
    """
    Calculate FROC curve points.
    
    Args:
        detections: List of all detections
        confidence_thresholds: List of confidence thresholds to evaluate
    
    Returns:
        fps_per_scan: List of average FPs per scan
        sensitivities: List of sensitivities
    """
    print("Calculating FROC curve...")
    
    fps_per_scan = []
    sensitivities = []
    
    for threshold in tqdm(confidence_thresholds, desc="Evaluating thresholds"):
        tps, fps, total_gt, num_scans = evaluate_detections_at_threshold(detections, threshold)
        
        if total_gt > 0:
            sensitivity = tps / total_gt
        else:
            sensitivity = 0.0
        
        avg_fps_per_scan = fps / num_scans if num_scans > 0 else 0.0
        
        fps_per_scan.append(avg_fps_per_scan)
        sensitivities.append(sensitivity)
    
    return fps_per_scan, sensitivities


def calculate_cpm(fps_per_scan, sensitivities):
    """
    Calculate CPM score by interpolating sensitivity at the 7 specific FP rates.
    
    Args:
        fps_per_scan: List of average FPs per scan
        sensitivities: List of sensitivities
    
    Returns:
        cpm_score: Average sensitivity at the 7 FP rates
        target_sensitivities: List of sensitivities at the 7 FP rates
    """
    # LUNA16 target FP rates
    target_fps = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    
    # Convert to numpy arrays
    fps_array = np.array(fps_per_scan)
    sens_array = np.array(sensitivities)
    
    # Sort by FP rate for interpolation
    sort_idx = np.argsort(fps_array)
    fps_sorted = fps_array[sort_idx]
    sens_sorted = sens_array[sort_idx]
    
    # Interpolate sensitivity at target FP rates
    target_sensitivities = []
    
    for target_fp in target_fps:
        if target_fp <= fps_sorted[0]:
            # Below minimum FP rate, use minimum sensitivity
            target_sens = sens_sorted[0]
        elif target_fp >= fps_sorted[-1]:
            # Above maximum FP rate, use maximum sensitivity
            target_sens = sens_sorted[-1]
        else:
            # Interpolate
            interp_func = interp1d(fps_sorted, sens_sorted, kind='linear', bounds_error=False, fill_value=(sens_sorted[0], sens_sorted[-1]))
            target_sens = interp_func(target_fp)
        
        target_sensitivities.append(float(target_sens))
    
    # Calculate CPM (average sensitivity)
    cpm_score = np.mean(target_sensitivities)
    
    return cpm_score, target_sensitivities


def run_model_on_entire_dataset(model, dataset, device, batch_size=128, max_candidates_per_scan=2000):
    """
    Run the model on entire dataset and collect all detections with confidence scores.
    Optimized for speed while maintaining accuracy.
    """
    print("Running model on entire dataset...")
    model.eval()
    all_detections = []
    
    for seriesuid, scan_path, candidates, annotations in tqdm(dataset, desc="Processing scans"):
        try:
            # Load scan
            volume, origin, spacing = load_scan(scan_path)
            
            # Get ground truth coordinates and radii
            gt_nodules = []
            for _, ann in annotations.iterrows():
                gt_coord = world_to_voxel([ann['coordZ'], ann['coordY'], ann['coordX']], origin, spacing)
                gt_radius = ann['diameter_mm'] / 2.0
                gt_nodules.append({
                    'coord': gt_coord,
                    'radius': gt_radius,
                    'matched': False
                })
            
            # Limit candidates per scan for speed (but keep more for accuracy)
            if len(candidates) > max_candidates_per_scan:
                candidates = candidates.sample(n=max_candidates_per_scan, random_state=42)
            
            # Process candidates in batches
            candidate_batches = []
            batch_coords = []
            
            for _, candidate in candidates.iterrows():
                # Extract patch
                candidate_coord = world_to_voxel([candidate['coordZ'], candidate['coordY'], candidate['coordX']], origin, spacing)
                patch = extract_patch(volume, candidate_coord)
                
                if patch is not None:
                    candidate_batches.append(patch)
                    batch_coords.append(candidate_coord)
                    
                    # Process batch when full
                    if len(candidate_batches) >= batch_size:
                        batch_tensor = torch.FloatTensor(np.array(candidate_batches)).unsqueeze(1).to(device)
                        
                        with torch.no_grad():
                            logits = model(batch_tensor).squeeze(1)
                            confidences = torch.sigmoid(logits).cpu().numpy()
                        
                        for i in range(len(candidate_batches)):
                            all_detections.append({
                                'seriesuid': seriesuid,
                                'coord': batch_coords[i],
                                'confidence': confidences[i],
                                'gt_nodules': gt_nodules.copy()
                            })
                        
                        candidate_batches = []
                        batch_coords = []
            
            # Process remaining candidates
            if len(candidate_batches) > 0:
                batch_tensor = torch.FloatTensor(np.array(candidate_batches)).unsqueeze(1).to(device)
                
                with torch.no_grad():
                    logits = model(batch_tensor).squeeze(1)
                    confidences = torch.sigmoid(logits).cpu().numpy()
                
                for i in range(len(candidate_batches)):
                    all_detections.append({
                        'seriesuid': seriesuid,
                        'coord': batch_coords[i],
                        'confidence': confidences[i],
                        'gt_nodules': gt_nodules.copy()
                    })
                    
        except Exception as e:
            print(f"Error processing scan {seriesuid}: {e}")
            continue
    
    print(f"Processed {len(all_detections)} total detections")
    return all_detections


def main():
    parser = argparse.ArgumentParser(description='Calculate LUNA16 CPM score')
    parser.add_argument('--model_path', type=str, default='pretrained_models/detector_best.pth',
                       help='Path to the best trained model')
    parser.add_argument('--annotations', type=str, default='data/annotations.csv',
                       help='Path to annotations.csv')
    parser.add_argument('--candidates', type=str, default='data/candidates.csv',
                       help='Path to candidates.csv')
    parser.add_argument('--scans_dirs', type=str, nargs='+', 
                       default=['data/subset0', 'data/subset1', 'data/subset2', 'data/subset3', 'data/subset4', 
                               'data/subset5', 'data/subset6', 'data/subset7', 'data/subset8', 'data/subset9'],
                       help='Directories containing scan files')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    parser.add_argument('--max_candidates_per_scan', type=int, default=2000,
                       help='Maximum candidates per scan')
    parser.add_argument('--num_thresholds', type=int, default=100,
                       help='Number of confidence thresholds to evaluate')
    parser.add_argument('--distance_threshold', type=float, default=5.0,
                       help='Distance threshold for matching detections to ground truth (mm)')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Enable fast mode for quick testing')
    args = parser.parse_args()
    
    # Fast mode overrides
    if args.fast_mode:
        args.batch_size = 256
        args.max_candidates_per_scan = 1000
        args.num_thresholds = 50
        print("Fast mode enabled")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = NoduleDetector(encoder_weights=None).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Load entire dataset
    dataset = load_entire_luna16_dataset(args.annotations, args.candidates, args.scans_dirs)
    
    # Run model on entire dataset
    detections = run_model_on_entire_dataset(model, dataset, device, args.batch_size, args.max_candidates_per_scan)
    
    if len(detections) == 0:
        print("No detections found. Check your data paths and model.")
        return
    
    # Generate confidence thresholds
    confidences = [det['confidence'] for det in detections]
    min_conf, max_conf = min(confidences), max(confidences)
    confidence_thresholds = np.linspace(max_conf, min_conf, args.num_thresholds) 
    
    print(f"Confidence range: {min_conf:.4f} to {max_conf:.4f}")
    print(f"Using {len(confidence_thresholds)} thresholds")
    
    # Calculate FROC curve
    fps_per_scan, sensitivities = calculate_froc_curve(detections, confidence_thresholds)
    
    # Calculate CPM
    cpm, target_sensitivities = calculate_cpm(fps_per_scan, sensitivities)
    
    # Print results
    print("\n" + "="*60)
    print("LUNA16 CPM CALCULATION RESULTS")
    print("="*60)
    print(f"CPM Score: {cpm:.4f}")
    print(f"Average Sensitivity: {cpm*100:.2f}%")
    print("\nSensitivity at each FP rate:")
    target_fps = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    for fp, sens in zip(target_fps, target_sensitivities):
        print(f"  {fp:.3f} FP/scan: {sens:.4f} ({sens*100:.2f}%)")
    
    # Save results
    results = {
        'cpm_score': float(cpm),
        'target_fps': target_fps,
        'target_sensitivities': target_sensitivities,
        'total_detections': len(detections),
        'total_scans': len(dataset),
        'confidence_range': [float(min_conf), float(max_conf)],
        'num_thresholds': len(confidence_thresholds),
        'max_candidates_per_scan': args.max_candidates_per_scan,
        'batch_size': args.batch_size,
        'distance_threshold': args.distance_threshold
    }
    
    results_file = args.model_path.replace('.pth', '_luna16_cpm_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    print("="*60)


if __name__ == '__main__':
    main()
