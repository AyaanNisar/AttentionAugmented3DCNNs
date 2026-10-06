import os
import torch
from torch.utils.data import DataLoader, random_split
from detection.dataset import DetectionPatchDataset
from detection.model import NoduleDetector
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from glob import glob
from utils.load_scan import load_scan 

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

def extract_positives(annotations_path, scans_dir, out_dir, meta_entries):
    ann = pd.read_csv(annotations_path)
    available_scans = set(os.path.basename(f).replace('.mhd', '') for f in glob(os.path.join(scans_dir, '*.mhd')))
    ann = ann[ann['seriesuid'].isin(available_scans)]
    for i, row in ann.iterrows():
        seriesuid = row['seriesuid']
        scan_path = os.path.join(scans_dir, f'{seriesuid}.mhd')
        volume, origin, spacing = load_scan(scan_path)
        center_world = [row['coordZ'], row['coordY'], row['coordX']]
        center_voxel = world_to_voxel(center_world, origin, spacing) 
        patch = extract_patch(volume, center_voxel)
        fname = f'{seriesuid}_ann_{i}.npy'
        np.save(os.path.join(out_dir, fname), patch)
        meta_entries.append({'filename': fname, 'label': 1, 'seriesuid': seriesuid})
    return meta_entries

def sample_negatives(meta_csv, patches_dir, n_samples, out_dir, meta_entries):
    meta = pd.read_csv(meta_csv)
    negatives = meta[meta['label'] == 0]
    sampled = negatives.sample(n=n_samples, random_state=42)
    for _, row in sampled.iterrows():
        src = os.path.join(patches_dir, row['filename'])
        dst = os.path.join(out_dir, row['filename'])
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)  # or use shutil.copy if you want copies
        meta_entries.append({'filename': row['filename'], 'label': 0, 'seriesuid': row['seriesuid']})
    return meta_entries

def train_detector(
    meta_csv='data/patches_detection_meta.csv',
    patches_dir='data/patches_detection',
    encoder_weights='pretrained_models/ssl_encoder.pth',
    save_path='pretrained_models/detector_best.pth',
    batch_size=16,
    epochs=20, 
    lr=1e-4,
    device=None 
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Load metadata
    meta = pd.read_csv(meta_csv)
    positives = meta[meta['label'] == 1]
    negatives = meta[meta['label'] == 0]
    n_pos = len(positives)
    n_neg = int(n_pos * 6)  # 15% positives, 85% negatives for better learning
    print(f"Preparing training set: {n_pos} positives, sampling {n_neg} negatives (15/85 split)")
    sampled_negatives = negatives.sample(n=n_neg, random_state=42) if n_neg <= len(negatives) else negatives
    balanced = pd.concat([positives, sampled_negatives]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Three-way split: 70% train, 15% validation, 15% test
    n_total = len(balanced)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val
    
    train_meta = balanced.iloc[:n_train].reset_index(drop=True)
    val_meta = balanced.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test_meta = balanced.iloc[n_train+n_val:].reset_index(drop=True)
    
    print(f"Train set: {len(train_meta)} samples ({train_meta['label'].sum()} positives, {len(train_meta)-train_meta['label'].sum()} negatives)")
    print(f"Val set: {len(val_meta)} samples ({val_meta['label'].sum()} positives, {len(val_meta)-val_meta['label'].sum()} negatives)")
    print(f"Test set: {len(test_meta)} samples ({test_meta['label'].sum()} positives, {len(test_meta)-test_meta['label'].sum()} negatives)")
    
    # Save for reproducibility
    train_meta.to_csv(meta_csv.replace('.csv', '_train.csv'), index=False)
    val_meta.to_csv(meta_csv.replace('.csv', '_val.csv'), index=False)
    test_meta.to_csv(meta_csv.replace('.csv', '_test.csv'), index=False)
    
    # Dataset
    train_set = DetectionPatchDataset(meta_csv.replace('.csv', '_train.csv'), patches_dir)
    val_set = DetectionPatchDataset(meta_csv.replace('.csv', '_val.csv'), patches_dir)
    test_set = DetectionPatchDataset(meta_csv.replace('.csv', '_test.csv'), patches_dir)
    
    print(f"Actual train dataset size (after filtering): {len(train_set)}")
    print(f"Actual val dataset size (after filtering): {len(val_set)}")
    print(f"Actual test dataset size (after filtering): {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = NoduleDetector(encoder_weights=encoder_weights).to(device)
    
    # Calculate class weights to address imbalance
    pos_weight = torch.tensor([len(negatives) / len(positives)], device=device)
    print(f"Using positive class weight: {pos_weight.item():.2f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Add L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_auc = 0
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            train_iter.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        all_logits, all_labels = [], []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
        
        with torch.no_grad():
            for x, y in val_iter:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(1)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        if len(all_logits) > 0:
            all_logits = np.concatenate(all_logits)
            all_labels = np.concatenate(all_labels)
            preds = (all_logits > 0).astype(int)
            acc = accuracy_score(all_labels, preds)
            
            try:
                auc = roc_auc_score(all_labels, all_logits)
            except:
                auc = float('nan')
            
            # Calculate additional metrics
            pos_preds = preds[all_labels == 1]
            neg_preds = preds[all_labels == 0]
            pos_acc = np.mean(pos_preds) if len(pos_preds) > 0 else 0
            neg_acc = 1 - np.mean(neg_preds) if len(neg_preds) > 0 else 0
            
            # Calculate F1 score for better balance assessment
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, preds)
            
            print(f"Epoch {epoch+1} | Val Acc: {acc:.4f} | ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Pos Acc: {pos_acc:.4f} | Neg Acc: {neg_acc:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(auc)
            
            # Early stopping based on AUC
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best model to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            print(f"Epoch {epoch+1} | No validation data available")
    
    print(f"Training completed. Best ROC-AUC: {best_auc:.4f}")
    
    # Final comprehensive evaluation on TEST set (never seen during training)
    print("\n" + "="*60)
    print("FINAL COMPREHENSIVE EVALUATION RESULTS (TEST SET)")
    print("="*60)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Final Evaluation (Test Set)"):
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    if len(all_logits) > 0:
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        preds = (all_logits > 0).astype(int)
        
        # Calculate all metrics
        accuracy = accuracy_score(all_labels, preds)
        try:
            auc = roc_auc_score(all_labels, all_logits)
        except:
            auc = float('nan')
        
        # Confusion matrix for detailed metrics
        cm = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate key metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate / Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR = 1 - Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Print comprehensive results
        print(f"Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"ROC-AUC Score:        {auc:.4f}")
        print(f"Sensitivity (TPR):    {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"Specificity (TNR):    {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"False Positive Rate:  {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
        print(f"Precision (PPV):      {precision:.4f} ({precision*100:.2f}%)")
        print(f"F1 Score:             {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                    Predicted")
        print(f"                  0 (No Nodule)  1 (Nodule)")
        print(f"Actual 0 (No Nodule)     {tn:8d}      {fp:8d}")
        print(f"Actual 1 (Nodule)        {fn:8d}      {tp:8d}")
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_labels, preds, target_names=['No Nodule', 'Nodule']))
        
        # Save final results to file
        results = {
            'accuracy': float(accuracy),
            'roc_auc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'false_positive_rate': float(false_positive_rate),
            'precision': float(precision),
            'f1_score': float(f1),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        import json
        results_file = save_path.replace('.pth', '_final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFinal results saved to: {results_file}")
        
    else:
        print("No test data available for final evaluation")
    
    print("="*60)

if __name__ == '__main__':
    # Simply call the training function - the metadata is already balanced from extract_detection_patches.py
    train_detector() 