# Attention-Augmented 3D CNNs with Self-Supervision for Lung Nodule Detection

Code for the paper **Attention-Augmented 3D CNNs with Self-Supervision for Lung Nodule Detection**.

Author: **Ayaan Nisar**

This repository contains the LUNA16 training and evaluation pipeline used for that paper: SimCLR pretraining on 3D CT patches, an attention-augmented 3D ResNet-18 detector, and CPM evaluation.

## Setup

Place the LUNA16 scans in `data/subset0` through `data/subset9`, and put `candidates.csv` in `data/`.

```bash
pip install -r requirements.txt
python -m utils.extract_patches
python -m pretraining.train_simclr
python -m utils.extract_detection_patches
python -m detection.train_detector
python calculate_cpm.py
```
