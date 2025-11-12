#!/usr/bin/env python3

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

def read_raw_data(filepath, columns=[0, 1]):
    """Read raw float values from a file for specified columns."""
    raw_data = {col: [] for col in columns}
    
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or ',' not in line:
                continue
            try:
                parts = line.strip().split(', ')
                for col in columns:
                    if col < len(parts):
                        raw_data[col].append(float(parts[col]))
                    else:
                        raw_data[col].append(0.0)
            except:
                continue

    # Convert to numpy arrays
    for col in columns:
        raw_data[col] = np.array(raw_data[col])
    
    return raw_data

def extract_features(raw_data, columns, window_size=3, step=0):
    """Extract features from raw data with specified window and step."""
    if step == 0:
        step = window_size
        
    features = []
    min_length = min(len(raw_data[col]) for col in columns)

    for i in range(0, min_length - window_size + 1, step):
        window_features = []
        window_data = {}
        
        # Calculate statistics for each column
        for col in columns:
            window = raw_data[col][i:i+window_size]
            window_data[col] = window
            stats = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.percentile(window, 25),
                np.percentile(window, 50),
                np.percentile(window, 75),
                np.percentile(window, 90),
                np.percentile(window, 95),
            ]
            window_features.extend(stats)
        
        # Add correlation and cross-column features if we have multiple columns
        if len(columns) >= 2:
            # Pairwise correlations between columns
            for i_col, col1 in enumerate(columns):
                for col2 in columns[i_col+1:]:
                    correlation = np.corrcoef(window_data[col1], window_data[col2])[0, 1]
                    # Handle NaN correlations (e.g., when std is 0)
                    if np.isnan(correlation):
                        correlation = 0.0
                    window_features.append(correlation)
                    
                    # Ratio features
                    mean1, mean2 = np.mean(window_data[col1]), np.mean(window_data[col2])
                    std1, std2 = np.std(window_data[col1]), np.std(window_data[col2])
                    
                    # Mean ratio (handle division by zero)
                    mean_ratio = mean1 / mean2 if mean2 != 0 else 0.0
                    window_features.append(mean_ratio)
                    
                    # Std ratio (handle division by zero)
                    std_ratio = std1 / std2 if std2 != 0 else 0.0
                    window_features.append(std_ratio)
                    
                    # Difference features
                    window_features.append(mean1 - mean2)  # mean difference
                    window_features.append(std1 - std2)   # std difference
        
        features.append(window_features)

    return features

def parse_columns(columns_str):
    """Parse column specification from command line."""
    if isinstance(columns_str, list):
        return columns_str
    # Handle comma-separated values
    if ',' in columns_str:
        return [int(x.strip()) for x in columns_str.split(',')]
    else:
        return [int(columns_str)]

def main():
    parser = argparse.ArgumentParser(description='PCIe Throughput Classifier for LLM Fingerprinting')
    parser.add_argument("--logs_dir", required=True, help="Directory containing PCIe throughput logs")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for feature extraction (default: 3)")
    parser.add_argument("--step", type=int, default=0, help="Step size for windowing (default=0 means no overlap)")
    parser.add_argument("--columns", default="0,1", 
                       help="Columns to use for features. Can be: '0,1' or '0 1' or single column '0' (default: '0,1')")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (default: 0.2)")
    args = parser.parse_args()
    
    # Parse columns argument
    if isinstance(args.columns, str):
        if ' ' in args.columns:
            args.columns = [int(x) for x in args.columns.split()]
        else:
            args.columns = parse_columns(args.columns)
    elif not isinstance(args.columns, list):
        args.columns = [args.columns]

    print("="*60)
    print("PCIe Throughput Classifier")
    print("="*60)
    print(f"Logs directory: {args.logs_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Step size: {args.step if args.step > 0 else args.window_size} (no overlap)" if args.step == 0 else f"Step size: {args.step} (overlapping)")
    print(f"Using columns: {args.columns} (0=RX, 1=TX)")
    print(f"Train/Test split: {int((1-args.test_ratio)*100)}/{int(args.test_ratio*100)}")
    print("="*60)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for fname in os.listdir(args.logs_dir):
        fpath = os.path.join(args.logs_dir, fname)
        if not os.path.isfile(fpath):
            continue
        label = fname  # file name is the label
        
        raw_data = read_raw_data(fpath, columns=args.columns)
        min_length = min(len(raw_data[col]) for col in args.columns)
        
        if min_length < args.window_size:
            print(f"Warning: Skipping {fname} - insufficient data (only {min_length} samples, need {args.window_size})")
            continue

        # Split data into train/test portions
        split_idx = int(min_length * (1 - args.test_ratio))
        
        raw_train = {}
        raw_test = {}
        for col in args.columns:
            raw_train[col] = raw_data[col][:split_idx]
            raw_test[col] = raw_data[col][split_idx:]

        feats_train = extract_features(raw_train, args.columns, window_size=args.window_size, step=args.step)
        feats_test = extract_features(raw_test, args.columns, window_size=args.window_size, step=args.step)

        X_train.extend(feats_train)
        y_train.extend([label] * len(feats_train))
        X_test.extend(feats_test)
        y_test.extend([label] * len(feats_test))
        
        print(f"Loaded {fname}: {len(feats_train)} train windows, {len(feats_test)} test windows")

    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Total training windows: {len(X_train)}")
    print(f"Total testing windows: {len(X_test)}")
    print(f"Features per window: {len(X_train[0]) if X_train else 0}")
    print(f"\nTrain label distribution:")
    for label, count in sorted(Counter(y_train).items()):
        print(f"  {label}: {count}")
    print(f"\nTest label distribution:")
    for label, count in sorted(Counter(y_test).items()):
        print(f"  {label}: {count}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("\nError: Insufficient data for training/testing")
        return

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print("\n" + "="*60)
    print("Training XGBoost Classifier...")
    print("="*60)
    clf = XGBClassifier(eval_metric='mlogloss', random_state=42)
    clf.fit(X_train, y_train_encoded)

    y_pred = clf.predict(X_test)

    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    # Feature Importance
    importance = clf.feature_importances_
    
    # Create feature names for both columns and correlation features
    base_features = ['mean', 'std', 'min', 'max', 'percentile_25', 'median', 'percentile_75', 'percentile_90', 'percentile_95']
    feature_names = []
    
    # Map column numbers to readable names
    col_names = {0: 'RX', 1: 'TX'}
    
    # Individual column features
    for col in args.columns:
        col_name = col_names.get(col, f'col{col}')
        for base_feat in base_features:
            feature_names.append(f"{col_name}_{base_feat}")
    
    # Cross-column correlation features (if multiple columns)
    if len(args.columns) >= 2:
        for i_col, col1 in enumerate(args.columns):
            for col2 in args.columns[i_col+1:]:
                col1_name = col_names.get(col1, f'col{col1}')
                col2_name = col_names.get(col2, f'col{col2}')
                feature_names.extend([
                    f"corr_{col1_name}_{col2_name}",
                    f"mean_ratio_{col1_name}_{col2_name}",
                    f"std_ratio_{col1_name}_{col2_name}",
                    f"mean_diff_{col1_name}_{col2_name}",
                    f"std_diff_{col1_name}_{col2_name}"
                ])

    print("\n" + "="*60)
    print("Feature Importances (Top 15)")
    print("="*60)
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for name, score in feature_importance[:15]:
        print(f"{name:30s}: {score:.4f}")

if __name__ == "__main__":
    main()
