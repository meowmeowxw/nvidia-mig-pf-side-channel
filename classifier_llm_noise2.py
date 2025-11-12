#!/usr/bin/env python3

import os
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import pandas as pd

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

    for col in columns:
        raw_data[col] = np.array(raw_data[col])
    
    return raw_data

def extract_features(raw_data, columns, window_size=2000, step=0):
    """Extract features from raw data with specified window and step."""
    if step == 0:
        step = window_size
        
    features = []
    min_length = min(len(raw_data[col]) for col in columns)

    for i in range(0, min_length - window_size + 1, step):
        window_features = []
        window_data = {}
        
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
        
        if len(columns) >= 2:
            for i_col, col1 in enumerate(columns):
                for col2 in columns[i_col+1:]:
                    correlation = np.corrcoef(window_data[col1], window_data[col2])[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                    window_features.append(correlation)
                    
                    mean1, mean2 = np.mean(window_data[col1]), np.mean(window_data[col2])
                    std1, std2 = np.std(window_data[col1]), np.std(window_data[col2])
                    
                    mean_ratio = mean1 / mean2 if mean2 != 0 else 0.0
                    window_features.append(mean_ratio)
                    
                    std_ratio = std1 / std2 if std2 != 0 else 0.0
                    window_features.append(std_ratio)
                    
                    window_features.append(mean1 - mean2)
                    window_features.append(std1 - std2)
        
        features.append(window_features)

    return features

def parse_filename(fname):
    """Parse filename to extract model name and noise level."""
    # Format: ModelName_noiselevel
    if '_' not in fname:
        return None, None
    
    parts = fname.rsplit('_', 1)
    model_name = parts[0]
    noise_level = parts[1] if len(parts) > 1 else 'unknown'
    
    return model_name, noise_level

def load_data_by_noise_level(logs_dir, noise_level, columns, window_size, step, test_ratio=0.2):
    """Load data for a specific noise level."""
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for fname in os.listdir(logs_dir):
        fpath = os.path.join(logs_dir, fname)
        if not os.path.isfile(fpath):
            continue
        
        model_name, file_noise_level = parse_filename(fname)
        if model_name is None or file_noise_level != noise_level:
            continue
        
        raw_data = read_raw_data(fpath, columns=columns)
        min_length = min(len(raw_data[col]) for col in columns)
        
        if min_length < window_size:
            continue

        # Split data into train/test portions
        split_idx = int(min_length * (1 - test_ratio))
        
        raw_train = {}
        raw_test = {}
        for col in columns:
            raw_train[col] = raw_data[col][:split_idx]
            raw_test[col] = raw_data[col][split_idx:]

        feats_train = extract_features(raw_train, columns, window_size=window_size, step=step)
        feats_test = extract_features(raw_test, columns, window_size=window_size, step=step)

        X_train.extend(feats_train)
        y_train.extend([model_name] * len(feats_train))
        X_test.extend(feats_test)
        y_test.extend([model_name] * len(feats_test))
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def parse_columns(columns_str):
    """Parse column specification from command line."""
    if isinstance(columns_str, list):
        return columns_str
    if ',' in columns_str:
        return [int(x.strip()) for x in columns_str.split(',')]
    else:
        return [int(columns_str)]

def main():
    parser = argparse.ArgumentParser(description='Train on clean/none data, test on noise levels')
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--window_size", type=int, default=2000)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--columns", default="0,1")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--train_noise", default="none", 
                       help="Noise level to train on (default: none)")
    args = parser.parse_args()
    
    if isinstance(args.columns, str):
        if ' ' in args.columns:
            args.columns = [int(x) for x in args.columns.split()]
        else:
            args.columns = parse_columns(args.columns)
    elif not isinstance(args.columns, list):
        args.columns = [args.columns]

    print("="*60)
    print(f"Using columns: {args.columns}")
    print(f"Window size: {args.window_size}, Step: {args.step if args.step > 0 else args.window_size}")
    print("="*60)

    # Test on each noise level separately (train and test on same level)
    noise_levels = ['none', 'low', 'medium', 'high']
    
    for noise_level in noise_levels:
        print("\n" + "="*60)
        print(f"Noise level: {noise_level}")
        print("="*60)
        
        # Load training and testing data for this noise level
        X_train, y_train, X_test, y_test = load_data_by_noise_level(
            args.logs_dir, noise_level, args.columns,
            args.window_size, args.step, args.test_ratio
        )
        
        if len(X_train) == 0:
            print(f"No training data found for noise level '{noise_level}'")
            continue
        
        if len(X_test) == 0:
            print(f"No test data found for noise level '{noise_level}'")
            continue
        
        print(f"\nTraining set:")
        print(f"  Total windows: {len(X_train)}")
        print(f"  Features per window: {len(X_train[0])}")
        print(f"  Label distribution: {Counter(y_train)}")
        
        print(f"\nTest set:")
        print(f"  Total windows: {len(X_test)}")
        print(f"  Label distribution: {Counter(y_test)}")
        
        # Fit label encoder and train model
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        clf = XGBClassifier(eval_metric='mlogloss', random_state=42)
        clf.fit(X_train, y_train_encoded)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
        
        # Calculate per-class accuracy
        cm = confusion_matrix(y_test_encoded, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        print("\n--- Per-Class Accuracy ---")
        for class_name, acc in zip(le.classes_, per_class_acc):
            print(f"{class_name}: {acc:.4f}")
        
        overall_acc = np.sum(y_pred == y_test_encoded) / len(y_test_encoded)
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        
        # Feature Importance for this noise level
        if noise_level == 'none':  # Only print once to avoid repetition
            print("\n" + "="*60)
            print("Feature Importances (Top 20)")
            print("="*60)
    
    importance = clf.feature_importances_
    base_features = ['mean', 'std', 'min', 'max', 'percentile_25', 'median', 
                     'percentile_75', 'percentile_90', 'percentile_95']
    feature_names = []
    
    for col in args.columns:
        for base_feat in base_features:
            feature_names.append(f"col{col}_{base_feat}")
    
    if len(args.columns) >= 2:
        for i_col, col1 in enumerate(args.columns):
            for col2 in args.columns[i_col+1:]:
                feature_names.extend([
                    f"corr_col{col1}_col{col2}",
                    f"mean_ratio_col{col1}_col{col2}",
                    f"std_ratio_col{col1}_col{col2}",
                    f"mean_diff_col{col1}_col{col2}",
                    f"std_diff_col{col1}_col{col2}"
                ])

    # Sort and display top features
    feature_importance = sorted(zip(feature_names, importance), 
                               key=lambda x: x[1], reverse=True)
    for name, score in feature_importance[:20]:
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main()
