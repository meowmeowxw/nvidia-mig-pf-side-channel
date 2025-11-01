#!/usr/bin/env python3

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import matplotlib.pyplot as plt

def extract_features(raw, window_size=2000, step=0):
    """Extract statistical features from raw data with specified window and step."""
    features = []
    if step == 0:
        step = window_size
    for i in range(0, len(raw) - window_size + 1, step):
        window = raw[i:i+window_size]
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
        features.append(stats)
    return features

def read_raw_data(filepath, column=0):
    """Read raw float values from a file."""
    raw = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or ',' not in line:
                continue
            try:
                parts = line.strip().split(', ')
                raw.append(float(parts[column]))
            except:
                continue
    return np.array(raw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--window_size", type=int, default=2000)
    parser.add_argument("--step", type=int, default=0, help="Step size for windowing (default=0 means no overlap)")
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    X_train, y_train = [], []
    X_test, y_test = [], []

    for fname in os.listdir(args.logs_dir):
        fpath = os.path.join(args.logs_dir, fname)
        if not os.path.isfile(fpath):
            continue
        label = fname  # File name is the label

        raw = read_raw_data(fpath, column=args.column)
        if len(raw) < args.window_size:
            continue

        split_idx = int(len(raw) * (1 - args.test_ratio))
        raw_train = raw[:split_idx]
        raw_test = raw[split_idx:]

        feats_train = extract_features(raw_train, window_size=args.window_size, step=args.step)
        feats_test = extract_features(raw_test, window_size=args.window_size, step=args.step)

        X_train.extend(feats_train)
        y_train.extend([label] * len(feats_train))
        X_test.extend(feats_test)
        y_test.extend([label] * len(feats_test))

    print(f"Total training windows: {len(X_train)}")
    print(f"Total testing windows: {len(X_test)}")
    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    clf = XGBClassifier(eval_metric='mlogloss')
    clf.fit(X_train, y_train_encoded)

    y_pred = clf.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    # Feature Importance
    importance = clf.feature_importances_
    feature_names = [
        'mean', 'std', 'min', 'max', 'percentile_25', 'median', 'percentile_75', 'percentile_90', 'percentile_95'
    ]

    print("\n=== Feature Importances ===")
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score:.4f}")

    plt.figure(figsize=(8, 4))
    plt.bar(feature_names, importance)
    plt.title("Feature Importances")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

