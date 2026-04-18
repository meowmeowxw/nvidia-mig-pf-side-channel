#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Read results
results_file = "./window_experiments/results.csv"
df = pd.read_csv(results_file)

print("Loaded results:")
print(df.to_string(index=False))
print("\n")

# Create output directory for plots
output_dir = Path("./window_experiments/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PLOT 1: Main performance metrics
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Impact of Window Size on LLM Fingerprinting Performance', fontsize=14, fontweight='bold')

# Plot 1: Accuracy vs Window Size
ax1 = axes[0, 0]
ax1.plot(df['window_size'], df['accuracy'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Window Size (samples)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Classification Accuracy vs Window Size')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Add horizontal line at your current best result
if 30000 in df['window_size'].values:
    current_accuracy = df[df['window_size'] == 30000]['accuracy'].values[0]
    ax1.axhline(y=current_accuracy, color='red', linestyle='--', alpha=0.5, label=f'Current (30k): {current_accuracy:.3f}')
    ax1.legend()

# Plot 2: F1-Score vs Window Size
ax2 = axes[0, 1]
ax2.plot(df['window_size'], df['macro_avg_f1'], 'o-', linewidth=2, markersize=8, color='#A23B72')
ax2.set_xlabel('Window Size (samples)')
ax2.set_ylabel('Macro Avg F1-Score')
ax2.set_title('F1-Score vs Window Size')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

# Plot 3: Number of Windows vs Window Size
ax3 = axes[1, 0]
ax3.plot(df['window_size'], df['num_train'], 'o-', linewidth=2, markersize=8, color='#F18F01', label='Training')
ax3.plot(df['window_size'], df['num_test'], 's-', linewidth=2, markersize=8, color='#C73E1D', label='Testing')
ax3.set_xlabel('Window Size (samples)')
ax3.set_ylabel('Number of Windows')
ax3.set_title('Dataset Size vs Window Size')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Precision, Recall, F1 together
ax4 = axes[1, 1]
ax4.plot(df['window_size'], df['macro_avg_precision'], 'o-', linewidth=2, markersize=6, label='Precision', color='#06A77D')
ax4.plot(df['window_size'], df['macro_avg_recall'], 's-', linewidth=2, markersize=6, label='Recall', color='#D4AF37')
ax4.plot(df['window_size'], df['macro_avg_f1'], '^-', linewidth=2, markersize=6, label='F1-Score', color='#9D4EDD')
ax4.set_xlabel('Window Size (samples)')
ax4.set_ylabel('Score')
ax4.set_title('Performance Metrics vs Window Size')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1.05])
ax4.legend()

plt.tight_layout()
plt.savefig(output_dir / 'window_size_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'window_size_analysis.png'}")

# ============================================================================
# PLOT 2: Accuracy-dataset size tradeoff
# ============================================================================
fig2, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot with size representing dataset size
scatter = ax.scatter(df['window_size'], df['accuracy'], 
                     s=df['num_train']/10,  # Scale down for visualization
                     c=df['macro_avg_f1'], 
                     cmap='viridis', 
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=1)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Macro Avg F1-Score')

# Annotate points with window size
for idx, row in df.iterrows():
    ax.annotate(f"{int(row['window_size']/1000)}k", 
                (row['window_size'], row['accuracy']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                alpha=0.7)

ax.set_xlabel('Window Size (samples)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Window Size\n(Bubble size = number of training windows)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'accuracy_tradeoff.png'}")

# ============================================================================
# PLOT 3 & 4: Feature Importance Analysis
# ============================================================================

# Load all feature importance files
feature_importance_dir = Path("./window_experiments/feature_importances")
feature_files = sorted(glob.glob(str(feature_importance_dir / "features_*.csv")))

if feature_files:
    print("\nAnalyzing feature importances...")
    
    # Create a dictionary to store all feature importances
    all_features = {}
    window_sizes_feat = []
    
    for feat_file in feature_files:
        # Extract window size from filename
        window_size = int(Path(feat_file).stem.split('_')[1])
        window_sizes_feat.append(window_size)
        
        # Load feature importances
        feat_df = pd.read_csv(feat_file)
        
        for _, row in feat_df.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            
            if feature_name not in all_features:
                all_features[feature_name] = {}
            all_features[feature_name][window_size] = importance
    
    # Convert to DataFrame for easier plotting
    feature_importance_df = pd.DataFrame(all_features).T
    feature_importance_df = feature_importance_df[sorted(feature_importance_df.columns)]
    
    # Plot 3: Heatmap of feature importances
    fig3, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(feature_importance_df, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Feature Importance'},
                ax=ax,
                linewidths=0.5)
    
    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance Across Different Window Sizes', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    ax.set_xticklabels([f'{int(x/1000)}k' for x in sorted(window_sizes_feat)], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_importance_heatmap.png'}")
    
    # Plot 4: Top features evolution
    fig4, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate mean importance across all window sizes for each feature
    feature_importance_df['mean_importance'] = feature_importance_df.mean(axis=1)
    top_features = feature_importance_df.nlargest(10, 'mean_importance').index
    
    # Plot evolution of top features
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_features)))
    
    for i, feature in enumerate(top_features):
        x_vals = sorted(window_sizes_feat)
        y_vals = [feature_importance_df.loc[feature, ws] for ws in x_vals]
        ax.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=6, 
                label=feature, color=colors[i])
    
    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Feature Importance', fontsize=12)
    ax.set_title('Evolution of Top 10 Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_evolution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'top_features_evolution.png'}")
    
    # Plot 5: Feature category analysis
    fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle('Feature Category Importance Across Window Sizes', fontsize=14, fontweight='bold')
    
    # Categorize features
    categories = {
        'col0_stats': [],
        'col1_stats': [],
        'correlations': [],
        'ratios_diffs': []
    }
    
    for feature in feature_importance_df.index:
        if feature.startswith('col0_'):
            categories['col0_stats'].append(feature)
        elif feature.startswith('col1_'):
            categories['col1_stats'].append(feature)
        elif feature.startswith('corr_'):
            categories['correlations'].append(feature)
        else:
            categories['ratios_diffs'].append(feature)
    
    # Calculate category-wise importance
    category_importance = {}
    for cat_name, features in categories.items():
        if features:
            category_importance[cat_name] = feature_importance_df.loc[features].mean()
    
    # Plot each category
    cat_names = ['col0_stats', 'col1_stats', 'correlations', 'ratios_diffs']
    cat_labels = ['Column 0 Statistics', 'Column 1 Statistics', 'Correlations', 'Ratios & Differences']
    colors_cat = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for idx, (cat_name, cat_label, color) in enumerate(zip(cat_names, cat_labels, colors_cat)):
        ax = axes[idx // 2, idx % 2]
        
        if cat_name in category_importance:
            x_vals = sorted(window_sizes_feat)
            y_vals = [category_importance[cat_name][ws] for ws in x_vals]
            ax.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8, color=color)
            ax.set_xlabel('Window Size (samples)')
            ax.set_ylabel('Mean Importance')
            ax.set_title(cat_label)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_category_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_category_analysis.png'}")
    
    # Save feature importance summary
    summary_file = output_dir / 'feature_importance_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top 10 Most Important Features (averaged across all window sizes):\n")
        f.write("-"*80 + "\n")
        for i, (feature, importance) in enumerate(feature_importance_df['mean_importance'].nlargest(10).items(), 1):
            f.write(f"{i:2d}. {feature:30s}: {importance:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Category-wise Importance (averaged across all window sizes):\n")
        f.write("-"*80 + "\n")
        
        for cat_name, cat_label in zip(cat_names, cat_labels):
            if cat_name in category_importance:
                mean_cat_imp = category_importance[cat_name].mean()
                f.write(f"{cat_label:30s}: {mean_cat_imp:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Feature Stability Analysis:\n")
        f.write("-"*80 + "\n")
        f.write("(Standard deviation of importance across window sizes)\n\n")
        
        feature_std = feature_importance_df.drop('mean_importance', axis=1).std(axis=1).sort_values(ascending=False)
        f.write("Most Stable Features (low std):\n")
        for feature, std in feature_std.nsmallest(5).items():
            f.write(f"  {feature:30s}: std={std:.4f}\n")
        
        f.write("\nLeast Stable Features (high std):\n")
        for feature, std in feature_std.nlargest(5).items():
            f.write(f"  {feature:30s}: std={std:.4f}\n")
    
    print(f"Saved: {summary_file}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

best_accuracy_idx = df['accuracy'].idxmax()
best_f1_idx = df['macro_avg_f1'].idxmax()

print(f"\nBest Accuracy: {df.loc[best_accuracy_idx, 'accuracy']:.4f}")
print(f"  Window Size: {df.loc[best_accuracy_idx, 'window_size']}")
print(f"  Training Windows: {df.loc[best_accuracy_idx, 'num_train']}")
print(f"  F1-Score: {df.loc[best_accuracy_idx, 'macro_avg_f1']:.4f}")

print(f"\nBest F1-Score: {df.loc[best_f1_idx, 'macro_avg_f1']:.4f}")
print(f"  Window Size: {df.loc[best_f1_idx, 'window_size']}")
print(f"  Training Windows: {df.loc[best_f1_idx, 'num_train']}")
print(f"  Accuracy: {df.loc[best_f1_idx, 'accuracy']:.4f}")

# Find optimal trade-off (high accuracy with reasonable dataset size)
# Define "reasonable" as having at least 1000 training windows
reasonable_df = df[df['num_train'] >= 1000]
if not reasonable_df.empty:
    optimal_idx = reasonable_df['accuracy'].idxmax()
    print(f"\nOptimal Trade-off (≥1000 training windows):")
    print(f"  Window Size: {df.loc[optimal_idx, 'window_size']}")
    print(f"  Accuracy: {df.loc[optimal_idx, 'accuracy']:.4f}")
    print(f"  F1-Score: {df.loc[optimal_idx, 'macro_avg_f1']:.4f}")
    print(f"  Training Windows: {df.loc[optimal_idx, 'num_train']}")

print("\n" + "="*60)
print("\nRecommendations for paper:")
print("="*60)

# Calculate performance variance
acc_std = df['accuracy'].std()
f1_std = df['macro_avg_f1'].std()

if acc_std < 0.05:
    print("\n✓ Low performance variance across window sizes suggests the model")
    print("  is robust to this hyperparameter. You can justify your choice based")
    print("  on computational efficiency (smaller windows = more training data).")
else:
    print("\n✓ Significant performance variance observed. Include ablation study")
    print("  showing these results and justify your chosen window size.")

print("\n✓ In your paper, report:")
print("  - The range of window sizes tested")
print(f"  - That step size was kept proportional (10% of window size)")
print("  - Performance metrics AND dataset sizes for transparency")
print("  - Justify your choice based on accuracy-efficiency trade-off")

print("\n✓ Consider adding to paper:")
print("  - A figure showing this window size analysis")
print("  - Discussion of temporal resolution vs statistical stability")
print("  - Computational cost comparison (smaller windows = more samples)")
print("  - Feature importance analysis showing which features are most discriminative")

# Save summary to text file
with open(output_dir / 'summary.txt', 'w') as f:
    f.write("WINDOW SIZE EXPERIMENT SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\n" + "="*60 + "\n")
    f.write(f"Best Accuracy: {df.loc[best_accuracy_idx, 'accuracy']:.4f} ")
    f.write(f"(window_size={df.loc[best_accuracy_idx, 'window_size']})\n")
    f.write(f"Best F1-Score: {df.loc[best_f1_idx, 'macro_avg_f1']:.4f} ")
    f.write(f"(window_size={df.loc[best_f1_idx, 'window_size']})\n")

print(f"\nSummary saved to: {output_dir / 'summary.txt'}")
print("\nAll plots generated successfully!")
plt.show()
