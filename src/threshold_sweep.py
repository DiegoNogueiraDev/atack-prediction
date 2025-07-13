#!/usr/bin/env python3
"""
Threshold Sweep for Optimal Attack Detection
Performs fine-grained threshold sweeping to optimize precision/recall trade-off
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThresholdSweeper:
    """Optimizes detection threshold for better precision/recall balance"""
    
    def __init__(self, model_dir: str = "model_v1"):
        self.model_dir = Path(model_dir)
        self.results = []
        
    def load_artifacts(self):
        """Load model artifacts and test data"""
        logger.info("Loading model artifacts...")
        
        # Load test errors and labels
        test_errors_path = self.model_dir / "test_errors.pkl"
        test_labels_path = self.model_dir / "test_labels.pkl"
        
        if not test_errors_path.exists() or not test_labels_path.exists():
            raise FileNotFoundError("Test errors/labels not found. Run detection script first.")
            
        with open(test_errors_path, 'rb') as f:
            self.test_errors = pickle.load(f)
        with open(test_labels_path, 'rb') as f:
            self.test_labels = pickle.load(f)
            
        # Load current threshold info
        threshold_path = self.model_dir / "threshold_info.pkl"
        with open(threshold_path, 'rb') as f:
            threshold_info = pickle.load(f)
            self.base_threshold = threshold_info['suggested_threshold']
            
        logger.info(f"Loaded {len(self.test_errors)} test samples")
        logger.info(f"Base threshold: {self.base_threshold:.4f}")
        
    def sweep_thresholds(self, multipliers=None):
        """Perform threshold sweep with different multipliers"""
        if multipliers is None:
            # Fine-grained sweep around base threshold
            multipliers = np.arange(0.3, 2.0, 0.05)
            
        logger.info(f"Testing {len(multipliers)} threshold values...")
        
        for multiplier in multipliers:
            threshold = self.base_threshold * multiplier
            predictions = (self.test_errors > threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(self.test_labels, predictions, zero_division=0)
            recall = recall_score(self.test_labels, predictions, zero_division=0)
            f1 = f1_score(self.test_labels, predictions, zero_division=0)
            
            # Calculate False Positive Rate
            tn, fp, fn, tp = confusion_matrix(self.test_labels, predictions).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Calculate additional metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            result = {
                'multiplier': multiplier,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'fpr': fpr,
                'specificity': specificity,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
            
            self.results.append(result)
            
        self.results_df = pd.DataFrame(self.results)
        logger.info("Threshold sweep completed")
        
    def find_optimal_thresholds(self):
        """Find optimal thresholds based on different criteria"""
        optimal_thresholds = {}
        
        # 1. Best F1 score
        best_f1_idx = self.results_df['f1_score'].idxmax()
        optimal_thresholds['best_f1'] = self.results_df.loc[best_f1_idx]
        
        # 2. Recall ≥ 50% with minimal FPR
        high_recall = self.results_df[self.results_df['recall'] >= 0.5]
        if not high_recall.empty:
            best_recall_idx = high_recall['fpr'].idxmin()
            optimal_thresholds['high_recall_low_fpr'] = high_recall.loc[best_recall_idx]
        
        # 3. FPR ≤ 20% with maximal recall
        low_fpr = self.results_df[self.results_df['fpr'] <= 0.2]
        if not low_fpr.empty:
            best_low_fpr_idx = low_fpr['recall'].idxmax()
            optimal_thresholds['low_fpr_high_recall'] = low_fpr.loc[best_low_fpr_idx]
        
        # 4. Balanced precision-recall (closest to diagonal)
        self.results_df['pr_balance'] = abs(self.results_df['precision'] - self.results_df['recall'])
        best_balance_idx = self.results_df['pr_balance'].idxmin()
        optimal_thresholds['balanced_pr'] = self.results_df.loc[best_balance_idx]
        
        return optimal_thresholds
        
    def plot_results(self, save_path=None):
        """Generate comprehensive plots of threshold sweep results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Threshold Sweep Analysis', fontsize=16, fontweight='bold')
        
        # 1. Precision vs Recall vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(self.results_df['threshold'], self.results_df['precision'], 'b-', label='Precision', linewidth=2)
        ax1.plot(self.results_df['threshold'], self.results_df['recall'], 'r-', label='Recall', linewidth=2)
        ax1.plot(self.results_df['threshold'], self.results_df['f1_score'], 'g-', label='F1-Score', linewidth=2)
        ax1.axvline(x=self.base_threshold, color='k', linestyle='--', alpha=0.7, label='Current Threshold')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        ax2 = axes[0, 1]
        ax2.plot(self.results_df['recall'], self.results_df['precision'], 'b-', linewidth=2)
        ax2.scatter(self.results_df['recall'], self.results_df['precision'], c=self.results_df['threshold'], 
                   cmap='viridis', s=30, alpha=0.7)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for threshold values
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Threshold')
        
        # 3. FPR vs Recall
        ax3 = axes[0, 2]
        ax3.plot(self.results_df['fpr'], self.results_df['recall'], 'r-', linewidth=2)
        ax3.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Target Recall ≥ 50%')
        ax3.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Max FPR ≤ 20%')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('Recall')
        ax3.set_title('FPR vs Recall Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Threshold vs FPR
        ax4 = axes[1, 0]
        ax4.plot(self.results_df['threshold'], self.results_df['fpr'], 'orange', linewidth=2)
        ax4.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='Max FPR = 20%')
        ax4.axvline(x=self.base_threshold, color='k', linestyle='--', alpha=0.7, label='Current Threshold')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('False Positive Rate')
        ax4.set_title('Threshold vs FPR')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Detection Performance Heatmap
        ax5 = axes[1, 1]
        
        # Create a grid for heatmap visualization
        recall_bins = np.linspace(0, 1, 11)
        fpr_bins = np.linspace(0, 1, 11)
        
        # Bin the data
        recall_binned = np.digitize(self.results_df['recall'], recall_bins)
        fpr_binned = np.digitize(self.results_df['fpr'], fpr_bins)
        
        # Create heatmap data
        heatmap_data = np.zeros((len(recall_bins)-1, len(fpr_bins)-1))
        for i, (r_bin, f_bin, f1) in enumerate(zip(recall_binned, fpr_binned, self.results_df['f1_score'])):
            if 1 <= r_bin <= len(recall_bins)-1 and 1 <= f_bin <= len(fpr_bins)-1:
                heatmap_data[r_bin-1, f_bin-1] = max(heatmap_data[r_bin-1, f_bin-1], f1)
        
        im = ax5.imshow(heatmap_data, cmap='RdYlBu_r', origin='lower', aspect='auto')
        ax5.set_xlabel('FPR Bins')
        ax5.set_ylabel('Recall Bins')
        ax5.set_title('F1-Score Heatmap (Recall vs FPR)')
        plt.colorbar(im, ax=ax5, label='F1-Score')
        
        # 6. Confusion Matrix Statistics
        ax6 = axes[1, 2]
        
        # Find best performing threshold for visualization
        best_f1_idx = self.results_df['f1_score'].idxmax()
        best_result = self.results_df.loc[best_f1_idx]
        
        cm_data = np.array([[int(best_result['tn']), int(best_result['fp'])], 
                           [int(best_result['fn']), int(best_result['tp'])]])
        
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax6,
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax6.set_title(f'Confusion Matrix (Best F1)\nThreshold: {best_result["threshold"]:.4f}')
        ax6.set_ylabel('True Label')
        ax6.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
        
    def generate_report(self, optimal_thresholds):
        """Generate detailed report of findings"""
        report = "\n" + "="*80 + "\n"
        report += "THRESHOLD SWEEP ANALYSIS REPORT\n"
        report += "="*80 + "\n\n"
        
        # Current performance
        current_result = self.results_df[self.results_df['multiplier'].round(2) == 1.0]
        if not current_result.empty:
            current = current_result.iloc[0]
            report += f"CURRENT PERFORMANCE (Threshold: {current['threshold']:.4f}):\n"
            report += f"  Precision: {current['precision']:.1%}\n"
            report += f"  Recall: {current['recall']:.1%}\n"
            report += f"  F1-Score: {current['f1_score']:.1%}\n"
            report += f"  FPR: {current['fpr']:.1%}\n\n"
        
        # Optimal configurations
        for name, result in optimal_thresholds.items():
            report += f"{name.upper().replace('_', ' ')} (Threshold: {result['threshold']:.4f}):\n"
            report += f"  Multiplier: {result['multiplier']:.2f}x\n"
            report += f"  Precision: {result['precision']:.1%}\n"
            report += f"  Recall: {result['recall']:.1%}\n"
            report += f"  F1-Score: {result['f1_score']:.1%}\n"
            report += f"  FPR: {result['fpr']:.1%}\n"
            report += f"  Accuracy: {result['accuracy']:.1%}\n\n"
        
        # Summary statistics
        report += "SUMMARY STATISTICS:\n"
        report += f"  Thresholds tested: {len(self.results_df)}\n"
        report += f"  Max Recall achieved: {self.results_df['recall'].max():.1%}\n"
        report += f"  Min FPR achieved: {self.results_df['fpr'].min():.1%}\n"
        report += f"  Max F1-Score: {self.results_df['f1_score'].max():.1%}\n"
        
        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        
        high_recall = self.results_df[self.results_df['recall'] >= 0.5]
        if not high_recall.empty:
            best_hr = high_recall.loc[high_recall['fpr'].idxmin()]
            report += f"  • For Recall ≥ 50%: Use threshold {best_hr['threshold']:.4f} "
            report += f"(Recall: {best_hr['recall']:.1%}, FPR: {best_hr['fpr']:.1%})\n"
        else:
            max_recall_result = self.results_df.loc[self.results_df['recall'].idxmax()]
            report += f"  • Max achievable recall: {max_recall_result['recall']:.1%} "
            report += f"at threshold {max_recall_result['threshold']:.4f}\n"
        
        low_fpr = self.results_df[self.results_df['fpr'] <= 0.2]
        if not low_fpr.empty:
            best_lf = low_fpr.loc[low_fpr['recall'].idxmax()]
            report += f"  • For FPR ≤ 20%: Use threshold {best_lf['threshold']:.4f} "
            report += f"(Recall: {best_lf['recall']:.1%}, FPR: {best_lf['fpr']:.1%})\n"
        
        return report
        
    def save_results(self, save_dir=None):
        """Save sweep results and recommendations"""
        if save_dir is None:
            save_dir = self.model_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_path = save_dir / "threshold_sweep_results.csv"
        self.results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Save optimal thresholds
        optimal_thresholds = self.find_optimal_thresholds()
        optimal_path = save_dir / "optimal_thresholds.pkl"
        with open(optimal_path, 'wb') as f:
            pickle.dump(optimal_thresholds, f)
        logger.info(f"Optimal thresholds saved to {optimal_path}")
        
        return optimal_thresholds

def main():
    parser = argparse.ArgumentParser(description='Perform threshold sweep for attack detection optimization')
    parser.add_argument('--model-dir', default='model_v1', help='Model directory path')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to file')
    parser.add_argument('--min-multiplier', type=float, default=0.3, help='Minimum threshold multiplier')
    parser.add_argument('--max-multiplier', type=float, default=2.0, help='Maximum threshold multiplier')
    parser.add_argument('--step', type=float, default=0.05, help='Step size for multiplier sweep')
    
    args = parser.parse_args()
    
    # Initialize sweeper
    sweeper = ThresholdSweeper(args.model_dir)
    
    try:
        # Load artifacts
        sweeper.load_artifacts()
        
        # Perform sweep
        multipliers = np.arange(args.min_multiplier, args.max_multiplier + args.step, args.step)
        sweeper.sweep_thresholds(multipliers)
        
        # Find optimal thresholds
        optimal_thresholds = sweeper.find_optimal_thresholds()
        
        # Generate plots
        plot_path = Path(args.model_dir) / "figures" / "threshold_sweep_analysis.png" if args.save_plots else None
        if plot_path:
            plot_path.parent.mkdir(exist_ok=True)
        sweeper.plot_results(plot_path)
        
        # Generate and display report
        report = sweeper.generate_report(optimal_thresholds)
        print(report)
        
        # Save results
        sweeper.save_results()
        
        logger.info("Threshold sweep analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during threshold sweep: {e}")
        raise

if __name__ == "__main__":
    main()