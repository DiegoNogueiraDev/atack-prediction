#!/usr/bin/env python3
"""
Threshold Sweep for Model V4 - LSTM-Attention Sequence Autoencoder
Find optimal threshold for sequence-based anomaly detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging
from sequence_preprocessor import SequencePreprocessor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define custom AttentionLayer for model loading
class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer for sequence autoencoder"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        
    def call(self, inputs):
        score = tf.nn.tanh(self.W1(inputs) + self.W2(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

class SequenceThresholdSweeper:
    """Threshold sweeper for sequence-based anomaly detection"""
    
    def __init__(self, model_dir='model_v4'):
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.config = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load model artifacts"""
        logger.info("🔧 Loading model artifacts...")
        
        # Load model
        model_path = self.model_dir / 'autoencoder.h5'
        custom_objects = {'AttentionLayer': AttentionLayer}
        self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Load preprocessor
        preprocessor_path = self.model_dir / 'sequence_preprocessor.pkl'
        self.preprocessor = SequencePreprocessor.load(str(preprocessor_path))
        
        # Load config
        config_path = self.model_dir / 'model_config.pkl'
        config_data = joblib.load(config_path)
        self.config = config_data['config_used']
        
        logger.info("✅ Model artifacts loaded successfully")
    
    def get_reconstruction_errors(self, df):
        """Get reconstruction errors for all sequences"""
        logger.info("🔄 Getting reconstruction errors...")
        
        # Transform to sequences
        X_sequences, y_true = self.preprocessor.transform(df, include_labels=True)
        
        if len(X_sequences) == 0:
            raise ValueError("No sequences created from data")
        
        # Get reconstruction errors
        X_reconstructed = self.model.predict(X_sequences, verbose=0)
        errors = np.mean(np.square(X_sequences - X_reconstructed), axis=(1, 2))
        
        logger.info(f"📊 Processed {len(errors)} sequences")
        logger.info(f"📈 Attack sequences: {np.sum(y_true)} ({np.mean(y_true):.1%})")
        
        return errors, y_true
    
    def sweep_thresholds(self, errors, y_true, multipliers=None):
        """Sweep through different threshold values"""
        if multipliers is None:
            # Test a wide range of multipliers for sequence data
            multipliers = np.arange(0.1, 2.0, 0.05)
        
        logger.info(f"🔍 Testing {len(multipliers)} threshold values...")
        
        # Base threshold (we'll use percentiles instead of the original threshold)
        base_threshold = np.percentile(errors, 95)  # Start with 95th percentile
        
        results = []
        
        for multiplier in multipliers:
            threshold = base_threshold * multiplier
            predictions = (errors > threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, predictions, zero_division=0)
            recall = recall_score(y_true, predictions, zero_division=0)
            f1 = f1_score(y_true, predictions, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, predictions)
            tn, fp, fn, tp = cm.ravel()
            
            # Additional metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Precision-Recall balance
            pr_balance = abs(precision - recall) if precision > 0 and recall > 0 else float('inf')
            
            results.append({
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
                'fn': fn,
                'pr_balance': pr_balance
            })
        
        results_df = pd.DataFrame(results)
        logger.info("✅ Threshold sweep completed")
        
        return results_df
    
    def find_optimal_thresholds(self, results_df):
        """Find optimal thresholds for different criteria"""
        
        # Best F1-score
        best_f1_idx = results_df['f1_score'].idxmax()
        best_f1 = results_df.loc[best_f1_idx]
        
        # High recall with low FPR (recall >= 50%, minimize FPR)
        high_recall_mask = results_df['recall'] >= 0.5
        if high_recall_mask.any():
            high_recall_candidates = results_df[high_recall_mask]
            high_recall_idx = high_recall_candidates['fpr'].idxmin()
            high_recall_low_fpr = results_df.loc[high_recall_idx]
        else:
            # If no 50%+ recall, get best recall with FPR <= 20%
            low_fpr_mask = results_df['fpr'] <= 0.2
            if low_fpr_mask.any():
                low_fpr_candidates = results_df[low_fpr_mask]
                high_recall_idx = low_fpr_candidates['recall'].idxmax()
                high_recall_low_fpr = results_df.loc[high_recall_idx]
            else:
                high_recall_low_fpr = best_f1  # Fallback
        
        # Balanced precision-recall
        valid_pr_mask = (results_df['precision'] > 0) & (results_df['recall'] > 0)
        if valid_pr_mask.any():
            valid_pr_results = results_df[valid_pr_mask]
            balanced_idx = valid_pr_results['pr_balance'].idxmin()
            balanced_pr = results_df.loc[balanced_idx]
        else:
            balanced_pr = best_f1  # Fallback
        
        optimal_thresholds = {
            'best_f1': best_f1,
            'high_recall_low_fpr': high_recall_low_fpr,
            'balanced_pr': balanced_pr
        }
        
        # Log results
        logger.info("🎯 OPTIMAL THRESHOLDS FOUND:")
        logger.info("-" * 50)
        for name, result in optimal_thresholds.items():
            logger.info(f"{name.upper()}:")
            logger.info(f"  Threshold: {result['threshold']:.6f}")
            logger.info(f"  Precision: {result['precision']:.1%}")
            logger.info(f"  Recall: {result['recall']:.1%}")
            logger.info(f"  F1-Score: {result['f1_score']:.1%}")
            logger.info(f"  FPR: {result['fpr']:.1%}")
            logger.info("")
        
        return optimal_thresholds
    
    def create_analysis_plots(self, results_df, optimal_thresholds, save_dir):
        """Create threshold analysis visualizations"""
        logger.info("📊 Creating threshold analysis plots...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model V4 - Sequence Threshold Sweep Analysis', fontsize=16, fontweight='bold')
        
        # 1. Metrics vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['recall'], 'r-', label='Recall', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['f1_score'], 'g-', label='F1-Score', linewidth=2)
        
        # Mark optimal thresholds
        for name, result in optimal_thresholds.items():
            ax1.axvline(result['threshold'], linestyle='--', alpha=0.7, 
                       label=f'{name}: {result["threshold"]:.3f}')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        ax2 = axes[0, 1]
        valid_mask = (results_df['precision'] > 0) | (results_df['recall'] > 0)
        valid_results = results_df[valid_mask]
        
        # Sort by recall for smooth curve
        valid_results = valid_results.sort_values('recall')
        ax2.plot(valid_results['recall'], valid_results['precision'], 'b-', linewidth=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        # 3. FPR vs Recall Trade-off
        ax3 = axes[0, 2]
        ax3.plot(results_df['fpr'], results_df['recall'], 'r-', linewidth=2)
        ax3.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Target Recall ≥ 50%')
        ax3.axvline(0.2, color='orange', linestyle='--', alpha=0.7, label='Max FPR ≤ 20%')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('Recall')
        ax3.set_title('FPR vs Recall Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Threshold vs FPR
        ax4 = axes[1, 0]
        ax4.plot(results_df['threshold'], results_df['fpr'], 'orange', linewidth=2)
        ax4.axhline(0.2, color='red', linestyle='--', alpha=0.7, label='Max FPR ≤ 20%')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('False Positive Rate')
        ax4.set_title('Threshold vs FPR')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. F1-Score Heatmap (Recall vs FPR)
        ax5 = axes[1, 1]
        
        # Create bins for heatmap
        recall_bins = np.linspace(0, 1, 11)
        fpr_bins = np.linspace(0, 1, 11)
        
        # Create heatmap data
        heatmap_data = np.zeros((len(recall_bins)-1, len(fpr_bins)-1))
        
        for i, (r1, r2) in enumerate(zip(recall_bins[:-1], recall_bins[1:])):
            for j, (f1, f2) in enumerate(zip(fpr_bins[:-1], fpr_bins[1:])):
                mask = ((results_df['recall'] >= r1) & (results_df['recall'] < r2) & 
                       (results_df['fpr'] >= f1) & (results_df['fpr'] < f2))
                if mask.any():
                    heatmap_data[i, j] = results_df[mask]['f1_score'].max()
        
        im = ax5.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
        ax5.set_xlabel('FPR Bins')
        ax5.set_ylabel('Recall Bins')
        ax5.set_title('F1-Score Heatmap (Recall vs FPR)')
        plt.colorbar(im, ax=ax5, label='F1-Score')
        
        # 6. Confusion Matrix for Best F1
        ax6 = axes[1, 2]
        best_f1_result = optimal_thresholds['best_f1']
        cm_data = [[best_f1_result['tn'], best_f1_result['fp']], 
                   [best_f1_result['fn'], best_f1_result['tp']]]
        
        cm_data_int = [[int(best_f1_result['tn']), int(best_f1_result['fp'])], 
                       [int(best_f1_result['fn']), int(best_f1_result['tp'])]]
        sns.heatmap(cm_data_int, annot=True, fmt='d', cmap='Blues', ax=ax6,
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax6.set_title(f'Confusion Matrix (Best F1)\nThreshold: {best_f1_result["threshold"]:.4f}')
        ax6.set_ylabel('True Label')
        ax6.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'threshold_sweep_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Plots saved to {save_dir}")
    
    def save_results(self, results_df, optimal_thresholds, save_dir):
        """Save threshold sweep results"""
        save_dir = Path(save_dir)
        
        # Save full results
        results_df.to_csv(save_dir / 'threshold_sweep_results.csv', index=False)
        
        # Save optimal thresholds
        joblib.dump(optimal_thresholds, save_dir / 'optimal_thresholds.pkl')
        
        logger.info(f"💾 Results saved to {save_dir}")
    
    def run_threshold_sweep(self, data_path='data/processed/flows.csv'):
        """Run complete threshold sweep analysis"""
        logger.info("🚀 Starting threshold sweep for Model V4...")
        logger.info("=" * 60)
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"📂 Loaded {len(df)} flows")
            
            # Get reconstruction errors
            errors, y_true = self.get_reconstruction_errors(df)
            
            # Sweep thresholds
            results_df = self.sweep_thresholds(errors, y_true)
            
            # Find optimal thresholds
            optimal_thresholds = self.find_optimal_thresholds(results_df)
            
            # Create visualizations
            save_dir = self.model_dir / 'figures'
            self.create_analysis_plots(results_df, optimal_thresholds, save_dir)
            
            # Save results
            self.save_results(results_df, optimal_thresholds, save_dir)
            
            logger.info("🎉 Threshold sweep completed successfully!")
            
            return results_df, optimal_thresholds
            
        except Exception as e:
            logger.error(f"❌ Error during threshold sweep: {e}")
            raise

def main():
    """Main function"""
    try:
        sweeper = SequenceThresholdSweeper('model_v4')
        results_df, optimal_thresholds = sweeper.run_threshold_sweep()
        
        # Print summary
        print("\n🎯 THRESHOLD SWEEP SUMMARY:")
        print("=" * 50)
        
        best_f1 = optimal_thresholds['best_f1']
        high_recall = optimal_thresholds['high_recall_low_fpr']
        
        print(f"Best F1-Score Configuration:")
        print(f"  Threshold: {best_f1['threshold']:.6f}")
        print(f"  Precision: {best_f1['precision']:.1%}")
        print(f"  Recall: {best_f1['recall']:.1%}")
        print(f"  F1-Score: {best_f1['f1_score']:.1%}")
        print(f"  FPR: {best_f1['fpr']:.1%}")
        print()
        
        print(f"High Recall Configuration:")
        print(f"  Threshold: {high_recall['threshold']:.6f}")
        print(f"  Precision: {high_recall['precision']:.1%}")
        print(f"  Recall: {high_recall['recall']:.1%}")
        print(f"  F1-Score: {high_recall['f1_score']:.1%}")
        print(f"  FPR: {high_recall['fpr']:.1%}")
        
        return results_df, optimal_thresholds
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        return None, None

if __name__ == "__main__":
    main()