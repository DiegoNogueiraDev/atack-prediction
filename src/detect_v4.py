#!/usr/bin/env python3
"""
Detection script for Model V4 - LSTM-Attention Sequence Autoencoder
Advanced temporal anomaly detection with sequence modeling
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import joblib
import logging
from sequence_preprocessor import SequencePreprocessor
import warnings
warnings.filterwarnings('ignore')

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
        # inputs shape: (batch_size, seq_len, features)
        score = tf.nn.tanh(self.W1(inputs) + self.W2(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SequenceAnomalyDetector:
    """Sequence-based anomaly detector using LSTM-Attention Autoencoder"""
    
    def __init__(self, model_dir='model_v4'):
        self.model_dir = Path(model_dir)
        self.fig_dir = self.model_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Load artifacts
        self.model = None
        self.preprocessor = None
        self.threshold_info = None
        self.config = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load all model artifacts"""
        logger.info("🔧 Loading model artifacts...")
        
        try:
            # Load model with custom objects
            model_path = self.model_dir / 'autoencoder.h5'
            custom_objects = {'AttentionLayer': AttentionLayer}
            self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
            
            # Load preprocessor
            preprocessor_path = self.model_dir / 'sequence_preprocessor.pkl'
            self.preprocessor = SequencePreprocessor.load(str(preprocessor_path))
            
            # Load threshold info
            threshold_path = self.model_dir / 'threshold_info.pkl'
            self.threshold_info = joblib.load(threshold_path)
            self.threshold = self.threshold_info['suggested_threshold']
            
            # Load config
            config_path = self.model_dir / 'model_config.pkl'
            config_data = joblib.load(config_path)
            self.config = config_data['config_used']
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"🎯 Threshold: {self.threshold:.6f}")
            logger.info(f"📋 Features: {self.config['features_used']}")
            
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame):
        """Detect anomalies in the given data"""
        logger.info("🔍 Running anomaly detection on sequences...")
        
        # Transform data to sequences
        X_sequences, y_true = self.preprocessor.transform(df, include_labels=True)
        
        if len(X_sequences) == 0:
            logger.warning("⚠️ No sequences created from input data")
            return None, None, None
        
        logger.info(f"📊 Processing {len(X_sequences)} sequences")
        logger.info(f"📈 Attack sequences in data: {np.sum(y_true)} ({np.mean(y_true):.1%})")
        
        # Get reconstruction errors
        X_reconstructed = self.model.predict(X_sequences, verbose=0)
        reconstruction_errors = np.mean(np.square(X_sequences - X_reconstructed), axis=(1, 2))
        
        # Binary predictions
        y_pred = (reconstruction_errors > self.threshold).astype(int)
        
        logger.info(f"🎯 Predictions: Normal={np.sum(y_pred==0)}, Attack={np.sum(y_pred==1)}")
        
        return reconstruction_errors, y_pred, y_true
    
    def calculate_metrics(self, y_true, y_pred, errors):
        """Calculate comprehensive performance metrics"""
        logger.info("📊 Calculating performance metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        roc_auc = roc_auc_score(y_true, errors)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, errors)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Attack'],
            output_dict=True,
            zero_division=0
        )
        
        # False Positive Rate
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'confusion_matrix': cm,
            'classification_report': report,
            'threshold': self.threshold,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        logger.info(f"🎯 RESULTS - Model V4 (LSTM-Attention Sequence):")
        logger.info(f"  • Accuracy: {accuracy:.1%}")
        logger.info(f"  • Precision: {precision:.1%}")
        logger.info(f"  • Recall: {recall:.1%}")
        logger.info(f"  • F1-Score: {f1:.1%}")
        logger.info(f"  • ROC-AUC: {roc_auc:.3f}")
        logger.info(f"  • FPR: {fpr:.1%}")
        
        return metrics
    
    def create_visualizations(self, errors, y_true, y_pred, metrics):
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Reconstruction Error Distribution
        ax1 = plt.subplot(3, 3, 1)
        normal_errors = errors[y_true == 0]
        attack_errors = errors[y_true == 1]
        
        ax1.hist(normal_errors, bins=30, alpha=0.7, label=f'Normal (n={len(normal_errors)})', 
                color='lightblue', density=True)
        ax1.hist(attack_errors, bins=20, alpha=0.7, label=f'Attack (n={len(attack_errors)})', 
                color='red', density=True)
        ax1.axvline(self.threshold, color='darkred', linestyle='--', linewidth=3,
                   label=f'Threshold: {self.threshold:.4f}')
        ax1.set_title('Reconstruction Error Distribution', fontweight='bold')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = plt.subplot(3, 3, 2)
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax2.set_title('Confusion Matrix', fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        # 3. ROC Curve
        ax3 = plt.subplot(3, 3, 3)
        fpr_roc, tpr_roc, _ = roc_curve(y_true, errors)
        ax3.plot(fpr_roc, tpr_roc, linewidth=3, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curve
        ax4 = plt.subplot(3, 3, 4)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, errors)
        ax4.plot(recall_curve, precision_curve, linewidth=3, 
                label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Metrics Comparison with Previous Models
        ax5 = plt.subplot(3, 3, 5)
        
        # Load previous model results for comparison
        comparison_data = self._load_comparison_data()
        comparison_data['Model V4 (Sequence)'] = {
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc']
        }
        
        models = list(comparison_data.keys())
        metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metric_names):
            values = [comparison_data[model][metric] for model in models]
            ax5.bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Score')
        ax5.set_title('Model Comparison', fontweight='bold')
        ax5.set_xticks(x + width * 1.5)
        ax5.set_xticklabels(models, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error Timeline (if temporal info available)
        ax6 = plt.subplot(3, 3, 6)
        sequence_ids = np.arange(len(errors))
        colors = ['red' if pred == 1 else 'blue' for pred in y_pred]
        ax6.scatter(sequence_ids, errors, c=colors, alpha=0.6, s=20)
        ax6.axhline(self.threshold, color='darkred', linestyle='--', linewidth=2)
        ax6.set_xlabel('Sequence ID')
        ax6.set_ylabel('Reconstruction Error')
        ax6.set_title('Error Timeline', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance Radar Chart
        ax7 = plt.subplot(3, 3, 7, projection='polar')
        
        # Select metrics for radar chart
        radar_metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        radar_values = [metrics['precision'], metrics['recall'], 
                       metrics['f1_score'], metrics['accuracy']]
        
        # Number of variables
        N = len(radar_metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values for each metric
        radar_values += radar_values[:1]  # Complete the circle
        
        # Plot
        ax7.plot(angles, radar_values, 'o-', linewidth=2, label='Model V4')
        ax7.fill(angles, radar_values, alpha=0.25)
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(radar_metrics)
        ax7.set_ylim(0, 1)
        ax7.set_title('Performance Radar', fontweight='bold', pad=20)
        ax7.grid(True)
        
        # 8. Sequence Length Analysis
        ax8 = plt.subplot(3, 3, 8)
        sequence_length = self.config['sequence_length']
        
        # Analyze error by position in sequence (if applicable)
        ax8.text(0.5, 0.5, f"""Model V4 Configuration:

• Architecture: LSTM-Attention
• Sequence Length: {sequence_length}
• Features: {len(self.config['features_used'])}
• Bottleneck: {self.config['architecture']['bottleneck']['units']}
• Total Sequences: {len(errors)}

Performance Highlights:
• Recall: {metrics['recall']:.1%}
• Precision: {metrics['precision']:.1%}
• FPR: {metrics['fpr']:.1%}
• ROC-AUC: {metrics['roc_auc']:.3f}""", 
                transform=ax8.transAxes, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax8.set_title('Model Configuration', fontweight='bold')
        ax8.axis('off')
        
        # 9. Comparison with Individual Models
        ax9 = plt.subplot(3, 3, 9)
        
        # Show improvement over baseline
        if 'Model V1' in comparison_data:
            baseline_recall = comparison_data['Model V1']['Recall']
            current_recall = metrics['recall']
            improvement = ((current_recall - baseline_recall) / baseline_recall) * 100
            
            improvements = [
                ('Recall', improvement),
                ('Precision', ((metrics['precision'] - comparison_data['Model V1']['Precision']) / 
                             comparison_data['Model V1']['Precision']) * 100),
                ('F1-Score', ((metrics['f1_score'] - comparison_data['Model V1']['F1-Score']) / 
                             comparison_data['Model V1']['F1-Score']) * 100),
                ('ROC-AUC', ((metrics['roc_auc'] - comparison_data['Model V1']['ROC-AUC']) / 
                            comparison_data['Model V1']['ROC-AUC']) * 100)
            ]
            
            metric_names = [imp[0] for imp in improvements]
            improvement_values = [imp[1] for imp in improvements]
            colors = ['green' if val > 0 else 'red' for val in improvement_values]
            
            bars = ax9.bar(metric_names, improvement_values, color=colors, alpha=0.7)
            ax9.set_ylabel('Improvement (%)')
            ax9.set_title('Improvement over Model V1', fontweight='bold')
            ax9.grid(True, alpha=0.3)
            ax9.axhline(0, color='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, improvement_values):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -10),
                        f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.suptitle('Model V4 - LSTM-Attention Sequence Autoencoder Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'model_v4_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualizations saved to {self.fig_dir}")
    
    def _load_comparison_data(self):
        """Load comparison data from previous models"""
        comparison_data = {}
        
        # Try to load data from previous models
        for model_name in ['model_v1', 'model_v2', 'model_v3']:
            try:
                summary_path = Path(model_name) / 'figures' / 'metrics_summary.csv'
                if summary_path.exists():
                    df = pd.read_csv(summary_path)
                    comparison_data[f'Model {model_name[-2:].upper()}'] = {
                        'Precision': df['precision'].iloc[0],
                        'Recall': df['recall'].iloc[0],
                        'F1-Score': df['f1_score'].iloc[0],
                        'ROC-AUC': df['roc_auc'].iloc[0]
                    }
            except:
                pass
        
        return comparison_data
    
    def save_results(self, metrics):
        """Save results to files"""
        logger.info("💾 Saving results...")
        
        # Save metrics summary
        summary_data = {
            'model': 'v4_sequence_lstm_attention',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'], 
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'fpr': metrics['fpr'],
            'threshold': metrics['threshold']
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(self.fig_dir / 'metrics_summary.csv', index=False)
        
        # Save confusion matrix
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'],
            index=['True_Normal', 'True_Attack'],
            columns=['Pred_Normal', 'Pred_Attack']
        )
        cm_df.to_csv(self.fig_dir / 'confusion_matrix.csv')
        
        # Save classification report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df.to_csv(self.fig_dir / 'classification_report.csv')
        
        logger.info(f"✅ Results saved to {self.fig_dir}")
    
    def run_evaluation(self, data_path='data/processed/flows.csv'):
        """Run complete evaluation"""
        logger.info("🚀 Starting Model V4 evaluation...")
        logger.info("=" * 60)
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"📂 Loaded {len(df)} flows")
            
            # Detect anomalies
            errors, y_pred, y_true = self.detect_anomalies(df)
            
            if errors is None:
                logger.error("❌ No sequences could be processed")
                return None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, errors)
            
            # Create visualizations
            self.create_visualizations(errors, y_true, y_pred, metrics)
            
            # Save results
            self.save_results(metrics)
            
            logger.info("🎉 Model V4 evaluation completed successfully!")
            logger.info(f"📁 Results saved in: {self.fig_dir}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            raise

def main():
    """Main function"""
    try:
        detector = SequenceAnomalyDetector('model_v4')
        metrics = detector.run_evaluation()
        return metrics
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        return None

if __name__ == "__main__":
    main()