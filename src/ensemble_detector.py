#!/usr/bin/env python3
"""
Ensemble Anomaly Detector
Combines Autoencoder with IsolationForest and LocalOutlierFactor for improved detection
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleAnomalyDetector:
    """Ensemble detector combining multiple algorithms"""
    
    def __init__(self, autoencoder_dir: str = "model_v1"):
        self.autoencoder_dir = Path(autoencoder_dir)
        self.results = {}
        
        # Load autoencoder components
        self.autoencoder = None
        self.preprocessor = None
        self.threshold_info = None
        self.features = None
        
        # Initialize other detectors
        self.isolation_forest = None
        self.lof_detector = None
        
        self._load_autoencoder_artifacts()
        
    def _load_autoencoder_artifacts(self):
        """Load pre-trained autoencoder and preprocessing pipeline"""
        logger.info("Loading autoencoder artifacts...")
        
        try:
            # Load preprocessor using joblib instead of pickle
            import joblib
            preprocessor_path = self.autoencoder_dir / 'preprocessor.pkl'
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Load model without compiling to avoid compatibility issues
            model_path = self.autoencoder_dir / 'autoencoder.h5'
            self.autoencoder = load_model(model_path, compile=False)
            
            # Load threshold info
            threshold_path = self.autoencoder_dir / 'threshold_info.pkl'
            self.threshold_info = joblib.load(threshold_path)
                
            # Load config to get features
            config_path = self.autoencoder_dir / 'model_config.pkl'
            config = joblib.load(config_path)
            self.features = config['config_used']['features_used']
                
            logger.info(f"Autoencoder loaded successfully")
            logger.info(f"Features: {self.features}")
            
        except Exception as e:
            logger.error(f"Error loading autoencoder artifacts: {e}")
            raise
            
    def fit_ensemble(self, X_train, contamination=0.1):
        """Fit ensemble detectors on training data"""
        logger.info("Training ensemble detectors...")
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0
        )
        self.isolation_forest.fit(X_train)
        
        # Train Local Outlier Factor
        self.lof_detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,  # Enable predict method
            algorithm='auto'
        )
        self.lof_detector.fit(X_train)
        
        logger.info("Ensemble detectors trained successfully")
        
    def predict_autoencoder(self, X):
        """Get autoencoder predictions"""
        # Reconstruct
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        
        # Calculate reconstruction error
        reconstruction_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        # Use suggested threshold
        threshold = self.threshold_info['suggested_threshold']
        predictions = (reconstruction_errors > threshold).astype(int)
        
        return predictions, reconstruction_errors
        
    def predict_isolation_forest(self, X):
        """Get Isolation Forest predictions"""
        predictions = self.isolation_forest.predict(X)
        # Convert from {-1, 1} to {1, 0} (anomaly, normal)
        predictions = (predictions == -1).astype(int)
        
        # Get decision scores (higher = more anomalous)
        scores = -self.isolation_forest.decision_function(X)
        
        return predictions, scores
        
    def predict_lof(self, X):
        """Get Local Outlier Factor predictions"""
        predictions = self.lof_detector.predict(X)
        # Convert from {-1, 1} to {1, 0} (anomaly, normal)
        predictions = (predictions == -1).astype(int)
        
        # Get decision scores (higher = more anomalous)
        scores = -self.lof_detector.decision_function(X)
        
        return predictions, scores
        
    def predict_ensemble(self, X, voting='majority'):
        """Get ensemble predictions"""
        # Get individual predictions
        ae_pred, ae_scores = self.predict_autoencoder(X)
        if_pred, if_scores = self.predict_isolation_forest(X)
        lof_pred, lof_scores = self.predict_lof(X)
        
        # Store individual results
        individual_results = {
            'autoencoder': {'predictions': ae_pred, 'scores': ae_scores},
            'isolation_forest': {'predictions': if_pred, 'scores': if_scores},
            'local_outlier_factor': {'predictions': lof_pred, 'scores': lof_scores}
        }
        
        if voting == 'majority':
            # Majority voting
            votes = ae_pred + if_pred + lof_pred
            ensemble_pred = (votes >= 2).astype(int)
            
        elif voting == 'any':
            # Any detector flags as anomaly
            ensemble_pred = np.maximum(np.maximum(ae_pred, if_pred), lof_pred)
            
        elif voting == 'all':
            # All detectors must agree
            ensemble_pred = np.minimum(np.minimum(ae_pred, if_pred), lof_pred)
            
        elif voting == 'weighted':
            # Weighted combination of scores (normalized)
            ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-8)
            if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
            lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
            
            # Equal weights for now - could be optimized
            combined_scores = (ae_scores_norm + if_scores_norm + lof_scores_norm) / 3
            threshold = np.percentile(combined_scores, 90)  # Top 10% as anomalies
            ensemble_pred = (combined_scores > threshold).astype(int)
            
        else:
            raise ValueError(f"Unknown voting method: {voting}")
            
        return ensemble_pred, individual_results
        
    def evaluate_ensemble(self, data_path='data/processed/flows.csv', voting_methods=None):
        """Evaluate ensemble performance with different voting methods"""
        if voting_methods is None:
            voting_methods = ['majority', 'any', 'all', 'weighted']
            
        logger.info("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        y_true = df['label'].values
        
        # Extract features and preprocess
        X_features = df[self.features]  # Keep as DataFrame for preprocessing
        X_processed = self.preprocessor.transform(X_features)
        
        # Split into normal (for training ensemble) and all data (for testing)
        normal_mask = (y_true == 0)
        X_normal = X_processed[normal_mask]
        
        # Train ensemble on normal data
        self.fit_ensemble(X_normal, contamination=0.1)
        
        # Evaluate on all data
        results = {}
        
        for voting in voting_methods:
            logger.info(f"Evaluating with {voting} voting...")
            
            ensemble_pred, individual_results = self.predict_ensemble(X_processed, voting=voting)
            
            # Calculate metrics
            metrics = {
                'accuracy': np.mean(ensemble_pred == y_true),
                'precision': precision_score(y_true, ensemble_pred, zero_division=0),
                'recall': recall_score(y_true, ensemble_pred, zero_division=0),
                'f1_score': f1_score(y_true, ensemble_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, ensemble_pred),
                'classification_report': classification_report(y_true, ensemble_pred, output_dict=True)
            }
            
            results[voting] = {
                'metrics': metrics,
                'predictions': ensemble_pred,
                'individual_results': individual_results
            }
            
        # Also evaluate individual detectors for comparison
        for detector_name in ['autoencoder', 'isolation_forest', 'local_outlier_factor']:
            pred = individual_results[detector_name]['predictions']
            
            metrics = {
                'accuracy': np.mean(pred == y_true),
                'precision': precision_score(y_true, pred, zero_division=0),
                'recall': recall_score(y_true, pred, zero_division=0),
                'f1_score': f1_score(y_true, pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, pred),
                'classification_report': classification_report(y_true, pred, output_dict=True)
            }
            
            results[detector_name] = {
                'metrics': metrics,
                'predictions': pred
            }
            
        self.results = results
        return results
        
    def plot_comparison(self, save_path=None):
        """Plot comparison of different methods"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ensemble Anomaly Detection Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        methods = []
        precision_vals = []
        recall_vals = []
        f1_vals = []
        accuracy_vals = []
        
        for method, result in self.results.items():
            methods.append(method.replace('_', ' ').title())
            precision_vals.append(result['metrics']['precision'])
            recall_vals.append(result['metrics']['recall'])
            f1_vals.append(result['metrics']['f1_score'])
            accuracy_vals.append(result['metrics']['accuracy'])
        
        # 1. Precision comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(methods, precision_vals, color='skyblue', alpha=0.7)
        ax1.set_title('Precision Comparison')
        ax1.set_ylabel('Precision')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(precision_vals):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Recall comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(methods, recall_vals, color='lightcoral', alpha=0.7)
        ax2.set_title('Recall Comparison')
        ax2.set_ylabel('Recall')
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(recall_vals):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. F1-Score comparison
        ax3 = axes[0, 2]
        bars = ax3.bar(methods, f1_vals, color='lightgreen', alpha=0.7)
        ax3.set_title('F1-Score Comparison')
        ax3.set_ylabel('F1-Score')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(f1_vals):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. Accuracy comparison
        ax4 = axes[1, 0]
        bars = ax4.bar(methods, accuracy_vals, color='gold', alpha=0.7)
        ax4.set_title('Accuracy Comparison')
        ax4.set_ylabel('Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        for i, v in enumerate(accuracy_vals):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 5. Precision vs Recall scatter
        ax5 = axes[1, 1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        for i, method in enumerate(methods):
            ax5.scatter(recall_vals[i], precision_vals[i], 
                       s=100, c=[colors[i]], label=method, alpha=0.7)
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision vs Recall')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Confusion matrix for best F1 method
        best_method = methods[np.argmax(f1_vals)]
        best_key = list(self.results.keys())[np.argmax(f1_vals)]
        cm = self.results[best_key]['metrics']['confusion_matrix']
        
        ax6 = axes[1, 2]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax6.set_title(f'Best Method: {best_method}\nF1-Score: {max(f1_vals):.3f}')
        ax6.set_ylabel('True Label')
        ax6.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
    def generate_report(self):
        """Generate detailed comparison report"""
        report = "\n" + "="*80 + "\n"
        report += "ENSEMBLE ANOMALY DETECTION REPORT\n"
        report += "="*80 + "\n\n"
        
        # Individual detector performance
        report += "INDIVIDUAL DETECTOR PERFORMANCE:\n"
        report += "-" * 40 + "\n"
        
        individual_methods = ['autoencoder', 'isolation_forest', 'local_outlier_factor']
        for method in individual_methods:
            if method in self.results:
                metrics = self.results[method]['metrics']
                report += f"\n{method.upper().replace('_', ' ')}:\n"
                report += f"  Precision: {metrics['precision']:.1%}\n"
                report += f"  Recall: {metrics['recall']:.1%}\n"
                report += f"  F1-Score: {metrics['f1_score']:.1%}\n"
                report += f"  Accuracy: {metrics['accuracy']:.1%}\n"
        
        # Ensemble performance
        report += "\n\nENSEMBLE PERFORMANCE:\n"
        report += "-" * 40 + "\n"
        
        ensemble_methods = ['majority', 'any', 'all', 'weighted']
        best_f1 = 0
        best_method = None
        
        for method in ensemble_methods:
            if method in self.results:
                metrics = self.results[method]['metrics']
                report += f"\n{method.upper()} VOTING:\n"
                report += f"  Precision: {metrics['precision']:.1%}\n"
                report += f"  Recall: {metrics['recall']:.1%}\n"
                report += f"  F1-Score: {metrics['f1_score']:.1%}\n"
                report += f"  Accuracy: {metrics['accuracy']:.1%}\n"
                
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_method = method
        
        # Recommendations
        report += "\n\nRECOMMENDATIONS:\n"
        report += "-" * 40 + "\n"
        
        if best_method:
            report += f"• Best ensemble method: {best_method.upper()} voting "
            report += f"(F1-Score: {best_f1:.1%})\n"
        
        # Compare with autoencoder alone
        if 'autoencoder' in self.results and best_method in self.results:
            ae_f1 = self.results['autoencoder']['metrics']['f1_score']
            ensemble_f1 = self.results[best_method]['metrics']['f1_score']
            improvement = ((ensemble_f1 - ae_f1) / ae_f1) * 100
            
            report += f"• Improvement over autoencoder alone: {improvement:+.1f}%\n"
            
            ae_recall = self.results['autoencoder']['metrics']['recall']
            ensemble_recall = self.results[best_method]['metrics']['recall']
            recall_improvement = ((ensemble_recall - ae_recall) / ae_recall) * 100
            
            report += f"• Recall improvement: {recall_improvement:+.1f}%\n"
        
        return report
        
    def save_results(self, save_dir=None):
        """Save ensemble results"""
        if save_dir is None:
            save_dir = self.autoencoder_dir / "ensemble_results"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True)
        
        # Save full results
        results_path = save_dir / "ensemble_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary CSV
        summary_data = []
        for method, result in self.results.items():
            metrics = result['metrics']
            summary_data.append({
                'method': method,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = save_dir / "ensemble_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Results saved to {save_dir}")
        
        return summary_df

def main():
    parser = argparse.ArgumentParser(description='Run ensemble anomaly detection')
    parser.add_argument('--autoencoder-dir', default='model_v1', help='Autoencoder model directory')
    parser.add_argument('--data-path', default='data/processed/flows.csv', help='Data file path')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to file')
    parser.add_argument('--voting-methods', nargs='+', 
                       default=['majority', 'any', 'all', 'weighted'],
                       help='Voting methods to evaluate')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = EnsembleAnomalyDetector(args.autoencoder_dir)
    
    try:
        # Run evaluation
        results = detector.evaluate_ensemble(args.data_path, args.voting_methods)
        
        # Generate plots
        plot_path = Path(args.autoencoder_dir) / "ensemble_results" / "comparison.png" if args.save_plots else None
        if plot_path:
            plot_path.parent.mkdir(exist_ok=True)
        detector.plot_comparison(plot_path)
        
        # Generate and display report
        report = detector.generate_report()
        print(report)
        
        # Save results
        summary_df = detector.save_results()
        
        logger.info("Ensemble evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ensemble evaluation: {e}")
        raise

if __name__ == "__main__":
    main()