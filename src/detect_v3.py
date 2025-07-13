# src/detect_v3.py - Versão melhorada para detecção de anomalias
# Melhorias: modularização, tratamento de erros, métricas avançadas, plots profissionais

import os
import joblib
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AnomalyDetector:
    """Classe para detecção de anomalias usando autoencoder treinado"""
    
    def __init__(self, model_dir='model', config_path=None):
        self.model_dir = Path(model_dir)
        self.fig_dir = self.model_dir / 'figures'
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar configuração
        self.config = self._load_config(config_path)
        
        # Carregar artefatos do modelo
        self.preprocessor = None
        self.autoencoder = None
        self.threshold_info = None
        self.threshold = None
        self.features = None
        
        self._load_model_artifacts()
        
    def _load_config(self, config_path):
        """Carregar configuração YAML ou do modelo salvo"""
        # Se config_path não especificado, tentar carregar do modelo
        if config_path is None:
            model_config_path = self.model_dir / 'model_config.pkl'
            if model_config_path.exists():
                try:
                    import pickle
                    with open(model_config_path, 'rb') as f:
                        saved_config = pickle.load(f)
                    print(f"✅ Configuração carregada do modelo salvo em {model_config_path}")
                    return saved_config['config_used']
                except Exception as e:
                    print(f"❌ Erro ao carregar config do modelo: {e}")
                    config_path = 'config/model_config.yaml'  # fallback
            else:
                config_path = 'config/model_config.yaml'  # fallback
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuração carregada de {config_path}")
            return config
        except Exception as e:
            print(f"❌ Erro ao carregar configuração: {e}")
            raise
    
    def _load_model_artifacts(self):
        """Carregar todos os artefatos do modelo"""
        try:
            # Preprocessor
            preprocessor_path = self.model_dir / 'preprocessor.pkl'
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Autoencoder
            model_path = self.model_dir / 'autoencoder.h5'
            self.autoencoder = load_model(model_path, compile=False)
            
            # Threshold info
            threshold_path = self.model_dir / 'threshold_info.pkl'
            self.threshold_info = joblib.load(threshold_path)
            self.threshold = self.threshold_info['suggested_threshold']
            
            # Features
            self.features = self.config['features_used']
            
            print(f"🔧 Modelo carregado com threshold: {self.threshold:.6f}")
            print(f"📋 Features utilizadas: {self.features}")
            
        except Exception as e:
            print(f"❌ Erro ao carregar artefatos do modelo: {e}")
            raise
    
    def load_data(self, data_path='data/processed/flows.csv'):
        """Carregar dados para detecção"""
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Dados carregados: {df.shape[0]} amostras, {df.shape[1]} features")
            
            # Verificar se existem as colunas necessárias
            if 'label' not in df.columns:
                raise ValueError("Coluna 'label' não encontrada nos dados")
            
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Features não encontradas: {missing_features}")
            
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def preprocess_data(self, df):
        """Pré-processar dados usando o preprocessor treinado"""
        try:
            X = df[self.features]
            X_proc = self.preprocessor.transform(X)
            y_true = df['label'].values
            
            print(f"✅ Dados pré-processados: {X_proc.shape}")
            print(f"📊 Distribuição de labels: Normal={np.sum(y_true==0)}, Attack={np.sum(y_true==1)}")
            
            return X_proc, y_true
        except Exception as e:
            print(f"❌ Erro no pré-processamento: {e}")
            raise
    
    def predict_anomalies(self, X_proc):
        """Detectar anomalias usando o autoencoder"""
        try:
            # Reconstrução
            X_rec = self.autoencoder.predict(X_proc, verbose=0)
            
            # Calcular erros de reconstrução
            errors = np.mean(np.square(X_proc - X_rec), axis=1)
            
            # Classificação binária
            y_pred = (errors > self.threshold).astype(int)
            
            print(f"✅ Predições realizadas")
            print(f"📊 Detecções: Normal={np.sum(y_pred==0)}, Attack={np.sum(y_pred==1)}")
            
            return errors, y_pred
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            raise
    
    def calculate_metrics(self, y_true, y_pred, errors):
        """Calcular métricas de desempenho"""
        try:
            # Métricas básicas
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Métricas de área sob a curva
            roc_auc = roc_auc_score(y_true, errors)
            
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, errors)
            pr_auc = auc(recall_curve, precision_curve)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, errors)
            
            # Classification report
            report_dict = classification_report(
                y_true, y_pred, 
                target_names=['Normal', 'Attack'], 
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'threshold': self.threshold,
                'fpr': fpr,
                'tpr': tpr,
                'precision_curve': precision_curve,
                'recall_curve': recall_curve,
                'classification_report': report_dict,
                'confusion_matrix': cm
            }
            
            print(f"🔢 Métricas calculadas:")
            print(f"  • Accuracy: {accuracy:.3f}")
            print(f"  • Precision: {precision:.3f}")
            print(f"  • Recall: {recall:.3f}")
            print(f"  • F1-Score: {f1:.3f}")
            print(f"  • ROC-AUC: {roc_auc:.3f}")
            print(f"  • PR-AUC: {pr_auc:.3f}")
            
            return metrics
        except Exception as e:
            print(f"❌ Erro no cálculo de métricas: {e}")
            raise
    
    def save_reports(self, metrics):
        """Salvar relatórios em CSV"""
        try:
            # Classification report
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            report_path = self.fig_dir / 'classification_report.csv'
            report_df.to_csv(report_path)
            
            # Confusion matrix
            cm_df = pd.DataFrame(
                metrics['confusion_matrix'],
                index=['True_Normal', 'True_Attack'],
                columns=['Pred_Normal', 'Pred_Attack']
            )
            cm_path = self.fig_dir / 'confusion_matrix.csv'
            cm_df.to_csv(cm_path)
            
            # Metrics summary
            summary_metrics = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'threshold': metrics['threshold']
            }
            summary_df = pd.DataFrame([summary_metrics])
            summary_path = self.fig_dir / 'metrics_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            
            print(f"✅ Relatórios salvos em {self.fig_dir}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar relatórios: {e}")
            raise
    
    def save_test_artifacts(self, errors, y_true):
        """Salvar artifacts para threshold sweep"""
        import pickle
        try:
            # Salvar errors e labels para threshold sweep
            errors_path = self.model_dir / 'test_errors.pkl'
            labels_path = self.model_dir / 'test_labels.pkl'
            
            with open(errors_path, 'wb') as f:
                pickle.dump(errors, f)
            with open(labels_path, 'wb') as f:
                pickle.dump(y_true, f)
                
            print(f"✅ Test artifacts salvos para threshold sweep")
            
        except Exception as e:
            print(f"❌ Erro ao salvar test artifacts: {e}")
            raise
    
    def create_plots(self, y_true, errors, metrics):
        """Criar visualizações profissionais"""
        try:
            # 1. Error histogram
            self._plot_error_histogram(y_true, errors)
            
            # 2. ROC curve
            self._plot_roc_curve(metrics)
            
            # 3. Precision-Recall curve
            self._plot_pr_curve(metrics)
            
            # 4. Confusion matrix heatmap
            self._plot_confusion_matrix(metrics)
            
            # 5. Metrics dashboard
            self._plot_metrics_dashboard(metrics)
            
            print(f"📊 Visualizações salvas em {self.fig_dir}")
            
        except Exception as e:
            print(f"❌ Erro ao criar plots: {e}")
            raise
    
    def _plot_error_histogram(self, y_true, errors):
        """Histograma de erros de reconstrução"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Separar erros por classe
        normal_errors = errors[y_true == 0]
        attack_errors = errors[y_true == 1]
        
        # Histogramas
        ax.hist(normal_errors, bins=50, alpha=0.7, label=f'Normal (n={len(normal_errors)})', 
                color='lightblue', density=True)
        ax.hist(attack_errors, bins=30, alpha=0.7, label=f'Attack (n={len(attack_errors)})', 
                color='red', density=True)
        
        # Threshold
        ax.axvline(self.threshold, color='darkred', linestyle='--', linewidth=3,
                   label=f'Threshold: {self.threshold:.4f}')
        
        # Estatísticas
        ax.axvline(np.mean(normal_errors), color='blue', linestyle=':', alpha=0.8,
                   label=f'Normal Mean: {np.mean(normal_errors):.4f}')
        ax.axvline(np.mean(attack_errors), color='red', linestyle=':', alpha=0.8,
                   label=f'Attack Mean: {np.mean(attack_errors):.4f}')
        
        ax.set_title('Reconstruction Error Distribution\nNormal vs Attack Traffic', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Reconstruction Error (MSE)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'error_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, metrics):
        """Curva ROC"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.plot(metrics['fpr'], metrics['tpr'], linewidth=3, 
                label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curve - Anomaly Detection Performance', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, metrics):
        """Curva Precision-Recall"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.plot(metrics['recall_curve'], metrics['precision_curve'], linewidth=3,
                label=f'PR Curve (AUC = {metrics["pr_auc"]:.3f})')
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curve - Anomaly Detection', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, metrics):
        """Matriz de confusão como heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred Normal', 'Pred Attack'],
                    yticklabels=['True Normal', 'True Attack'],
                    ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_dashboard(self, metrics):
        """Dashboard com todas as métricas principais"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Métricas principais (barplot)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score']]
        
        bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_title('Classification Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Threshold analysis
        thresholds = [metrics['threshold'] * i for i in [0.5, 0.75, 1.0, 1.25, 1.5]]
        threshold_names = ['0.5×T', '0.75×T', 'T', '1.25×T', '1.5×T']
        
        ax2.bar(threshold_names, thresholds, color='lightcoral')
        ax2.axhline(metrics['threshold'], color='red', linestyle='--', linewidth=2, 
                   label=f'Selected: {metrics["threshold"]:.4f}')
        ax2.set_title('Threshold Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Threshold Value', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. AUC comparison
        auc_names = ['ROC-AUC', 'PR-AUC']
        auc_values = [metrics['roc_auc'], metrics['pr_auc']]
        
        bars = ax3.bar(auc_names, auc_values, color=['steelblue', 'darkorange'])
        ax3.set_title('Area Under Curve Metrics', fontsize=14, fontweight='bold')
        ax3.set_ylabel('AUC Score', fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, auc_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Model info
        ax4.axis('off')
        info_text = f"""
        Model Configuration:
        
        • Features: {', '.join(self.features)}
        • Threshold: {metrics['threshold']:.6f}
        • Architecture: Autoencoder
        • Total Samples: {metrics['confusion_matrix'].sum()}
        
        Performance Summary:
        • Best Metric: ROC-AUC = {metrics['roc_auc']:.3f}
        • Detection Rate: {metrics['recall']:.1%}
        • False Positive Rate: {1-metrics['precision']:.1%}
        """
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Anomaly Detection - Performance Dashboard', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'metrics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_evaluation(self, data_path='data/processed/flows.csv'):
        """Executar avaliação completa"""
        print("🚀 Iniciando avaliação completa de detecção de anomalias")
        print("=" * 60)
        
        try:
            # 1. Carregar dados
            df = self.load_data(data_path)
            
            # 2. Pré-processar
            X_proc, y_true = self.preprocess_data(df)
            
            # 3. Detectar anomalias
            errors, y_pred = self.predict_anomalies(X_proc)
            
            # 4. Calcular métricas
            metrics = self.calculate_metrics(y_true, y_pred, errors)
            
            # 5. Salvar relatórios
            self.save_reports(metrics)
            
            # 6. Criar visualizações
            self.create_plots(y_true, errors, metrics)
            
            # 7. Salvar artifacts para threshold sweep
            self.save_test_artifacts(errors, y_true)
            
            print("\n🎉 Avaliação completa finalizada com sucesso!")
            print(f"📁 Resultados salvos em: {self.fig_dir}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Erro na avaliação: {e}")
            raise

def main():
    """Função principal"""
    try:
        # Criar detector
        detector = AnomalyDetector(model_dir='model_v3')
        
        # Executar avaliação completa
        metrics = detector.run_full_evaluation()
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        return None

if __name__ == "__main__":
    main()