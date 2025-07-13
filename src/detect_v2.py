# src/detect_v2.py

import os
import joblib
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    classification_report, confusion_matrix
)
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# --- 1) Configurações de diretórios ---
MODEL_DIR = 'model'
FIG_DIR = os.path.join(MODEL_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# --- 2) Carregando artefatos e configuração ---
config_path = 'config/model_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
ae = load_model(os.path.join(MODEL_DIR, 'autoencoder.h5'), compile=False)
threshold_info = joblib.load(os.path.join(MODEL_DIR, 'threshold_info.pkl'))

THRESHOLD = threshold_info['suggested_threshold']
FEATURES = config['features_used']

print(f"🔧 Loaded model, preprocessor and threshold {THRESHOLD:.6f}")
print(f"📋 Using features: {FEATURES}")

# --- 3) Carregando dados e separando labels ---
df = pd.read_csv('data/processed/flows.csv')
y_true = df['label'].values
X = df[FEATURES]

# --- 4) Pré-processamento ---
X_proc = preprocessor.transform(X)
print(f"✅ Data preprocessed: {X_proc.shape[0]} samples, {X_proc.shape[1]} features")

# --- 5) Predição e cálculo de erro de reconstrução ---
X_rec = ae.predict(X_proc, verbose=0)
errors = np.mean(np.square(X_proc - X_rec), axis=1)

# --- 6) Classificação via threshold ---
y_pred = (errors > THRESHOLD).astype(int)

# --- 7) Cálculo de métricas ---
roc_auc = roc_auc_score(y_true, errors)
fpr, tpr, _ = roc_curve(y_true, errors)
precision, recall, _ = precision_recall_curve(y_true, errors)
pr_auc = auc(recall, precision)

# Classification report e matriz de confusão
report_dict = classification_report(y_true, y_pred, target_names=['Normal','Attack'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=['Normal','Attack'], columns=['Pred_Normal','Pred_Attack'])

# Salvar relatórios
report_df.to_csv(os.path.join(FIG_DIR, 'classification_report.csv'))
cm_df.to_csv(os.path.join(FIG_DIR, 'confusion_matrix.csv'))

print(f"🔢 ROC-AUC: {roc_auc:.3f}, PR-AUC: {pr_auc:.3f}")

# --- 8) Plotagem e salvamento de figuras ---

# Histograma de erros
plt.figure(figsize=(8,6))
plt.hist(errors[y_true==0], bins=50, alpha=0.6, label='Normal')
plt.hist(errors[y_true==1], bins=50, alpha=0.6, label='Attack')
plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold={THRESHOLD:.4f}')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'error_histogram.png'))
plt.close()

# Curva ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curve.png'))
plt.close()

# Curva Precision-Recall
plt.figure(figsize=(8,6))
plt.plot(recall, precision, label=f'PR curve (AUC={pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'pr_curve.png'))
plt.close()

print(f"📊 Figures saved in {FIG_DIR}")

# --- 9) Salvar métricas gerais ---
metrics = {
    'roc_auc': roc_auc,
    'pr_auc': pr_auc,
    'threshold': THRESHOLD
}
pd.DataFrame([metrics]).to_csv(os.path.join(FIG_DIR, 'metrics_summary.csv'), index=False)

print("🎉 Detection evaluation completed!")
