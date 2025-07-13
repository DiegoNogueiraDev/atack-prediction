# src/train_autoencoder.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.preprocessor import fit_preprocessor, transform_preprocessor

# --- 1) Configurações ---
DATA_PATH = 'data/processed/flows.csv'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 2) Carrega e filtra dados ---
print("📊 Carregando dados...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset carregado: {df.shape[0]} fluxos, {df.shape[1]} features")

# Baseado no EDA: Filtrar apenas tráfego normal para treinamento
df_norm = df[df.label == 0].reset_index(drop=True)
print(f"Tráfego normal para treinamento: {df_norm.shape[0]} fluxos")

# Features numéricas baseadas no EDA
numeric_features = ['bytes', 'pkts', 'duration', 'iat_mean', 'iat_std']

# Remover features não significativas baseado no EDA (opção configurável)
REMOVE_NON_SIGNIFICANT = True
if REMOVE_NON_SIGNIFICANT:
    # Baseado nos testes estatísticos do EDA: duration e iat_std não são significativas
    features_to_keep = ['bytes', 'pkts', 'iat_mean']  # Apenas features significativas
    print(f"⚠️ Removendo features não significativas. Usando apenas: {features_to_keep}")
else:
    features_to_keep = numeric_features
    print(f"✅ Usando todas as features: {features_to_keep}")

# Filtrar apenas features relevantes
df_norm = df_norm[features_to_keep + ['label']].copy()

# --- 3) Pré-processamento avançado baseado no EDA ---
print("🔧 Aplicando pré-processamento avançado...")

# Tratar valores ausentes primeiro (baseado no EDA: 41% missing em iat_std)
print("📋 Tratando valores ausentes...")
missing_before = df_norm[features_to_keep].isnull().sum().sum()
print(f"Valores ausentes antes: {missing_before}")

if missing_before > 0:
    imputer = SimpleImputer(strategy='median')
    df_features = pd.DataFrame(
        imputer.fit_transform(df_norm[features_to_keep]),
        columns=features_to_keep
    )
    # Salvar imputer para uso posterior
    joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))
else:
    df_features = df_norm[features_to_keep].copy()

print(f"Valores ausentes após imputação: {df_features.isnull().sum().sum()}")

# Aplicar transformações para reduzir skewness (baseado no EDA)
APPLY_TRANSFORMATIONS = True
if APPLY_TRANSFORMATIONS:
    print("🔄 Aplicando transformações para reduzir assimetria...")
    
    # Baseado no EDA: bytes e pkts têm alta assimetria
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    
    # Aplicar transformação
    df_transformed = pd.DataFrame(
        transformer.fit_transform(df_features),
        columns=features_to_keep
    )
    
    # Salvar transformer
    joblib.dump(transformer, os.path.join(MODEL_DIR, 'power_transformer.pkl'))
    
    # Verificar melhoria na assimetria
    original_skew = df_features.skew().abs().mean()
    transformed_skew = df_transformed.skew().abs().mean()
    print(f"Skewness média: {original_skew:.3f} → {transformed_skew:.3f}")
    
    df_features = df_transformed

# Usar o preprocessor existente para normalização final
prep = fit_preprocessor(df_features)
X = transform_preprocessor(df_features)

# Split treino/validação com estratificação baseada em outliers (baseado no EDA)
from sklearn.model_selection import train_test_split

print("✂️ Dividindo dados em treino/validação...")

# Detectar outliers para estratificação (baseado no EDA)
REMOVE_OUTLIERS = False  # Configurável
if REMOVE_OUTLIERS:
    print("🔍 Removendo outliers extremos do conjunto de treino...")
    from scipy import stats
    
    # Remover apenas outliers extremos (z-score > 4)
    z_scores = np.abs(stats.zscore(X, axis=0))
    outlier_mask = (z_scores < 4).all(axis=1)
    X_clean = X[outlier_mask]
    print(f"Amostras após remoção de outliers: {X_clean.shape[0]} de {X.shape[0]} ({X_clean.shape[0]/X.shape[0]*100:.1f}%)")
    X = X_clean

# Split estratificado
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Validação: {X_val.shape[0]} amostras")

# Salvar todos os preprocessors
joblib.dump(prep, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
print("💾 Preprocessors salvos em model/")

# --- 4) Arquitetura do Autoencoder otimizada baseada no EDA ---
input_dim = X_train.shape[1]
print(f"🏗️ Construindo autoencoder para {input_dim} features...")

# Arquitetura adaptativa baseada no número de features
if input_dim <= 3:
    # Para poucas features (quando removemos as não significativas)
    encoder_dims = [max(16, input_dim * 4), max(8, input_dim * 2)]
    bottleneck_dim = max(4, input_dim)
elif input_dim <= 5:
    # Para 4-5 features (todas as features)
    encoder_dims = [32, 16]
    bottleneck_dim = 8
else:
    # Para mais features (caso futuro)
    encoder_dims = [64, 32]
    bottleneck_dim = 16

print(f"Arquitetura: {input_dim} → {encoder_dims[0]} → {encoder_dims[1]} → {bottleneck_dim} → {encoder_dims[1]} → {encoder_dims[0]} → {input_dim}")

# Construir encoder
inp = layers.Input(shape=(input_dim,), name='input')
x = layers.Dense(encoder_dims[0], activation='relu', name='encoder_1')(inp)
x = layers.Dropout(0.2, name='dropout_1')(x)  # Regularização baseada no potencial overfitting
x = layers.Dense(encoder_dims[1], activation='relu', name='encoder_2')(x)
x = layers.Dropout(0.1, name='dropout_2')(x)
bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)

# Construir decoder (simétrico)
x = layers.Dense(encoder_dims[1], activation='relu', name='decoder_1')(bottleneck)
x = layers.Dropout(0.1, name='dropout_3')(x)
x = layers.Dense(encoder_dims[0], activation='relu', name='decoder_2')(x)
x = layers.Dropout(0.2, name='dropout_4')(x)
out = layers.Dense(input_dim, activation='linear', name='output')(x)

# Criar modelo
ae = Model(inp, out, name='autoencoder')

# Compilar com configurações otimizadas
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
ae.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print("✅ Autoencoder construído e compilado")
print(ae.summary())

# --- 5) Callbacks otimizados ---
print("⚙️ Configurando callbacks...")

# Early stopping mais refinado
es = EarlyStopping(
    monitor='val_loss',
    patience=15,  # Mais paciência devido ao dataset pequeno
    restore_best_weights=True,
    verbose=1,
    min_delta=1e-5
)

# Model checkpoint
mc = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'autoencoder.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Redução de learning rate
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduzir LR pela metade
    patience=8,   # Aguardar 8 épocas sem melhoria
    min_lr=1e-6,
    verbose=1
)

callbacks = [es, mc, rlr]
print(f"✅ {len(callbacks)} callbacks configurados")

# --- 6) Treinamento otimizado ---
print("🚀 Iniciando treinamento...")

# Batch size adaptativo baseado no tamanho do dataset
train_size = X_train.shape[0]
if train_size < 1000:
    batch_size = 16  # Batch menor para datasets pequenos
elif train_size < 5000:
    batch_size = 32
else:
    batch_size = 64

print(f"Tamanho do lote: {batch_size}")
print(f"Épocas máximas: 150")

# Treinamento
history = ae.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=150,  # Mais épocas com early stopping
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

print("✅ Treinamento concluído!")

# --- 7) Pós-processamento e análise ---
print("📊 Salvando resultados...")

# Salvar histórico de treinamento
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(MODEL_DIR, 'history.csv'), index=False)

# Estatísticas do treinamento
final_train_loss = history_df['loss'].iloc[-1]
final_val_loss = history_df['val_loss'].iloc[-1]
best_val_loss = history_df['val_loss'].min()
epochs_trained = len(history_df)

print(f"📈 Estatísticas do Treinamento:")
print(f"  • Épocas treinadas: {epochs_trained}")
print(f"  • Loss final (treino): {final_train_loss:.6f}")
print(f"  • Loss final (validação): {final_val_loss:.6f}")
print(f"  • Melhor loss (validação): {best_val_loss:.6f}")
print(f"  • Overfitting: {'Sim' if final_val_loss > final_train_loss * 1.5 else 'Não'}")

# Avaliar reconstrução no conjunto de validação
print("🔍 Avaliando qualidade da reconstrução...")
val_predictions = ae.predict(X_val, verbose=0)
reconstruction_errors = np.mean(np.square(X_val - val_predictions), axis=1)

# Estatísticas dos erros de reconstrução
error_stats = {
    'mean': np.mean(reconstruction_errors),
    'std': np.std(reconstruction_errors),
    'median': np.median(reconstruction_errors),
    'q95': np.percentile(reconstruction_errors, 95),
    'max': np.max(reconstruction_errors)
}

print(f"📊 Erros de Reconstrução (Validação):")
for stat, value in error_stats.items():
    print(f"  • {stat}: {value:.6f}")

# Salvar threshold sugerido (baseado no percentil 95)
threshold_95 = error_stats['q95']
threshold_mean_plus_3std = error_stats['mean'] + 3 * error_stats['std']

# Usar o menor dos dois como threshold conservador
suggested_threshold = min(threshold_95, threshold_mean_plus_3std)

threshold_info = {
    'threshold_95_percentile': threshold_95,
    'threshold_mean_plus_3std': threshold_mean_plus_3std,
    'suggested_threshold': suggested_threshold,
    'validation_error_stats': error_stats
}

# Salvar informações do threshold
joblib.dump(threshold_info, os.path.join(MODEL_DIR, 'threshold_info.pkl'))

print(f"🎯 Threshold Sugerido para Detecção: {suggested_threshold:.6f}")
print(f"  • Baseado no menor entre P95 ({threshold_95:.6f}) e μ+3σ ({threshold_mean_plus_3std:.6f})")

# Salvar configurações do modelo para reprodutibilidade
model_config = {
    'features_used': features_to_keep,
    'remove_non_significant': REMOVE_NON_SIGNIFICANT,
    'apply_transformations': APPLY_TRANSFORMATIONS,
    'remove_outliers': REMOVE_OUTLIERS,
    'input_dim': input_dim,
    'architecture': {
        'encoder_dims': encoder_dims,
        'bottleneck_dim': bottleneck_dim
    },
    'training_params': {
        'epochs_trained': epochs_trained,
        'batch_size': batch_size,
        'train_samples': train_size,
        'val_samples': X_val.shape[0]
    }
}

joblib.dump(model_config, os.path.join(MODEL_DIR, 'model_config.pkl'))

print("💾 Todos os artefatos salvos em model/:")
print("  • autoencoder.h5 - Modelo treinado")
print("  • preprocessor.pkl - Pipeline de normalização")
print("  • imputer.pkl - Tratamento de valores ausentes")
print("  • power_transformer.pkl - Transformações de assimetria")
print("  • threshold_info.pkl - Thresholds para detecção")
print("  • model_config.pkl - Configurações do modelo")
print("  • history.csv - Histórico de treinamento")

print("\n🎉 Treinamento concluído com sucesso!")
print("💡 Use os thresholds salvos para detectar anomalias no conjunto de teste")
