# src/train_autoencoder_v2.py
# Versão refatorada com pipeline unificado e configuração YAML

import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from scipy import stats

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.preprocessor import (
    fit_preprocessor, 
    transform_preprocessor, 
    get_eda_config,
    SELECTED_FEATURES
)

def set_seeds(numpy_seed=42, tf_seed=42):
    """Configurar seeds para reprodutibilidade"""
    np.random.seed(numpy_seed)
    tf.random.set_seed(tf_seed)
    # Para Python builtin random (se usado)
    import random
    random.seed(numpy_seed)

def load_config(config_path="config/model_config.yaml"):
    """Carregar configurações do arquivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️ Arquivo de configuração não encontrado: {config_path}")
        print("Usando configurações padrão...")
        return get_default_config()

def get_default_config():
    """Configurações padrão caso o YAML não seja encontrado"""
    return {
        'data': {'path': 'data/processed/flows.csv', 'normal_label': 0},
        'features': {'significant': ['bytes', 'pkts', 'iat_mean'], 'use_only_significant': True},
        'preprocessing': {'apply_transformations': True, 'remove_outliers': False, 'outlier_threshold': 4.0},
        'training': {'validation_split': 0.2, 'epochs': 150, 'random_state': 42},
        'output': {'model_dir': 'model', 'figures_dir': 'model/figures'},
        'reproducibility': {'numpy_seed': 42, 'tensorflow_seed': 42}
    }

def create_directories(config):
    """Criar diretórios necessários"""
    dirs_to_create = [
        config['output']['model_dir'],
        config['output'].get('figures_dir', 'model/figures'),
        config['output'].get('reports_dir', 'reports')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def load_and_filter_data(config):
    """Carregar e filtrar dados baseado na configuração"""
    print("📊 Carregando dados...")
    
    # Carregar dados
    df = pd.read_csv(config['data']['path'])
    print(f"Dataset carregado: {df.shape[0]} fluxos, {df.shape[1]} features")
    
    # Filtrar apenas tráfego normal para treinamento
    normal_label = config['data']['normal_label']
    df_norm = df[df.label == normal_label].reset_index(drop=True)
    print(f"Tráfego normal para treinamento: {df_norm.shape[0]} fluxos")
    
    # Determinar features a usar
    if config['features']['use_only_significant']:
        features_to_use = config['features']['significant']
        print(f"✅ Usando apenas features significativas: {features_to_use}")
    else:
        # Usar todas as features numéricas disponíveis
        features_to_use = SELECTED_FEATURES
        print(f"✅ Usando todas as features: {features_to_use}")
    
    return df_norm, features_to_use

def remove_outliers(X, threshold=4.0):
    """Remover outliers extremos usando z-score"""
    z_scores = np.abs(stats.zscore(X, axis=0))
    outlier_mask = (z_scores < threshold).all(axis=1)
    return X[outlier_mask], outlier_mask

def create_autoencoder(input_dim, config):
    """Criar arquitetura do autoencoder adaptativa"""
    print(f"🏗️ Construindo autoencoder para {input_dim} features...")
    
    # Determinar arquitetura baseada no número de features
    if input_dim <= 3:
        arch = config['model']['architecture']['three_features']
    elif input_dim <= 5:
        arch = config['model']['architecture']['five_features']
    else:
        arch = config['model']['architecture']['many_features']
    
    encoder_dims = arch['encoder_dims']
    bottleneck_dim = arch['bottleneck_dim']
    dropout_rates = config['model']['dropout_rates']
    
    print(f"Arquitetura: {input_dim} → {encoder_dims[0]} → {encoder_dims[1]} → {bottleneck_dim} → {encoder_dims[1]} → {encoder_dims[0]} → {input_dim}")
    
    # Construir modelo
    inp = layers.Input(shape=(input_dim,), name='input')
    
    # Encoder
    x = layers.Dense(encoder_dims[0], activation='relu', name='encoder_1')(inp)
    x = layers.Dropout(dropout_rates[0], name='dropout_1')(x)
    x = layers.Dense(encoder_dims[1], activation='relu', name='encoder_2')(x)
    x = layers.Dropout(dropout_rates[1], name='dropout_2')(x)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)
    
    # Decoder
    x = layers.Dense(encoder_dims[1], activation='relu', name='decoder_1')(bottleneck)
    x = layers.Dropout(dropout_rates[2], name='dropout_3')(x)
    x = layers.Dense(encoder_dims[0], activation='relu', name='decoder_2')(x)
    x = layers.Dropout(dropout_rates[3], name='dropout_4')(x)
    out = layers.Dense(input_dim, activation='linear', name='output')(x)
    
    # Compilar modelo
    ae = Model(inp, out, name='autoencoder')
    
    optimizer_config = config['model']['optimizer']
    optimizer = Adam(
        learning_rate=optimizer_config['learning_rate'],
        beta_1=optimizer_config['beta_1'],
        beta_2=optimizer_config['beta_2']
    )
    
    ae.compile(
        optimizer=optimizer,
        loss=config['model']['loss'],
        metrics=config['model']['metrics']
    )
    
    print("✅ Autoencoder construído e compilado")
    if config.get('flags', {}).get('verbose', True):
        print(ae.summary())
    
    return ae

def create_callbacks(config):
    """Criar callbacks para treinamento"""
    print("⚙️ Configurando callbacks...")
    
    model_dir = config['output']['model_dir']
    training_config = config['training']
    
    # Early stopping
    es_config = training_config['early_stopping']
    es = EarlyStopping(
        monitor='val_loss',
        patience=es_config['patience'],
        restore_best_weights=es_config['restore_best_weights'],
        verbose=1,
        min_delta=es_config['min_delta']
    )
    
    # Model checkpoint
    mc_config = training_config['model_checkpoint']
    mc = ModelCheckpoint(
        os.path.join(model_dir, 'autoencoder.h5'),
        monitor='val_loss',
        save_best_only=mc_config['save_best_only'],
        save_weights_only=mc_config['save_weights_only'],
        verbose=1
    )
    
    # Reduce learning rate
    rlr_config = training_config['reduce_lr']
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=rlr_config['factor'],
        patience=rlr_config['patience'],
        min_lr=rlr_config['min_lr'],
        verbose=1
    )
    
    callbacks = [es, mc, rlr]
    print(f"✅ {len(callbacks)} callbacks configurados")
    
    return callbacks

def determine_batch_size(train_size, config):
    """Determinar batch size baseado no tamanho do dataset"""
    batch_config = config['training']['batch_size']
    
    if train_size < 1000:
        return batch_config['small_dataset']
    elif train_size < 5000:
        return batch_config['medium_dataset']
    else:
        return batch_config['large_dataset']

def calculate_thresholds(reconstruction_errors):
    """Calcular thresholds para detecção de anomalias"""
    error_stats = {
        'mean': np.mean(reconstruction_errors),
        'std': np.std(reconstruction_errors),
        'median': np.median(reconstruction_errors),
        'q95': np.percentile(reconstruction_errors, 95),
        'q99': np.percentile(reconstruction_errors, 99),
        'max': np.max(reconstruction_errors)
    }
    
    # Métodos de threshold
    threshold_95 = error_stats['q95']
    threshold_mean_plus_3std = error_stats['mean'] + 3 * error_stats['std']
    
    # Threshold conservador (menor dos dois)
    suggested_threshold = min(threshold_95, threshold_mean_plus_3std)
    
    threshold_info = {
        'threshold_95_percentile': threshold_95,
        'threshold_mean_plus_3std': threshold_mean_plus_3std,
        'suggested_threshold': suggested_threshold,
        'validation_error_stats': error_stats
    }
    
    return threshold_info

def save_training_plots(history, config):
    """Salvar plots do treinamento"""
    if not config.get('flags', {}).get('save_plots', True):
        return
    
    figures_dir = config['output'].get('figures_dir', 'model/figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot de loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Treino', color='blue')
    ax1.plot(history.history['val_loss'], label='Validação', color='red')
    ax1.set_title('Loss durante o Treinamento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE
    if 'mae' in history.history:
        ax2.plot(history.history['mae'], label='Treino', color='blue')
        ax2.plot(history.history['val_mae'], label='Validação', color='red')
        ax2.set_title('MAE durante o Treinamento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot de treinamento salvo em {figures_dir}/training_history.png")

def save_reconstruction_analysis(reconstruction_errors, config):
    """Salvar análise dos erros de reconstrução"""
    if not config.get('flags', {}).get('save_plots', True):
        return
    
    figures_dir = config['output'].get('figures_dir', 'model/figures')
    
    # Histograma dos erros de reconstrução
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(reconstruction_errors), color='red', linestyle='--', 
                label=f'Média: {np.mean(reconstruction_errors):.6f}')
    plt.axvline(np.percentile(reconstruction_errors, 95), color='orange', linestyle='--',
                label=f'P95: {np.percentile(reconstruction_errors, 95):.6f}')
    plt.title('Distribuição dos Erros de Reconstrução')
    plt.xlabel('Erro de Reconstrução (MSE)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(reconstruction_errors)
    plt.title('Boxplot dos Erros de Reconstrução')
    plt.ylabel('Erro de Reconstrução (MSE)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'reconstruction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Análise de reconstrução salva em {figures_dir}/reconstruction_analysis.png")

def main():
    """Função principal"""
    print("🚀 Iniciando treinamento do autoencoder v2.0")
    print("=" * 60)
    
    # 1. Carregar configuração
    config = load_config()
    
    # 2. Configurar reprodutibilidade
    repro_config = config['reproducibility']
    set_seeds(repro_config['numpy_seed'], repro_config['tensorflow_seed'])
    print(f"✅ Seeds configuradas: numpy={repro_config['numpy_seed']}, tf={repro_config['tensorflow_seed']}")
    
    # 3. Criar diretórios
    create_directories(config)
    
    # 4. Carregar e filtrar dados
    df_norm, features_to_use = load_and_filter_data(config)
    
    # 5. Pré-processamento unificado
    print("🔧 Aplicando pré-processamento unificado...")
    preprocessor = fit_preprocessor(
        df_norm, 
        features_to_use=features_to_use,
        apply_transformations=config['preprocessing']['apply_transformations']
    )
    X = transform_preprocessor(df_norm, preprocessor, features_to_use)
    
    print(f"✅ Dados pré-processados: {X.shape}")
    
    # 6. Remover outliers se configurado
    if config['preprocessing']['remove_outliers']:
        print("🔍 Removendo outliers extremos...")
        X_clean, outlier_mask = remove_outliers(X, config['preprocessing']['outlier_threshold'])
        removed_count = len(X) - len(X_clean)
        print(f"Outliers removidos: {removed_count} ({removed_count/len(X)*100:.1f}%)")
        X = X_clean
    
    # 7. Split treino/validação
    print("✂️ Dividindo dados...")
    val_split = config['training']['validation_split']
    random_state = config['training']['random_state']
    
    X_train, X_val = train_test_split(
        X, 
        test_size=val_split, 
        random_state=random_state, 
        shuffle=True
    )
    
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Validação: {X_val.shape[0]} amostras")
    
    # 8. Criar modelo
    input_dim = X_train.shape[1]
    autoencoder = create_autoencoder(input_dim, config)
    
    # 9. Configurar callbacks
    callbacks = create_callbacks(config)
    
    # 10. Treinar modelo
    print("🚀 Iniciando treinamento...")
    
    batch_size = determine_batch_size(X_train.shape[0], config)
    epochs = config['training']['epochs']
    
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    print("✅ Treinamento concluído!")
    
    # 11. Análise pós-treinamento
    print("📊 Analisando resultados...")
    
    # Salvar histórico
    model_dir = config['output']['model_dir']
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dir, 'history.csv'), index=False)
    
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
    
    # 12. Análise de reconstrução e thresholds
    print("🔍 Calculando thresholds para detecção...")
    val_predictions = autoencoder.predict(X_val, verbose=0)
    reconstruction_errors = np.mean(np.square(X_val - val_predictions), axis=1)
    
    threshold_info = calculate_thresholds(reconstruction_errors)
    
    print(f"📊 Erros de Reconstrução (Validação):")
    for stat, value in threshold_info['validation_error_stats'].items():
        print(f"  • {stat}: {value:.6f}")
    
    print(f"🎯 Threshold Sugerido: {threshold_info['suggested_threshold']:.6f}")
    print(f"  • P95: {threshold_info['threshold_95_percentile']:.6f}")
    print(f"  • μ+3σ: {threshold_info['threshold_mean_plus_3std']:.6f}")
    
    # 13. Salvar artefatos
    print("💾 Salvando artefatos...")
    
    # Preprocessor
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    
    # Threshold info
    joblib.dump(threshold_info, os.path.join(model_dir, 'threshold_info.pkl'))
    
    # Configuração do modelo
    model_config = {
        'features_used': features_to_use,
        'input_dim': input_dim,
        'architecture': config['model']['architecture'],
        'training_params': {
            'epochs_trained': epochs_trained,
            'batch_size': batch_size,
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0]
        },
        'config_used': config
    }
    
    joblib.dump(model_config, os.path.join(model_dir, 'model_config.pkl'))
    
    # 14. Gerar visualizações
    save_training_plots(history, config)
    save_reconstruction_analysis(reconstruction_errors, config)
    
    print("💾 Artefatos salvos:")
    artifacts = [
        "autoencoder.h5 - Modelo treinado",
        "preprocessor.pkl - Pipeline completo de pré-processamento",
        "threshold_info.pkl - Thresholds para detecção",
        "model_config.pkl - Configurações e metadados",
        "history.csv - Histórico de treinamento"
    ]
    
    if config.get('flags', {}).get('save_plots', True):
        artifacts.extend([
            "figures/training_history.png - Curvas de treinamento",
            "figures/reconstruction_analysis.png - Análise de erros"
        ])
    
    for artifact in artifacts:
        print(f"  • {artifact}")
    
    print("\n🎉 Treinamento concluído com sucesso!")
    print("💡 Pipeline totalmente unificado e reproduzível")
    print("📊 Use threshold_info.pkl para detectar anomalias")

if __name__ == "__main__":
    main()