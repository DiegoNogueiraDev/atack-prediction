# src/train_autoencoder_v3.py - Versão melhorada com correções
# Melhorias: suporte a argumentos, compatibilidade com novo config, tratamento de erros

import os
import yaml
import joblib
import argparse
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

from preprocessor import (
    fit_preprocessor, 
    transform_preprocessor, 
    get_eda_config,
    SELECTED_FEATURES
)

# Configurar estilo dos plots
plt.style.use('default')
sns.set_palette("husl")

class AutoencoderTrainer:
    """Classe para treinamento de autoencoder para detecção de anomalias"""
    
    def __init__(self, config_path='config/model_config.yaml'):
        self.config_path = config_path
        self.config = None
        self.model_dir = None
        self.figures_dir = None
        
        # Carregar configuração
        self._load_config()
        self._setup_directories()
        
    def _load_config(self):
        """Carregar configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Configuração carregada de {self.config_path}")
        except FileNotFoundError:
            print(f"⚠️ Arquivo de configuração não encontrado: {self.config_path}")
            print("Usando configurações padrão...")
            self.config = self._get_default_config()
        except Exception as e:
            print(f"❌ Erro ao carregar configuração: {e}")
            raise
    
    def _get_default_config(self):
        """Configurações padrão caso o YAML não seja encontrado"""
        return {
            'data': {'path': 'data/processed/flows.csv', 'normal_label': 0},
            'features_used': ['bytes', 'pkts', 'iat_mean'],
            'preprocessing': {
                'apply_transformations': True, 
                'remove_outliers': True, 
                'outlier_threshold': 3.5
            },
            'training': {
                'validation_split': 0.25, 
                'epochs': 200, 
                'batch_size': 32,
                'random_state': 42,
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 20,
                    'min_delta': 1e-5,
                    'restore_best_weights': True
                },
                'reduce_lr': {
                    'factor': 0.5,
                    'patience': 10,
                    'min_lr': 1e-6
                },
                'model_checkpoint': {
                    'save_best_only': True,
                    'save_weights_only': False
                }
            },
            'architecture': {
                'multipliers': [4, 2],
                'bottleneck': 3,
                'dropout': [0.3, 0.2, 0.2, 0.3]
            },
            'model': {
                'optimizer': {
                    'name': 'adam',
                    'learning_rate': 0.001,
                    'beta_1': 0.9,
                    'beta_2': 0.999
                },
                'loss': 'mse',
                'metrics': ['mae']
            },
            'output': {
                'model_dir': 'model',
                'figures_dir': 'model/figures'
            },
            'reproducibility': {
                'numpy_seed': 42,
                'tensorflow_seed': 42
            },
            'flags': {
                'verbose': True,
                'save_plots': True
            }
        }
    
    def _setup_directories(self):
        """Configurar diretórios de saída"""
        self.model_dir = Path(self.config['output']['model_dir'])
        self.figures_dir = Path(self.config['output']['figures_dir'])
        
        # Criar diretórios
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Diretórios configurados: {self.model_dir}, {self.figures_dir}")
    
    def set_seeds(self):
        """Configurar seeds para reprodutibilidade"""
        repro_config = self.config['reproducibility']
        numpy_seed = repro_config['numpy_seed']
        tf_seed = repro_config['tensorflow_seed']
        
        np.random.seed(numpy_seed)
        tf.random.set_seed(tf_seed)
        import random
        random.seed(numpy_seed)
        
        print(f"✅ Seeds configuradas: numpy={numpy_seed}, tf={tf_seed}")
    
    def load_and_filter_data(self):
        """Carregar e filtrar dados baseado na configuração"""
        try:
            print("📊 Carregando dados...")
            
            # Carregar dados
            df = pd.read_csv(self.config['data']['path'])
            print(f"Dataset carregado: {df.shape[0]} fluxos, {df.shape[1]} features")
            
            # Filtrar apenas tráfego normal para treinamento
            normal_label = self.config['data']['normal_label']
            df_norm = df[df.label == normal_label].reset_index(drop=True)
            print(f"Tráfego normal para treinamento: {df_norm.shape[0]} fluxos")
            
            # Usar features especificadas no config
            features_to_use = self.config['features_used']
            print(f"✅ Usando features: {features_to_use}")
            
            return df_norm, features_to_use
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def remove_outliers(self, X, threshold=3.5):
        """Remover outliers extremos usando z-score"""
        try:
            z_scores = np.abs(stats.zscore(X, axis=0))
            outlier_mask = (z_scores < threshold).all(axis=1)
            return X[outlier_mask], outlier_mask
        except Exception as e:
            print(f"❌ Erro na remoção de outliers: {e}")
            raise
    
    def create_autoencoder(self, input_dim):
        """Criar arquitetura do autoencoder baseada nos multiplicadores"""
        try:
            print(f"🏗️ Construindo autoencoder para {input_dim} features...")
            
            # Arquitetura baseada nos multiplicadores (4×D, 2×D, D)
            multipliers = self.config['architecture']['multipliers']
            encoder_dims = [input_dim * mult for mult in multipliers]  # [4×D, 2×D]
            bottleneck_dim = self.config['architecture']['bottleneck']
            dropout_rates = self.config['architecture']['dropout']
            
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
            
            optimizer_config = self.config['model']['optimizer']
            optimizer = Adam(
                learning_rate=float(optimizer_config['learning_rate']),
                beta_1=float(optimizer_config['beta_1']),
                beta_2=float(optimizer_config['beta_2'])
            )
            
            ae.compile(
                optimizer=optimizer,
                loss=self.config['model']['loss'],
                metrics=self.config['model']['metrics']
            )
            
            print("✅ Autoencoder construído e compilado")
            if self.config.get('flags', {}).get('verbose', True):
                print(ae.summary())
            
            return ae
        except Exception as e:
            print(f"❌ Erro na criação do autoencoder: {e}")
            raise
    
    def create_callbacks(self):
        """Criar callbacks para treinamento"""
        try:
            print("⚙️ Configurando callbacks...")
            
            training_config = self.config['training']
            
            # Early stopping
            es_config = training_config['early_stopping']
            es = EarlyStopping(
                monitor=es_config['monitor'],
                patience=es_config['patience'],
                restore_best_weights=es_config['restore_best_weights'],
                verbose=1,
                min_delta=float(es_config['min_delta'])
            )
            
            # Model checkpoint
            mc_config = training_config['model_checkpoint']
            mc = ModelCheckpoint(
                str(self.model_dir / 'autoencoder.h5'),
                monitor='val_loss',
                save_best_only=mc_config['save_best_only'],
                save_weights_only=mc_config['save_weights_only'],
                verbose=1
            )
            
            # Reduce learning rate
            rlr_config = training_config['reduce_lr']
            rlr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=float(rlr_config['factor']),
                patience=rlr_config['patience'],
                min_lr=float(rlr_config['min_lr']),
                verbose=1
            )
            
            callbacks = [es, mc, rlr]
            print(f"✅ {len(callbacks)} callbacks configurados")
            
            return callbacks
        except Exception as e:
            print(f"❌ Erro na criação dos callbacks: {e}")
            raise
    
    def calculate_thresholds(self, reconstruction_errors):
        """Calcular thresholds para detecção de anomalias - Configuração BALANCEADA"""
        try:
            error_stats = {
                'mean': np.mean(reconstruction_errors),
                'std': np.std(reconstruction_errors),
                'median': np.median(reconstruction_errors),
                'q95': np.percentile(reconstruction_errors, 95),
                'q97': np.percentile(reconstruction_errors, 97),  # Mais conservador
                'q99': np.percentile(reconstruction_errors, 99),
                'max': np.max(reconstruction_errors)
            }
            
            # Métodos de threshold conservadores
            threshold_97 = error_stats['q97']                    # P97 em vez de P95
            threshold_mean_plus_2std = error_stats['mean'] + 2 * error_stats['std']  # 2σ em vez de 3σ
            
            # Threshold conservador (menor dos dois - mais restritivo)
            suggested_threshold = min(threshold_97, threshold_mean_plus_2std)
            
            threshold_info = {
                'threshold_95_percentile': error_stats['q95'],
                'threshold_97_percentile': threshold_97,
                'threshold_mean_plus_2std': threshold_mean_plus_2std,
                'threshold_mean_plus_3std': error_stats['mean'] + 3 * error_stats['std'],
                'suggested_threshold': suggested_threshold,
                'validation_error_stats': error_stats
            }
            
            return threshold_info
        except Exception as e:
            print(f"❌ Erro no cálculo de thresholds: {e}")
            raise
    
    def save_training_plots(self, history):
        """Salvar curva de loss como loss_curve.png"""
        try:
            if not self.config.get('flags', {}).get('save_plots', True):
                return
            
            # Plot de loss (training vs validation)
            plt.figure(figsize=(12, 8))
            plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
            plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
            plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('MSE Loss', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Loss curve salva em {self.figures_dir}/loss_curve.png")
        except Exception as e:
            print(f"❌ Erro ao salvar plots: {e}")
    
    def train(self):
        """Executar treinamento completo"""
        print("🚀 Iniciando treinamento do autoencoder v3.0")
        print("=" * 60)
        
        try:
            # 1. Configurar reprodutibilidade
            self.set_seeds()
            
            # 2. Carregar e filtrar dados
            df_norm, features_to_use = self.load_and_filter_data()
            
            # 3. Pré-processamento unificado
            print("🔧 Aplicando pré-processamento unificado...")
            preprocessor = fit_preprocessor(
                df_norm, 
                features_to_use=features_to_use,
                apply_transformations=self.config['preprocessing']['apply_transformations']
            )
            X = transform_preprocessor(df_norm, preprocessor, features_to_use)
            
            print(f"✅ Dados pré-processados: {X.shape}")
            
            # 4. Remover outliers se configurado
            if self.config['preprocessing']['remove_outliers']:
                print("🔍 Removendo outliers extremos...")
                X_clean, outlier_mask = self.remove_outliers(X, self.config['preprocessing']['outlier_threshold'])
                removed_count = len(X) - len(X_clean)
                print(f"Outliers removidos: {removed_count} ({removed_count/len(X)*100:.1f}%)")
                X = X_clean
            
            # 5. Split treino/validação
            print("✂️ Dividindo dados...")
            val_split = self.config['training']['validation_split']
            random_state = self.config['training']['random_state']
            
            X_train, X_val = train_test_split(
                X, 
                test_size=val_split, 
                random_state=random_state, 
                shuffle=True
            )
            
            print(f"Treino: {X_train.shape[0]} amostras")
            print(f"Validação: {X_val.shape[0]} amostras")
            
            # 6. Criar modelo
            input_dim = X_train.shape[1]
            autoencoder = self.create_autoencoder(input_dim)
            
            # 7. Configurar callbacks
            callbacks = self.create_callbacks()
            
            # 8. Treinar modelo
            print("🚀 Iniciando treinamento...")
            
            batch_size = self.config['training']['batch_size']
            epochs = self.config['training']['epochs']
            
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
            
            # 9. Análise pós-treinamento
            print("📊 Analisando resultados...")
            
            # Salvar histórico
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(self.model_dir / 'history.csv', index=False)
            
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
            
            # 10. Análise de reconstrução e thresholds
            print("🔍 Calculando thresholds para detecção...")
            val_predictions = autoencoder.predict(X_val, verbose=0)
            reconstruction_errors = np.mean(np.square(X_val - val_predictions), axis=1)
            
            threshold_info = self.calculate_thresholds(reconstruction_errors)
            
            print(f"📊 Erros de Reconstrução (Validação):")
            for stat, value in threshold_info['validation_error_stats'].items():
                print(f"  • {stat}: {value:.6f}")
            
            print(f"🎯 Threshold Sugerido: {threshold_info['suggested_threshold']:.6f}")
            print(f"  • P97: {threshold_info['threshold_97_percentile']:.6f}")
            print(f"  • μ+2σ: {threshold_info['threshold_mean_plus_2std']:.6f}")
            
            # 11. Salvar artefatos
            print("💾 Salvando artefatos...")
            
            # Preprocessor
            joblib.dump(preprocessor, self.model_dir / 'preprocessor.pkl')
            
            # Threshold info
            joblib.dump(threshold_info, self.model_dir / 'threshold_info.pkl')
            
            # Configuração do modelo
            model_config = {
                'features_used': features_to_use,
                'input_dim': input_dim,
                'architecture': self.config['architecture'],
                'training_params': {
                    'epochs_trained': epochs_trained,
                    'batch_size': batch_size,
                    'train_samples': X_train.shape[0],
                    'val_samples': X_val.shape[0]
                },
                'config_used': self.config
            }
            
            joblib.dump(model_config, self.model_dir / 'model_config.pkl')
            
            # 12. Gerar visualizações
            self.save_training_plots(history)
            
            print("💾 Artefatos salvos:")
            artifacts = [
                "autoencoder.h5 - Modelo treinado",
                "preprocessor.pkl - Pipeline completo de pré-processamento",
                "threshold_info.pkl - Thresholds para detecção",
                "model_config.pkl - Configurações e metadados",
                "history.csv - Histórico de treinamento"
            ]
            
            if self.config.get('flags', {}).get('save_plots', True):
                artifacts.append("figures/loss_curve.png - Curva de treinamento")
            
            for artifact in artifacts:
                print(f"  • {artifact}")
            
            print("\n🎉 Treinamento concluído com sucesso!")
            print("💡 Pipeline totalmente unificado e reproduzível")
            print("📊 Use threshold_info.pkl para detectar anomalias")
            
            return {
                'model': autoencoder,
                'history': history,
                'threshold_info': threshold_info,
                'preprocessor': preprocessor
            }
            
        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            raise

def main():
    """Função principal"""
    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Treinar autoencoder para detecção de anomalias')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Caminho para o arquivo de configuração YAML')
    
    args = parser.parse_args()
    
    try:
        # Criar trainer
        trainer = AutoencoderTrainer(config_path=args.config)
        
        # Executar treinamento
        results = trainer.train()
        
        return results
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        return None

if __name__ == "__main__":
    main()