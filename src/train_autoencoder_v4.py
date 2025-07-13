#!/usr/bin/env python3
"""
Training script for Model V4 - LSTM-Attention Sequence Autoencoder
Advanced temporal pattern detection for network anomaly detection
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, RepeatVector, 
    TimeDistributed, Attention, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from sequence_preprocessor import SequencePreprocessor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

class SequenceAutoencoder:
    """LSTM-Attention Sequence Autoencoder for anomaly detection"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        
        # Model parameters
        self.sequence_length = self.config['sequence_length']
        self.n_features = len(self.config['features_used'])
        
        # Enhanced features if enabled
        if self.config.get('feature_engineering', {}).get('sequence_stats', False):
            self.n_features *= 6  # original + 5 statistical features
            
        logger.info(f"🔧 Initialized SequenceAutoencoder")
        logger.info(f"📐 Input shape: ({self.sequence_length}, {self.n_features})")
        
    def build_model(self):
        """Build LSTM-Attention Autoencoder architecture"""
        logger.info("🏗️ Building LSTM-Attention Autoencoder...")
        
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # Encoder
        encoder_arch = self.config['architecture']['encoder']
        x = input_layer
        
        # Encoder LSTM layers
        for i, units in enumerate(encoder_arch['lstm_units']):
            return_sequences = (i < len(encoder_arch['lstm_units']) - 1) or encoder_arch.get('return_sequences', True)
            
            x = LSTM(
                units, 
                return_sequences=return_sequences,
                dropout=encoder_arch.get('dropout', 0.3),
                name=f'encoder_lstm_{i+1}'
            )(x)
            
            if encoder_arch.get('dropout', 0.3) > 0:
                x = Dropout(encoder_arch['dropout'], name=f'encoder_dropout_{i+1}')(x)
        
        # Attention mechanism
        if encoder_arch.get('attention', False):
            attention_out = AttentionLayer(encoder_arch['lstm_units'][-1], name='encoder_attention')(x)
            x = attention_out
        else:
            # If not using attention, take the last output
            if len(x.shape) == 3:
                x = x[:, -1, :]  # Take last timestep
        
        # Bottleneck
        bottleneck_arch = self.config['architecture']['bottleneck']
        encoded = Dense(
            bottleneck_arch['units'], 
            activation=bottleneck_arch.get('activation', 'tanh'),
            name='bottleneck'
        )(x)
        
        # Create encoder model
        self.encoder = Model(input_layer, encoded, name='encoder')
        
        # Decoder
        decoder_arch = self.config['architecture']['decoder']
        
        # Repeat vector to create sequence
        x = RepeatVector(self.sequence_length, name='repeat_vector')(encoded)
        
        # Decoder LSTM layers
        for i, units in enumerate(decoder_arch['lstm_units']):
            return_sequences = decoder_arch.get('return_sequences', True)
            
            x = LSTM(
                units,
                return_sequences=return_sequences,
                dropout=decoder_arch.get('dropout', 0.3),
                name=f'decoder_lstm_{i+1}'
            )(x)
            
            if decoder_arch.get('dropout', 0.3) > 0:
                x = Dropout(decoder_arch['dropout'], name=f'decoder_dropout_{i+1}')(x)
        
        # Output layer
        output_arch = self.config['architecture']['output']
        decoded = TimeDistributed(
            Dense(
                output_arch['units'],
                activation=output_arch.get('activation', 'linear')
            ),
            name='output'
        )(x)
        
        # Create full autoencoder
        self.model = Model(input_layer, decoded, name='sequence_autoencoder')
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )
        
        logger.info("✅ Model built successfully")
        logger.info(f"📊 Model summary:")
        self.model.summary()
        
        return self.model
    
    def prepare_callbacks(self, model_dir: Path):
        """Prepare training callbacks"""
        callbacks = []
        
        # Early stopping
        if self.config['training'].get('early_stopping', True):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config['training'].get('patience', 15),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        if self.config['training'].get('model_checkpoint', True):
            checkpoint = ModelCheckpoint(
                filepath=str(model_dir / 'autoencoder.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['training'].get('reduce_lr_factor', 0.5),
            patience=self.config['training'].get('reduce_lr_patience', 8),
            min_lr=self.config['training'].get('min_lr', 1e-6),
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        if self.config['training'].get('tensorboard', True):
            tensorboard = TensorBoard(
                log_dir=str(model_dir / 'logs'),
                histogram_freq=1,
                write_graph=True
            )
            callbacks.append(tensorboard)
        
        return callbacks
    
    def train(self, X_sequences: np.ndarray, model_dir: str = 'model_v4'):
        """Train the sequence autoencoder"""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Starting training for model_v4...")
        logger.info(f"📊 Training data shape: {X_sequences.shape}")
        
        # Split data (only use normal sequences for training autoencoder)
        X_train, X_val = train_test_split(
            X_sequences, 
            test_size=self.config['training']['validation_split'],
            random_state=42
        )
        
        logger.info(f"📊 Train sequences: {X_train.shape}")
        logger.info(f"📊 Validation sequences: {X_val.shape}")
        
        # Prepare callbacks
        callbacks = self.prepare_callbacks(model_dir)
        
        # Train model
        self.history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = output
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Training completed!")
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(model_dir / 'history.csv', index=False)
        
        return self.history
    
    def save_model_artifacts(self, model_dir: str, preprocessor: SequencePreprocessor):
        """Save all model artifacts"""
        model_dir = Path(model_dir)
        
        # Save config
        config_data = {
            'config_used': self.config,
            'model_type': 'sequence_autoencoder',
            'features_used': self.config['features_used'],
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }
        joblib.dump(config_data, model_dir / 'model_config.pkl')
        
        # Save preprocessor
        preprocessor.save(str(model_dir / 'sequence_preprocessor.pkl'))
        
        logger.info(f"💾 Model artifacts saved to {model_dir}")
    
    def calculate_threshold(self, X_sequences: np.ndarray, method: str = 'percentile', percentile: float = 95):
        """Calculate anomaly detection threshold"""
        logger.info("🔍 Calculating anomaly detection threshold...")
        
        # Get reconstruction errors
        X_pred = self.model.predict(X_sequences, verbose=0)
        errors = np.mean(np.square(X_sequences - X_pred), axis=(1, 2))
        
        if method == 'percentile':
            threshold = np.percentile(errors, percentile)
        elif method == 'statistical':
            threshold = np.mean(errors) + 2 * np.std(errors)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        threshold_info = {
            'suggested_threshold': threshold,
            'method': method,
            'percentile': percentile if method == 'percentile' else None,
            'error_stats': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'percentiles': {
                    '90': np.percentile(errors, 90),
                    '95': np.percentile(errors, 95),
                    '99': np.percentile(errors, 99)
                }
            }
        }
        
        logger.info(f"✅ Threshold calculated: {threshold:.6f}")
        logger.info(f"📊 Error statistics: mean={np.mean(errors):.6f}, std={np.std(errors):.6f}")
        
        return threshold_info
    
    def create_loss_plots(self, model_dir: str):
        """Create training loss visualization"""
        if self.history is None:
            logger.warning("No training history available for plotting")
            return
        
        model_dir = Path(model_dir)
        figures_dir = model_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Plot training history
        plt.figure(figsize=(12, 8))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Model Loss (MSE)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in self.history.history:
            plt.subplot(2, 2, 2)
            plt.plot(self.history.history['mae'], 'b-', label='Training MAE', linewidth=2)
            plt.plot(self.history.history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
            plt.title('Model MAE', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        plt.subplot(2, 2, 3)
        if 'lr' in self.history.history:
            plt.plot(self.history.history['lr'], 'g-', linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        # Training summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = f"""
        Training Summary:
        
        • Epochs: {len(self.history.history['loss'])}
        • Final Train Loss: {self.history.history['loss'][-1]:.6f}
        • Final Val Loss: {self.history.history['val_loss'][-1]:.6f}
        • Best Val Loss: {min(self.history.history['val_loss']):.6f}
        • Architecture: LSTM-Attention
        • Sequence Length: {self.sequence_length}
        • Features: {self.n_features}
        """\n        \n        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, \n                fontsize=11, verticalalignment='top',\n                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))\n        \n        plt.suptitle('LSTM-Attention Sequence Autoencoder - Training Results', \n                     fontsize=16, fontweight='bold', y=0.98)\n        plt.tight_layout()\n        plt.savefig(figures_dir / 'training_history.png', dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        logger.info(f\"📊 Training plots saved to {figures_dir}\")\n\ndef main():\n    \"\"\"Main training function\"\"\"\n    try:\n        # Load data\n        logger.info(\"📂 Loading data...\")\n        df = pd.read_csv('data/processed/flows.csv')\n        \n        # Initialize preprocessor\n        logger.info(\"🔧 Initializing sequence preprocessor...\")\n        with open('config/model_config_sequence_v4.yaml', 'r') as f:\n            config = yaml.safe_load(f)\n        \n        preprocessor = SequencePreprocessor(config)\n        \n        # Create sequences (only use normal flows for autoencoder training)\n        logger.info(\"🔄 Creating sequences...\")\n        normal_df = df[df['label'] == 0].copy()  # Only normal flows\n        X_sequences, _ = preprocessor.fit_transform(normal_df)\n        \n        if len(X_sequences) == 0:\n            raise ValueError(\"No sequences created. Check your data and configuration.\")\n        \n        # Initialize and build model\n        logger.info(\"🏗️ Building model...\")\n        autoencoder = SequenceAutoencoder('config/model_config_sequence_v4.yaml')\n        autoencoder.build_model()\n        \n        # Train model\n        logger.info(\"🚀 Starting training...\")\n        history = autoencoder.train(X_sequences, 'model_v4')\n        \n        # Calculate threshold on all data (including attacks)\n        logger.info(\"🔍 Processing all data for threshold calculation...\")\n        X_all_sequences, y_all_sequences = preprocessor.transform(df)\n        normal_sequences = X_all_sequences[y_all_sequences == 0]  # Only normal for threshold\n        \n        threshold_info = autoencoder.calculate_threshold(\n            normal_sequences,\n            method=config['anomaly_detection']['threshold_method'],\n            percentile=config['anomaly_detection']['threshold_percentile']\n        )\n        \n        # Save threshold info\n        joblib.dump(threshold_info, 'model_v4/threshold_info.pkl')\n        \n        # Save all artifacts\n        autoencoder.save_model_artifacts('model_v4', preprocessor)\n        \n        # Create visualizations\n        autoencoder.create_loss_plots('model_v4')\n        \n        logger.info(\"🎉 Model V4 training completed successfully!\")\n        logger.info(f\"📁 All artifacts saved in: model_v4/\")\n        \n        return autoencoder, preprocessor, threshold_info\n        \n    except Exception as e:\n        logger.error(f\"❌ Error in training: {e}\")\n        raise\n\nif __name__ == \"__main__\":\n    main()