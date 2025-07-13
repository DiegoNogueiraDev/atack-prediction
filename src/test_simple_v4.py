#!/usr/bin/env python3
"""
Simple test version of model_v4 to debug NaN issues
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sequence_preprocessor import SequencePreprocessor
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_model(sequence_length, n_features):
    """Create a simple LSTM autoencoder without attention"""
    
    # Input
    input_layer = Input(shape=(sequence_length, n_features))
    
    # Encoder
    encoded = LSTM(32, activation='tanh')(input_layer)
    
    # Bottleneck
    bottleneck = Dense(8, activation='tanh')(encoded)
    
    # Decoder
    decoded = RepeatVector(sequence_length)(bottleneck)
    decoded = LSTM(32, return_sequences=True, activation='tanh')(decoded)
    decoded = TimeDistributed(Dense(n_features, activation='linear'))(decoded)
    
    # Model
    model = Model(input_layer, decoded)
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    # Load config and data
    with open('config/model_config_sequence_v4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv('data/processed/flows.csv')
    normal_df = df[df['label'] == 0].copy()
    
    # Create preprocessor
    preprocessor = SequencePreprocessor(config)
    X_sequences, _ = preprocessor.fit_transform(normal_df)
    
    logger.info(f"Sequences shape: {X_sequences.shape}")
    logger.info(f"Data range: {X_sequences.min():.3f} to {X_sequences.max():.3f}")
    logger.info(f"Data mean: {X_sequences.mean():.3f}, std: {X_sequences.std():.3f}")
    
    # Check for NaN or inf values
    if np.any(np.isnan(X_sequences)):
        logger.error("NaN values found in sequences!")
        return
    if np.any(np.isinf(X_sequences)):
        logger.error("Inf values found in sequences!")
        return
    
    # Create simple model
    model = create_simple_model(X_sequences.shape[1], X_sequences.shape[2])
    model.summary()
    
    # Train for a few epochs
    history = model.fit(
        X_sequences, X_sequences,
        epochs=5,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    logger.info("Simple model training completed successfully!")
    logger.info(f"Final loss: {history.history['loss'][-1]:.6f}")

if __name__ == "__main__":
    main()