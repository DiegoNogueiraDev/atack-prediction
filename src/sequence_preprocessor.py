#!/usr/bin/env python3
"""
Sequence Preprocessor for Model V4
Converts flow data into temporal sequences for LSTM-Attention Autoencoder
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SequencePreprocessor:
    """Preprocessor for converting flows into sequences"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features = config['features_used']
        self.sequence_length = config['sequence_length']
        self.overlap = config['overlap']
        
        # Scalers for different feature types
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        # Sequence statistics
        self.sequence_stats = {}
        
    def _group_flows_by_connection(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group flows by connection (src-dst pair)"""
        # Create connection identifier
        df['connection_id'] = df['src'].astype(str) + '_' + df['dst'].astype(str)
        
        # Group by connection and sort by duration (proxy for time)
        grouped = {}
        for conn_id, group in df.groupby('connection_id'):
            # Sort by duration to get temporal order
            group_sorted = group.sort_values('duration').reset_index(drop=True)
            grouped[conn_id] = group_sorted
            
        logger.info(f"📊 Grouped flows into {len(grouped)} connections")
        return grouped
        
    def _create_sequences(self, grouped_flows: Dict[str, pd.DataFrame], 
                         include_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from grouped flows"""
        sequences = []
        labels = []
        
        for conn_id, flows in grouped_flows.items():
            if len(flows) < self.sequence_length:
                continue
                
            # Extract features
            flow_features = flows[self.features].values
            flow_labels = flows['label'].values if 'label' in flows.columns else None
            
            # Create overlapping sequences
            step = max(1, self.sequence_length - self.overlap)
            for i in range(0, len(flows) - self.sequence_length + 1, step):
                seq_features = flow_features[i:i + self.sequence_length]
                sequences.append(seq_features)
                
                if include_labels and flow_labels is not None:
                    # Label sequence as attack if any flow in sequence is attack
                    seq_label = int(np.any(flow_labels[i:i + self.sequence_length]))
                    labels.append(seq_label)
        
        sequences = np.array(sequences)
        labels = np.array(labels) if labels else None
        
        logger.info(f"✅ Created {len(sequences)} sequences of length {self.sequence_length}")
        if labels is not None:
            attack_seqs = np.sum(labels)
            logger.info(f"📈 Attack sequences: {attack_seqs} ({attack_seqs/len(labels):.1%})")
            
        return sequences, labels
        
    def _add_sequence_features(self, sequences: np.ndarray) -> np.ndarray:
        """Add statistical features across sequence dimension"""
        if not self.config.get('feature_engineering', {}).get('sequence_stats', False):
            return sequences
            
        # Calculate statistics across time dimension
        seq_mean = np.mean(sequences, axis=1, keepdims=True)
        seq_std = np.std(sequences, axis=1, keepdims=True)
        seq_min = np.min(sequences, axis=1, keepdims=True)
        seq_max = np.max(sequences, axis=1, keepdims=True)
        
        # Compute trends (difference between last and first)
        seq_trend = sequences[:, -1:, :] - sequences[:, :1, :]
        
        # Broadcast statistics to all time steps
        batch_size, seq_len, n_features = sequences.shape
        
        stats_features = np.concatenate([
            np.repeat(seq_mean, seq_len, axis=1),
            np.repeat(seq_std, seq_len, axis=1),
            np.repeat(seq_min, seq_len, axis=1),
            np.repeat(seq_max, seq_len, axis=1),
            np.repeat(seq_trend, seq_len, axis=1)
        ], axis=2)
        
        # Combine original and statistical features
        enhanced_sequences = np.concatenate([sequences, stats_features], axis=2)
        
        logger.info(f"🔧 Enhanced sequences: {sequences.shape} -> {enhanced_sequences.shape}")
        return enhanced_sequences
        
    def fit(self, df: pd.DataFrame) -> 'SequencePreprocessor':
        """Fit the preprocessor on training data"""
        logger.info("🔧 Fitting sequence preprocessor...")
        
        # Group flows by connection
        grouped_flows = self._group_flows_by_connection(df)
        
        # Create sequences
        sequences, labels = self._create_sequences(grouped_flows, include_labels=True)
        
        if len(sequences) == 0:
            raise ValueError("No sequences could be created. Check sequence_length parameter.")
        
        # Fit scaler on flattened sequences (all timesteps and features)
        flattened_sequences = sequences.reshape(-1, sequences.shape[-1])
        self.feature_scaler.fit(flattened_sequences)
        
        # Store statistics
        self.sequence_stats = {
            'n_sequences': len(sequences),
            'sequence_shape': sequences.shape,
            'attack_rate': np.mean(labels) if labels is not None else 0.0,
            'n_connections': len(grouped_flows),
            'avg_flows_per_connection': np.mean([len(flows) for flows in grouped_flows.values()])
        }
        
        self.is_fitted = True
        logger.info("✅ Sequence preprocessor fitted successfully")
        logger.info(f"📊 Statistics: {self.sequence_stats}")
        
        return self
        
    def transform(self, df: pd.DataFrame, include_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data into sequences"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        logger.info("🔄 Transforming data into sequences...")
        
        # Group flows
        grouped_flows = self._group_flows_by_connection(df)
        
        # Create sequences
        sequences, labels = self._create_sequences(grouped_flows, include_labels=include_labels)
        
        if len(sequences) == 0:
            logger.warning("⚠️ No sequences created from input data")
            return np.array([]), np.array([])
        
        # Scale features
        original_shape = sequences.shape
        flattened_sequences = sequences.reshape(-1, sequences.shape[-1])
        scaled_flattened = self.feature_scaler.transform(flattened_sequences)
        sequences = scaled_flattened.reshape(original_shape)
        
        # Add enhanced features if configured
        sequences = self._add_sequence_features(sequences)
        
        logger.info(f"✅ Transformed to {sequences.shape} sequences")
        
        return sequences, labels
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
        
    def save(self, filepath: str):
        """Save preprocessor to file"""
        save_data = {
            'config': self.config,
            'features': self.features,
            'sequence_length': self.sequence_length,
            'overlap': self.overlap,
            'feature_scaler': self.feature_scaler,
            'is_fitted': self.is_fitted,
            'sequence_stats': self.sequence_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"💾 Sequence preprocessor saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'SequencePreprocessor':
        """Load preprocessor from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create instance
        preprocessor = cls(save_data['config'])
        
        # Restore state
        preprocessor.features = save_data['features']
        preprocessor.sequence_length = save_data['sequence_length']
        preprocessor.overlap = save_data['overlap']
        preprocessor.feature_scaler = save_data['feature_scaler']
        preprocessor.is_fitted = save_data['is_fitted']
        preprocessor.sequence_stats = save_data['sequence_stats']
        
        logger.info(f"📂 Sequence preprocessor loaded from {filepath}")
        return preprocessor
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features including enhanced ones"""
        base_features = self.features.copy()
        
        if self.config.get('feature_engineering', {}).get('sequence_stats', False):
            # Add statistical feature names
            for stat in ['mean', 'std', 'min', 'max', 'trend']:
                for feature in self.features:
                    base_features.append(f'{feature}_{stat}')
                    
        return base_features

def main():
    """Test the sequence preprocessor"""
    import yaml
    
    # Load config
    with open('config/model_config_sequence_v4.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_csv('data/processed/flows.csv')
    
    # Create and test preprocessor
    preprocessor = SequencePreprocessor(config)
    sequences, labels = preprocessor.fit_transform(df)
    
    print(f"📊 RESULTS:")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Attack sequences: {np.sum(labels)} ({np.mean(labels):.1%})")
    print(f"Feature names: {preprocessor.get_feature_names()}")
    
if __name__ == "__main__":
    main()