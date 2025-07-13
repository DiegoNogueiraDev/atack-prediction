# src/preprocessor.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer

# Configuração das features baseada no EDA
# Features significativas: bytes, pkts, iat_mean
# Features removidas: duration, iat_std (não significativas)
SELECTED_FEATURES = ['bytes', 'pkts', 'iat_mean']

# Features por tipo de transformação (baseado no EDA)
FEATURES_BOXCOX = ['bytes']  # Apenas valores positivos
FEATURES_YEOJOHNSON = ['pkts', 'iat_mean']  # Pode ter zeros
FEATURES_ALL = FEATURES_BOXCOX + FEATURES_YEOJOHNSON

def create_preprocessor(features_to_use=None, apply_transformations=True):
    """
    Cria pipeline de pré-processamento unificado baseado no EDA.
    
    Args:
        features_to_use: Lista de features a usar (default: SELECTED_FEATURES)
        apply_transformations: Se aplicar transformações de skewness (default: True)
    
    Returns:
        Pipeline completo de pré-processamento
    """
    if features_to_use is None:
        features_to_use = SELECTED_FEATURES.copy()
    
    # Verificar quais features estão disponíveis
    boxcox_features = [f for f in FEATURES_BOXCOX if f in features_to_use]
    yj_features = [f for f in FEATURES_YEOJOHNSON if f in features_to_use]
    
    transformers = []
    
    if apply_transformations:
        # Imputação + Box-Cox para features que precisam (apenas valores positivos)
        if boxcox_features:
            transformers.append((
                'boxcox_pipeline',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('boxcox', PowerTransformer(method='box-cox', standardize=False))
                ]),
                boxcox_features
            ))
        
        # Imputação + Yeo-Johnson para features que podem ter zeros
        if yj_features:
            transformers.append((
                'yeojohnson_pipeline', 
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=False))
                ]),
                yj_features
            ))
    else:
        # Apenas imputação sem transformações
        all_features = features_to_use
        transformers.append((
            'imputer_only',
            SimpleImputer(strategy='median'),
            all_features
        ))
    
    # Pipeline completo
    preprocessor = Pipeline([
        ('preprocessing', ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop features não especificadas
            sparse_threshold=0  # Retorna array denso
        )),
        ('scaler', StandardScaler())
    ])
    
    return preprocessor

# Pipeline padrão (backward compatibility)
preprocessor = create_preprocessor()

def fit_preprocessor(df: pd.DataFrame, features_to_use=None, apply_transformations=True):
    """
    Ajusta o preprocessador aos dados.
    
    Args:
        df: DataFrame com os dados
        features_to_use: Features a usar (default: SELECTED_FEATURES)  
        apply_transformations: Se aplicar transformações (default: True)
    
    Returns:
        Pipeline ajustado
    """
    if features_to_use is None:
        features_to_use = SELECTED_FEATURES.copy()
    
    # Verificar se todas as features existem
    missing_features = [f for f in features_to_use if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features não encontradas no DataFrame: {missing_features}")
    
    # Criar e ajustar pipeline
    pipeline = create_preprocessor(features_to_use, apply_transformations)
    X = df[features_to_use]
    
    return pipeline.fit(X)

def transform_preprocessor(df: pd.DataFrame, fitted_preprocessor=None, features_to_use=None):
    """
    Transforma os dados usando preprocessador ajustado.
    
    Args:
        df: DataFrame para transformar
        fitted_preprocessor: Pipeline já ajustado (se None, usa o global)
        features_to_use: Features a usar (default: SELECTED_FEATURES)
    
    Returns:
        Array numpy transformado
    """
    if features_to_use is None:
        features_to_use = SELECTED_FEATURES.copy()
    
    if fitted_preprocessor is None:
        fitted_preprocessor = preprocessor
    
    X = df[features_to_use]
    return fitted_preprocessor.transform(X)

# Função auxiliar para carregar configuração do EDA
def get_eda_config():
    """Retorna configuração baseada nos resultados do EDA"""
    return {
        'significant_features': ['bytes', 'pkts', 'iat_mean'],
        'non_significant_features': ['duration', 'iat_std'],
        'high_skew_features': ['bytes', 'pkts'],
        'missing_value_features': ['iat_std', 'sport', 'dport'],
        'outlier_threshold': 4.0,  # Z-score threshold
        'correlation_threshold': 0.7
    }
