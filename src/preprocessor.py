# src/preprocessor.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer

# 1) Escolha de features (baseado na EDA)
#    - bytes (Box-Cox)
#    - iat_mean (Yeo-Johnson)
selected_features = ['bytes', 'iat_mean']

# 2) Separar colunas enviesadas e tratar
skewed = ['bytes']
normal = ['iat_mean']

# Box-Cox para bytes (λ será ajustado internamente)
boxcox_tr = Pipeline([
    ('select', FunctionTransformer(lambda df: df[skewed], validate=False)),
    ('boxcox', PowerTransformer(method='box-cox', standardize=False))
])

# Yeo-Johnson para iat_mean
yj_tr = Pipeline([
    ('select', FunctionTransformer(lambda df: df[normal], validate=False)),
    ('yeo', PowerTransformer(method='yeo-johnson', standardize=False))
])

# escala final
scaler = StandardScaler()

preprocessor = Pipeline([
    ('features', ColumnTransformer([
        ('boxcox_bytes', boxcox_tr, skewed),
        ('yj_iatmean', yj_tr, normal),
    ])),
    ('scale', scaler)
])

def fit_preprocessor(df: pd.DataFrame):
    X = df[selected_features]
    return preprocessor.fit(X)

def transform_preprocessor(df: pd.DataFrame):
    X = df[selected_features]
    return preprocessor.transform(X)
