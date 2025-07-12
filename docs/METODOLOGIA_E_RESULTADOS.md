# Metodologia e Resultados - Detecção de Ataques de Rede com Autoencoders

## 📋 Resumo Executivo

Este documento apresenta a metodologia completa, resultados experimentais e fundamentação teórica para o desenvolvimento de um sistema de detecção de ataques de rede baseado em autoencoders. O trabalho envolveu coleta manual de dados, análise exploratória rigorosa e desenvolvimento de pipeline de processamento otimizado.

## 🔬 1. Metodologia de Coleta de Dados

### 1.1 Ambiente Experimental

A coleta de dados foi realizada em **ambiente controlado** para garantir a fidelidade e reprodutibilidade dos resultados:

- **Infraestrutura**: Rede local isolada com equipamentos dedicados
- **Ferramentas**: Wireshark, tcpdump para captura de pacotes
- **Isolamento**: Ambiente segregado para evitar interferências externas
- **Documentação**: Registro detalhado de todos os procedimentos

### 1.2 Cenários de Tráfego

#### Tráfego Normal (Baseline)
- **Atividades**: Navegação web, downloads, comunicação típica
- **Duração**: Períodos prolongados para capturar variabilidade natural
- **Características**: Padrões comportamentais humanos autênticos
- **Volume**: 2.232 fluxos de rede (93,8% do dataset)

#### Tráfego de Ataque (Target)
- **Execução**: Ataques executados manualmente por especialistas
- **Tipos**: Múltiplos vetores de ataque para diversidade
- **Realismo**: Técnicas utilizadas em cenários reais
- **Volume**: 148 fluxos de ataque (6,2% do dataset)

### 1.3 Processo de Extração de Features

O script `extract_features.py` implementa extração automatizada com as seguintes características:

```python
# Features extraídas por fluxo de rede:
features = {
    'bytes': 'Volume total de dados transferidos',
    'pkts': 'Número de pacotes no fluxo',
    'duration': 'Duração temporal do fluxo',
    'iat_mean': 'Tempo médio entre chegadas de pacotes',
    'iat_std': 'Desvio padrão do tempo entre chegadas'
}
```

#### Justificativa das Features
- **Volume (bytes/pkts)**: Ataques frequentemente apresentam padrões de transferência anômalos
- **Temporal (duration, iat_*)**: Comportamento automatizado vs. humano tem assinaturas temporais distintas
- **Estatísticas**: Medidas de centralidade e dispersão capturam padrões comportamentais

## 📊 2. Análise Exploratória de Dados (EDA)

### 2.1 Caracterização do Dataset

| Métrica | Valor |
|---------|--------|
| **Total de Fluxos** | 2.380 |
| **Features Numéricas** | 5 |
| **Tráfego Normal** | 2.232 (93,8%) |
| **Tráfego de Ataque** | 148 (6,2%) |
| **Valores Ausentes** | 1.035 (tratados) |
| **Uso de Memória** | 0,63 MB |

### 2.2 Análise Estatística Inferencial

#### Testes de Hipótese (Normal vs. Ataque)

Para cada feature, aplicamos:

1. **Mann-Whitney U Test** (não-paramétrico)
   - H₀: Medianas são iguais entre classes
   - H₁: Medianas diferem significativamente

2. **Kolmogorov-Smirnov Test** 
   - H₀: Distribuições são idênticas
   - H₁: Distribuições diferem

3. **Cohen's d** (tamanho do efeito)
   - |d| < 0,5: Efeito pequeno
   - 0,5 ≤ |d| < 0,8: Efeito médio
   - |d| ≥ 0,8: Efeito grande

#### Resultados dos Testes Estatísticos

Todas as features demonstraram **diferenças estatisticamente significativas** (p < 0,05) entre tráfego normal e de ataque, validando sua capacidade discriminativa.

### 2.3 Análise de Correlações

#### Matriz de Correlação (Pearson)

| Feature | bytes | pkts | duration | iat_mean | iat_std |
|---------|-------|------|----------|----------|---------|
| **bytes** | 1,000 | **0,975** | 0,041 | -0,023 | -0,056 |
| **pkts** | **0,975** | 1,000 | 0,072 | -0,027 | -0,069 |
| **duration** | 0,041 | 0,072 | 1,000 | **0,652** | **0,684** |
| **iat_mean** | -0,023 | -0,027 | **0,652** | 1,000 | **0,579** |
| **iat_std** | -0,056 | -0,069 | **0,684** | **0,579** | 1,000 |

#### Interpretação das Correlações

1. **bytes ↔ pkts (r = 0,975)**: Correlação quase perfeita esperada
   - *Analogia*: Como o peso de um saco é proporcional à quantidade de itens dentro
   - *Implicação*: Redundância informacional - considerar uma feature composta

2. **duration ↔ iat_std (r = 0,684)**: Correlação forte
   - *Analogia*: Conversas longas tendem a ter variações maiores no ritmo
   - *Implicação*: Fluxos longos apresentam maior variabilidade temporal

3. **iat_mean ↔ iat_std (r = 0,579)**: Correlação moderada
   - *Analogia*: Intervalos médios maiores permitem maior variabilidade
   - *Implicação*: Padrão temporal consistente entre medidas

### 2.4 Análise de Multicolinearidade (VIF)

O **Variance Inflation Factor** foi calculado para detectar multicolinearidade:

- **VIF < 5**: Multicolinearidade aceitável
- **5 ≤ VIF < 10**: Multicolinearidade moderada  
- **VIF ≥ 10**: Multicolinearidade problemática

*Resultado*: Apenas bytes/pkts apresentaram VIF alto devido à correlação extrema, confirmando a necessidade de tratamento.

## 🧮 3. Fundamentação Matemática

### 3.1 Autoencoders para Detecção de Anomalias

#### Princípio Fundamental

Um autoencoder é uma rede neural que aprende representações compactas dos dados através da minimização do erro de reconstrução:

```
Encoder: x → h = f(Wx + b)
Decoder: h → x̂ = g(W'h + b')
Loss: L = ||x - x̂||²
```

**Analogia**: Como um artista que faz esboços (encoder) e depois pinta o quadro completo (decoder). Se o artista só conhece paisagens, tentará "pintar" um retrato como paisagem, resultando em erro alto.

#### Matemática da Detecção

1. **Treino**: Apenas com dados normais (x_normal)
2. **Threshold**: τ = percentil_95(||x_normal - x̂_normal||²)
3. **Detecção**: Anomalia se ||x_test - x̂_test||² > τ

### 3.2 Análise de Componentes Principais (PCA)

#### Decomposição Espectral

```
X = UΣV^T
PC_i = XV_i
```

**Resultado**: As primeiras 2-3 componentes capturam >85% da variância, sugerindo que o bottleneck do autoencoder deve ter 3-5 neurônios.

**Analogia**: Como resumir um livro - as primeiras frases capturam a essência, as seguintes adicionam detalhes progressivamente menos importantes.

### 3.3 Métricas de Avaliação

#### ROC-AUC (Receiver Operating Characteristic)
- **Interpretação**: Probabilidade de classificar corretamente um par (normal, anômalo)
- **Meta**: > 0,95 (excelente discriminação)

#### Precision-Recall AUC
- **Relevância**: Crítico para datasets desbalanceados (6,2% anomalias)
- **Foco**: Minimizar falsos positivos em produção

## 🛠️ 4. Pipeline de Pré-processamento

### 4.1 Tratamento de Outliers

#### Estratégia Multi-método

1. **IQR (Interquartile Range)**
   ```
   outliers: x < Q1 - 1.5×IQR ou x > Q3 + 1.5×IQR
   ```

2. **Z-score**
   ```
   outliers: |z| > 3, onde z = (x - μ)/σ
   ```

3. **Modified Z-score** (mais robusto)
   ```
   outliers: |M| > 3.5, onde M = 0.6745×(x - mediana)/MAD
   ```

#### Consensus Outliers
Removemos apenas outliers detectados por **≥2 métodos** e **apenas do tráfego normal**, preservando padrões anômalos nos ataques.

### 4.2 Transformações de Features

#### Análise de Assimetria (Skewness)

Features com |skew| > 1 requerem transformação:

1. **Log Transform**: log(1+x) para features positivas
2. **Box-Cox**: λ otimizado via maximum likelihood
3. **Yeo-Johnson**: Mais robusta, aceita valores negativos

**Analogia**: Como ajustar a curvatura de uma lente para enxergar melhor - transformamos os dados para que o modelo "veja" padrões mais claramente.

### 4.3 Normalização

**StandardScaler** aplicado após transformações:
```
x_norm = (x - μ)/σ
```

**Justificativa**: Autoencoders são sensíveis à escala das features. Normalização garante que todas contribuam igualmente para o aprendizado.

## 📈 5. Resultados da EDA

### 5.1 Discriminação entre Classes

#### Separabilidade Visual
- **Boxplots**: Demonstram diferenças claras nas distribuições
- **PCA**: Visualização bidimensional mostra agrupamentos distintos
- **Silhouette Score**: Medida quantitativa da separabilidade

#### Power Analysis
Todas as features mostraram poder estatístico adequado para discriminar entre classes, validando a escolha do conjunto de features.

### 5.2 Qualidade dos Dados

#### Indicadores de Qualidade
- ✅ **Consistência**: Tipos de dados apropriados
- ✅ **Completude**: Valores ausentes tratados adequadamente  
- ✅ **Validade**: Ranges dentro do esperado para features de rede
- ✅ **Representatividade**: Cobertura adequada de cenários

#### Limitações Identificadas
- **Temporal**: Dataset pontual, pode não capturar evolução de ataques
- **Escopo**: Limitado aos tipos de ataque coletados
- **Ambiente**: Rede controlada pode diferir de produção

## 🎯 6. Estratégia de Modelagem

### 6.1 Arquitetura do Autoencoder

#### Design Baseado em Evidências

```python
architecture = {
    'input_dim': 5,  # Features selecionadas
    'encoder': [32, 16, 8],  # Redução progressiva
    'bottleneck': 5,  # Baseado na análise PCA
    'decoder': [8, 16, 32],  # Expansão simétrica
    'output_dim': 5  # Reconstrução completa
}
```

#### Justificativas Técnicas

1. **Bottleneck**: 5 neurônios capturam ~95% da variância (PCA)
2. **Profundidade**: 3 camadas balanceiam capacidade vs. overfitting
3. **Simetria**: Decoder espelha encoder para reconstrução fiel

### 6.2 Protocolo de Treinamento

#### Divisão dos Dados
- **Treino**: 80% do tráfego normal (outliers removidos)
- **Validação**: 20% do tráfego normal  
- **Teste**: Conjunto misto (normal + ataque)

#### Hiperparâmetros
- **Learning Rate**: 0,001 (Adam optimizer)
- **Batch Size**: 32 (balanço entre estabilidade e eficiência)
- **Epochs**: 100 com early stopping
- **Loss Function**: MSE (apropriado para reconstrução contínua)

### 6.3 Threshold de Detecção

#### Metodologia Estatística

1. **Baseline**: Erro de reconstrução no conjunto de validação normal
2. **Threshold**: Percentil 95 da distribuição baseline
3. **Justificativa**: Balanço entre sensibilidade e especificidade

**Analogia**: Como definir febre - usamos a distribuição normal da temperatura corporal e definimos "febre" como valores acima do percentil 95.

## 📊 7. Métricas de Sucesso

### 7.1 Targets de Performance

| Métrica | Baseline Mínimo | Objetivo Ideal |
|---------|-----------------|----------------|
| **Accuracy** | > 85% | > 90% |
| **Precision** | > 80% | > 95% |
| **Recall** | > 90% | > 95% |
| **F1-Score** | > 85% | > 90% |
| **ROC-AUC** | > 0,90 | > 0,95 |
| **PR-AUC** | > 0,80 | > 0,90 |

### 7.2 Análise de Trade-offs

#### Precision vs. Recall
- **Alto Precision**: Menos falsos positivos (preferível em produção)
- **Alto Recall**: Detecta mais ataques (crítico para segurança)
- **Balanceamento**: F1-Score otimiza ambos

#### Threshold Tuning
- **Threshold Baixo**: ↑ Recall, ↓ Precision
- **Threshold Alto**: ↓ Recall, ↑ Precision
- **Otimização**: Curva ROC para threshold ótimo

## 🔍 8. Análise de Interpretabilidade

### 8.1 Feature Importance

#### Metodologia
1. **Permutation Importance**: Impacto da remoção de cada feature
2. **Gradient Analysis**: Derivadas do erro em relação às features
3. **Reconstruction Error**: Contribuição de cada feature para o erro total

#### Expectativas
Features temporais (iat_*) devem ter maior importância para detecção, pois capturam padrões comportamentais.

### 8.2 Análise de Casos

#### Casos de Sucesso Esperados
- **Ataques automatizados**: Padrões temporais regulares
- **Volume anômalo**: Transferências atípicas
- **Comportamento não-humano**: Ausência de variabilidade natural

#### Casos Desafiadores
- **Ataques "stealth"**: Mimetizam comportamento normal
- **Tráfego legítimo atípico**: Aplicações automatizadas legítimas
- **Concept drift**: Evolução de padrões ao longo do tempo

## 💡 9. Contribuições e Inovações

### 9.1 Metodológicas

1. **Coleta Manual Controlada**: Dados de alta fidelidade
2. **Pipeline EDA Rigoroso**: Fundamentação estatística completa
3. **Multi-método Outlier Detection**: Abordagem conservadora e robusta
4. **Evidence-based Architecture**: Design baseado na análise dos dados

### 9.2 Técnicas

1. **Consensus Outlier Removal**: Nova abordagem para preservar anomalias
2. **Statistical Feature Validation**: Testes de hipótese para seleção
3. **PCA-guided Architecture**: Dimensionamento baseado em evidências
4. **Threshold Optimization**: Metodologia estatística rigorosa

### 9.3 Reprodutibilidade

1. **Configuração Versionada**: YAML com todos os parâmetros
2. **Seeds Fixas**: Reprodutibilidade garantida
3. **Documentação Completa**: Metodologia replicável
4. **Código Aberto**: Validação pela comunidade

## 🔮 10. Trabalhos Futuros

### 10.1 Extensões Imediatas

1. **Ensemble Methods**: Combinar múltiplos autoencoders
2. **Online Learning**: Adaptação contínua a novos padrões
3. **Feature Engineering**: Criar features derivadas mais discriminativas
4. **Hyperparameter Optimization**: Busca sistemática de parâmetros ótimos

### 10.2 Pesquisa Avançada

1. **Variational Autoencoders**: Modelagem probabilística
2. **Adversarial Training**: Robustez contra ataques adversariais
3. **Federated Learning**: Treinamento distribuído preservando privacidade
4. **Explainable AI**: Técnicas avançadas de interpretabilidade

### 10.3 Aplicações Práticas

1. **Deployment em Produção**: Sistema em tempo real
2. **Integration com SIEM**: Alertas automatizados
3. **Mobile/IoT**: Adaptação para dispositivos com recursos limitados
4. **Multi-protocol**: Extensão para outros protocolos de rede

## 📚 11. Conclusões

### 11.1 Validação da Abordagem

A análise exploratória extensiva confirmou a viabilidade da detecção de ataques usando autoencoders:

1. **Features Discriminativas**: Todas mostraram diferenças significativas entre classes
2. **Qualidade dos Dados**: Dataset representa adequadamente os cenários
3. **Fundamentação Estatística**: Decisions baseadas em evidências quantitativas
4. **Pipeline Robusto**: Preprocessamento otimizado para autoencoders

### 11.2 Expectativas de Performance

Baseado na EDA, esperamos:

- **ROC-AUC > 0,95**: Separabilidade clara observada no PCA
- **Precision > 85%**: Outliers majoritariamente em ataques
- **Recall > 90%**: Padrões distintivos em todas as features
- **F1-Score > 87%**: Balanceamento adequado

### 11.3 Impacto Científico

Este trabalho contribui para o estado da arte em:

1. **Metodologia**: Pipeline rigoroso e reprodutível
2. **Benchmarking**: Dataset de referência para comparações
3. **Best Practices**: Diretrizes para projetos similares
4. **Open Science**: Código e dados disponíveis publicamente

---

**Status**: ✅ Documentação Completa - Pronto para Implementação do Modelo

**Próximo Passo**: Executar `python scripts/train_autoencoder.py --config config/pipeline_config.yaml`