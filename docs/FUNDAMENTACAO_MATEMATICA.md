# Fundamentação Matemática - Autoencoders para Detecção de Anomalias

## 🧮 1. Introdução Teórica

### 1.1 O Problema Matemático

A detecção de anomalias em tráfego de rede pode ser formulada como um problema de **estimação de densidade** em alta dimensionalidade. Dado um conjunto de observações normais X_normal ⊂ ℝᵈ, queremos aprender uma função f: ℝᵈ → ℝ⁺ que estime a densidade p(x) dos dados normais.

**Analogia**: Como um sommelier que aprende a identificar vinhos de qualidade. Após provar muitos vinhos excelentes, consegue detectar quando algo está "fora do padrão" - mesmo sem nunca ter provado aquele defeito específico.

### 1.2 Limitações de Métodos Tradicionais

#### Curse of Dimensionality
Em espaços de alta dimensão (d >> 1), métodos baseados em distância falham:

```
Volume da esfera unitária em d dimensões:
V_d = π^(d/2) / Γ(d/2 + 1)

Para d >> 1: V_d → 0
```

**Implicação**: Pontos ficam uniformemente distantes, perdendo discriminação.

#### Estimação Paramétrica vs. Não-paramétrica

| Método | Vantagens | Desvantagens |
|--------|-----------|--------------|
| **Paramétrico** (Gaussian) | Eficiente, interpretável | Assume distribuição específica |
| **Não-paramétrico** (KDE) | Flexível | Curse of dimensionality |
| **Neural** (Autoencoder) | Aprende manifolds | Black-box, requer dados |

## 🔬 2. Teoria dos Autoencoders

### 2.1 Definição Formal

Um autoencoder é um par de funções (f_enc, f_dec) que minimizam a distância entre entrada e reconstrução:

```
Encoder: f_enc: ℝᵈ → ℝᵏ, h = f_enc(x)
Decoder: f_dec: ℝᵏ → ℝᵈ, x̂ = f_dec(h)

Objetivo: min_θ 𝔼[||x - f_dec(f_enc(x))||²]
```

onde k << d (bottleneck) força aprendizado de representação compacta.

### 2.2 Manifold Learning Perspective

#### Hipótese do Manifold

Dados reais frequentemente residem em **manifolds de baixa dimensionalidade** embebidos em espaços de alta dimensão:

```
X ⊂ M ⊂ ℝᵈ, onde dim(M) = k << d
```

**Analogia**: Como uma folha de papel (2D) amassada no espaço 3D. O papel mantém sua estrutura bidimensional intrínseca, mesmo deformado no espaço maior.

#### Autoencoder como Aproximador de Manifold

O encoder aprende uma **função de projeção** φ: M → ℝᵏ, e o decoder aprende a **função inversa** ψ: ℝᵏ → M.

Para pontos x ∈ M (normais):
```
||x - ψ(φ(x))||² ≈ 0
```

Para pontos x ∉ M (anômalos):
```
||x - ψ(φ(x))||² >> 0
```

### 2.3 Capacidade e Generalização

#### Universal Approximation Theorem

Redes neurais com uma camada oculta podem aproximar qualquer função contínua, mas:

1. **Largura necessária** pode ser exponencial na dimensão
2. **Otimização** pode não encontrar o mínimo global
3. **Generalização** depende da complexidade dos dados

#### Bias-Variance Tradeoff

- **Underfitting** (high bias): Modelo muito simples, erro alto em treino e teste
- **Overfitting** (high variance): Modelo muito complexo, erro baixo em treino, alto em teste
- **Sweet spot**: Balanço ótimo via regularização e validação

**Analogia**: Como uma roupa - muito apertada (underfitting) não serve, muito larga (overfitting) não define bem a forma. O tamanho certo fica perfeito.

## 📊 3. Matemática da Detecção de Anomalias

### 3.1 Reconstruction Error como Métrica

#### Definição
Para uma observação x, o erro de reconstrução é:

```
RE(x) = ||x - f_dec(f_enc(x))||²
```

onde ||·|| pode ser norma L2, L1, ou outras métricas.

#### Propriedades Estatísticas

Para dados normais X_normal ~ p_normal(x):

```
RE_normal = {RE(x_i) : x_i ∈ X_normal}
```

**Assumindo** RE_normal segue uma distribuição conhecida (ex: Gamma), podemos calcular quantis:

```
τ_α = Q_α(RE_normal)
```

onde Q_α é o quantil α da distribuição.

### 3.2 Threshold Selection

#### Abordagem Estatística

1. **Método do Percentil**:
   ```
   τ = percentile(RE_normal, α)
   Exemplo: α = 95% → 5% falsos positivos esperados
   ```

2. **Método σ-multiplier**:
   ```
   τ = μ(RE_normal) + k·σ(RE_normal)
   Exemplo: k = 2.5 → Chebyshev bound
   ```

3. **Método da Máxima Verossimilhança**:
   ```
   Assumir RE_normal ~ Gamma(α, β)
   Estimar parâmetros via MLE
   τ = Q_p(Gamma(α̂, β̂))
   ```

#### Trade-off Fundamental

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

Threshold ↑: Precision ↑, Recall ↓
Threshold ↓: Precision ↓, Recall ↑
```

**Analogia**: Como ajustar a sensibilidade de um detector de metal. Muito sensível encontra tudo (alto recall), mas também lixo (baixo precision). Pouco sensível perde tesouros (baixo recall), mas o que encontra é valioso (alto precision).

### 3.3 Análise ROC e PR

#### Curva ROC (Receiver Operating Characteristic)

```
TPR = TP / (TP + FN)  (Sensitivity)
FPR = FP / (FP + TN)  (1 - Specificity)

AUC_ROC = ∫₀¹ TPR(FPR) d(FPR)
```

**Interpretação**: Probabilidade de classificar corretamente um par (normal, anômalo) escolhido aleatoriamente.

#### Curva Precision-Recall

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

AUC_PR = ∫₀¹ Precision(Recall) d(Recall)
```

**Vantagem para dados desbalanceados**: Foca no desempenho na classe minoritária (anomalias).

## 🏗️ 4. Arquitetura e Design

### 4.1 Dimensionamento do Bottleneck

#### Análise via PCA

O PCA fornece uma baseline para o dimensionamento:

```
X = UΣV^T

Variância explicada pelos primeiros k componentes:
R²(k) = Σᵢ₌₁ᵏ σᵢ² / Σᵢ₌₁ᵈ σᵢ²
```

**Regra prática**: Escolher k tal que R²(k) ≥ 0.95.

#### Information Theory Perspective

O bottleneck força **compressão com perda**:

```
I(X; H) ≤ H(H) ≤ k·log₂(n_neurons)
```

onde I(X; H) é a informação mútua entre entrada e representação oculta.

**Analogia**: Como fazer uma mala - só cabe o essencial. O autoencoder aprende quais aspectos dos dados são "essenciais" para reconstrução.

### 4.2 Função de Ativação

#### ReLU vs. Alternatives

```
ReLU: f(x) = max(0, x)
Leaky ReLU: f(x) = max(αx, x), α ∈ (0, 1)
ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
```

**Para autoencoders**:
- **Encoder**: ReLU (sparsity, computational efficiency)
- **Decoder final**: Linear (permite valores negativos na reconstrução)

#### Normalização

**Batch Normalization**:
```
BN(x) = γ·(x - μ)/σ + β
```

**Benefícios**: Estabiliza gradientes, acelera convergência, regularização implícita.

### 4.3 Função de Perda

#### Mean Squared Error (MSE)

```
L_MSE = (1/n)Σᵢ₌₁ⁿ ||xᵢ - x̂ᵢ||²
```

**Vantagens**: Diferenciável, penaliza outliers quadraticamente.
**Desvantagens**: Sensível a outliers extremos.

#### Mean Absolute Error (MAE)

```
L_MAE = (1/n)Σᵢ₌₁ⁿ ||xᵢ - x̂ᵢ||₁
```

**Vantagens**: Robusto a outliers.
**Desvantagens**: Não diferenciável em 0, gradientes constantes.

#### Huber Loss (Hybrid)

```
L_Huber = {
  (1/2)(x - x̂)² if |x - x̂| ≤ δ
  δ|x - x̂| - (1/2)δ² if |x - x̂| > δ
}
```

**Benefício**: Combina robustez (MAE) com suavidade (MSE).

## 📈 5. Otimização e Convergência

### 5.1 Gradient Descent Variants

#### Adam Optimizer

```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²

m̂_t = m_t/(1-β₁^t)
v̂_t = v_t/(1-β₂^t)

θ_{t+1} = θ_t - α·m̂_t/(√v̂_t + ε)
```

**Parâmetros típicos**: α=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸

**Analogia**: Como um esquiador descendo uma montanha na neblina. O momento (m_t) mantém a direção geral, a adaptação (v_t) ajusta a velocidade baseada na inclinação local.

### 5.2 Regularização

#### Weight Decay (L2 Regularization)

```
L_total = L_reconstruction + λ·Σ_i θᵢ²
```

**Efeito**: Penaliza pesos grandes, força suavidade na função aprendida.

#### Dropout

```
During training: y = f(x ⊙ mask), mask ~ Bernoulli(p)
During inference: y = p·f(x)
```

**Benefício**: Reduz co-adaptação entre neurônios, melhora generalização.

#### Early Stopping

Monitora erro de validação:
```
if val_loss[t] > val_loss[t-patience]:
    stop_training()
```

**Analogia**: Como estudar para uma prova - há um ponto ótimo onde parar de estudar evita "overtraining" e cansaço mental.

### 5.3 Learning Rate Scheduling

#### Decaimento Exponencial

```
lr(t) = lr₀·γ^t, onde γ ∈ (0, 1)
```

#### Cosine Annealing

```
lr(t) = lr_min + (lr_max - lr_min)·(1 + cos(πt/T))/2
```

**Vantagem**: Permite "re-aquecimento" para escapar de mínimos locais.

## 🎯 6. Métricas de Avaliação Avançadas

### 6.1 Silhouette Score para Representações

```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}

onde:
a(i) = distância média para pontos da mesma classe
b(i) = distância média para pontos da classe mais próxima
```

**Interpretação**: s ∈ [-1, 1], valores próximos de 1 indicam boa separação.

### 6.2 Calinski-Harabasz Index

```
CH = (trace(B_k)/(k-1)) / (trace(W_k)/(n-k))

onde:
B_k = between-cluster sum of squares
W_k = within-cluster sum of squares
```

**Interpretação**: Razão entre dispersão inter-cluster e intra-cluster.

### 6.3 Adjusted Rand Index (ARI)

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

onde RI é o Rand Index e E[RI] é seu valor esperado.

**Benefício**: Corrigido para concordância casual, ARI ∈ [-1, 1].

## 🔧 7. Aspectos Computacionais

### 7.1 Complexidade Temporal

#### Forward Pass
```
Encoder: O(d·h₁ + h₁·h₂ + ... + h_{n-1}·k)
Decoder: O(k·h_{n-1} + ... + h₂·h₁ + h₁·d)

Total: O(d·Σhᵢ + k·Σhᵢ)
```

#### Backward Pass (Backpropagation)
Complexidade similar ao forward pass, mas com overhead adicional para cálculo de gradientes.

### 7.2 Complexidade Espacial

```
Memória para pesos: O(d·h₁ + Σᵢhᵢ·hᵢ₊₁ + hₙ·k)
Memória para ativações (batch B): O(B·Σhᵢ)
```

### 7.3 Paralelização

#### Mini-batch Processing
```
Batch size B: balanço entre:
- Gradiente mais estável (B ↑)
- Menor uso de memória (B ↓)
- Maior paralelização (B ↑)
```

**Regra prática**: B ∈ [16, 128] para a maioria dos casos.

## 🌊 8. Análise de Convergência

### 8.1 Landscape de Perda

#### Problema Non-convex

Função de perda de redes neurais é **não-convexa**:
- Múltiplos mínimos locais
- Platôs e saddle points
- Garantias de convergência limitadas

#### Teorema de Convergência (Simplificado)

Para step size apropriado α:
```
lim_{t→∞} ||∇L(θ_t)|| = 0
```

**Interpretação**: Convergência para ponto crítico (não necessariamente mínimo global).

### 8.2 Estratégias para Mínimos Locais

1. **Múltiplas Inicializações**: Random restarts
2. **Momentum**: Escapar de vales rasos
3. **Learning Rate Scheduling**: Refinamento fino
4. **Ensemble**: Combinar múltiplos modelos

**Analogia**: Como procurar o ponto mais baixo em uma cordilheira montanhosa usando múltiplos helicópteros (inicializações) e diferentes estratégias de busca.

## 🎨 9. Interpretabilidade Matemática

### 9.1 Feature Attribution

#### Gradient-based Methods

```
Attribution(xᵢ) = ∂L/∂xᵢ
```

**Interpretação**: Quanto a mudança em xᵢ afeta a perda.

#### Integrated Gradients

```
IG(xᵢ) = (xᵢ - x'ᵢ) × ∫₀¹ ∂L/∂x|_{x'+(x-x')t} dt
```

onde x' é uma baseline (ex: zero vector).

### 9.2 Análise do Espaço Latente

#### t-SNE para Visualização

```
Minimizar: KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)

onde:
pᵢⱼ = similaridade em alta dimensão
qᵢⱼ = similaridade em baixa dimensão
```

#### UMAP (Uniform Manifold Approximation)

```
Objective: min ∑ᵢⱼ wᵢⱼ log(wᵢⱼ/qᵢⱼ) + (1-wᵢⱼ)log((1-wᵢⱼ)/(1-qᵢⱼ))
```

**Vantagem**: Preserva estrutura global e local melhor que t-SNE.

## 🔮 10. Extensões Teóricas

### 10.1 Variational Autoencoders (VAE)

#### Framework Probabilístico

```
Encoder: q_φ(z|x) ≈ p(z|x)
Decoder: p_θ(x|z)
Prior: p(z) = N(0, I)

Loss: L = E[log p_θ(x|z)] - KL(q_φ(z|x)||p(z))
```

**Benefício**: Geração de amostras, incerteza quantificada.

### 10.2 Adversarial Autoencoders

#### Min-max Game

```
Generator (Decoder): min_G L_reconstruction
Discriminator: max_D L_adversarial

Total: min_G max_D V(D,G)
```

**Benefício**: Prior mais flexível que Gaussian.

### 10.3 Transformer-based Autoencoders

#### Self-attention Mechanism

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V

MultiHead = Concat(head₁,...,head_h)W^O
```

**Aplicação**: Sequências temporais de tráfego de rede.

## 📚 11. Conexões com Outras Áreas

### 11.1 Information Theory

#### Mutual Information

```
I(X;Z) = ∫∫ p(x,z) log(p(x,z)/(p(x)p(z))) dxdz
```

**Interpretação**: Quantidade de informação compartilhada entre entrada e representação.

#### Rate-Distortion Theory

```
R(D) = min_{p(ẑ|z):E[d(z,ẑ)]≤D} I(Z;Ẑ)
```

**Aplicação**: Trade-off entre compressão (rate) e qualidade (distortion).

### 11.2 Differential Geometry

#### Riemannian Manifolds

```
Metric tensor: g_ij = ∂φ/∂u^i · ∂φ/∂u^j
```

**Conexão**: Autoencoder aprende mapeamento entre manifolds Riemannianos.

### 11.3 Control Theory

#### Stability Analysis

```
Sistema dinâmico: θ_{t+1} = θ_t - α∇L(θ_t)

Estabilidade de Lyapunov: V(θ) ≥ 0, V̇(θ) ≤ 0
```

## 🏁 12. Conclusões Matemáticas

### 12.1 Teoremas Fundamentais

1. **Universal Approximation**: Capacidade expressiva suficiente
2. **Manifold Learning**: Fundamentação geométrica
3. **Convergência**: Garantias sob condições específicas
4. **Generalização**: Bounds via teoria PAC-learning

### 12.2 Limitações Teóricas

1. **Não-convexidade**: Sem garantias de ótimo global
2. **Curse of Dimensionality**: Ainda presente em espaços intermediários
3. **Interpretabilidade**: Trade-off com performance
4. **Robustez**: Sensibilidade a perturbações adversariais

### 12.3 Direções Futuras

1. **Teoria de Otimização**: Novos algoritmos para landscapes não-convexos
2. **Geometria Diferencial**: Métricas adaptativas para manifolds
3. **Teoria da Informação**: Bounds mais apertados para compressão
4. **Robustez Teórica**: Garantias formais contra adversários

---

**Nota**: Esta fundamentação matemática serve como base teórica sólida para o desenvolvimento e análise do sistema de detecção de anomalias proposto. Cada conceito foi escolhido por sua relevância direta ao problema e implementação prática.