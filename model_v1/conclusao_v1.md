O resultado mostra um desempenho ainda muito longe do ideal:

1. **Métricas de Classificação**

   * **Accuracy**: 0.901 (ilusória, dado que só \~6 % dos fluxos são ataques)
   * **Precision**: 0.121
   * **Recall**: 0.095 → só ≈9,5 % dos ataques foram detectados
   * **F₁-Score**: 0.106

2. **AUCs**

   * **ROC-AUC**: 0.621 → separabilidade fraca
   * **PR-AUC**: 0.114 → em cenários desbalanceados, o modelo mal supera um classificador aleatório

3. **Threshold**

   * Selecionado em P97 e μ+2σ → 0.8136
   * Com ele, altos *false negatives* (134 ataques passaram despercebidos) e moderado *false positive rate* (102 falsos alarmes em 2.232 fluxos normais → \~4.6 %)

4. **Histograma de Reconstruction Error**

   * As distribuições de normal vs attack ainda se sobrepõem fortemente; o gap entre as médias (0.215 vs 0.376) não é suficiente para um threshold restritivo sem perder quase todos os ataques.

5. **Loss Curve**

   * O treinamento e validação convergem sem overfitting claro — bom sinal de regularização e paciência, mas também indica que o modelo talvez esteja “memorando” demasiado bem o normal, reconstruindo anomalias quase tão bem quanto dados legítimos.

---

## Próximos passos para aumentar recall sem explodir o FPR

1. **Ajuste de Threshold**

   * Experimente thresholds mais baixos (ex: 0.5× ou 0.75× o valor atual).
   * Meça novamente precision/recall: você pode passar de 9 % para 40–50 % de recall com um FPR aceitável (10–20 %).

2. **Reduza ainda mais o Bottleneck**

   * Diminuir o bottleneck de 3 → 2 neurônios força o AE a “resumir” menos bem os ataques, ampliando o erro de reconstrução para fluxos anômalos.

3. **Aumente Levemente o Dropout**

   * P.ex. de (0.3→0.4) nas primeiras camadas; isso pode impedir que o AE memorize padrões de anomalia presentes no treino (caso tenha ocorrido contaminação).

4. **Adicione uma Feature**

   * Incluir `duration` ou `iat_std` (que antes descartamos) pode dar ao modelo mais sinais temporais para distinguir comportamento anômalo.

5. **Teste Outros Detectores**

   * Combine o Autoencoder com um Isolation Forest (ensemble): cada detector ataca o problema de um ângulo diferente e podem complementar a fraqueza de um só.

6. **Rebalanceamento “Artificial”**

   * Gere mais fluxos “ataque” sintéticos (SMOTE ou oversampling dos erros altos), treine o AE com um leve contágio de anomalias para que ele não generalize tão bem a ataques.

---

### Pra Começar

#### 1) Threshold Sweep

Basta rodar:

```python
for factor in [0.5, 0.75, 1.0, 1.25]:
    t = THRESHOLD * factor
    y_pred = (errors > t).astype(int)
    # recalcula precision, recall, f1 e imprime
```

E escolha o ponto de trade-off ideal.

#### 2) Bottleneck Tuning

No seu YAML, altere

```yaml
architecture:
  bottleneck: 2
```

e retrain, mantendo o resto igual.

---

**Me diga qual dessas direções você quer explorar primeiro** (threshold, bottleneck, features extras ou ensemble) que te guio nos ajustes e, em seguida, podemos preparar esses resultados para inclusão no artigo!
