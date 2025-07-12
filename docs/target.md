# Objetivos e Métricas do Projeto de Previsão de Dengue

## 1. Objetivo Principal

**Desenvolver um sistema de previsão de notificações municipais de dengue para apoiar campanhas preventivas de saúde pública.**

### Pergunta-guia
"Com que antecedência e precisão preciso prever notificações municipais de dengue para acionar campanhas preventivas eficazes?"

## 2. Métricas de Sucesso

### Métrica Primária
- **RMSE (Root Mean Square Error)** em horizonte de **4 semanas**
- **Meta inicial**: ≤ 20 casos/100 mil habitantes
- **Justificativa**: 4 semanas permite tempo adequado para planejamento e execução de campanhas

### Métricas Secundárias
- **MAE (Mean Absolute Error)**: Para interpretabilidade mais direta
- **MAPE (Mean Absolute Percentage Error)**: Para avaliar erro relativo
- **Acurácia de tendência**: Percentual de acertos na direção da mudança (aumento/diminuição)
- **Recall para surtos**: Capacidade de detectar aumentos significativos (>50% da média histórica)

## 3. Critérios de Avaliação

### Horizonte Temporal
- **Primário**: 4 semanas à frente
- **Secundário**: 1-2 semanas (validação de curto prazo)
- **Exploratório**: 6-8 semanas (planejamento estratégico)

### Granularidade
- **Espacial**: Municipal (foco inicial em municípios com >50k habitantes)
- **Temporal**: Semanal (alinhado com ciclos epidemiológicos)

### Benchmarks
- **Baseline simples**: Média móvel de 4 semanas
- **Baseline sazonal**: Modelo SARIMA
- **Meta aspiracional**: Superar modelos de referência em 15-20%

## 4. Casos de Uso Prioritários

1. **Alerta precoce**: Identificar municípios com risco elevado nas próximas 4 semanas
2. **Alocação de recursos**: Priorizar distribuição de insumos e equipes
3. **Planejamento de campanhas**: Definir timing e intensidade de ações preventivas
4. **Monitoramento regional**: Acompanhar evolução epidemiológica em tempo real

## 5. Limitações e Considerações

### Limitações Conhecidas
- Dependência da qualidade dos dados de notificação
- Variabilidade sazonal e climática regional
- Subnotificação em períodos de baixa incidência

### Considerações Éticas
- Transparência na comunicação de incertezas
- Evitar alarmes desnecessários
- Respeitar privacidade e confidencialidade dos dados

## 6. Definição de Sucesso por Fase

### Fase 1: Prova de Conceito (MVP)
- **Objetivo**: Demonstrar viabilidade técnica
- **Critério**: RMSE < 25 casos/100k hab em pelo menos 3 municípios piloto
- **Prazo**: 8 semanas

### Fase 2: Validação Operacional
- **Objetivo**: Validar utilidade prática para gestores de saúde
- **Critério**: Feedback positivo de 80% dos usuários piloto + RMSE < 22 casos/100k hab
- **Prazo**: 12 semanas

### Fase 3: Implementação Escalável
- **Objetivo**: Sistema robusto para uso em produção
- **Critério**: RMSE ≤ 20 casos/100k hab + disponibilidade >95%
- **Prazo**: 20 semanas

## 7. Riscos e Mitigações

### Riscos Técnicos
- **Qualidade dos dados**: Implementar validação automática e limpeza de dados
- **Overfitting**: Usar validação cruzada temporal e regularização
- **Deriva de conceito**: Monitoramento contínuo e retreinamento automático

### Riscos Operacionais
- **Adoção pelos usuários**: Envolver gestores desde o início do desenvolvimento
- **Interpretabilidade**: Desenvolver explicações visuais e relatórios claros
- **Manutenção**: Documentar processos e criar testes automatizados

## 8. Próximos Passos

### Imediatos (1-2 semanas)
- [ ] Documentar escolhas detalhadas em `docs/objetivo.md`
- [ ] Definir pipeline de avaliação contínua
- [ ] Estabelecer protocolo de validação com especialistas

### Curto prazo (3-4 semanas)
- [ ] Criar dashboard de monitoramento de performance
- [ ] Implementar baseline models (média móvel, SARIMA)
- [ ] Definir protocolo de coleta de feedback dos usuários

### Médio prazo (5-8 semanas)
- [ ] Desenvolver modelos de machine learning
- [ ] Criar interface de usuário para visualização
- [ ] Estabelecer pipeline de retreinamento automático

## 9. Monitoramento e Qualidade

### Métricas de Sistema
- **Latência**: Tempo de resposta < 2 segundos para previsões
- **Disponibilidade**: Uptime > 95% durante horário comercial
- **Throughput**: Capacidade de processar 1000+ municípios simultaneamente

### Métricas de Qualidade dos Dados
- **Completude**: % de dados faltantes por município/semana
- **Consistência**: Detecção de anomalias e outliers
- **Atualidade**: Delay entre ocorrência e disponibilidade dos dados

### Alertas Automáticos
- Degradação de performance > 10% em relação ao baseline
- Falha na atualização de dados por > 48h
- Detecção de drift nos padrões de dados

## 10. Glossário

- **RMSE**: Root Mean Square Error - métrica que penaliza erros grandes
- **MAE**: Mean Absolute Error - erro médio absoluto, mais interpretável
- **MAPE**: Mean Absolute Percentage Error - erro percentual médio
- **Surto**: Aumento de casos > 50% da média histórica do período
- **Horizonte de previsão**: Período futuro para o qual se faz a previsão
- **Baseline**: Modelo simples usado como referência de comparação

---

**Última atualização**: 2025-07-11
**Responsável**: Equipe de Desenvolvimento
**Status**: Em definição
**Revisão**: Pendente aprovação dos stakeholders

> **Nota**: Este documento deve ser revisado e atualizado conforme o projeto evolui e novos requisitos são identificados.
