# Arquiteturas Neurais dos Modelos - Representação como Neurônios Reais

## Modelo V1 - Autoencoder Básico
```mermaid
graph TB
    subgraph "INPUT LAYER - Dados de Entrada"
        I1[bytes 🔢]
        I2[pkts 📦]
        I3[iat_mean ⏱️]
    end

    subgraph "ENCODER - Compressão Progressiva"
        subgraph "Hidden Layer 1 (4×D = 12 neurônios)"
            H11[N1 🧠] 
            H12[N2 🧠]
            H13[N3 🧠]
            H14[N4 🧠]
            H15[...] 
            H16[N12 🧠]
        end
        
        subgraph "Hidden Layer 2 (2×D = 6 neurônios)"
            H21[N1 🧠]
            H22[N2 🧠]
            H23[N3 🧠]
            H24[N4 🧠]
            H25[N5 🧠]
            H26[N6 🧠]
        end
        
        subgraph "BOTTLENECK (D = 3 neurônios) - Representação Compacta"
            B1[Feature 1 💎]
            B2[Feature 2 💎]
            B3[Feature 3 💎]
        end
    end

    subgraph "DECODER - Reconstrução Progressiva"
        subgraph "Hidden Layer 3 (2×D = 6 neurônios)"
            D21[N1 🧠]
            D22[N2 🧠]
            D23[N3 🧠]
            D24[N4 🧠]
            D25[N5 🧠]
            D26[N6 🧠]
        end
        
        subgraph "Hidden Layer 4 (4×D = 12 neurônios)"
            D11[N1 🧠]
            D12[N2 🧠]
            D13[N3 🧠]
            D14[N4 🧠]
            D15[...]
            D16[N12 🧠]
        end
    end

    subgraph "OUTPUT LAYER - Reconstrução"
        O1[bytes' 🔢]
        O2[pkts' 📦]
        O3[iat_mean' ⏱️]
    end

    subgraph "ANOMALY DETECTION - Detecção"
        AD1[Erro MSE 📊]
        AD2[Threshold 🎯]
        AD3[Normal/Anomalia ⚠️]
    end

    %% Conexões Input -> Encoder
    I1 --> H11
    I1 --> H12
    I1 --> H13
    I2 --> H11
    I2 --> H12
    I2 --> H13
    I3 --> H11
    I3 --> H12
    I3 --> H13

    %% Conexões Encoder
    H11 --> H21
    H12 --> H21
    H13 --> H22
    H14 --> H22
    H21 --> B1
    H22 --> B2
    H23 --> B3

    %% Conexões Decoder
    B1 --> D21
    B2 --> D22
    B3 --> D23
    D21 --> D11
    D22 --> D12
    D23 --> D13

    %% Conexões Output
    D11 --> O1
    D12 --> O2
    D13 --> O3

    %% Detecção de Anomalia
    O1 --> AD1
    O2 --> AD1
    O3 --> AD1
    I1 -.-> AD1
    I2 -.-> AD1
    I3 -.-> AD1
    AD1 --> AD2
    AD2 --> AD3

    style I1 fill:#e3f2fd
    style I2 fill:#e3f2fd
    style I3 fill:#e3f2fd
    style B1 fill:#ffcdd2
    style B2 fill:#ffcdd2
    style B3 fill:#ffcdd2
    style AD3 fill:#c8e6c9
```

## Modelo V2/V3 - Pipeline Configurável
```mermaid
graph TB
    subgraph "DATA PREPROCESSING - Pré-processamento"
        PP1[Imputação 🔧]
        PP2[Transformações 🔄]
        PP3[Normalização 📏]
    end

    subgraph "CONFIGURABLE INPUT - Entrada Configurável"
        CI1[Feature 1 📊]
        CI2[Feature 2 📊]
        CI3[Feature N 📊]
    end

    subgraph "ENCODER CONFIGURÁVEL"
        subgraph "Layer 1 (Multiplicador × D)"
            E11[N1 🧠]
            E12[N2 🧠]
            E13[Dropout 🎲]
            E14[...NM 🧠]
        end
        
        subgraph "Layer 2 (Multiplicador × D)"
            E21[N1 🧠]
            E22[N2 🧠]
            E23[Dropout 🎲]
            E24[...NK 🧠]
        end
        
        subgraph "BOTTLENECK ADAPTATIVO"
            EB1[Feature 1 💎]
            EB2[Feature 2 💎]
            EB3[Feature K 💎]
        end
    end

    subgraph "DECODER SIMÉTRICO"
        subgraph "Layer 3 (Espelho Layer 2)"
            D21[N1 🧠]
            D22[N2 🧠]
            D23[Dropout 🎲]
            D24[...NK 🧠]
        end
        
        subgraph "Layer 4 (Espelho Layer 1)"
            D11[N1 🧠]
            D12[N2 🧠]
            D13[Dropout 🎲]
            D14[...NM 🧠]
        end
    end

    subgraph "RECONSTRUCTION OUTPUT"
        RO1[Feature 1' 📊]
        RO2[Feature 2' 📊]
        RO3[Feature N' 📊]
    end

    subgraph "ADVANCED ANOMALY DETECTION"
        AAD1[Múltiplos Thresholds 🎯]
        AAD2[P95, P97, μ+2σ 📈]
        AAD3[Threshold Conservador 🛡️]
        AAD4[Normal/Anomalia ⚠️]
    end

    %% Fluxo de dados
    PP1 --> PP2
    PP2 --> PP3
    PP3 --> CI1
    PP3 --> CI2
    PP3 --> CI3

    CI1 --> E11
    CI2 --> E12
    CI3 --> E14

    E11 --> E21
    E12 --> E22
    E14 --> E24

    E21 --> EB1
    E22 --> EB2
    E24 --> EB3

    EB1 --> D21
    EB2 --> D22
    EB3 --> D24

    D21 --> D11
    D22 --> D12
    D24 --> D14

    D11 --> RO1
    D12 --> RO2
    D14 --> RO3

    RO1 --> AAD1
    RO2 --> AAD1
    RO3 --> AAD1
    AAD1 --> AAD2
    AAD2 --> AAD3
    AAD3 --> AAD4

    style PP1 fill:#f3e5f5
    style PP2 fill:#f3e5f5
    style PP3 fill:#f3e5f5
    style EB1 fill:#ffcdd2
    style EB2 fill:#ffcdd2
    style EB3 fill:#ffcdd2
    style AAD4 fill:#c8e6c9
```

## Modelo V4 - LSTM-Attention Temporal
```mermaid
graph TB
    subgraph "SEQUENCE INPUT - Entrada Temporal"
        subgraph "Timestep 1"
            T11[bytes 🔢]
            T12[pkts 📦]
            T13[iat_mean ⏱️]
            T14[duration 🕐]
            T15[iat_std 📏]
        end
        subgraph "Timestep 2"
            T21[bytes 🔢]
            T22[pkts 📦]
            T23[iat_mean ⏱️]
            T24[duration 🕐]
            T25[iat_std 📏]
        end
        subgraph "..."
            TX1[... ⋯]
        end
        subgraph "Timestep 10"
            T101[bytes 🔢]
            T102[pkts 📦]
            T103[iat_mean ⏱️]
            T104[duration 🕐]
            T105[iat_std 📏]
        end
    end

    subgraph "LSTM ENCODER - Processamento Temporal"
        subgraph "LSTM Layer 1 (64 unidades)"
            L11[LSTM Cell 1 🧠📚]
            L12[LSTM Cell 2 🧠📚]
            L13[Hidden State h₁ 🧭]
            L14[Cell State c₁ 📖]
        end
        
        subgraph "LSTM Layer 2 (32 unidades)"
            L21[LSTM Cell 1 🧠📚]
            L22[LSTM Cell 2 🧠📚]
            L23[Hidden State h₂ 🧭]
            L24[Cell State c₂ 📖]
        end
        
        subgraph "ATTENTION MECHANISM - Mecanismo de Atenção"
            A1[Query 🔍]
            A2[Key 🗝️]
            A3[Value 💎]
            A4[Attention Weights 🎯]
            A5[Context Vector 🌟]
        end
    end

    subgraph "BOTTLENECK TEMPORAL"
        TB1[Temporal Feature 1 ⏰]
        TB2[Temporal Feature 2 ⏰]
        TB3[Temporal Feature 8 ⏰]
    end

    subgraph "LSTM DECODER - Reconstrução Temporal"
        subgraph "Repeat Vector"
            RV1[Broadcast 📡]
        end
        
        subgraph "LSTM Layer 3 (32 unidades)"
            LD1[LSTM Cell 1 🧠📚]
            LD2[LSTM Cell 2 🧠📚]
            LD3[Hidden State h₃ 🧭]
        end
        
        subgraph "LSTM Layer 4 (64 unidades)"
            LD4[LSTM Cell 1 🧠📚]
            LD5[LSTM Cell 2 🧠📚]
            LD6[Hidden State h₄ 🧭]
        end
    end

    subgraph "SEQUENCE OUTPUT - Saída Temporal"
        subgraph "Reconstructed Timestep 1"
            RT11[bytes' 🔢]
            RT12[pkts' 📦]
            RT13[iat_mean' ⏱️]
            RT14[duration' 🕐]
            RT15[iat_std' 📏]
        end
        subgraph "..."
            RTX1[... ⋯]
        end
        subgraph "Reconstructed Timestep 10"
            RT101[bytes' 🔢]
            RT102[pkts' 📦]
            RT103[iat_mean' ⏱️]
            RT104[duration' 🕐]
            RT105[iat_std' 📏]
        end
    end

    subgraph "TEMPORAL ANOMALY DETECTION"
        TAD1[Sequence MSE 📊]
        TAD2[Temporal Patterns 🌊]
        TAD3[Attention Analysis 🔍]
        TAD4[Anomaly Score ⚠️]
    end

    %% Conexões Temporais
    T11 --> L11
    T21 --> L11
    T101 --> L11
    
    L11 --> L21
    L12 --> L21
    L13 --> L21
    
    L21 --> A1
    L22 --> A2
    L23 --> A3
    A1 --> A4
    A2 --> A4
    A3 --> A4
    A4 --> A5
    
    A5 --> TB1
    A5 --> TB2
    A5 --> TB3
    
    TB1 --> RV1
    TB2 --> RV1
    TB3 --> RV1
    
    RV1 --> LD1
    LD1 --> LD4
    LD4 --> RT11
    LD4 --> RT101
    
    RT11 --> TAD1
    RT101 --> TAD1
    TAD1 --> TAD2
    TAD2 --> TAD3
    TAD3 --> TAD4

    style T11 fill:#e8eaf6
    style T21 fill:#e8eaf6
    style T101 fill:#e8eaf6
    style A5 fill:#fff3e0
    style TB1 fill:#ffcdd2
    style TB2 fill:#ffcdd2
    style TB3 fill:#ffcdd2
    style TAD4 fill:#c8e6c9
```

## Comparação de Processamento Neural

### Modelo V1-V3: Processamento Pontual
- **Entrada**: 1 fluxo → 3-5 features
- **Processamento**: Neurônios densos com ativação ReLU
- **Memória**: Sem memória temporal
- **Detecção**: Erro de reconstrução instantâneo

### Modelo V4: Processamento Temporal
- **Entrada**: Sequência de 10 fluxos → 50 features temporais
- **Processamento**: LSTM cells com gates (forget, input, output)
- **Memória**: Hidden states + Cell states para contexto temporal
- **Atenção**: Foco dinâmico em timesteps relevantes
- **Detecção**: Padrões temporais anômalos

## Analogia com Neurônios Reais

### Neurônios Densos (V1-V3)
- 🧠 **Como neurônios simples**: Recebem sinais, processam com função de ativação
- ⚡ **Sinapses**: Pesos conectam todas as entradas a todas as saídas
- 🔄 **Processamento**: Instantâneo, sem memória

### LSTM Cells (V4)
- 🧠📚 **Como neurônios com memória**: Mantêm informação ao longo do tempo
- 🚪 **Gates**: Controlam fluxo de informação (como canais iônicos)
- 📖 **Cell State**: Memória de longo prazo (como potencial de repouso)
- 🧭 **Hidden State**: Memória de curto prazo (como potencial de ação)

### Attention Mechanism (V4)
- 🔍 **Como atenção seletiva**: Foca em informações relevantes
- 🎯 **Pesos de atenção**: Determina importância de cada timestep
- 🌟 **Context vector**: Resumo ponderado das informações importantes