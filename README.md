# CPU Temperature Monitor

Aplicativo desktop para monitoramento e detecção de anomalias em temperatura de CPU usando Machine Learning.

## Estrutura do Projeto

```
cpu_temp_detector/
├── src/                    # Código fonte
│   ├── __init__.py
│   ├── cpu_temp_bundled.py     # Interface com LibreHardwareMonitor
│   ├── core_regressor.py       # Modelos de regressão para detecção de anomalias
│   ├── conv_autoencoder.py     # Autoencoder convolucional (CNN) para erro de reconstrução
│   ├── data_extractor.py       # Extração e engenharia de features
│   ├── tray_monitor.py         # Monitor de bandeja do sistema
│   └── lib/                    # DLLs do LibreHardwareMonitor
│       ├── HidSharp.dll
│       └── LibreHardwareMonitorLib.dll
├── data/                   # Dados de treino
│   └── *.csv
├── models/                 # Modelos treinados
│   ├── cpu_temp_model_linear.joblib
│   ├── cpu_temp_model_xgb.joblib
│   ├── cpu_temp_model_lightgbm.joblib
│   └── cpu_temp_model_autoencoder.pt
├── notebooks/              # Jupyter notebooks (experimentação)
│   └── cpu_temp.ipynb
├── app.py                  # Aplicativo GUI principal
├── config.yaml             # Arquivo de configuração
└── requirements.txt        # Dependências Python
```

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

Dependências principais:
- `pystray`, `Pillow` - System tray
- `plyer` - Notificações Windows
- `PyYAML` - Configuração
- `scikit-learn`, `xgboost`, `lightgbm` - Machine Learning
- `torch` - PyTorch (autoencoder)
- `pandas`, `numpy` - Processamento de dados
- `matplotlib` - Gráficos em tempo real

## Como Usar

O aplicativo possui 6 abas organizadas em um fluxo lógico:

**Guide → Collect → Train → Monitor → Log → Settings**

### 1. Guide (Guia)

Aba de boas-vindas com explicações para novos usuários:
- Como o sistema funciona
- Passo a passo para começar (Coletar → Treinar → Monitorar → Analisar)
- Conceitos-chave: Anomalia, Threshold, Erro de Reconstrução, Window Size
- Dicas de uso
- Botão "Get Started" para ir direto à coleta

### 2. Collect (Coleta de Dados)

Colete dados dos sensores do computador para treinar o modelo:
- **Coleta em background**: Defina duração e intervalo, acompanhe em tempo real
- **Carregar CSVs**: Importe dados previamente coletados
- **Gráfico ao vivo**: Visualize os sensores sendo coletados em tempo real, com seletor de sensor
- Botão "Go to Train →" para avançar ao treinamento

### 3. Train (Treinamento)

Treine o modelo de detecção de anomalias:
- **Step 1 — Escolha o modelo**:
  - **Regressor**: Linear (Ridge), XGBoost ou LightGBM
  - **Autoencoder**: CNN com parâmetros configuráveis (window size, épocas, learning rate, batch size)
- **Step 2 — Treine**: Clique em "Train From Data" e acompanhe o progresso
- **Step 3 — Salve**: Salve o modelo treinado e use-o no monitoramento

### 4. Monitor (Monitoramento)

Monitore a CPU em tempo real com detecção de anomalias:
- Selecione o modelo treinado (`.joblib` para regressor, `.pt` para autoencoder)
- Configure o threshold e a janela de anomalia
- **Gráfico em tempo real**: Visualize dados reais vs reconstruídos por sensor (autoencoder)
- **Painel de saúde dos sensores**: 7 indicadores coloridos mostrando estado de cada sensor
  - Verde (Healthy) / Vermelho (Anomaly)
  - Sensores monitorados: CPU Temp, CPU Load, CPU Power, GPU Temp, GPU Load, GPU Power, RAM Load
- **Classificação de anomalias**: O sistema categoriza automaticamente o tipo de anomalia:
  - **Cooling problem** — Temperaturas altas sem carga correspondente
  - **Heavy workload** — Temperaturas e cargas altas simultaneamente
  - **GPU isolated** — Anomalia isolada na GPU
  - **Power anomaly** — Anomalia nos sensores de energia
  - **Memory pressure** — Anomalia isolada na RAM
  - **Single sensor spike** — Apenas um sensor anômalo
  - **Unknown pattern** — Combinação não classificada
- Minimize para a bandeja do sistema com ícone colorido (verde/vermelho)

### 5. Log (Registro de Anomalias)

Histórico completo das anomalias detectadas:
- **Estatísticas resumidas**: Total de anomalias, categoria mais frequente, último evento
- **Tabela detalhada**: Horário, categoria, sensores afetados, temperaturas, erro de reconstrução
- **Linhas coloridas** por categoria para identificação visual rápida
- **Exportar para CSV**: Salve o histórico para análise posterior
- **Limpar log**: Reinicie o registro

### 6. Settings (Configurações)

- Caminho do modelo
- Threshold de anomalia
- Intervalo de verificação
- Janela de anomalia (anomalias consecutivas para confirmar)
- Notificações habilitadas/desabilitadas
- Minimizar para bandeja

## Configuração

Edite `config.yaml` ou use a aba **Settings** na GUI:

```yaml
model_path: models/cpu_temp_model_autoencoder.pt
model_approach: autoencoder        # 'regressor' ou 'autoencoder'
threshold_std: 1.5
check_interval: 5
monitor_anomaly_window: 1          # Anomalias consecutivas para confirmar alerta
multi_variable: true               # Usar todos os sensores ou apenas tempo
notifications_enabled: true
minimize_to_tray: true
```

## Como Funciona

1. **Coleta de Dados**: Coleta em background ou carregamento de CSVs (sensores via LibreHardwareMonitor)
2. **Feature Engineering**: Cria features de lag, rolling statistics e diferenças
3. **Treinamento**: Treina modelo de regressão ou autoencoder para aprender o comportamento normal
4. **Detecção**: Identifica anomalias quando o erro excede o threshold
5. **Classificação**: Categoriza a anomalia com base nos sensores afetados
6. **Janela de Anomalia**: Requer N anomalias consecutivas antes de alertar, evitando falsos positivos

## Abordagens de Detecção

### Regressão
Prevê a temperatura "normal" da CPU com base nos outros sensores. Anomalias são detectadas quando `|real - previsto| > threshold`.

- **Linear (Ridge)**: Rápido, simples, bom baseline
- **XGBoost**: Ótimo para padrões complexos
- **LightGBM**: Balanceado entre velocidade e precisão (recomendado)

### Autoencoder (CNN)
Usa erro de reconstrução em janelas temporais para detectar anomalias multivariadas. Dados que não se encaixam no padrão aprendido geram alto erro de reconstrução.

- Aprende padrões multivariados ao longo do tempo (7 sensores simultâneos)
- Erro de reconstrução global e **por sensor** (per-feature)
- Thresholds individuais por sensor para detecção granular
- Visualização de dados reais vs reconstruídos em tempo real
- Classificação automática do tipo de anomalia

## Desenvolvimento

Para experimentar no Jupyter:
```bash
jupyter notebook notebooks/cpu_temp.ipynb
```

## Requisitos do Sistema

- Windows 10/11
- Python 3.8+
- Permissões de administrador (para acessar sensores de hardware)

## Troubleshooting

### "Permission denied" ao acessar hardware
Execute o Python/terminal como Administrador.

### Modelo não detecta anomalias
- Ajuste o threshold_std nas configurações
- Treine com mais dados (colete por mais tempo ou combine mais CSVs)
- Verifique se os dados de treino são representativos do uso normal
- Para autoencoder: experimente diferentes window sizes

### Notificações não aparecem
- Verifique se notificações estão habilitadas no Windows
- Ative `notifications_enabled: true` no config.yaml

### Erro de reconstrução alto mesmo em condições normais
- Retreine o modelo com dados mais representativos
- Verifique se o scaler está adequado aos seus dados (Standard, MinMax, Robust)
