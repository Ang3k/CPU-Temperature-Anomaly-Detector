# CPU Temperature Monitor

Aplicativo desktop para monitoramento e detecção de anomalias em temperatura de CPU usando Machine Learning.

## Estrutura do Projeto

```
cpu_temp_detector/
├── src/                    # Código fonte
│   ├── __init__.py
│   ├── cpu_temp_bundled.py     # Interface com LibreHardwareMonitor
│   ├── core_regressor.py       # Modelos de regressão para detecção de anomalias
│   ├── conv_autoencoder.py     # Autoencoder convolucional para erro de reconstrução
│   ├── data_extractor.py       # Extração e engenharia de features
│   ├── tray_monitor.py         # Monitor de bandeja do sistema
│   └── lib/                    # DLLs do LibreHardwareMonitor
│       ├── HidSharp.dll
│       └── LibreHardwareMonitorLib.dll
├── data/                   # Dados de treino
│   └── *.csv
├── models/                 # Modelos treinados (.joblib)
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
- `pandas`, `numpy` - Processamento de dados

## Como Usar

### 1. Execute o Aplicativo

```bash
python app.py
```

### 2. Treine um Modelo

Na aba **Train**:
- Selecione a abordagem: **Regressor** (regressão) ou **Autoencoder** (erro de reconstrução)
- Para Regressor: escolha o modelo (Linear, XGBoost, LightGBM)
- Para Autoencoder: configure janela, épocas, learning rate e batch size
- Selecione o scaler (Standard, MinMax, Robust)
- Colete dados usando a **coleta em background** ou carregue de **arquivo(s) CSV**
- Opcionalmente, defina **mean time** para reamostrar os dados por janelas de tempo
- Clique em "Train From Data"
- Clique em "Save Model" quando terminar

### 3. Inicie o Monitoramento

Na aba **Monitor**:
- Selecione o modelo treinado (`.joblib` ou `.pt`) — detecta automaticamente se é Regressor ou Autoencoder
- Configure o threshold de anomalia (padrão: 1.5 std)
- Configure a **janela de anomalia** (número de anomalias consecutivas antes de alertar)
- Clique em "Start Monitoring"
- Minimize para a bandeja do sistema

### 4. Receba Alertas

- O ícone na bandeja fica:
  - **Verde**: Temperatura normal
  - **Vermelho**: Anomalia detectada
- Notificações Windows aparecem quando anomalias são detectadas
- Clique com botão direito no ícone para acessar o menu

## Configuração

Edite `config.yaml` ou use a aba **Settings** na GUI:

```yaml
model_path: models/cpu_temp_model_lightgbm.joblib
model_approach: regressor          # 'regressor' ou 'autoencoder'
threshold_std: 1.5
check_interval: 5
mean_time: 5                       # Janela de reamostragem em segundos (opcional)
monitor_anomaly_window: 1          # Anomalias consecutivas para confirmar alerta
multi_variable: true               # Usar todos os sensores ou apenas tempo
notifications_enabled: true
minimize_to_tray: true
```

## Como Funciona

1. **Coleta de Dados**: Coleta em background ou carregamento de CSVs (sensores via LibreHardwareMonitor)
2. **Reamostragem** (opcional): Agrega dados por janelas de tempo (mean time)
3. **Feature Engineering**: Cria features de lag, rolling statistics, e diferenças
4. **Treinamento**: Treina modelo de regressão ou autoencoder para detectar comportamento normal
5. **Detecção**: Identifica anomalias quando a diferença ou o erro de reconstrução excede o threshold
6. **Janela de Anomalia**: Requer N anomalias consecutivas antes de alertar, evitando falsos positivos

## Abordagens de Detecção

### Regressão
Prevê a temperatura "normal" da CPU com base nos outros sensores. Anomalias são detectadas quando `|real - previsto| > threshold`.

- **Linear (Ridge)**: Rápido, simples, bom baseline
- **XGBoost**: Ótimo para padrões complexos
- **LightGBM**: Balanceado entre velocidade e precisão (recomendado)

### Autoencoder
Usa erro de reconstrução em janelas temporais para detectar anomalias. Dados que não se encaixam no padrão aprendido geram alto erro de reconstrução.

- Aprende padrões multivariados ao longo do tempo
- Permite acompanhar erro global e erro por sensor

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
- Aumente o threshold_std nas configurações
- Treine com mais dados (colete por mais tempo ou combine mais CSVs)
- Verifique se os dados de treino são representativos

### Notificações não aparecem
- Verifique se notificações estão habilitadas no Windows
- Ative `notifications_enabled: true` no config.yaml
