# CPU Temperature Monitor

Aplicativo desktop para monitoramento e detecção de anomalias em temperatura de CPU usando Machine Learning.

## Estrutura do Projeto

```
cpu_temp_detector/
├── src/                    # Código fonte
│   ├── __init__.py
│   ├── cpu_temp_bundled.py     # Interface com LibreHardwareMonitor
│   ├── core_regressor.py       # Classe de regressão e detecção de anomalias
│   ├── tray_monitor.py         # Monitor de bandeja do sistema
│   └── lib/                    # DLLs do LibreHardwareMonitor
│       ├── HidSharp.dll
│       └── LibreHardwareMonitorLib.dll
├── data/                   # Dados de treino
│   └── data.csv
├── models/                 # Modelos treinados (.joblib)
│   ├── cpu_temp_model_linear.joblib
│   ├── cpu_temp_model_xgb.joblib
│   └── cpu_temp_model_lightgbm.joblib
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
- Selecione o tipo de modelo (linear, xgb, lightgbm)
- Configure o número de iterações e intervalo
- Clique em "Start Training"
- Aguarde a coleta de dados e treinamento
- Clique em "Save Model" quando terminar

### 3. Inicie o Monitoramento

Na aba **Monitor**:
- Selecione o modelo treinado (`.joblib`)
- Configure o threshold de anomalia (padrão: 1.5 std)
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
threshold_std: 1.5
check_interval: 5
notifications_enabled: true
minimize_to_tray: true
```

## Como Funciona

1. **Coleta de Dados**: Usa LibreHardwareMonitor para ler sensores (CPU, GPU, RAM, etc.)
2. **Feature Engineering**: Cria features de lag, rolling statistics, e diferenças
3. **Treinamento**: Treina um modelo de regressão para prever temperatura normal
4. **Detecção**: Compara temperatura real vs prevista - se diferença > threshold, alerta

## Modelos Disponíveis

- **Linear (Ridge)**: Rápido, simples, bom baseline
- **XGBoost**: Ótimo para padrões complexos
- **LightGBM**: Balanceado entre velocidade e precisão (recomendado)

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
- Treine com mais dados (mais iterações)
- Verifique se os dados de treino são representativos

### Notificações não aparecem
- Verifique se notificações estão habilitadas no Windows
- Ative `notifications_enabled: true` no config.yaml
