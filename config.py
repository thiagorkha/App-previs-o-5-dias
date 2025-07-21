import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """
    Classe de configuração para o aplicativo de previsão de ações.
    Define parâmetros para download de dados, modelo LSTM, caminhos, etc.
    """
    TICKER: str = "ITUB4.SA" # Ticker padrão
    DIAS_PREVISAO: int = 5 # Quantos dias no futuro prever
    LOOKBACK: int = 60 # Quantos dias históricos usar para cada previsão
    EPOCHS: int = 50 # Épocas de treinamento do LSTM
    BATCH_SIZE: int = 32 # Tamanho do batch para treinamento
    START_DATE_OFFSET_YEARS: int = 5 # Quantos anos de dados históricos baixar
    FEATURES: List[str] = field(default_factory=lambda: [
        "Close", "SMA_20", "RSI", "MACD", "BollingerBands_Upper", "BollingerBands_Lower"
    ]) # Features (colunas) a serem usadas pelo modelo

    # Caminhos para arquivos e diretórios
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
    MODEL_PATH: str = os.path.join(BASE_DIR, "lstm_model.h5")
    SCALER_PATH: str = os.path.join(BASE_DIR, "lstm_scaler.pkl") # Caminho para o scaler
    PLOT_PATH: str = os.path.join(BASE_DIR, "previsao_acao.png") # Para salvar o plot, se necessário
    LOG_FILE: str = os.path.join(BASE_DIR, "stock_predictor.log")

    TRAIN_SPLIT_RATIO: float = 0.8 # Proporção dos dados para treinamento
    PLOT_LAST_DAYS: int = 100 # Quantos dias históricos mostrar no gráfico

    # Parâmetros do modelo LSTM
    EARLY_STOPPING_PATIENCE: int = 15
    LSTM_UNITS: List[int] = field(default_factory=lambda: [100, 50])
    DENSE_UNITS: List[int] = field(default_factory=lambda: [50])
    DROPOUT_RATE: float = 0.3

# Instancia a configuração para ser usada em outros módulos
app_config = Config()