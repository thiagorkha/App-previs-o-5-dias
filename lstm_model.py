import numpy as np
import pandas as pd
import logging
import pickle
import os
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Importa a configuração
from config import app_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(app_config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

@dataclass
class ModelMetrics:
    """Classe para armazenar métricas do modelo."""
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:
        return f"RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}, R²: {self.r2:.4f}"


class LSTMModel:
    """Modelo LSTM para previsão de preços de ações."""

    def __init__(self, input_shape: Tuple[int, int], config):
        self.config = config
        self.input_shape = input_shape
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def build_model(self) -> None:
        """Constrói a arquitetura do modelo LSTM."""
        logger.info("Construindo modelo LSTM...")

        self.model = Sequential()

        self.model.add(LSTM(
            self.config.LSTM_UNITS[0],
            return_sequences=len(self.config.LSTM_UNITS) > 1,
            input_shape=self.input_shape
        ))
        self.model.add(Dropout(self.config.DROPOUT_RATE))

        for i, units in enumerate(self.config.LSTM_UNITS[1:], 1):
            return_sequences = i < len(self.config.LSTM_UNITS) - 1
            self.model.add(LSTM(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config.DROPOUT_RATE))

        for units in self.config.DENSE_UNITS:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(self.config.DROPOUT_RATE))

        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        logger.info(f"Modelo construído com {self.model.count_params()} parâmetros")
        # self.model.summary(print_fn=logger.info) # Comentado para evitar log excessivo no Streamlit

    def prepare_data(self, df: pd.DataFrame, features: List[str],
                     lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepara dados para treinamento do LSTM."""
        logger.info("Preparando dados para LSTM...")

        data_for_scaling = df[features].values
        scaled_data = self.scaler.fit_transform(data_for_scaling)

        X, y = [], []
        close_idx = features.index("Close")
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, close_idx])

        X, y = np.array(X), np.array(y)

        train_size = int(len(X) * self.config.TRAIN_SPLIT_RATIO)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        logger.info(f"Dados preparados - Treino: {X_train.shape}, Teste: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Treina o modelo LSTM."""
        if self.model is None:
            raise ValueError("Modelo não foi construído. Chame build_model() primeiro.")

        logger.info("Iniciando treinamento do modelo...")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.config.MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0 # Mudar para 0 para não lotar o log do Streamlit
        )

        self.is_trained = True
        logger.info("Treinamento concluído")

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões com o modelo."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ou carregado.")
        return self.model.predict(X, verbose=0)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """Avalia o modelo usando múltiplas métricas."""
        y_pred = self.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return ModelMetrics(rmse=rmse, mae=mae, r2=r2)

    def save_model(self, filepath: str, scaler_filepath: str) -> None:
        """Salva o modelo treinado e o scaler."""
        if self.model is None:
            raise ValueError("Modelo não foi construído ou treinado.")
        self.model.save(filepath)
        with open(scaler_filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Modelo e scaler salvos em {filepath} e {scaler_filepath}")

    def load_model(self, filepath: str, scaler_filepath: str) -> None:
        """Carrega modelo e scaler salvos."""
        try:
            self.model = load_model(filepath)
            with open(scaler_filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            logger.info(f"Modelo e scaler carregados de {filepath}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo ou scaler de {filepath}: {str(e)}")
            raise