import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import List, Optional
import logging

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

class Visualizer:
    """Classe para visualização de resultados."""

    def __init__(self, config):
        self.config = config

    def plot_predictions(self, df: pd.DataFrame, predictions: List[float],
                         future_dates: List[str], ticker: str,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota previsões de preços e indicadores técnicos.
        Retorna o objeto Figure do Matplotlib para o Streamlit.
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Gráfico de Preço Histórico e Previsão
        historical_data = df['Close'].tail(self.config.PLOT_LAST_DAYS)
        axes[0].plot(historical_data.index, historical_data.values,
                     label='Preço Histórico', color='blue', linewidth=2)

        future_dates_dt = [datetime.strptime(date, '%Y-%m-%d') for date in future_dates]
        axes[0].plot(future_dates_dt, predictions,
                     label='Previsão', color='red', marker='o', linewidth=2, markersize=6)

        axes[0].axvline(x=df.index[-1], color='black', linestyle='--',
                        label='Último Dado Histórico', alpha=0.7)

        axes[0].set_title(f'Previsão de Preços - {ticker}', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Data', fontsize=12)
        axes[0].set_ylabel('Preço (R$)', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # Gráfico RSI
        if 'RSI' in df.columns:
            rsi_data = df['RSI'].tail(self.config.PLOT_LAST_DAYS)
            axes[1].plot(rsi_data.index, rsi_data.values,
                         label='RSI (14)', color='purple', linewidth=1.5)
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Sobrecomprado (70)')
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Sobrevendido (30)')
            axes[1].set_ylabel('RSI', fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Índice de Força Relativa (RSI)', fontsize=14)
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].set_visible(False) # Esconde o subplot se a coluna não existe

        # Gráfico MACD
        if 'MACD' in df.columns:
            macd_data = df['MACD'].tail(self.config.PLOT_LAST_DAYS)
            axes[2].plot(macd_data.index, macd_data.values,
                         label='MACD', color='orange', linewidth=1.5)
            axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Linha Zero')
            axes[2].set_ylabel('MACD', fontsize=12)
            axes[2].set_xlabel('Data', fontsize=12)
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('MACD (Moving Average Convergence Divergence)', fontsize=14)
            axes[2].tick_params(axis='x', rotation=45)
        else:
            axes[2].set_visible(False) # Esconde o subplot se a coluna não existe

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Gráfico salvo em {save_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar gráfico: {e}")

        # Retorna a figura para o Streamlit
        return fig

    @staticmethod
    def plot_training_history(history: Dict[str, List[float]]) -> plt.Figure:
        """
        Plota histórico de treinamento e retorna o objeto Figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history['loss'], label='Treino', linewidth=2)
        ax1.plot(history['val_loss'], label='Validação', linewidth=2)
        ax1.set_title('Perda do Modelo', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Perda')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history['mae'], label='Treino', linewidth=2)
        ax2.plot(history['val_mae'], label='Validação', linewidth=2)
        ax2.set_title('Erro Absoluto Médio', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig