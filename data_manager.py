import yfinance as yf
import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime, timedelta
from typing import List, Tuple

# Importa a configuração
from config import app_config

# Configuração de logging (simplificada para o Streamlit, mas ainda útil para depuração)
logger = logging.getLogger(__name__)
# Certifique-se de que o logger não esteja adicionando vários handlers se este arquivo for importado várias vezes
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, # Pode ser DEBUG para mais detalhes
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(app_config.LOG_FILE),
            logging.StreamHandler()
        ]
    )


class DataManager:
    """Gerencia download, processamento e cache de dados de ações."""

    def __init__(self, config):
        self.config = config
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_stock_data(self, ticker: str, start_date: str, end_date: str,
                            use_cache: bool = True) -> pd.DataFrame:
        """
        Baixa dados de ações com cache opcional.
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.pkl")

        if use_cache and os.path.exists(cache_file):
            logger.info(f"Carregando dados do cache: {cache_file}")
            try:
                df = pd.read_pickle(cache_file)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"Cache '{cache_file}' inválido (não é DataFrame ou está vazio). Baixando novamente.")
                    os.remove(cache_file)
                    return self.download_stock_data(ticker, start_date, end_date, use_cache=False)

                last_cache_date = df.index.max().date()
                current_date_for_comparison = datetime.now().date()
                if current_date_for_comparison > last_cache_date + timedelta(days=1):
                     logger.warning("Dados do cache estão desatualizados, baixando novamente.")
                     os.remove(cache_file)
                     return self.download_stock_data(ticker, start_date, end_date, use_cache=False)

                if isinstance(df.columns, pd.MultiIndex):
                    logger.warning("Cache carregado com MultiIndex de colunas. Achatando...")
                    df.columns = df.columns.droplevel(1)
                    df = df.loc[:,~df.columns.duplicated()].copy()
                
                df.columns.name = None # Remove o nome do índice das colunas
                return df
            except Exception as e:
                logger.warning(f"Erro ao carregar cache '{cache_file}': {str(e)}. Baixando dados novamente.")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                return self.download_stock_data(ticker, start_date, end_date, use_cache=False)

        try:
            logger.info(f"Baixando dados para {ticker} de {start_date} até {end_date}")
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError(f"Não foi possível obter dados válidos para {ticker} no período especificado. DataFrame vazio ou inválido.")

            if isinstance(df.columns, pd.MultiIndex):
                logger.info("DataFrame baixado possui MultiIndex nas colunas. Achatando...")
                df.columns = df.columns.droplevel(1)
            
            df.columns.name = None # Remove o nome do índice das colunas

            if use_cache:
                df.to_pickle(cache_file)
                logger.info(f"Dados salvos no cache: {cache_file}")

            return df.copy()

        except Exception as e:
            logger.error(f"Erro ao baixar dados: {str(e)}")
            raise ValueError(f"Falha no download de dados para {ticker}: {str(e)}")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos (SMA, RSI, MACD, Bollinger Bands).
        """
        logger.info("Calculando indicadores técnicos...")
        df = df.copy()

        # Importa apenas aqui para evitar importação circular e grandes dependências no topo
        try:
            from ta.momentum import RSIIndicator
            from ta.volatility import BollingerBands
            from ta.trend import MACD
        except ImportError:
            logger.error("Bibliotecas 'ta' não instaladas. Por favor, instale com 'pip install ta'.")
            raise

        if 'Close' not in df.columns:
            raise ValueError("Coluna 'Close' não encontrada no DataFrame para cálculo de indicadores.")

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Close'].fillna(method='ffill', inplace=True)
        df['Close'].fillna(method='bfill', inplace=True)
        
        if df['Close'].isnull().all() or df['Close'].empty:
            logger.warning("Coluna 'Close' está toda NaN ou vazia após limpeza inicial. Indicadores podem falhar.")

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        macd_indicator = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd_indicator.macd()
        bollinger_bands = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BollingerBands_Upper"] = bollinger_bands.bollinger_hband()
        df["BollingerBands_Lower"] = bollinger_bands.bollinger_lband()
        df['Volatility'] = df['Close'].rolling(window=20).std()

        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

        logger.info("Indicadores técnicos calculados")
        return df

    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Valida se os dados têm as colunas necessárias e qualidade adequada.
        Levanta ValueError se os dados não forem válidos.
        """
        logger.info("Iniciando validação de dados...")

        if df.empty:
            raise ValueError("DataFrame está vazio após o download ou processamento.")

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colunas requeridas faltando no DataFrame: {missing_columns}")

        for col in required_columns:
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        df[required_columns] = df[required_columns].fillna(method='ffill')
        df[required_columns] = df[required_columns].fillna(method='bfill')

        initial_rows = len(df)
        df.dropna(subset=required_columns, inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logger.warning(f"Removidas {rows_dropped} linhas devido a NaNs remanescentes nas features requeridas.")

        if df.empty:
            raise ValueError("DataFrame ficou vazio após limpeza de NaNs nas features requeridas.")
        
        if len(df) < self.config.LOOKBACK + self.config.DIAS_PREVISAO:
            raise ValueError(f"Dados insuficientes após limpeza: {len(df)} registros. "
                             f"Necessário pelo menos {self.config.LOOKBACK + self.config.DIAS_PREVISAO}")
        logger.info("Validação de dados concluída.")