import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import matplotlib.pyplot as plt # Importar aqui para st.pyplot()
import tensorflow as tf # Para tf.keras.backend.clear_session()

# Importar as classes refatoradas
from config import app_config
from data_manager import DataManager
from lstm_model import LSTMModel, ModelMetrics
from visualizer import Visualizer

# Configuração de logging
# O logger será configurado automaticamente pelos módulos importados, mas podemos ajustar o nível aqui
logger = logging.getLogger(__name__)
if not logger.handlers: # Previne handlers duplicados se o script for recarregado
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(app_config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

st.set_page_config(layout="wide", page_title="Previsor de Ações com LSTM")

# --- Instâncias das Classes ---
data_manager = DataManager(app_config)
visualizer = Visualizer(app_config)

# --- Funções para serem cacheadas pelo Streamlit ---
# Use st.cache_data para dados que não mudam com frequência ou que levam tempo para processar
# Ou para o resultado de um modelo que não muda a cada execução, a menos que os parâmetros mudem
@st.cache_data(ttl=timedelta(days=1)) # Cacheia por 1 dia
def get_and_process_data(ticker, start_date, end_date, use_cache):
    """
    Baixa e processa os dados, calculando indicadores técnicos.
    Esta função é cacheada para evitar downloads repetidos.
    """
    df = data_manager.download_stock_data(ticker, start_date, end_date, use_cache)
    df = data_manager.calculate_technical_indicators(df)
    data_manager.validate_data(df, app_config.FEATURES)
    return df

@st.cache_resource # Cacheia o modelo LSTM treinado e o scaler
def get_trained_model(input_shape, lookback, df_data, features, retrain):
    """
    Constrói, treina ou carrega o modelo LSTM.
    Esta função é cacheada para evitar retreinamento desnecessário.
    """
    model_instance = LSTMModel(input_shape=input_shape, config=app_config)

    model_exists = os.path.exists(app_config.MODEL_PATH) and os.path.exists(app_config.SCALER_PATH)

    force_retrain_due_to_mismatch = False
    if model_exists:
        try:
            # Tentar carregar temporariamente para verificar a compatibilidade
            temp_model = tf.keras.models.load_model(app_config.MODEL_PATH)
            loaded_input_features = temp_model.input_shape[2]
            current_expected_features = len(features)
            if loaded_input_features != current_expected_features:
                logger.warning(f"Modelo salvo espera {loaded_input_features} features, mas o app tem {current_expected_features}. Forçando retreinamento.")
                force_retrain_due_to_mismatch = True
            del temp_model
            tf.keras.backend.clear_session() # Limpar sessão Keras após uso do temp_model
        except Exception as e:
            logger.warning(f"Erro ao inspecionar modelo existente ({e}). Forçando retreinamento.")
            force_retrain_due_to_mismatch = True

    if retrain or not model_exists or force_retrain_due_to_mismatch:
        st.info("Treinando novo modelo... Isso pode levar alguns minutos.")
        model_instance.build_model()
        X_train, X_test, y_train, y_test = model_instance.prepare_data(df_data, features, lookback)

        val_split = int(len(X_train) * app_config.TRAIN_SPLIT_RATIO)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train_final = X_train[:val_split]
        y_train_final = y_train[:val_split]

        history = model_instance.train(X_train_final, y_train_final, X_val, y_val)
        model_instance.save_model(app_config.MODEL_PATH, app_config.SCALER_PATH)
        st.session_state['training_history'] = history # Salva o histórico para plotar
        st.success("Modelo treinado e salvo com sucesso!")
    else:
        st.info("Carregando modelo e scaler existentes...")
        model_instance.load_model(app_config.MODEL_PATH, app_config.SCALER_PATH)
        st.success("Modelo e scaler carregados!")
    
    return model_instance

# --- Interface do Usuário Streamlit ---
st.title("📈 Previsor de Ações com LSTM")
st.markdown("Uma aplicação para prever preços de ações usando redes neurais LSTM.")

with st.sidebar:
    st.header("Configurações da Previsão")
    ticker_input = st.text_input("Ticker da Ação (ex: ITUB4.SA, PETR4.SA)", value=app_config.TICKER).upper()
    dias_previsao_input = st.slider("Dias para Previsão Futura", 1, 30, app_config.DIAS_PREVISAO)
    lookback_input = st.slider("Período de Lookback (dias)", 30, 120, app_config.LOOKBACK)
    use_cache = st.checkbox("Usar Dados do Cache (se disponíveis)", value=True, help="Recomendado para evitar downloads repetidos.")
    retrain_model = st.checkbox("Forçar Retreinamento do Modelo (pode demorar)", value=False, help="Marque para treinar um novo modelo do zero. Útil se você mudou features ou deseja um modelo atualizado.")
    
    st.markdown("---")
    st.info(
        "**Observações:**\n"
        "- Os dados são do Yahoo Finance.\n"
        "- O modelo é um LSTM treinado nas features selecionadas.\n"
        "- Retreinamento pode levar alguns minutos dependendo dos dados e hardware."
    )

if st.button("Executar Previsão"):
    if not ticker_input:
        st.error("Por favor, insira um ticker de ação válido.")
    else:
        # Atualiza a configuração global com os valores da UI
        app_config.TICKER = ticker_input
        app_config.DIAS_PREVISAO = dias_previsao_input
        app_config.LOOKBACK = lookback_input

        # Limpa o cache do modelo se o retrain for forçado
        if retrain_model:
            get_trained_model.clear()
            st.session_state['training_history'] = None # Limpa histórico de treinamento anterior

        with st.status("Preparando e executando a previsão...", expanded=True) as status_box:
            try:
                today = datetime.now()
                end_date = today.strftime('%Y-%m-%d')
                start_date = (today - timedelta(days=app_config.START_DATE_OFFSET_YEARS * 365)).strftime('%Y-%m-%d')

                status_box.write("1. Baixando e processando dados históricos...")
                df_data = get_and_process_data(ticker_input, start_date, end_date, use_cache)
                status_box.update(label="1. Dados processados.", state="running", expanded=True)

                status_box.write("2. Preparando o modelo LSTM...")
                # O input_shape deve ser (LOOKBACK, número de features)
                input_shape = (app_config.LOOKBACK, len(app_config.FEATURES))
                
                # O modelo será treinado/carregado por esta função cacheada
                current_model = get_trained_model(input_shape, app_config.LOOKBACK, df_data, app_config.FEATURES, retrain_model)
                status_box.update(label="2. Modelo pronto.", state="running", expanded=True)

                status_box.write("3. Gerando previsões futuras...")
                
                # --- Lógica de Previsão Futura ---
                if current_model is None or not current_model.is_trained:
                     st.error("Modelo não disponível para previsão. Tente retreiná-lo.")
                     status_box.update(label="Falha na previsão.", state="error", expanded=False)
                     logger.error("Modelo não está treinado ou carregado para previsão.")
                     st.stop() # Parar a execução aqui

                if len(df_data) < app_config.LOOKBACK:
                    st.error(f"Dados insuficientes para previsão futura. Necessário pelo menos {app_config.LOOKBACK} dias, mas tem {len(df_data)}.")
                    status_box.update(label="Falha na previsão.", state="error", expanded=False)
                    st.stop()

                last_data_point_scaled = current_model.scaler.transform(df_data[app_config.FEATURES].tail(app_config.LOOKBACK).values)
                actual_last_price = df_data['Close'].iloc[-1]
                
                predictions = []
                current_batch = last_data_point_scaled # Shape (lookback, num_features)

                for i in range(app_config.DIAS_PREVISAO):
                    predicted_scaled_price = current_model.predict(current_batch[np.newaxis, :, :])[0, 0]
                    
                    dummy_row_for_inverse = np.zeros((1, len(app_config.FEATURES)))
                    close_idx = app_config.FEATURES.index("Close")
                    dummy_row_for_inverse[0, close_idx] = predicted_scaled_price
                    
                    predicted_price = current_model.scaler.inverse_transform(dummy_row_for_inverse)[0, close_idx]
                    predictions.append(predicted_price)

                    next_day_features_scaled = current_batch[-1].copy() 
                    next_day_features_scaled[close_idx] = predicted_scaled_price
                    
                    current_batch = np.vstack([current_batch[1:], next_day_features_scaled[np.newaxis, :]])
                # --- Fim da Lógica de Previsão Futura ---

                future_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, app_config.DIAS_PREVISAO + 1)]
                
                metrics = current_model.evaluate_model(
                    *current_model.prepare_data(df_data, app_config.FEATURES, app_config.LOOKBACK)[1:4] # X_test, y_test
                )

                status_box.update(label="Previsão concluída!", state="complete", expanded=False)

                st.subheader(f"📊 Resultados da Previsão para {ticker_input}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Último Preço Histórico ({df_data.index[-1].strftime('%Y-%m-%d')})", value=f"R$ {actual_last_price:.2f}")
                with col2:
                    st.metric(label=f"Preço Previsto para {future_dates[-1]}", value=f"R$ {predictions[-1]:.2f}", 
                              delta=f"{(predictions[-1] - actual_last_price):+.2f} ({((predictions[-1] - actual_last_price)/actual_last_price)*100:+.1f}%)")

                st.markdown("---")
                st.subheader("📈 Previsões Futuras Detalhadas")
                prediction_df = pd.DataFrame({
                    "Data": future_dates,
                    "Preço Previsto (R$)": [f"{p:.2f}" for p in predictions]
                })
                st.dataframe(prediction_df, use_container_width=True)

                st.markdown("---")
                st.subheader("📉 Métricas de Avaliação do Modelo (no Conjunto de Teste)")
                st.write(f"**RMSE (Erro Quadrático Médio):** {metrics.rmse:.4f}")
                st.write(f"**MAE (Erro Absoluto Médio):** {metrics.mae:.4f}")
                st.write(f"**R² (Coeficiente de Determinação):** {metrics.r2:.4f}")

                st.markdown("---")
                st.subheader("Gráficos de Resultados")
                # Plota as previsões e dados históricos
                fig_predictions = visualizer.plot_predictions(
                    df_data, predictions, future_dates, ticker_input
                )
                st.pyplot(fig_predictions)
                plt.clf() # Limpar a figura atual do matplotlib para evitar sobreposição

                # Plota o histórico de treinamento se disponível
                if 'training_history' in st.session_state and st.session_state['training_history']:
                    st.subheader("Histórico de Treinamento do Modelo")
                    fig_history = visualizer.plot_training_history(st.session_state['training_history'])
                    st.pyplot(fig_history)
                    plt.clf()

                logger.info("Previsão e display no Streamlit concluídos com sucesso.")

            except ValueError as ve:
                st.error(f"Erro de Validação: {ve}")
                logger.error(f"Erro de validação no Streamlit: {ve}", exc_info=True)
                status_box.update(label="Falha: Erro de validação.", state="error", expanded=False)
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}. Verifique o log para mais detalhes.")
                logger.error(f"Erro geral no Streamlit: {e}", exc_info=True)
                status_box.update(label="Falha: Erro inesperado.", state="error", expanded=False)