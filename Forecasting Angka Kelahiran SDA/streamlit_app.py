# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go

# page config
st.set_page_config(page_title="Forecasting AHH Sidoarjo", layout="wide", page_icon="📈")

# path dataset
path_df_pria = 'data/format/df_pria.csv'
path_df_wanita = 'data/format/df_wanita.csv'

# load dataset
df_pria = pd.read_csv(path_df_pria)
df_wanita = pd.read_csv(path_df_wanita)

# Preprocessing
scaler_wanita = MinMaxScaler()
scaler_pria = MinMaxScaler()

scaled_wanita = scaler_wanita.fit_transform(df_wanita['Jumlah'].values.reshape(-1,1))
scaled_pria = scaler_pria.fit_transform(df_pria['Jumlah'].values.reshape(-1,1))

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 3

X_wanita, y_wanita = create_sequences(scaled_wanita, seq_length)
X_pria, y_pria = create_sequences(scaled_pria, seq_length)

X_wanita = X_wanita.reshape((X_wanita.shape[0], X_wanita.shape[1], 1))
X_pria = X_pria.reshape((X_pria.shape[0], X_pria.shape[1], 1))

# Train Model LSTM
@st.cache_resource
def train_models(X_wanita, y_wanita, X_pria, y_pria):
    model_wanita = Sequential([
        LSTM(150, activation='relu', input_shape=(X_wanita.shape[1], 1)),
        Dense(1)
    ])
    model_wanita.compile(optimizer='adam', loss='mse')
    model_wanita.fit(X_wanita, y_wanita, epochs=100, verbose=0, batch_size=2)

    model_pria = Sequential([
        LSTM(150, activation='relu', input_shape=(X_pria.shape[1], 1)),
        Dense(1)
    ])
    model_pria.compile(optimizer='adam', loss='mse')
    model_pria.fit(X_pria, y_pria, epochs=100, verbose=0, batch_size=2)

    return model_wanita, model_pria

# called function to train models
model_wanita, model_pria = train_models(X_wanita, y_wanita, X_pria, y_pria)

# Streamlit App
st.title('📈 Prediksi AHH Kota Sidoarjo per Gender')
st.markdown('---')

# Input tahun target
tahun_terakhir_data = max(df_pria['Tahun'].max(), df_wanita['Tahun'].max())
tahun_target = st.number_input(
    label=f"Masukkan Tahun Target Prediksi (min {tahun_terakhir_data + 1}):",
    min_value=tahun_terakhir_data + 1,
    max_value=2050,
    step=1,
    value=2030
)

predict_button = st.button('🔮 Prediksi!')

if predict_button:
    if tahun_target < 2025:
        st.error('❌ Tahun target prediksi harus >= 2025.')
        st.stop()
    else:
        with st.spinner('🔮 Sedang Melakukan Prediksi...'):
            tahun_prediksi = list(range(tahun_terakhir_data + 1, tahun_target + 1))

            # Prediksi Wanita
            last_seq_wanita = scaled_wanita[-seq_length:].reshape((1, seq_length, 1))
            future_preds_wanita = []
            for _ in tahun_prediksi:
                pred = model_wanita.predict(last_seq_wanita, verbose=0)[0][0]
                future_preds_wanita.append(pred)
                last_seq_wanita = np.append(last_seq_wanita[:,1:,:], [[[pred]]], axis=1)
            future_preds_wanita = scaler_wanita.inverse_transform(np.array(future_preds_wanita).reshape(-1,1)).flatten()

            # Prediksi Pria
            last_seq_pria = scaled_pria[-seq_length:].reshape((1, seq_length, 1))
            future_preds_pria = []
            for _ in tahun_prediksi:
                pred = model_pria.predict(last_seq_pria, verbose=0)[0][0]
                future_preds_pria.append(pred)
                last_seq_pria = np.append(last_seq_pria[:,1:,:], [[[pred]]], axis=1)
            future_preds_pria = scaler_pria.inverse_transform(np.array(future_preds_pria).reshape(-1,1)).flatten()
        st.success('✅ Prediksi selesai!')
    st.markdown('---')
    st.subheader('🎨 Visualisasi Prediksi')

    fig = go.Figure()

    # plot Wanita
    fig.add_trace(go.Scatter(
        x=df_wanita['Tahun'],
        y=df_wanita['Jumlah'],
        mode='lines+markers',
        name='Perempuan (Data Aktual)',
        line=dict(color='cyan')
    ))
    fig.add_trace(go.Scatter(
        x=tahun_prediksi,
        y=future_preds_wanita,
        mode='lines+markers',
        name='Perempuan (Prediksi)',
        line=dict(color='magenta', dash='dash')
    ))

    # plot Pria
    fig.add_trace(go.Scatter(
        x=df_pria['Tahun'],
        y=df_pria['Jumlah'],
        mode='lines+markers',
        name='Laki-laki (Data Aktual)',
        line=dict(color='lightgreen')
    ))
    fig.add_trace(go.Scatter(
        x=tahun_prediksi,
        y=future_preds_pria,
        mode='lines+markers',
        name='Laki-laki (Prediksi)',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title='Prediksi AHH Kota Sidoarjo per Gender',
        template='plotly_dark',
        xaxis_title='Tahun',
        yaxis_title='AHH (Angka Harapan Hidup)',
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('📊 Hasil Prediksi per Tahun')

    hasil_prediksi = pd.DataFrame({
        'Tahun': tahun_prediksi,
        'Prediksi AHH Perempuan': np.round(future_preds_wanita, 2),
        'Prediksi AHH Laki-laki': np.round(future_preds_pria, 2)
    })

    st.dataframe(hasil_prediksi, use_container_width=True)