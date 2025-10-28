import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

def run_trend_prediction(stock_symbol_or_name: str) -> str:
    stock_symbol = stock_symbol_or_name.upper().strip() + ".NS"
    start_date_str = '2022-09-01'
    end_date = datetime.now()
    
    try:
        df = yf.download(stock_symbol, start=start_date_str, end=end_date, progress=False)
        if df.empty:
            return f"No data returned for '{stock_symbol}'"
    except Exception as e:
        return f"Error downloading data: {e}"

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features]
    data.fillna(method='ffill', inplace=True)
    
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    time_step = 60
    train_len = int(np.ceil(len(scaled_data) * 0.95))
    train_data = scaled_data[:train_len, :]
    test_data = scaled_data[train_len - time_step:, :]
    
    def create_dataset(dataset, time_step):
        x, y = [], []
        for i in range(time_step, len(dataset)):
            x.append(dataset[i - time_step:i, :])
            y.append(dataset[i, 3])
        return np.array(x), np.array(y)
    
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], len(features))),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=200, batch_size=50, validation_split=0.2, callbacks=[es], verbose=0)
    
    predictions = model.predict(x_test)
    dummy = np.zeros((len(predictions), len(features)))
    dummy[:, 3] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy)[:, 3]

    trend_window = 5
    if len(predictions) >= trend_window:
        start_price = predictions[-trend_window]
        end_price = predictions[-1]
        change = end_price - start_price
        if change > 0:
            trend = "Upward 📈"
        elif change < 0:
            trend = "Downward 📉"
        else:
            trend = "Neutral ↔"
        return f"Based on last {trend_window} days, the trend for {stock_symbol_or_name.title()} is: {trend}"
    return f"Not enough data to analyze trend for {stock_symbol_or_name.title()}"
