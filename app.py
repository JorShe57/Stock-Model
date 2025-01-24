import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
import yfinance as yf
from flask import Flask, request, jsonify, render_template
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime, timedelta
import numpy as np
from flask_caching import Cache

# Initialize Flask app and caching
app = Flask(__name__)
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

# Set random seed for reproducibility
import tensorflow as tf
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed()

# Step 1: Fetch Data from yfinance (with caching)
@cache.memoize(timeout=3600)  # Cache for 1 hour
def fetch_data_from_api(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
        if df.empty:
            raise ValueError(f"No data found for ticker {symbol}. Please check the symbol.")
        return df['Close']
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}")

# Step 2: Preprocess data
def preprocess_data(data):
    global_scaler = MinMaxScaler(feature_range=(0, 1))  # Consistent scaling range
    data = data.values.reshape(-1, 1)
    global_scaler.fit(data)  # Fit to full dataset for consistent scaling
    data_normalized = global_scaler.transform(data)
    return data_normalized, global_scaler

# Step 3: Create sequences for LSTM
def create_sequences(data, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# Step 4: Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5.1: Predict Future Prices with Confidence Intervals
def predict_future_prices_with_intervals(model, data, scaler, sequence_length=60, num_days=30, simulations=1000):
    future_predictions = []
    last_sequence = data[-sequence_length:]  # Get the last sequence from the data

    for day in range(num_days):
        # Monte Carlo Simulations with reduced noise
        input_data = np.tile(last_sequence.reshape(1, sequence_length, 1), (simulations, 1, 1))
        simulated_predictions = model.predict(input_data, verbose=0).flatten()
        simulated_predictions += np.random.normal(0, 0.001, size=simulations)  # Reduced std deviation

        mean_prediction = np.mean(simulated_predictions)
        lower_bound = np.percentile(simulated_predictions, 2.5)
        upper_bound = np.percentile(simulated_predictions, 97.5)

        future_predictions.append((mean_prediction, lower_bound, upper_bound))

        # Update the sequence
        new_data_point = np.array([[mean_prediction]])
        last_sequence = np.vstack((last_sequence[1:], new_data_point))

    future_predictions = np.array(future_predictions)
    mean_predictions = scaler.inverse_transform(future_predictions[:, 0].reshape(-1, 1))
    lower_bounds = scaler.inverse_transform(future_predictions[:, 1].reshape(-1, 1))
    upper_bounds = scaler.inverse_transform(future_predictions[:, 2].reshape(-1, 1))

    return mean_predictions, lower_bounds, upper_bounds

# Step 5.2: Create Plot with Confidence Intervals
def create_plot_with_intervals(dates, actual_prices, mean_predictions, lower_bounds, upper_bounds, future_dates):
    historical_dates = dates[-60:]
    historical_prices = actual_prices[-60:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode='lines',
        name='Historical Prices',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=mean_predictions.flatten(),
        mode='lines+markers',
        name='Mean Predictions',
        line=dict(color='green'),
        marker=dict(size=8),
        hovertemplate='Date: %{x}<br>Mean Predicted Price: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bounds.flatten(),
        mode='lines+markers',
        name='Upper Band',
        line=dict(color='red', dash='dash'),
        marker=dict(size=8),
        hovertemplate='Date: %{x}<br>Upper Band Price: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bounds.flatten(),
        mode='lines+markers',
        name='Lower Band',
        line=dict(color='purple', dash='dash'),
        marker=dict(size=8),
        hovertemplate='Date: %{x}<br>Lower Band Price: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper_bounds.flatten(), lower_bounds.flatten()[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 200, 100, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='Stock Price Prediction with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified'
    )

    return pio.to_html(fig, full_html=False)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker']
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        # Fetch and preprocess data
        data = fetch_data_from_api(ticker, start_date, end_date)
        data_normalized, scaler = preprocess_data(data)

        sequence_length = 60
        X, y = create_sequences(data_normalized, sequence_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]

        # Build and train model
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Predict future prices
        mean_predictions, lower_bounds, upper_bounds = predict_future_prices_with_intervals(
            model, data_normalized, scaler, sequence_length=sequence_length, num_days=30, simulations=1000
        )

        # Generate future dates
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='B')

        # Create plot
        plot_html = create_plot_with_intervals(
            dates=data.index,
            actual_prices=data.values.flatten(),
            mean_predictions=mean_predictions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            future_dates=future_dates
        )

        return render_template('prediction.html', plot_html=plot_html)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)

