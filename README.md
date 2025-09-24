# ForecastTest: Temporal Fusion Transformer for Cryptocurrency Forecasting

## Usage

```bash
# Run the full pipeline end-to-end
python scripts/exec.py --full \
  --symbols BTCUSDT ETHUSDT \
  --interval 1m \
  --days 3 \
  --lookback 168 \
  --horizon 24 \
  --epochs 15 \
  --batch-size 64 \
  --device cpu
  ```


Artifacts will be saved in:
•	artifacts/*.ckpt — model checkpoints
•	artifacts/qualitative_forecast.png — forecast plot
•	artifacts/evaluation.csv — metrics
•	data/raw/, data/processed/merged.parquet — raw & processed data

Introduction

Forecasting financial time series is challenging due to volatility, noise, and structural breaks. Traditional methods such as ARIMA or exponential smoothing struggle to capture nonlinearities. Deep learning approaches, such as LSTMs, improve by modeling sequential dependencies, but still fall short when long-term dependencies and dynamic covariates are present.

The Temporal Fusion Transformer (TFT) addresses these challenges:
•	Uses LSTM encoders/decoders for local dependencies.
•	Uses multi-head attention for long-range relationships.
•	Uses variable selection networks (VSNs) for dynamic feature relevance.
•	Produces quantile forecasts (not just point predictions).

This makes TFT particularly suitable for cryptocurrency forecasting.

Forecasting Mechanics
1.	Time Indexing
Each record is assigned a monotonically increasing time_idx, derived from timestamps, enabling the model to understand sequential order.
2.	Encoder–Decoder Windowing
With lookback L and horizon H, the model encodes the past L steps and predicts the next H.
Example: with L=3, H=2
y[t-2], y[t-1], y[t] → predict y[t+1], y[t+2]

3.	Variable Selection Networks (VSNs)
Dynamically weight input features at each time step to focus on relevant signals.
4.	LSTM Sequence Modeling
Captures short-term sequential dependencies:
h_t = LSTM(x_t, h_{t-1})

5.	Attention Mechanism
Captures long-range relationships across the entire lookback window.
6.	Quantile Forecasting
Rather than a single scalar output, TFT predicts multiple quantiles (e.g., 0.1, 0.5, 0.9) to express uncertainty.

Toy Mathematical Example

We model a simple process:

[
y_t = 0.5 \cdot y_{t-1} + \sin(\mathrm{dow}_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)
]
•	( y_{t-1} ) captures autoregressive dependence.
•	(\mathrm{dow}_t) (day of week) introduces cyclic behavior.
•	(\epsilon_t) is noise.

Example Setup:
•	Past 3 values: [100, 102, 103]
•	Days: [Mon, Tue, Wed]
•	Horizon to predict: [Thu, Fri]

TFT does:
•	LSTM: model the AR(1) effect ((0.5 \cdot y_{t-1}))
•	Attention: identify cyclical influence of day-of-week
•	Quantile outputs: provide prediction intervals rather than a single guess


Pipeline Overview
flowchart TD
  A[Raw OHLCV Data] --> B[Feature Engineering<br/>(lags, RSI, volatility)]
  B --> C[TimeSeriesDataSet Construction]
  C --> D[Temporal Fusion Transformer]
  D --> E[Probabilistic Forecasts]

Scripts
•	scripts/setup_data.py — Fetch raw OHLCV data (e.g., BTC, ETH)
•	scripts/make_features.py — Generate features (lags, RSI, volatility, calendar)
•	scripts/train_tft.py — Define and train the TFT model, save checkpoints, produce forecast plots
•	scripts/exec.py — Orchestrator for the full pipeline (flags for fetch, features, train, eval)
•	scripts/evaluate.py — Evaluate trained models and output metrics

Artifacts
•	Checkpoints: artifacts/*.ckpt
•	Forecast plots: artifacts/qualitative_forecast.png
•	Evaluation metrics: artifacts/evaluation.csv
•	Data output: data/raw/ and data/processed/merged.parquet

Results

Forecast Plot

Placeholder: forecast plot showing encoder history, actual future, and predicted median

Evaluation Plot

Placeholder: metrics such as validation MAPE/RMSE and quantile coverage

Conclusion

We present a complete pipeline for cryptocurrency forecasting using the Temporal Fusion Transformer. TFT combines sequence modeling, attention, and dynamic feature selection to produce interpretable and probabilistic forecasts. The system demonstrates how raw OHLCV data can be transformed into meaningful predictions with deep learning.

Future work could include:
•	Real-time, streaming inference
•	Incorporation of sentiment, news, or social signals
•	Multi-resolution forecasting (e.g. 1m, 5m, 1h intervals)