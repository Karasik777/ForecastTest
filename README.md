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

Forecasting financial time series is difficult due to volatility, noise, and structural breaks. Traditional methods such as ARIMA or exponential smoothing struggle to capture nonlinearities. Deep learning approaches, such as LSTMs, improve by modeling sequential dependencies, but still fall short when long-term dependencies and dynamic covariates are present.

The Temporal Fusion Transformer (TFT) addresses these challenges:
	•	Uses LSTM encoders/decoders for local dependencies.
	•	Uses multi-head attention for long-range relationships.
	•	Uses variable selection networks (VSNs) for dynamic feature relevance.
	•	Produces quantile forecasts (not just point predictions).

This makes TFT particularly suitable for cryptocurrency forecasting.

⸻

Forecasting Mechanics
	1.	Time Indexing
Each record has a monotonically increasing time_idx (built from timestamps). This allows the model to learn sequential structure.
	2.	Encoder–Decoder Windowing
With lookback $L$ and horizon $H$, the model encodes the past $L$ steps and predicts the next $H$.
Example: with $L=3$, $H=2$

y[t-2], y[t-1], y[t] → predict y[t+1], y[t+2]

3.	Variable Selection Networks (VSNs)
Learn dynamic weights for each input feature at every step.
4.	LSTM Sequence Modeling
Captures short-term sequential dependencies:

h_t = LSTM(x_t, h_{t-1})

5.	Attention Mechanism
Captures long-range dependencies across the entire lookback.
6.	Quantile Forecasting
Instead of a single output, TFT predicts multiple quantiles (e.g., 0.1, 0.5, 0.9) for uncertainty estimation.

Toy Mathematical Example
Let’s model:
y_t = 0.5 * y_{t-1} + sin(dow_t) + ε_t
ε_t ~ N(0,1)

	•	y_{t-1} is autoregressive dependence.
	•	dow_t (day of week) introduces a weekly cycle.
	•	ε_t adds noise.

If the past 3 values are [100, 102, 103] and the dow are [Mon, Tue, Wed], the model predicts Thu/Fri values using both autoregression and cyclical features. TFT will:
	•	LSTM: capture the AR(1)-like dependence (0.5 * y_{t-1}).
	•	Attention: highlight the role of weekday cycles.
	•	Quantiles: output prediction intervals around the median.

Pipeline Overview
flowchart TD
  A[Raw OHLCV Data] --> B[Feature Engineering<br/>(lags, RSI, volatility)]
  B --> C[TimeSeriesDataSet Construction]
  C --> D[Temporal Fusion Transformer]
  D --> E[Probabilistic Forecasts]

Scripts
•	scripts/setup_data.py
Fetch raw OHLCV data (e.g., BTC, ETH).
•	scripts/make_features.py
Generate lagged returns, RSI, volatility, calendar features.
•	scripts/train_tft.py
Define and train the TFT model, save checkpoints, produce plots.
•	scripts/exec.py
Orchestrator for the full pipeline (flags for fetch, features, train, eval).
•	scripts/evaluate.py (optional)
Evaluate checkpoints and output metrics.

Artifacts
	•	Checkpoints: artifacts/*.ckpt
	•	Plots: artifacts/qualitative_forecast.png
	•	Metrics: artifacts/evaluation.csv
	•	Data: data/raw/ and data/processed/merged.parquet

Results

Forecast Plot

Placeholder: qualitative forecast plot showing:
	•	Encoder history
	•	True future
	•	Predicted median

Evaluation Plot

Placeholder: metrics such as validation loss, quantile coverage.

Conclusion

We presented a full pipeline for cryptocurrency forecasting using the Temporal Fusion Transformer. TFT combines sequence modeling, attention, and feature selection to produce interpretable and probabilistic forecasts. The system demonstrates how raw OHLCV data can be transformed into actionable insights with deep learning. Future extensions may include live streaming, sentiment features, and multi-resolution forecasting.
