# Pneumonia-Detection---CNN
The notebook builds an end‑to‑end LSTM model to forecast stock closing prices (AAPL) from historical data. It downloads daily prices, prepares a supervised learning dataset with sliding 60‑day windows, trains a stacked LSTM network, and uses the trained model to predict and inverse‑scale future prices, achieving low training and validation loss on the test period.

# Objective and Data
- The stated goal is to predict future closing prices using an LSTM network that learns from past closing price sequences.​
- Historical Apple (AAPL) data from 2016‑01‑01 to 2024‑01‑01 is fetched via the yfinance API, and only the Close column is used for modeling.

# Preprocessing and Sequence Creation
- The close prices are reshaped to a single-column series and scaled to [0,1] using MinMaxScaler to stabilize and speed up LSTM training.​
- A lookback window of 60 days is applied: for each time step, the previous 60 scaled values form the input sequence 
X, and the next value is the target y; these are then reshaped to the 3D shape required by Keras LSTMs.​

# Model Architecture and Training
- The model is a Keras Sequential network with two stacked LSTM layers (50 units each, the first returning sequences) followed by a dense output layer with a single neuron for the next‑day price.​
- The dataset is split 80/20 into train and test sets, the model is compiled with Adam optimizer and MSE loss, and trained for 10 epochs with a validation split of 0.1, producing very low training and validation loss values.

# Prediction and Inverse Scaling
- After training, the model predicts on the test sequences X_test, and the predicted scaled values are inverse‑transformed back to the original price range using the fitted scaler.​
- This allows direct comparison of predicted prices with actual closing prices in real currency units, illustrating how well temporal patterns in the series are captured.

# Final Insights
- The notebook concludes that the LSTM achieves low error on historical data and demonstrates the ability of sequence models to learn temporal dependencies in stock prices for trend forecasting.​
- It also notes that despite good in‑sample performance, real‑world stock prediction remains uncertain due to volatility and external market factors beyond what the model sees in past prices.

