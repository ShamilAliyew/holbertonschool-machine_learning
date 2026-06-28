# 📈 BTC Price Forecasting with RNNs

Bitcoin (BTC) became a major topic of interest after its price surge in 2018. Since then, many have attempted to predict its future value using historical market data.  
This project explores the use of **Recurrent Neural Networks (RNNs)** to forecast the **Bitcoin closing price** based on minute-level trading data.

The model uses the **past 24 hours of BTC data** to predict the **closing price one hour into the future**, which roughly aligns with the average transaction confirmation time.

---

## 🧠 Objective

- Use historical BTC/USD data to perform time-series forecasting  
- Predict the BTC **closing price 60 minutes ahead**  
- Train and validate a Keras-based RNN model  
- Feed data efficiently using `tf.data.Dataset`  
- Properly preprocess raw financial time-series data  

---

## 📊 Dataset

Two public datasets are used and combined:

- **Coinbase BTC/USD**
- **Bitstamp BTC/USD**

Each row represents a **60-second time window** and includes:

1. Timestamp (Unix time)
2. Open price (USD)
3. High price (USD)
4. Low price (USD)
5. Close price (USD)
6. BTC volume
7. USD volume
8. Volume-weighted average price (VWAP)

---

## 🧹 Data Preprocessing (`preprocess_data.py`)

Since the datasets are raw, a separate preprocessing script is used.

### Data Cleaning
- Rows with missing timestamps are removed  
- Only strictly **60-second intervals** are kept  
- Irregular or broken time sequences are discarded  
- Data is sorted chronologically  

### Feature Selection
The following features are used as inputs:

- Open  
- High  
- Low  
- Close  
- BTC volume  
- USD volume  
- Weighted price (VWAP)  

The target variable is the **future Close price**.

### Scaling
- Features are standardized using `StandardScaler`  
- Scalers are fit **only on training data** to avoid data leakage  
- Target values are scaled separately  

---

## ⏱ Time-Series Windowing

The time-series data is converted into supervised learning samples using a sliding window approach:

- **Input window:** 1440 minutes (24 hours)  
- **Prediction horizon:** 60 minutes  
- Input shape: `(1440, 7)`  
- Output: single future closing price  

A custom generator is implemented to stream data in batches without loading the entire dataset into memory.

---

## 🔁 TensorFlow Data Pipeline

The generator is wrapped using:

- `tf.data.Dataset.from_generator`
- Prefetching with `AUTOTUNE`

This allows efficient and scalable training on large time-series datasets.

---

## 🧩 Model Architecture (`forecast_btc.py`)

The forecasting model uses an LSTM-based RNN architecture:



This structure enables the model to capture both short-term and long-term temporal dependencies in the data.

---

## ⚙️ Training Configuration

- Optimizer: Adam  
- Loss function: Mean Squared Error (MSE)  
- Metric: Mean Absolute Error (MAE)  
- Batch size: 32  
- Epochs: 10  

### Callbacks
- EarlyStopping (to reduce overfitting)  
- ModelCheckpoint (to save the best model)  

---

## 📉 Evaluation

Training and validation performance is monitored using:

- Loss (MSE)
- Mean Absolute Error (MAE)

Results are visualized to analyze convergence and generalization behavior.

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  

---

## 🚀 Notes

This project focuses on **correct data preprocessing, temporal modeling, and training discipline**, rather than claiming precise market prediction.  
It serves as a practical application of **RNNs for financial time-series forecasting** using real-world data.
