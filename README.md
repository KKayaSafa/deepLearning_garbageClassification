# Jupyter Notebooks for Price Prediction and Time Series Classification

## Overview
This repository contains Jupyter notebooks focused on price prediction and time series classification using various machine learning techniques. The notebooks leverage PySpark and other machine learning libraries to analyze and model financial data.

## Notebooks Included

### 1. PricePredictingSparkRegression.ipynb
**Description:**
This notebook utilizes PySpark's regression models to predict prices based on historical data. It involves data preprocessing, feature engineering, and model training using regression algorithms.

**Key Features:**
- Data loading and preprocessing using PySpark.
- Feature selection and transformation.
- Implementation of regression models for price prediction.
- Model evaluation using performance metrics.

**Dependencies:**
- PySpark
- Pandas
- NumPy
- Matplotlib/Seaborn (for visualization)

### 2. StockTimeSeriesClassification.ipynb
**Description:**
This notebook applies machine learning techniques to classify stock price movements based on historical time series data.

**Key Features:**
- Data preprocessing and feature extraction from stock market time series.
- Implementation of classification models (e.g., Random Forest, XGBoost, or Neural Networks).
- Model evaluation and performance analysis.

**Dependencies:**
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn

### 3. time_series_classification_with_daily_data.ipynb
**Description:**
This notebook focuses on daily time series data classification, leveraging deep learning models and statistical approaches.

**Key Features:**
- Data handling and visualization of time series data.
- Implementation of deep learning models (e.g., LSTMs, CNNs) for classification.
- Model training, evaluation, and performance comparison.

**Dependencies:**
- TensorFlow/Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn

## Usage
1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```
2. Install dependencies:
   ```sh
   pip install pyspark scikit-learn tensorflow pandas numpy matplotlib seaborn
   ```
3. Open Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
4. Navigate to the desired notebook and run the cells sequentially.

## Notes
- Ensure that your system has sufficient memory and computational power for training deep learning models.
- Data sources used in the notebooks should be placed in the appropriate directories before execution.


