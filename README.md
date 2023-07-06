# Stock Price Prediction Project

This project aims to predict the future prices of stocks using historical data and machine learning techniques. The prediction models are trained on historical stock market data to identify patterns and trends that can be used to make informed predictions about future stock prices.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models and Techniques](#models-and-techniques)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

Stock market prediction is a challenging and complex task due to the volatility and randomness of stock prices. This project aims to tackle this problem by leveraging historical stock market data and applying machine learning algorithms to make predictions about future stock prices. By analyzing patterns and trends in the data, the models can capture the underlying relationships and make predictions with a certain level of accuracy.

## Dataset

The project utilizes a historical dataset of stock prices. The dataset includes the following information:

- Date: The date of the stock price observation.
- Open: The opening price of the stock on that day.
- High: The highest price of the stock on that day.
- Low: The lowest price of the stock on that day.
- Close: The closing price of the stock on that day.
- Volume: The trading volume of the stock on that day.

The dataset is split into training and testing sets. The training set is used to train the prediction models, while the testing set is used to evaluate the performance of the trained models.

## Models and Techniques

The project employs the following models and techniques for stock price prediction:

1. **Linear Regression**: Linear regression is a simple but effective model for predicting continuous values. It assumes a linear relationship between the input features and the target variable (stock price).

2. **Long Short-Term Memory (LSTM)**: LSTM is a type of recurrent neural network (RNN) that can capture temporal dependencies in sequential data. It is well-suited for time series forecasting tasks like stock price prediction.

## Requirements

To run the project, the following dependencies are required:

- Python 3.x
- pandas
- numpy
- scikit-learn
- TensorFlow
- Keras

You can install the dependencies using pip by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Prepare the dataset:

   - Download the historical stock price dataset.
   - Place the dataset file in the `data/` directory.

3. Train the models:

   - Run the Jupyter notebook `main.ipynb`.
   - The notebook will load the dataset, preprocess the data, train the models, and save the trained models.

4. Make predictions:

   - Run the Jupyter notebook `main.ipynb`.
   - The notebook will load the trained models, preprocess the data, and make predictions on the test dataset.
   - The predicted stock prices will be saved in a file.

## Results

The project evaluates the performance of the prediction models using various evaluation metrics such as mean squared error (MSE) and mean absolute error (MAE). The results of the models are presented and compared in the project report
