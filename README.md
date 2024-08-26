# Social Media Sentiment Analysis and Time Series Prediction for Financial Markets

## Overview

This project combines social media sentiment analysis with time series prediction to gain insights into the financial markets, particularly focusing on the Toronto Stock Exchange (TSX) and Canadian economy. The code utilizes various libraries including NLTK, snscrape, matplotlib, seaborn, pandas, and machine learning frameworks like TensorFlow and PyTorch.

## Table of Contents

1. Data Collection and Preprocessing
2. Sentiment Analysis
3. Time Series Analysis
4. Machine Learning Models
   - LSTM Model
   - Transformer Model
   - GAN Model
5. Visualization and Results

## Dependencies

- nltk
- snscrape
- matplotlib
- seaborn
- pandas
- datetime
- pickle
- numpy
- tensorflow
- pytorch
- transformers

## Data Collection and Preprocessing

The project starts by scraping tweets related to the Toronto Stock Exchange, Canadian economy, business, finance, stocks, and investing using snscrape. The scraped tweets are then preprocessed and stored in a pandas DataFrame.


## Sentiment Analysis

Sentiment analysis is performed using NLTK's VADER sentiment analyzer. The code calculates sentiment scores for each tweet and adds them to the DataFrame.


## Time Series Analysis

The code performs time series analysis by grouping tweets by date and calculating the average sentiment score for each day.


## Machine Learning Models

### LSTM Model

An LSTM model is trained to predict stock prices based on historical data and sentiment analysis results.


## Visualization and Results

The project includes various visualizations to display the results of the sentiment analysis and machine learning models.


## Conclusion

This project demonstrates how social media sentiment analysis and time series prediction can be combined to gain insights into financial markets. The code showcases various techniques including data preprocessing, sentiment analysis, and advanced machine learning models. The results provide valuable information about market trends and sentiment shifts over time, which can be useful for investors and financial analysts.

## Future Work

1. Expand the dataset to include more sources of financial news and data.
2. Implement real-time sentiment analysis for current market conditions.
3. Develop a web interface for users to interact with the generated data and predictions.
4. Explore other machine learning architectures for improved performance.
5. Conduct a thorough evaluation of the model's performance against actual market data.
