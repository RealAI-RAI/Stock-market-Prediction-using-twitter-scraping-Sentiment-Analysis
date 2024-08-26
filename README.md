# Social Media Sentiment Analysis and Time Series Prediction for Financial Markets




## Overview

![Figure 5](https://github.com/user-attachments/assets/e5ee81db-b999-42c9-8576-ab6700e8eec2) 
![Figure 6](https://github.com/user-attachments/assets/50ae3eb2-3007-4726-8131-d3abd2cf3a96)


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

## Methodology

The methodology employed in this project involves the following steps:

-Data Collection: Gather 22 years of true historical stock data from the Toronto Stock Exchange.

-GAN Model Training: Utilize generative adversarial networks (GANs) to train the deep learning model. GANs consist of two neural networks - a generator network that generates synthetic data and a discriminator network that distinguishes between real and synthetic data. The GAN model is trained to generate realistic stock price predictions.

-Model Evaluation: Compare the performance of the GAN model against various machine learning and deep learning methods such as Random Forests (RF), gradient boosting (GB), Support Vector Machines (SVM), and long-short term memory (LSTM). Several evaluation metrics, including root mean square error (RMSE), mean squared error (MSE), maximum error (ME), R-squared (R2), explained variance score (EVS), and mean absolute error (MAE), are used to assess the performance of the models.

-Enhancement with a Temporal Attention Layer: Improve the GAN model by incorporating a Temporal Attention Layer. This layer aids in identifying and forecasting market movements, enhancing the accuracy of the predictions.

-Incorporation of External Data: Enhance the accuracy of projections by incorporating data from news and social networking sites. This additional data can provide valuable insights and help improve the forecasting capabilities of the model.

## Installation

To run this project locally, follow these steps:

Clone the repository: git clone https://github.com/RealAI-RAI/Stock-market-Prediction-using-twitter-scraping-Sentiment-Analysis.git

Install the required dependencies: pip install -r requirements.txt

Run the data preprocessing script: python scripts/preprocess_data.py

Train the GAN model: python scripts/train_gan.py

Evaluate the model and compare against other methods: python scripts/evaluate_models.py

Access the Jupyter notebooks in the notebooks/ directory for a detailed explanation and implementation of the models.

## Future Work

1. Expand the dataset to include more sources of financial news and data.
2. Implement real-time sentiment analysis for current market conditions.
3. Develop a web interface for users to interact with the generated data and predictions.
4. Explore other machine learning architectures for improved performance.
5. Conduct a thorough evaluation of the model's performance against actual market data.

## Contact
For any questions or suggestions, feel free to reach out to the project maintainer:

Name: Inam Ullah
Email: Inamullahiiufet@gmail.com
Note: This README provides an overview of the project and its structure. For detailed explanations and step-by-step implementation, please refer to the Jupyter notebooks in the notebooks/ directory.
