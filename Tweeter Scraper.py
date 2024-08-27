# Install Libries

**Snscrape** is a Python package that allows users to easily scrape and extract data from various social media platforms, including Twitter, Instagram, and Tumblr. It provides a simple and efficient way to retrieve data from these platforms using their respective APIs without needing to write complex web scrapers. Snscrape can be installed through pip, and once installed, it can be used to retrieve data such as tweets, user profiles, hashtags, and more, making it a powerful tool for social media analysis and research.
"""

!pip install snscrape

"""# Import Libries

**nltk:**
 The Natural Language Toolkit (NLTK) is a leading platform for building Python programs to work with human language data. It provides a suite of text processing libraries for tasks such as tokenization, stemming, tagging, parsing, semantic reasoning, and wrappers for industrial-strength NLP libraries.
SentimentIntensityAnalyzer: It is a part of the nltk library and is used for sentiment analysis. It is a pre-trained model that uses a lexicon-based approach to score the sentiment of a given text. It returns a sentiment score ranging from -1 (negative) to +1 (positive).

**snscrape: **
snscrape is a Python package that provides an easy way to access Twitter data. It allows you to scrape Twitter content, such as tweets, profiles, and trends, without requiring any API credentials. It can retrieve both historical and real-time data.

**matplotlib:**
 matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.
WordCloud: WordCloud is a Python library for generating word clouds, which are visual representations of text data. It is an easy-to-use package that takes a list of words and generates a word cloud image based on the frequency of each word.

**seaborn:** seaborn is a data visualization library based on matplotlib. It provides a high-level interface for creating informative and attractive statistical graphics. It includes functions for visualizing univariate and bivariate data, categorical data, and timeseries data.
unicodedata: The unicodedata module provides access to the Unicode Character Database, which defines character properties for all Unicode characters. It is used in the code snippet to normalize the text data.
pandas: pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool. It is used in the code snippet to store and manipulate the data.

**datetime:** The datetime module supplies classes for working with dates and times. It is used in the code snippet to retrieve and manipulate the date and time data.

**pickle:** pickle is a Python module used for serializing and de-serializing Python object structures. It is used in the code snippet to save and load the data.
"""
# Import the libraries used in the code
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import unicodedata
import pandas as pd
import datetime
import pickle
import nltk
import math

"""# scraping steps

The variable **query_terms** contains a list of search terms that can be used to query social media platforms, specifically Twitter, for posts related to the Toronto Stock Exchange, Shopify, the Canadian economy, business, finance, stocks, and investing. The terms include keywords and hashtags commonly used in social media discussions related to these topics. Additionally, the query includes specific Twitter handles, such as @FinancialPost and @CBCBusiness, that may provide relevant and authoritative information related to the Canadian financial landscape.
"""

# define Query accordin to your Project
query_terms = ['TSX OR "Toronto Stock Exchange" OR "TorontoStockExchange"', 'Shopify', '#CanadaEconomy', '#CanadianBusiness', '#CanadianFinance', '#CanadaStocks', '#CanadaInvesting', '@FinancialPost', '@CBCBusiness']

"""This code defines a function to extract stock and company names from tweet text, and then proceeds to scrape tweets using the **snscrape** library based on a list of query terms, a start date, and an end date. It specifies the maximum number of tweets to scrape per query and appends the date, tweet text, stock name, and company name to a list of tweets. A progress message is printed every 1000 tweets, as well as a message when all tweets have been scraped for a given query term. The resulting list of tweets is saved to a file with a filename containing the current date and time.


"""

# Define function to extract stock and company names from tweet text
def extract_names(tweet_text):
    stock_names = ['TSX']
    company_names = ['Toronto Stock Exchange']
    return stock_names, company_names

# Define filename with current date and time
now = datetime.datetime.now()
filename = f"tweets_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
start_date = '2010-01-01'
end_date = '2023-02-28'
num_tweets_per_query = 100000

# Scrape tweets and append to list
tweets = []
for query_term in query_terms:
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query_term + ' since:' + start_date + ' until:' + end_date + ' lang:en').get_items()):
        if i >= num_tweets_per_query:
            break
        date = tweet.date.strftime('%Y-%m-%d')
        text = tweet.content
        stock_names, company_names = extract_names(text)
        for stock_name, company_name in zip(stock_names, company_names):
            tweets.append([date, text, stock_name, company_name])

        # Print progress message every 1000 tweets
        if len(tweets) % 1000 == 0:
            print(f"Scraped {len(tweets)} tweets...")

    # Print progress message for each query term
    print(f"Finished scraping tweets for query term: {query_term}")

"""The code saves the list of scraped tweets to a file using pickle serialization. The "wb" flag in the "open" function specifies that the file should be opened in binary mode for writing. The **"pickle.dump"** function is used to write the tweets list to the file.


"""

# Save tweets to pickle file
with open(filename, "wb") as f:
    pickle.dump(tweets, f)

"""loading a pickle file that contains previously scraped tweets using the pickle module. The file is located in the directory and is named  Once the file is opened, the pickle.load() function is used to deserialize the data from the file and load it into the tweets variable."""

# Print message indicating number of tweets scraped
print(f"Scraped a total of {len(tweets)} tweets.")
with open('/content/tweets_2023-03-16_20-58-09.pkl', 'rb') as f:
    tweets = pickle.load(f)

"""The list of tweets has been converted to a pandas DataFrame with columns 'Date', 'Tweet', 'Stock Name', and 'Company Name'. This will allow for easier data analysis and manipulation going forward."""

# Convert tweets to pandas DataFrame
df = pd.DataFrame(tweets, columns=['Date', 'Tweet', 'Stock Name', 'Company Name'])

# Save tweets to CSV file
df.to_csv('/content/millions_tweets.csv', index=False)

with open('/content/tweets_2023-03-16_20-58-09.pkl', 'rb') as f:
    tweets = pickle.load(f)

# Convert tweets to pandas DataFrame
df = pd.DataFrame(tweets, columns=['Date', 'Tweet', 'Stock Name', 'Company Name'])

# Save tweets to CSV file
df.to_csv('millions_tweets.csv', index=False)

df=pd.read_csv('/content/millions_tweets.csv')

# Show First 5 tweets of CSV file
df.head(12)
