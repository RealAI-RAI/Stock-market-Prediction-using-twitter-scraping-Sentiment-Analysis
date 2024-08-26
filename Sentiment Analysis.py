"""# Sentiment Analysis By vader lexicon

using the **SentimentIntensityAnalyzer** class from the nltk.sentiment.vader module to analyze the sentiment of each tweet in the df DataFrame.

The SentimentIntensityAnalyzer is a pre-trained model that uses a lexicon of words and their associated sentiment scores to calculate a compound score, which is a value between -1 and 1 that represents the overall sentiment of the text. A score of **-1** represents extremely **negative** **sentiment**, a score of **0** represents **neutral** **sentiment**, and a score of 1 represents extremely **positive** **sentiment**.

The for loop iterates over each row in the DataFrame and uses the polarity_scores() method of the SentimentIntensityAnalyzer class to calculate the sentiment scores for the tweet text. The sentiment scores are then added to the DataFrame as new columns: sentiment_score, Negative, Neutral, and Positive.

The try and except statements are used to handle any TypeError exceptions that may occur when attempting to normalize the tweet text using the unicodedata.normalize() method.
"""

nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()
for indx, row in df.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', df.loc[indx, 'Tweet'])
        sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
        df.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
        df.at[indx, 'Negative'] = sentence_sentiment['neg']
        df.at[indx, 'Neutral'] = sentence_sentiment['neu']
        df.at[indx, 'Positive'] = sentence_sentiment['pos']
    except TypeError:
        print (df.loc[indx, 'Tweet'])
        print (indx)
        break

head_tweets = df.head(12)  # Get the first 5 tweets
tail_tweets = df.tail(10)  # Get the last 5 tweets

# Display the head tweets with sentiment scores
print("Head 5 Tweets:")
for index, row in head_tweets.iterrows():
    print("Tweet:", row['Tweet'])
    print("Sentiment Score:", row['sentiment_score'])
    print("------------------------------")

# Display the tail tweets with sentiment scores
print("Tail 5 Tweets:")
for index, row in tail_tweets.iterrows():
    print("Tweet:", row['Tweet'])
    print("Sentiment Score:", row['sentiment_score'])
    print("------------------------------")

"""to group tweets by date and calculate the average sentiment score for each day. It first defines a new DataFrame called "daily_sentiment" by grouping the original DataFrame "df" by the "Date" column and calculating the mean sentiment score for each group using the "mean()" method."""

# Group the tweets by date and calculate the average sentiment score for each day
daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()

# Print the resulting DataFrame
df.head()

"""Exports the daily sentiment scores to a CSV file named . The resulting CSV file will be saved in the current working directory."""

daily_sentiment.to_csv("millions_tweets_sentiment_score .csv")

print(df['sentiment_score'].mean())

""" **Date**  column of the DataFrame to a datetime object and drop any rows with missing or invalid date values. The dates are then converted again to a datetime object to ensure consistency, and a new column 'Year' is added to the DataFrame containing the year value of each date."""

# Convert 'Date' column to datetime object
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].apply(lambda x: x.year)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.barplot(x='Year', y='sentiment_score', data=df)

print(df['sentiment_score'].max())

"""# Plot WordCloud of TSX

In this project, we scraped millions of tweets related to the **Toronto Stock Exchange (TSX)**, Canadian economy, business, and finance using the snscrape Python library. We then used the Natural Language Toolkit (NLTK) library to perform sentiment analysis on the tweets and analyzed the sentiment trends over time. Additionally, we used the WordCloud library to visualize the most commonly used words in the tweets. The insights gained from this analysis could potentially be useful for investors, analysts, and financial professionals interested in understanding the sentiment and trends surrounding the Canadian financial market.
"""

# Concatenate all tweets into a single string
text = ' '.join(df['Tweet'])

# Create WordCloud object
wordcloud = WordCloud(width=800, height=800, background_color='white',
                      min_font_size=10).generate(text)

# Plot WordCloud
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

"""# Calculate sentiment scores

"""

sentiment_scores = df['sentiment_score'].value_counts()

# Create bar chart of sentiment scores
plt.bar(sentiment_scores.index, sentiment_scores.values)

# Set title and axis labels
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment Score')
plt.ylabel('TSX Tweets ')

# Show plot
plt.show()

