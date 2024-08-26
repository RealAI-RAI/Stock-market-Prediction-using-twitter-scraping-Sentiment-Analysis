import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(df):
    # Analyze sentiment for each tweet
    df['sentiment_score'] = None
    df['Negative'] = None
    df['Neutral'] = None
    df['Positive'] = None
    
    for indx, row in df.iterrows():
        try:
            sentence_i = unicodedata.normalize('NFKD', row['Tweet'])
            sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
            
            df.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
            df.at[indx, 'Negative'] = sentence_sentiment['neg']
            df.at[indx, 'Neutral'] = sentence_sentiment['neu']
            df.at[indx, 'Positive'] = sentence_sentiment['pos']
        except TypeError:
            print(f"Error processing tweet: {row['Tweet']}")
    
    return df

def plot_sentiment_over_time(df):
    daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='sentiment_score', data=daily_sentiment)
    plt.title('Daily Average Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.show()

def visualize_wordcloud(text):
    wordcloud = WordCloud(width=800, height=800, background_color='white',
                          min_font_size=10).generate(text)
    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def main():
    # Assuming df is already loaded with tweets
    df = analyze_sentiment(df)
    
    head_tweets = df.head(12)
    tail_tweets = df.tail(10)
    
    print("Head 5 Tweets:")
    for index, row in head_tweets.iterrows():
        print(f"Tweet: {row['Tweet']}")
        print(f"Sentiment Score: {row['sentiment_score']}")
        print("------------------------")
    
    print("\nTail 5 Tweets:")
    for index, row in tail_tweets.iterrows():
        print(f"Tweet: {row['Tweet']}")
        print(f"Sentiment Score: {row['sentiment_score']}")
        print("------------------------")
    
    # Group tweets by date and calculate average sentiment score
    daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()
    
    # Export daily sentiment scores to CSV
    daily_sentiment.to_csv("daily_sentiment.csv", index=False)
    
    print(f"\nAverage sentiment score: {df['sentiment_score'].mean()}")
    
    # Convert Date column to datetime object and drop rows with missing dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    
    # Plot sentiment over time
    plot_sentiment_over_time(df)
    
    # Calculate maximum sentiment score
    max_sentiment = df['sentiment_score'].max()
    print(f"\nMaximum sentiment score: {max_sentiment}")
    
    # Create word cloud
    text = ' '.join(df['Tweet'])
    visualize_wordcloud(text)
    
    # Plot sentiment distribution
    sentiment_scores = df['sentiment_score'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(sentiment_scores.index, sentiment_scores.values)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()
