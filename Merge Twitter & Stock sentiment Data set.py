"""# merge the twitter and stock on the date

reads in two dataframes (df_twitter and df_stock) from CSV files, converts their 'Date' column to datetime format, and then merges the two dataframes on the 'Date' column to create a new dataframe df_merged.
"""

# read in your dataframes
df_twitter = pd.read_csv('/content/millions_tweets_sentiments .csv')
df_stock = pd.read_csv('/content/Toronto_stock_data .csv')

# convert date column to datetime format for both dataframes
df_twitter['Date'] = pd.to_datetime(df_twitter['Date'])
df_stock['Date'] = pd.to_datetime(df_stock['Date'])

# merge the two dataframes on the date column
df_merged = pd.merge(df_twitter, df_stock, on='Date')

df_merged.head()

df_merged.tail()

# This will remove the Unnamed: 0 column from both dataframes.
df_merged = df_merged.drop(['Unnamed: 0'], axis=1)

3 show the merged data frame
df_merged.head()

# check for null values
null_values = df_merged.isnull().sum()
print(null_values)

# Save the combined data frame
df_merged.to_csv('combine_twitter_tsx. csv')

