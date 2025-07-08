import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Set Seaborn styling
sns.set_style("whitegrid")  # Use Seaborn's whitegrid style

# Load the dataset with relative paths
train_df = pd.read_csv('twitter_training.csv', 
                       names=['Tweet_ID', 'entity', 'sentiment', 'Tweet_content'])
val_df = pd.read_csv('twitter_validation.csv', 
                     names=['Tweet_ID', 'entity', 'sentiment', 'Tweet_content'])

# Basic data inspection
print("Training Data Info:")
print(train_df.info())
print("\nSentiment Distribution:")
print(train_df['sentiment'].value_counts())

# Clean tweet text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.lower().strip()

train_df['cleaned_text'] = train_df['Tweet_content'].apply(clean_text)

# Sentiment distribution by entity
entity_sentiment = train_df.groupby(['entity', 'sentiment']).size().unstack(fill_value=0)
print("\nSentiment Counts by Entity:")
print(entity_sentiment)

# Visualization 1: Bar plot of sentiment distribution
plt.figure(figsize=(10, 6))
train_df['sentiment'].value_counts().plot(kind='bar', color=['gray', 'red', 'green'])
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.close()

# Visualization 2: Stacked bar plot for top 5 entities
top_entities = train_df['entity'].value_counts().head(5).index
entity_subset = entity_sentiment.loc[top_entities]
entity_subset.plot(kind='bar', stacked=True, figsize=(12, 6), color=['gray', 'red', 'green'])
plt.title('Sentiment Distribution for Top 5 Entities')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.savefig('entity_sentiment.png')
plt.close()

# Visualization 3: Word cloud for positive sentiment
positive_tweets = ' '.join(train_df[train_df['sentiment'] == 'Positive']['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Sentiment Tweets')
plt.savefig('positive_wordcloud.png')
plt.close()

# Most common words
all_words = ' '.join(train_df['cleaned_text']).split()
word_freq = Counter(all_words).most_common(10)
print("\nTop 10 Most Common Words:")
print(word_freq)