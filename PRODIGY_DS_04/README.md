# Task 04: Sentiment Analysis of Twitter Data

**Description**: Analyzed sentiment patterns in the Twitter Entity Sentiment Analysis dataset to understand public opinions towards specific entities (topics or brands). The task involved data preprocessing, sentiment analysis, and visualization of patterns.

* *Dataset**: Twitter Entity Sentiment Analysis from Kaggle (https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). Contains \~74K training tweets and 1K validation tweets with sentiment labels (Positive, Negative, Neutral, Irrelevant).

**Files**:

- `sentiment_analysis.py`: Python script for data loading, text preprocessing, analysis, and visualizations.
- `twitter_training.csv`: Training dataset.
- `twitter_validation.csv`: Validation dataset.
- `sentiment_distribution.png`: Bar plot of overall sentiment distribution.
- `entity_sentiment.png`: Stacked bar plot of sentiment for top 5 entities.
- `positive_wordcloud.png`: Word cloud for positive sentiment tweets.

**Key Steps Performed**:

1. **Data Loading & Inspection**: Loaded `twitter_training.csv` and `twitter_validation.csv`, checked data structure and sentiment distribution.
2. **Text Preprocessing**: Cleaned tweet text by removing URLs, mentions, hashtags, and punctuation, and converting to lowercase.
3. **Analysis**: Calculated sentiment distribution by entity and identified top words in tweets.
4. **Visualizations**:
   - Bar plot of overall sentiment distribution.
   - Stacked bar plot for sentiment of top 5 entities.
   - Word cloud for positive sentiment tweets.

**Output**:

- **Sentiment Distribution**:
  - Negative: 22,542
  - Positive: 20,832
  - Neutral: 18,318
  - Irrelevant: 12,990
- **Top Entities Analyzed**: MaddenNFL, NBA2K, FIFA, TomClancysRainbowSix, Verizon
- **Common Words**: 'the', 'i', 'to', 'and', 'a', 'of', 'is', 'for', 'in', 'this'

**Next Steps**:

- Explore sentiment trends over time (if timestamps are available).

Apply NLP techniques like TF-IDF or sentiment intensity scoring for deeper insights.