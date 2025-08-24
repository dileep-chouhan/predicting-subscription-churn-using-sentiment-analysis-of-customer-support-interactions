import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_interactions = 200
data = {
    'customer_id': np.arange(num_interactions),
    'interaction_text': [f"Interaction {i}: This is a sample interaction text. "
                         f"{'Positive' if np.random.rand() > 0.5 else 'Negative'} sentiment."
                         for i in range(num_interactions)],
    'churned': np.random.choice([True, False], size=num_interactions, p=[0.2, 0.8]) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
df['sentiment_polarity'] = df['interaction_text'].apply(analyze_sentiment)
# --- 3. Data Cleaning (if needed) ---
# In a real-world scenario, this section would include handling missing values, 
# outliers, and other data quality issues.  For this synthetic data, it's omitted.
# --- 4. Analysis ---
# Group by churn status and calculate average sentiment
average_sentiment_by_churn = df.groupby('churned')['sentiment_polarity'].mean()
print("Average Sentiment by Churn Status:")
print(average_sentiment_by_churn)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.barplot(x=average_sentiment_by_churn.index.astype(str), y=average_sentiment_by_churn.values)
plt.title('Average Sentiment Polarity by Churn Status')
plt.xlabel('Churned')
plt.ylabel('Average Sentiment Polarity')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'sentiment_churn.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve correlation analysis between sentiment and churn, 
#building a predictive model (e.g., logistic regression), etc.  This example 
#focuses on the core aspects of sentiment analysis and visualization.