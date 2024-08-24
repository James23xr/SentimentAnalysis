import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from tqdm import tqdm
import random

# Your existing code for data loading and initial processing
df = pd.read_csv('Reviews.csv')
df = df.head(500)  # Using 500 reviews as in your original code

# Your existing RoBERTa model setup
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Modified sentiment analysis function to return both RoBERTa and VADER scores
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    # RoBERTa sentiment
    encoded_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    roberta_sentiment = ['Negative', 'Neutral', 'Positive'][np.argmax(scores)]
    
    # VADER sentiment
    vader_scores = sia.polarity_scores(text)
    
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
        'roberta_sentiment': roberta_sentiment,
        'vader_neg': vader_scores['neg'],
        'vader_neu': vader_scores['neu'],
        'vader_pos': vader_scores['pos'],
        'vader_compound': vader_scores['compound']
    }

# Simulate customer service inquiries
df['CustomerInquiry'] = df['Text'].apply(lambda x: f"Customer Inquiry: {x}")

# Process customer inquiries
def process_customer_inquiries(inquiries):
    results = []
    for inquiry in tqdm(inquiries):
        sentiment_result = get_sentiment(inquiry)
        response_time = simulate_response_time(sentiment_result['roberta_sentiment'])
        results.append({
            'inquiry': inquiry,
            **sentiment_result,
            'response_time': response_time
        })
    return pd.DataFrame(results)

def simulate_response_time(sentiment):
    if sentiment == 'Negative':
        return random.randint(1, 10)
    elif sentiment == 'Neutral':
        return random.randint(5, 20)
    else:
        return random.randint(10, 30)

# Process inquiries
inquiry_results = process_customer_inquiries(df['CustomerInquiry'])

# Simulate customer satisfaction
def simulate_satisfaction(sentiment, response_time):
    base_satisfaction = {'Positive': 8, 'Neutral': 6, 'Negative': 4}[sentiment]
    time_factor = max(0, 10 - response_time) / 10
    return min(10, base_satisfaction + time_factor * 2)

inquiry_results['satisfaction'] = inquiry_results.apply(lambda x: simulate_satisfaction(x['roberta_sentiment'], x['response_time']), axis=1)

# Simulate customer retention
def simulate_retention(satisfaction):
    retention_prob = satisfaction / 10
    return random.random() < retention_prob

inquiry_results['retained'] = inquiry_results['satisfaction'].apply(simulate_retention)

# Calculate metrics
avg_response_time = inquiry_results['response_time'].mean()
avg_satisfaction = inquiry_results['satisfaction'].mean()
retention_rate = inquiry_results['retained'].mean()

print(f"Average Response Time: {avg_response_time:.2f} minutes")
print(f"Average Satisfaction Score: {avg_satisfaction:.2f} / 10")
print(f"Customer Retention Rate: {retention_rate:.2%}")

# Visualize results (using your existing plotting code)
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Original plots
sns.barplot(data=inquiry_results, x='roberta_sentiment', y='roberta_pos', ax=axs[0, 0])
sns.barplot(data=inquiry_results, x='roberta_sentiment', y='roberta_neu', ax=axs[0, 1])
sns.barplot(data=inquiry_results, x='roberta_sentiment', y='roberta_neg', ax=axs[0, 2])
axs[0, 0].set_title('RoBERTa Positive')
axs[0, 1].set_title('RoBERTa Neutral')
axs[0, 2].set_title('RoBERTa Negative')

# New plots
sns.histplot(inquiry_results['response_time'], kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Response Time Distribution')
sns.histplot(inquiry_results['satisfaction'], kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Satisfaction Score Distribution')
sns.countplot(x='retained', data=inquiry_results, ax=axs[1, 2])
axs[1, 2].set_title('Customer Retention')

plt.tight_layout()
plt.show()

# Simulate improvement
def improve_response_time(time):
    return max(1, time * 0.6)  # 40% improvement, minimum 1 minute

inquiry_results['improved_response_time'] = inquiry_results['response_time'].apply(improve_response_time)
inquiry_results['improved_satisfaction'] = inquiry_results.apply(lambda x: simulate_satisfaction(x['roberta_sentiment'], x['improved_response_time']), axis=1)
inquiry_results['improved_retained'] = inquiry_results['improved_satisfaction'].apply(simulate_retention)

# Calculate improved metrics
improved_avg_response_time = inquiry_results['improved_response_time'].mean()
improved_avg_satisfaction = inquiry_results['improved_satisfaction'].mean()
improved_retention_rate = inquiry_results['improved_retained'].mean()

print("\nAfter Improvement:")
print(f"Average Response Time: {improved_avg_response_time:.2f} minutes")
print(f"Average Satisfaction Score: {improved_avg_satisfaction:.2f} / 10")
print(f"Customer Retention Rate: {improved_retention_rate:.2%}")

# Calculate improvement percentages
response_time_improvement = (avg_response_time - improved_avg_response_time) / avg_response_time * 100
satisfaction_improvement = (improved_avg_satisfaction - avg_satisfaction) / avg_satisfaction * 100
retention_improvement = (improved_retention_rate - retention_rate) / retention_rate * 100

print(f"\nResponse Time Improvement: {response_time_improvement:.2f}%")
print(f"Satisfaction Score Improvement: {satisfaction_improvement:.2f}%")
print(f"Customer Retention Improvement: {retention_improvement:.2f}%")

# Additional analysis: Compare RoBERTa and VADER
plt.figure(figsize=(10, 6))
plt.scatter(inquiry_results['vader_compound'], inquiry_results['roberta_pos'], alpha=0.5)
plt.xlabel('VADER Compound Score')
plt.ylabel('RoBERTa Positive Score')
plt.title('VADER vs RoBERTa Sentiment Scores')
plt.show()

# Print some example disagreements between VADER and RoBERTa
disagreements = inquiry_results[
    (inquiry_results['vader_compound'] > 0.5) & (inquiry_results['roberta_sentiment'] == 'Negative') |
    (inquiry_results['vader_compound'] < -0.5) & (inquiry_results['roberta_sentiment'] == 'Positive')
]
print("\nExample disagreements between VADER and RoBERTa:")
for _, row in disagreements.head().iterrows():
    print(f"Text: {row['inquiry']}")
    print(f"VADER Compound: {row['vader_compound']:.2f}")
    print(f"RoBERTa Sentiment: {row['roberta_sentiment']}")
    print(f"RoBERTa Positive Score: {row['roberta_pos']:.2f}")
    print("---")