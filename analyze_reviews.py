import pandas as pd
import numpy as np
from pathlib import Path
import os
from textblob import TextBlob
from googletrans import Translator
import re
import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Initialize translator
translator = Translator()

def examine_file_structure():
    """Examine the structure of the review files"""
    files = [
        'output/welab_bank_combined_20250730_161901.xlsx',
        'output/za_bank_combined_20250730_160113.xlsx', 
        'output/mox_bank_combined_20250730_155023.xlsx'
    ]
    
    for file in files:
        print(f"\n{'='*50}")
        print(f"Examining: {file}")
        print(f"{'='*50}")
        
        try:
            df = pd.read_excel(file)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print("\nFirst few rows:")
            print(df.head(3))
            print("\nData types:")
            print(df.dtypes)
            
            # Check for review content column
            for col in df.columns:
                if 'review' in col.lower() or 'content' in col.lower() or 'comment' in col.lower():
                    print(f"\nSample reviews from '{col}':")
                    sample_reviews = df[col].dropna().head(3)
                    for i, review in enumerate(sample_reviews):
                        print(f"{i+1}. {review[:100]}...")
                        
        except Exception as e:
            print(f"Error reading {file}: {e}")

def translate_text(text, max_length=5000):
    """Translate text with error handling"""
    if pd.isna(text) or text == "":
        return ""
    
    try:
        # Truncate if too long
        text_str = str(text)[:max_length]
        result = translator.translate(text_str, dest='en')
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return str(text)

def categorize_review(review_text):
    """Categorize review into issue types"""
    if pd.isna(review_text):
        return "unknown"
    
    text_lower = str(review_text).lower()
    
    # Keywords for different categories
    account_opening_keywords = ['open account', 'register', 'signup', 'sign up', 'create account', 'registration', 'onboarding']
    verification_keywords = ['verify', 'verification', 'identity', 'kyc', 'document', 'passport', 'id card', 'selfie']
    reward_keywords = ['reward', 'bonus', 'cashback', 'points', 'promotion', 'offer', 'gift', 'referral']
    app_keywords = ['app', 'crash', 'bug', 'loading', 'slow', 'interface', 'ui', 'ux', 'design', 'feature']
    
    # Check categories
    if any(keyword in text_lower for keyword in account_opening_keywords):
        return "account_opening"
    elif any(keyword in text_lower for keyword in verification_keywords):
        return "account_verification"
    elif any(keyword in text_lower for keyword in reward_keywords):
        return "reward_issues"
    elif any(keyword in text_lower for keyword in app_keywords):
        return "app_issues"
    else:
        return "general"

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if pd.isna(text) or text == "":
        return {
            'sentiment_score': 0,
            'sentiment_category': 'neutral',
            'sentiment_words': []
        }
    
    blob = TextBlob(str(text))
    score = blob.sentiment.polarity
    
    # Categorize sentiment
    if score > 0.1:
        category = 'positive'
    elif score < -0.1:
        category = 'negative'
    else:
        category = 'neutral'
    
    # Extract sentiment-bearing words (simple approach)
    words = str(text).lower().split()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'easy', 'fast', 'convenient']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'slow', 'difficult', 'problem', 'issue', 'error', 'bug']
    
    sentiment_words = []
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in positive_words or word_clean in negative_words:
            sentiment_words.append(word_clean)
    
    return {
        'sentiment_score': round(score, 3),
        'sentiment_category': category,
        'sentiment_words': sentiment_words
    }

def create_problem_summary(review_text, max_length=100):
    """Create a concise problem summary"""
    if pd.isna(review_text) or review_text == "":
        return ""
    
    text = str(review_text)
    # Simple summarization - take first sentence or truncate
    sentences = re.split(r'[.!?]', text)
    if sentences and len(sentences[0]) > 10:
        summary = sentences[0][:max_length]
    else:
        summary = text[:max_length]
    
    return summary.strip()

if __name__ == "__main__":
    print("Starting review analysis...")
    examine_file_structure() 