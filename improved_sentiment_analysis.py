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
import time

# Initialize translator
translator = Translator()

def safe_translate(text, max_retries=3, delay=1):
    """Safely translate text with retries and rate limiting"""
    if pd.isna(text) or text == "" or text is None:
        return ""
    
    # If text is already in English, return as is
    text_str = str(text)
    if text_str.isascii() and len([c for c in text_str if c.isalpha()]) > len(text_str) * 0.7:
        return text_str
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)  # Rate limiting
            result = translator.translate(text_str[:4000], dest='en')  # Limit length
            return result.text
        except Exception as e:
            print(f"Translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * 2)  # Exponential backoff
    
    return text_str  # Return original if all attempts fail

def categorize_review(review_text, title_text=""):
    """Categorize review into issue types based on content and title"""
    if pd.isna(review_text):
        review_text = ""
    if pd.isna(title_text):
        title_text = ""
    
    text_lower = (str(review_text) + " " + str(title_text)).lower()
    
    # Enhanced keywords for different categories
    account_opening_keywords = [
        'open account', 'register', 'signup', 'sign up', 'create account', 
        'registration', 'onboarding', 'new user', 'account creation',
        'apply', 'application', 'eligibility', 'rejected application'
    ]
    
    verification_keywords = [
        'verify', 'verification', 'identity', 'kyc', 'document', 'passport', 
        'id card', 'selfie', 'upload', 'identity check', 'proof', 'address',
        'income', 'employment', 'facial recognition', 'face id', 'biometric'
    ]
    
    reward_keywords = [
        'reward', 'bonus', 'cashback', 'points', 'promotion', 'offer', 'gift', 
        'referral', 'welcome bonus', 'spending reward', 'loyalty', 'discount',
        'voucher', 'coupon', 'incentive'
    ]
    
    app_keywords = [
        'app', 'crash', 'bug', 'loading', 'slow', 'interface', 'ui', 'ux', 
        'design', 'feature', 'update', 'version', 'performance', 'lag',
        'freeze', 'error', 'glitch', 'navigation', 'usability', 'login', 'log in'
    ]
    
    # Check categories with scoring
    scores = {
        'account_opening': sum(1 for keyword in account_opening_keywords if keyword in text_lower),
        'account_verification': sum(1 for keyword in verification_keywords if keyword in text_lower),
        'reward_issues': sum(1 for keyword in reward_keywords if keyword in text_lower),
        'app_issues': sum(1 for keyword in app_keywords if keyword in text_lower)
    }
    
    if max(scores.values()) == 0:
        return "general"
    
    return max(scores, key=scores.get)

def analyze_sentiment_improved(text, rating=None):
    """Improved sentiment analysis specifically for banking app reviews"""
    if pd.isna(text) or text == "" or text is None:
        return {
            'sentiment_score': 0,
            'sentiment_category': 'neutral',
            'sentiment_words': [],
            'positive_words': [],
            'negative_words': [],
            'rating_adjusted': False
        }
    
    text_str = str(text).lower()
    
    # Banking-specific positive indicators
    positive_indicators = [
        'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'easy', 
        'fast', 'convenient', 'helpful', 'smooth', 'reliable', 'secure',
        'efficient', 'user-friendly', 'fantastic', 'awesome', 'satisfied',
        'recommend', 'useful', 'works well', 'seamless', 'quick', 'simple'
    ]
    
    # Banking-specific negative indicators (enhanced)
    negative_indicators = [
        'bad', 'terrible', 'awful', 'hate', 'slow', 'difficult', 'problem', 
        'issue', 'error', 'bug', 'crash', 'frustrating', 'disappointing',
        'useless', 'horrible', 'worst', 'annoying', 'confusing', 'broken',
        'scam', 'fraud', 'locked', 'frozen', 'blocked', 'suspended',
        'cant', "can't", 'cannot', 'unable', 'fail', 'failed', 'failing',
        'stuck', 'glitch', 'freeze', 'hang', 'not working', 'doesnt work',
        "doesn't work", 'waste', 'money stuck', 'money locked', 'lost money',
        'terrible experience', 'poor service', 'customer service',
        'rejected', 'denied', 'declined', 'timeout', 'session expired'
    ]
    
    # Count sentiment indicators
    positive_count = sum(1 for indicator in positive_indicators if indicator in text_str)
    negative_count = sum(1 for indicator in negative_indicators if indicator in text_str)
    
    # TextBlob baseline score
    blob = TextBlob(str(text))
    textblob_score = blob.sentiment.polarity
    
    # Calculate adjusted sentiment score
    indicator_score = (positive_count - negative_count) * 0.3
    combined_score = (textblob_score * 0.7) + indicator_score
    
    # Rating-based adjustment (strong signal for banking apps)
    rating_adjusted = False
    if rating is not None:
        if rating <= 2:
            # 1-2 star reviews should lean negative unless clearly positive
            if combined_score > -0.3 and negative_count > positive_count:
                combined_score = min(combined_score - 0.4, -0.3)
                rating_adjusted = True
            elif combined_score > 0 and positive_count == 0:
                combined_score = min(combined_score - 0.2, -0.1)
                rating_adjusted = True
        elif rating >= 4:
            # 4-5 star reviews should lean positive unless clearly negative
            if combined_score < 0.3 and positive_count > negative_count:
                combined_score = max(combined_score + 0.3, 0.3)
                rating_adjusted = True
    
    # More aggressive sentiment categorization
    if combined_score > 0.15:
        category = 'positive'
    elif combined_score < -0.15:
        category = 'negative'
    else:
        category = 'neutral'
    
    # Extract found sentiment words
    found_positive = [word for word in positive_indicators if word in text_str]
    found_negative = [word for word in negative_indicators if word in text_str]
    
    return {
        'sentiment_score': round(combined_score, 3),
        'sentiment_category': category,
        'sentiment_words': found_positive + found_negative,
        'positive_words': found_positive,
        'negative_words': found_negative,
        'rating_adjusted': rating_adjusted
    }

def create_problem_summary(review_text, title_text="", max_length=150):
    """Create a concise problem summary from review content"""
    if pd.isna(review_text):
        review_text = ""
    if pd.isna(title_text):
        title_text = ""
    
    # Combine title and content
    full_text = f"{title_text}. {review_text}" if title_text else review_text
    text = str(full_text).strip()
    
    if not text:
        return ""
    
    # Simple summarization - take first meaningful sentence
    sentences = re.split(r'[.!?。！？]', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and any(c.isalpha() for c in sentence):
            summary = sentence[:max_length]
            return summary.strip()
    
    # Fallback to truncated text
    return text[:max_length].strip()

def process_reviews_file_improved(file_path, bank_name):
    """Process a single review file with improved sentiment analysis"""
    print(f"\n🏦 Processing {bank_name} reviews (IMPROVED METHOD)...")
    
    # Read the file
    df = pd.read_excel(file_path)
    print(f"Found {len(df)} reviews")
    
    # Initialize new columns
    df['translated_content'] = ""
    df['translated_title'] = ""
    df['category'] = ""
    df['problem_summary'] = ""
    df['sentiment_score'] = 0.0
    df['sentiment_category'] = ""
    df['sentiment_words'] = ""
    df['positive_words'] = ""
    df['negative_words'] = ""
    df['rating_adjusted'] = False
    
    # Process each review
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processing review {idx + 1}/{len(df)}...")
        
        # Translate content and title
        try:
            translated_content = safe_translate(row['content'])
            translated_title = safe_translate(row['title'])
            
            df.at[idx, 'translated_content'] = translated_content
            df.at[idx, 'translated_title'] = translated_title
            
            # Use translated text for analysis
            analysis_text = f"{translated_title} {translated_content}".strip()
            
            # Categorize
            category = categorize_review(analysis_text, translated_title)
            df.at[idx, 'category'] = category
            
            # Create summary
            summary = create_problem_summary(translated_content, translated_title)
            df.at[idx, 'problem_summary'] = summary
            
            # IMPROVED Sentiment analysis (using rating as signal)
            sentiment = analyze_sentiment_improved(analysis_text, row['rating'])
            df.at[idx, 'sentiment_score'] = sentiment['sentiment_score']
            df.at[idx, 'sentiment_category'] = sentiment['sentiment_category']
            df.at[idx, 'sentiment_words'] = ', '.join(sentiment['sentiment_words'])
            df.at[idx, 'positive_words'] = ', '.join(sentiment['positive_words'])
            df.at[idx, 'negative_words'] = ', '.join(sentiment['negative_words'])
            df.at[idx, 'rating_adjusted'] = sentiment['rating_adjusted']
            
        except Exception as e:
            print(f"  Error processing review {idx}: {e}")
            continue
    
    # Save processed file
    output_path = f"analysis/{bank_name}_sentiment_analysis_IMPROVED.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Saved improved analysis to {output_path}")
    
    return df

def generate_comparison_report(original_df, improved_df, bank_name):
    """Generate a comparison report between original and improved analysis"""
    print(f"\n📊 Generating comparison report for {bank_name}...")
    
    # Compare sentiment distributions
    original_sentiment = original_df['sentiment_category'].value_counts()
    improved_sentiment = improved_df['sentiment_category'].value_counts()
    
    comparison = {
        'bank_name': bank_name,
        'total_reviews': len(improved_df),
        'average_rating': improved_df['rating'].mean(),
        'original_sentiment': original_sentiment.to_dict(),
        'improved_sentiment': improved_sentiment.to_dict(),
        'rating_adjustments': improved_df['rating_adjusted'].sum(),
        'sentiment_changes': {},
        'low_rating_sentiment_fix': {}
    }
    
    # Check how many low-rating reviews are now correctly classified
    low_rating_reviews = improved_df[improved_df['rating'] <= 2]
    if len(low_rating_reviews) > 0:
        low_rating_sentiment = low_rating_reviews['sentiment_category'].value_counts()
        comparison['low_rating_sentiment_fix'] = {
            'total_1_2_star_reviews': len(low_rating_reviews),
            'negative_sentiment': low_rating_sentiment.get('negative', 0),
            'neutral_sentiment': low_rating_sentiment.get('neutral', 0),
            'positive_sentiment': low_rating_sentiment.get('positive', 0),
            'percentage_negative': round(low_rating_sentiment.get('negative', 0) / len(low_rating_reviews) * 100, 1)
        }
    
    # Save comparison report
    comparison_path = f"analysis/{bank_name}_sentiment_comparison.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved comparison report to {comparison_path}")
    return comparison

def main_improved():
    """Main function for improved analysis"""
    print("🚀 Starting IMPROVED sentiment analysis...")
    
    # Ensure analysis directory exists
    os.makedirs("analysis", exist_ok=True)
    
    # File configurations
    files_config = [
        {
            'path': 'output/welab_bank_combined_20250730_161901.xlsx',
            'name': 'WeLab_Bank'
        },
        {
            'path': 'output/za_bank_combined_20250730_160113.xlsx',
            'name': 'ZA_Bank'
        },
        {
            'path': 'output/mox_bank_combined_20250730_155023.xlsx',
            'name': 'Mox_Bank'
        }
    ]
    
    all_comparisons = []
    
    # Process each file with improved method
    for config in files_config:
        if os.path.exists(config['path']):
            # Process with improved method
            improved_df = process_reviews_file_improved(config['path'], config['name'])
            
            # Load original analysis for comparison if it exists
            original_path = f"analysis/{config['name']}_sentiment_analysis.xlsx"
            if os.path.exists(original_path):
                original_df = pd.read_excel(original_path)
                comparison = generate_comparison_report(original_df, improved_df, config['name'])
                all_comparisons.append(comparison)
            
        else:
            print(f"⚠️ File not found: {config['path']}")
    
    print("\n🎉 Improved analysis complete!")
    
    # Print summary of improvements
    if all_comparisons:
        print("\n📈 SENTIMENT ANALYSIS IMPROVEMENTS:")
        for comp in all_comparisons:
            print(f"\n{comp['bank_name']}:")
            print(f"  Rating adjustments made: {comp['rating_adjustments']}")
            if comp['low_rating_sentiment_fix']:
                fix_data = comp['low_rating_sentiment_fix']
                print(f"  1-2 star reviews: {fix_data['total_1_2_star_reviews']}")
                print(f"  Now classified as negative: {fix_data['negative_sentiment']} ({fix_data['percentage_negative']}%)")

if __name__ == "__main__":
    main_improved() 