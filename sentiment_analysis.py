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
        'apply', 'application', 'eligibility'
    ]
    
    verification_keywords = [
        'verify', 'verification', 'identity', 'kyc', 'document', 'passport', 
        'id card', 'selfie', 'upload', 'identity check', 'proof', 'address',
        'income', 'employment', 'facial recognition'
    ]
    
    reward_keywords = [
        'reward', 'bonus', 'cashback', 'points', 'promotion', 'offer', 'gift', 
        'referral', 'welcome bonus', 'spending reward', 'loyalty', 'discount',
        'voucher', 'coupon', 'incentive'
    ]
    
    app_keywords = [
        'app', 'crash', 'bug', 'loading', 'slow', 'interface', 'ui', 'ux', 
        'design', 'feature', 'update', 'version', 'performance', 'lag',
        'freeze', 'error', 'glitch', 'navigation', 'usability'
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

def analyze_sentiment(text):
    """Enhanced sentiment analysis using TextBlob"""
    if pd.isna(text) or text == "" or text is None:
        return {
            'sentiment_score': 0,
            'sentiment_category': 'neutral',
            'sentiment_words': [],
            'positive_words': [],
            'negative_words': []
        }
    
    blob = TextBlob(str(text))
    score = blob.sentiment.polarity
    
    # Categorize sentiment with more nuanced thresholds
    if score > 0.2:
        category = 'positive'
    elif score < -0.2:
        category = 'negative'
    else:
        category = 'neutral'
    
    # Enhanced sentiment word lists
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'easy', 
        'fast', 'convenient', 'helpful', 'smooth', 'reliable', 'secure',
        'efficient', 'user-friendly', 'fantastic', 'awesome', 'satisfied'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'hate', 'slow', 'difficult', 'problem', 
        'issue', 'error', 'bug', 'crash', 'frustrating', 'disappointing',
        'useless', 'horrible', 'worst', 'annoying', 'confusing', 'broken'
    ]
    
    words = str(text).lower().split()
    found_positive = []
    found_negative = []
    
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in positive_words:
            found_positive.append(word_clean)
        elif word_clean in negative_words:
            found_negative.append(word_clean)
    
    return {
        'sentiment_score': round(score, 3),
        'sentiment_category': category,
        'sentiment_words': found_positive + found_negative,
        'positive_words': found_positive,
        'negative_words': found_negative
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

def process_reviews_file(file_path, bank_name):
    """Process a single review file"""
    print(f"\n🏦 Processing {bank_name} reviews...")
    
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
            
            # Sentiment analysis
            sentiment = analyze_sentiment(analysis_text)
            df.at[idx, 'sentiment_score'] = sentiment['sentiment_score']
            df.at[idx, 'sentiment_category'] = sentiment['sentiment_category']
            df.at[idx, 'sentiment_words'] = ', '.join(sentiment['sentiment_words'])
            df.at[idx, 'positive_words'] = ', '.join(sentiment['positive_words'])
            df.at[idx, 'negative_words'] = ', '.join(sentiment['negative_words'])
            
        except Exception as e:
            print(f"  Error processing review {idx}: {e}")
            continue
    
    # Save processed file
    output_path = f"analysis/{bank_name}_sentiment_analysis.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Saved processed reviews to {output_path}")
    
    return df

def generate_summary_report(df, bank_name):
    """Generate summary statistics and insights"""
    print(f"\n📊 Generating summary report for {bank_name}...")
    
    summary = {
        'bank_name': bank_name,
        'total_reviews': len(df),
        'average_rating': df['rating'].mean(),
        'rating_distribution': df['rating'].value_counts().to_dict(),
        'sentiment_distribution': df['sentiment_category'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'top_positive_words': [],
        'top_negative_words': [],
        'most_common_issues': []
    }
    
    # Extract top sentiment words
    all_positive = []
    all_negative = []
    
    for _, row in df.iterrows():
        if row['positive_words']:
            all_positive.extend(row['positive_words'].split(', '))
        if row['negative_words']:
            all_negative.extend(row['negative_words'].split(', '))
    
    if all_positive:
        positive_counter = Counter(all_positive)
        summary['top_positive_words'] = positive_counter.most_common(10)
    
    if all_negative:
        negative_counter = Counter(all_negative)
        summary['top_negative_words'] = negative_counter.most_common(10)
    
    # Most common issues by category
    for category in df['category'].unique():
        if category != 'general':
            category_reviews = df[df['category'] == category]
            if len(category_reviews) > 0:
                avg_sentiment = category_reviews['sentiment_score'].mean()
                summary['most_common_issues'].append({
                    'category': category,
                    'count': len(category_reviews),
                    'avg_sentiment': round(avg_sentiment, 3),
                    'avg_rating': round(category_reviews['rating'].mean(), 2)
                })
    
    # Save summary as JSON
    summary_path = f"analysis/{bank_name}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved summary report to {summary_path}")
    return summary

def create_visualizations(df, bank_name):
    """Create visualizations for the analysis"""
    print(f"\n📈 Creating visualizations for {bank_name}...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{bank_name} Review Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Sentiment Distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Sentiment Distribution')
    
    # 2. Category Distribution
    category_counts = df['category'].value_counts()
    axes[0, 1].bar(category_counts.index, category_counts.values)
    axes[0, 1].set_title('Issue Categories')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Rating vs Sentiment
    rating_sentiment = df.groupby('rating')['sentiment_score'].mean()
    axes[1, 0].plot(rating_sentiment.index, rating_sentiment.values, marker='o')
    axes[1, 0].set_title('Average Sentiment by Rating')
    axes[1, 0].set_xlabel('Rating')
    axes[1, 0].set_ylabel('Sentiment Score')
    
    # 4. Category vs Sentiment
    cat_sentiment = df.groupby('category')['sentiment_score'].mean().sort_values()
    axes[1, 1].barh(cat_sentiment.index, cat_sentiment.values)
    axes[1, 1].set_title('Average Sentiment by Category')
    axes[1, 1].set_xlabel('Sentiment Score')
    
    plt.tight_layout()
    viz_path = f"analysis/{bank_name}_analysis_dashboard.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved dashboard to {viz_path}")

def create_wordcloud(df, bank_name):
    """Create word cloud from translated reviews"""
    print(f"\n☁️ Creating word cloud for {bank_name}...")
    
    # Combine all translated content
    all_text = " ".join(df['translated_content'].dropna().astype(str))
    
    if not all_text.strip():
        print("  ⚠️ No text available for word cloud")
        return
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    
    # Save word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{bank_name} Review Word Cloud', fontsize=16, fontweight='bold')
    
    wc_path = f"analysis/{bank_name}_wordcloud.png"
    plt.savefig(wc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved word cloud to {wc_path}")

def main():
    """Main analysis function"""
    print("🚀 Starting comprehensive sentiment analysis...")
    
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
    
    all_summaries = []
    
    # Process each file
    for config in files_config:
        if os.path.exists(config['path']):
            # Process reviews
            df = process_reviews_file(config['path'], config['name'])
            
            # Generate summary
            summary = generate_summary_report(df, config['name'])
            all_summaries.append(summary)
            
            # Create visualizations
            create_visualizations(df, config['name'])
            
            # Create word cloud
            create_wordcloud(df, config['name'])
            
        else:
            print(f"⚠️ File not found: {config['path']}")
    
    # Create combined summary
    if all_summaries:
        combined_summary_path = "analysis/combined_analysis_summary.json"
        with open(combined_summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved combined summary to {combined_summary_path}")
    
    print("\n🎉 Analysis complete! Check the 'analysis' folder for all results.")

if __name__ == "__main__":
    main() 