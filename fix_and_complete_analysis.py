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

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def generate_final_summary_report(df, bank_name):
    """Generate comprehensive summary report"""
    print(f"\n📊 Generating final summary for {bank_name}...")
    
    summary = {
        'bank_name': bank_name,
        'total_reviews': int(len(df)),
        'average_rating': float(df['rating'].mean()),
        'rating_distribution': df['rating'].value_counts().to_dict(),
        'sentiment_distribution': df['sentiment_category'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'rating_adjustments_made': int(df['rating_adjusted'].sum()),
        'low_rating_analysis': {},
        'high_rating_analysis': {},
        'top_issues_by_category': []
    }
    
    # Analyze low ratings (1-2 stars)
    low_rating = df[df['rating'] <= 2]
    if len(low_rating) > 0:
        low_sentiment = low_rating['sentiment_category'].value_counts()
        summary['low_rating_analysis'] = {
            'total_reviews': int(len(low_rating)),
            'sentiment_distribution': low_sentiment.to_dict(),
            'percentage_negative': float(low_sentiment.get('negative', 0) / len(low_rating) * 100)
        }
    
    # Analyze high ratings (4-5 stars)
    high_rating = df[df['rating'] >= 4]
    if len(high_rating) > 0:
        high_sentiment = high_rating['sentiment_category'].value_counts()
        summary['high_rating_analysis'] = {
            'total_reviews': int(len(high_rating)),
            'sentiment_distribution': high_sentiment.to_dict(),
            'percentage_positive': float(high_sentiment.get('positive', 0) / len(high_rating) * 100)
        }
    
    # Top issues by category
    for category in df['category'].unique():
        if category != 'general':
            category_reviews = df[df['category'] == category]
            if len(category_reviews) > 0:
                summary['top_issues_by_category'].append({
                    'category': category,
                    'count': int(len(category_reviews)),
                    'avg_sentiment': float(category_reviews['sentiment_score'].mean()),
                    'avg_rating': float(category_reviews['rating'].mean()),
                    'negative_percentage': float((category_reviews['sentiment_category'] == 'negative').sum() / len(category_reviews) * 100)
                })
    
    # Convert all numpy types to JSON serializable
    summary = convert_to_json_serializable(summary)
    
    # Save summary
    summary_path = f"analysis/{bank_name}_FINAL_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved final summary to {summary_path}")
    return summary

def create_improved_visualizations(df, bank_name):
    """Create improved visualizations"""
    print(f"\n📈 Creating improved visualizations for {bank_name}...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{bank_name} - IMPROVED Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Sentiment Distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Overall Sentiment Distribution')
    
    # 2. Category Distribution
    category_counts = df['category'].value_counts()
    axes[0, 1].bar(category_counts.index, category_counts.values, color='#95a5a6')
    axes[0, 1].set_title('Issue Categories')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Rating vs Sentiment Distribution
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment_category'], normalize='index') * 100
    rating_sentiment.plot(kind='bar', stacked=True, ax=axes[0, 2], color=colors)
    axes[0, 2].set_title('Sentiment % by Rating')
    axes[0, 2].set_xlabel('Rating')
    axes[0, 2].legend(title='Sentiment')
    
    # 4. Category vs Average Sentiment
    cat_sentiment = df.groupby('category')['sentiment_score'].mean().sort_values()
    axes[1, 0].barh(cat_sentiment.index, cat_sentiment.values, color='#e74c3c')
    axes[1, 0].set_title('Average Sentiment by Category')
    axes[1, 0].set_xlabel('Sentiment Score')
    
    # 5. Rating Distribution
    rating_counts = df['rating'].value_counts().sort_index()
    axes[1, 1].bar(rating_counts.index, rating_counts.values, color='#f39c12')
    axes[1, 1].set_title('Rating Distribution')
    axes[1, 1].set_xlabel('Rating')
    axes[1, 1].set_ylabel('Count')
    
    # 6. Sentiment Score Distribution
    axes[1, 2].hist(df['sentiment_score'], bins=30, alpha=0.7, color='#9b59b6')
    axes[1, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('Sentiment Score Distribution')
    axes[1, 2].set_xlabel('Sentiment Score')
    
    plt.tight_layout()
    viz_path = f"analysis/{bank_name}_IMPROVED_dashboard.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved improved dashboard to {viz_path}")

def create_wordcloud_improved(df, bank_name):
    """Create improved word cloud"""
    print(f"\n☁️ Creating improved word cloud for {bank_name}...")
    
    # Separate positive and negative reviews
    negative_text = " ".join(df[df['sentiment_category'] == 'negative']['translated_content'].dropna().astype(str))
    positive_text = " ".join(df[df['sentiment_category'] == 'positive']['translated_content'].dropna().astype(str))
    
    if negative_text.strip() and positive_text.strip():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Negative word cloud
        negative_wc = WordCloud(width=800, height=400, background_color='white', 
                               colormap='Reds', max_words=100).generate(negative_text)
        ax1.imshow(negative_wc, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('Negative Reviews Word Cloud', fontsize=14, fontweight='bold')
        
        # Positive word cloud
        positive_wc = WordCloud(width=800, height=400, background_color='white', 
                               colormap='Greens', max_words=100).generate(positive_text)
        ax2.imshow(positive_wc, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('Positive Reviews Word Cloud', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'{bank_name} - Sentiment-based Word Clouds', fontsize=16, fontweight='bold')
        
        wc_path = f"analysis/{bank_name}_IMPROVED_wordclouds.png"
        plt.savefig(wc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved improved word clouds to {wc_path}")

def main_complete():
    """Complete the improved analysis for all banks"""
    print("🚀 Starting COMPLETE IMPROVED sentiment analysis...")
    
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
            # Check if already processed
            improved_path = f"analysis/{config['name']}_sentiment_analysis_IMPROVED.xlsx"
            
            if os.path.exists(improved_path):
                print(f"\n✅ {config['name']} already processed, loading existing results...")
                df = pd.read_excel(improved_path)
            else:
                # Process with improved method
                df = process_reviews_file_improved(config['path'], config['name'])
            
            # Generate final summary and visualizations
            summary = generate_final_summary_report(df, config['name'])
            all_summaries.append(summary)
            
            create_improved_visualizations(df, config['name'])
            create_wordcloud_improved(df, config['name'])
            
        else:
            print(f"⚠️ File not found: {config['path']}")
    
    # Create overall comparison report
    if all_summaries:
        overall_path = "analysis/FINAL_COMPARISON_REPORT.json"
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved overall comparison to {overall_path}")
        
        # Print summary
        print("\n🎉 COMPLETE ANALYSIS FINISHED!")
        print("\n📊 SUMMARY OF IMPROVEMENTS:")
        for summary in all_summaries:
            print(f"\n{summary['bank_name']}:")
            print(f"  Total reviews: {summary['total_reviews']}")
            print(f"  Average rating: {summary['average_rating']:.2f}")
            print(f"  Sentiment: {summary['sentiment_distribution']}")
            if summary['low_rating_analysis']:
                low_data = summary['low_rating_analysis']
                print(f"  1-2⭐ reviews: {low_data['total_reviews']} ({low_data['percentage_negative']:.1f}% negative)")

if __name__ == "__main__":
    main_complete() 