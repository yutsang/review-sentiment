#!/usr/bin/env python3
"""
Analyze 3 specific categories for Mox Bank, ZA Bank, and WeLab Bank
"""

import pandas as pd

def analyze_three_categories():
    """Analyze 3 specific categories for the three banks"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Define the 3 categories we want to analyze
    categories = {
        'onboarding_verification': 'Onboarding & Verification',
        'platform_performance': 'Platform Performance & Reliability', 
        'accounts_interest': 'Accounts & Interest Terms'
    }
    
    banks = ['Mox Bank', 'ZA Bank', 'WeLab Bank']
    
    print("📊 ANALYSIS OF 3 CATEGORIES")
    print("=" * 80)
    
    for bank in banks:
        bank_df = df[df['bank'] == bank]
        
        print(f"\n🏦 {bank}:")
        print(f"   Total reviews: {len(bank_df)}")
        
        for category_key, category_name in categories.items():
            # Filter reviews for this category
            cat_reviews = bank_df[bank_df['ai_category'] == category_key]
            
            if len(cat_reviews) > 0:
                avg_rating = cat_reviews['rating'].mean()
                avg_sentiment = cat_reviews['ai_sentiment_score'].mean()
                count = len(cat_reviews)
                
                print(f"\n   📂 {category_name}:")
                print(f"     Reviews: {count}")
                print(f"     Average Rating: {avg_rating:.2f}/5.0")
                print(f"     Average Sentiment: {avg_sentiment:.3f}")
                
                # Get 3 example comments (around 15 words each)
                print(f"     Example Comments:")
                
                # Get negative comments for examples
                negative_comments = cat_reviews[cat_reviews['ai_sentiment'] == 'negative'].head(3)
                
                for i, (_, comment) in enumerate(negative_comments.iterrows(), 1):
                    content = comment['translated_content'] if pd.notna(comment['translated_content']) else comment['content']
                    # Truncate to around 15 words
                    words = str(content).split()[:15]
                    short_comment = ' '.join(words)
                    if len(words) == 15:
                        short_comment += '...'
                    print(f"       {i}. \"{short_comment}\"")
                
                if len(negative_comments) < 3:
                    # Get some neutral/positive comments to fill up
                    other_comments = cat_reviews[cat_reviews['ai_sentiment'] != 'negative'].head(3-len(negative_comments))
                    for i, (_, comment) in enumerate(other_comments.iterrows(), len(negative_comments)+1):
                        content = comment['translated_content'] if pd.notna(comment['translated_content']) else comment['content']
                        words = str(content).split()[:15]
                        short_comment = ' '.join(words)
                        if len(words) == 15:
                            short_comment += '...'
                        print(f"       {i}. \"{short_comment}\"")
            else:
                print(f"\n   📂 {category_name}: No reviews")
        
        print("-" * 80)

if __name__ == "__main__":
    analyze_three_categories() 