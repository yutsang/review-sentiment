#!/usr/bin/env python3
"""
Analyze specific categories for WeLab, Mox, and ZA banks with negative comment analysis
"""

import pandas as pd
import numpy as np

def analyze_bank_categories():
    """Analyze specific categories for the three banks"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Define categories for each bank
    bank_categories = {
        'WeLab Bank': ['onboarding_verification', 'platform_performance', 'accounts_interest'],
        'Mox Bank': ['promotion_fees', 'security_privacy', 'accounts_interest'],
        'ZA Bank': ['onboarding_verification', 'interface_functionality', 'transactions_payments']
    }
    
    category_names = {
        'onboarding_verification': 'Onboarding & Verification',
        'platform_performance': 'Platform Performance & Reliability',
        'accounts_interest': 'Accounts & Interest Terms',
        'promotion_fees': 'Promotion & Fees',
        'security_privacy': 'Security & Privacy',
        'interface_functionality': 'Interface & Functionality',
        'transactions_payments': 'Transactions & Payments'
    }
    
    results = {}
    
    for bank, categories in bank_categories.items():
        print(f"\n🏦 {bank.upper()}")
        print("=" * 80)
        
        bank_df = df[df['bank'] == bank]
        total_negative = len(bank_df[bank_df['ai_sentiment'] == 'negative'])
        
        results[bank] = {}
        
        for category in categories:
            cat_reviews = bank_df[bank_df['ai_category'] == category]
            negative_reviews = cat_reviews[cat_reviews['ai_sentiment'] == 'negative']
            
            if len(cat_reviews) > 0:
                avg_rating = cat_reviews['rating'].mean()
                negative_count = len(negative_reviews)
                negative_percentage = (negative_count / total_negative * 100) if total_negative > 0 else 0
                
                print(f"\n📂 {category_names[category]}:")
                print(f"   Total Reviews: {len(cat_reviews)}")
                print(f"   Negative Reviews: {negative_count} ({negative_percentage:.1f}% of total negative)")
                print(f"   Average Rating: {avg_rating:.2f}/5.0")
                
                # Get 2-3 example comments (10-15 words each)
                if len(negative_reviews) > 0:
                    print(f"   Example Negative Comments:")
                    
                    # Get content from negative reviews
                    comments = []
                    for _, review in negative_reviews.iterrows():
                        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                        if pd.notna(content) and content:
                            # Split into words and take first 10-15 words
                            words = str(content).split()
                            if len(words) > 5:  # Only include meaningful comments
                                short_comment = ' '.join(words[:min(15, len(words))])
                                if len(words) > 15:
                                    short_comment += '...'
                                comments.append(short_comment)
                    
                    # Remove duplicates and overlapping comments
                    unique_comments = []
                    for comment in comments:
                        is_duplicate = False
                        for existing in unique_comments:
                            # Check for significant overlap (more than 50% common words)
                            comment_words = set(comment.lower().split())
                            existing_words = set(existing.lower().split())
                            if len(comment_words.intersection(existing_words)) / len(comment_words.union(existing_words)) > 0.3:
                                is_duplicate = True
                                break
                        if not is_duplicate and len(unique_comments) < 3:
                            unique_comments.append(comment)
                    
                    for i, comment in enumerate(unique_comments[:3], 1):
                        print(f"     {i}. \"{comment}\"")
                
                results[bank][category_names[category]] = {
                    'total_reviews': len(cat_reviews),
                    'negative_count': negative_count,
                    'negative_percentage': negative_percentage,
                    'avg_rating': avg_rating,
                    'example_comments': unique_comments[:3] if len(negative_reviews) > 0 else []
                }
            else:
                print(f"\n📂 {category_names[category]}: No reviews")
                results[bank][category_names[category]] = {
                    'total_reviews': 0,
                    'negative_count': 0,
                    'negative_percentage': 0,
                    'avg_rating': 0,
                    'example_comments': []
                }
    
    return results

def write_summary_to_file(results):
    """Write summary to text file"""
    
    with open('three_banks_category_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("🏦 THREE BANKS - CATEGORY ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for bank, categories in results.items():
            f.write(f"🏦 {bank.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            for category_name, data in categories.items():
                f.write(f"📂 {category_name}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total Reviews: {data['total_reviews']}\n")
                f.write(f"Negative Reviews: {data['negative_count']} ({data['negative_percentage']:.1f}% of total negative)\n")
                f.write(f"Average Rating: {data['avg_rating']:.2f}/5.0\n")
                
                if data['example_comments']:
                    f.write("Example Negative Comments:\n")
                    for i, comment in enumerate(data['example_comments'], 1):
                        f.write(f"  {i}. \"{comment}\"\n")
                else:
                    f.write("Example Negative Comments: None\n")
                
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n💾 Analysis saved to: three_banks_category_analysis.txt")

if __name__ == "__main__":
    results = analyze_bank_categories()
    write_summary_to_file(results) 