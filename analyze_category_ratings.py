#!/usr/bin/env python3
"""
Analyze average ratings by category for Mox Bank, ZA Bank, and WeLab Bank
"""

import pandas as pd

def analyze_category_ratings():
    """Analyze ratings by category for the three banks"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Define category mappings
    category_mapping = {
        'account_opening_and_verification': ['onboarding_verification', 'security_privacy'],
        'app_operation': ['platform_performance', 'interface_functionality', 'transactions_payments', 'customer_service'],
        'marketing': ['promotion_fees', 'accounts_interest']
    }
    
    banks = ['Mox Bank', 'ZA Bank', 'WeLab Bank']
    
    print("📊 AVERAGE RATINGS BY CATEGORY")
    print("=" * 60)
    
    for bank in banks:
        bank_df = df[df['bank'] == bank]
        
        print(f"\n🏦 {bank}:")
        print(f"   Total reviews: {len(bank_df)}")
        print(f"   Overall average rating: {bank_df['rating'].mean():.2f}/5.0")
        print(f"   Overall average sentiment score: {bank_df['ai_sentiment_score'].mean():.3f}")
        
        print(f"\n   📂 By Category:")
        
        for category, subcategories in category_mapping.items():
            # Filter reviews for this category
            cat_reviews = bank_df[bank_df['ai_category'].isin(subcategories)]
            
            if len(cat_reviews) > 0:
                avg_rating = cat_reviews['rating'].mean()
                avg_sentiment = cat_reviews['ai_sentiment_score'].mean()
                count = len(cat_reviews)
                
                print(f"     {category.replace('_', ' ').title()}:")
                print(f"       Reviews: {count}")
                print(f"       Average Rating: {avg_rating:.2f}/5.0")
                print(f"       Average Sentiment: {avg_sentiment:.3f}")
            else:
                print(f"     {category.replace('_', ' ').title()}: No reviews")
        
        print("-" * 40)

if __name__ == "__main__":
    analyze_category_ratings() 