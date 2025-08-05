#!/usr/bin/env python3
"""
Create comprehensive analysis table for all 3 banks and all 8 categories
"""

import pandas as pd
import numpy as np

def create_comprehensive_analysis_table():
    """Create comprehensive analysis table for all 3 banks and all 8 categories"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Define all categories
    categories = {
        'onboarding_verification': 'Onboarding & Verification',
        'platform_performance': 'Platform Performance & Reliability',
        'customer_service': 'Customer Service Quality & Accessibility',
        'transactions_payments': 'Transactions & Payments',
        'promotion_fees': 'Promotion & Fees',
        'interface_functionality': 'Interface & Functionality',
        'security_privacy': 'Security & Privacy',
        'accounts_interest': 'Accounts & Interest Terms'
    }
    
    # Three banks
    banks = ['WeLab Bank', 'Mox Bank', 'ZA Bank']
    
    print("🏦 THREE BANKS - COMPREHENSIVE CATEGORY ANALYSIS")
    print("=" * 120)
    
    # Create comprehensive table
    table_data = []
    
    for bank in banks:
        print(f"\n🏦 {bank.upper()}")
        print("=" * 120)
        
        bank_df = df[df['bank'] == bank]
        total_negative = len(bank_df[bank_df['ai_sentiment'] == 'negative'])
        
        for category_key, category_name in categories.items():
            cat_reviews = bank_df[bank_df['ai_category'] == category_key]
            
            if len(cat_reviews) > 0:
                # Basic metrics
                total_reviews = len(cat_reviews)
                positive_reviews = len(cat_reviews[cat_reviews['ai_sentiment'] == 'positive'])
                negative_reviews = len(cat_reviews[cat_reviews['ai_sentiment'] == 'negative'])
                neutral_reviews = len(cat_reviews[cat_reviews['ai_sentiment'] == 'neutral'])
                
                # Sentiment metrics
                avg_rating = cat_reviews['rating'].mean()
                avg_sentiment = cat_reviews['ai_sentiment_score'].mean()
                negative_percentage = (negative_reviews / total_negative * 100) if total_negative > 0 else 0
                
                # Get example comments
                negative_comments = cat_reviews[cat_reviews['ai_sentiment'] == 'negative']
                example_comments = []
                
                if len(negative_comments) > 0:
                    for _, review in negative_comments.head(3).iterrows():
                        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                        if pd.notna(content) and content:
                            words = str(content).split()
                            if len(words) > 5:
                                short_comment = ' '.join(words[:min(15, len(words))])
                                if len(words) > 15:
                                    short_comment += '...'
                                example_comments.append(short_comment)
                
                print(f"\n📂 {category_name}")
                print("-" * 50)
                print(f"Total Reviews: {total_reviews}")
                print(f"Negative Reviews: {negative_reviews} ({negative_percentage:.1f}% of total negative)")
                print(f"Average Rating: {avg_rating:.2f}/5.0")
                print(f"Average Sentiment: {avg_sentiment:.3f}")
                
                if example_comments:
                    print("Example Negative Comments:")
                    for i, comment in enumerate(example_comments, 1):
                        print(f"  {i}. \"{comment}\"")
                
                # Store data for table
                table_data.append({
                    'Bank': bank,
                    'Category': category_name,
                    'Total Reviews': total_reviews,
                    'Negative Reviews': negative_reviews,
                    'Negative % of Total': f"{negative_percentage:.1f}%",
                    'Average Rating': f"{avg_rating:.2f}/5.0",
                    'Average Sentiment': f"{avg_sentiment:.3f}",
                    'Example Comment 1': example_comments[0] if len(example_comments) > 0 else "N/A",
                    'Example Comment 2': example_comments[1] if len(example_comments) > 1 else "N/A",
                    'Example Comment 3': example_comments[2] if len(example_comments) > 2 else "N/A"
                })
            else:
                print(f"\n📂 {category_name}")
                print("-" * 50)
                print("No reviews")
                
                table_data.append({
                    'Bank': bank,
                    'Category': category_name,
                    'Total Reviews': 0,
                    'Negative Reviews': 0,
                    'Negative % of Total': "0.0%",
                    'Average Rating': "N/A",
                    'Average Sentiment': "N/A",
                    'Example Comment 1': "N/A",
                    'Example Comment 2': "N/A",
                    'Example Comment 3': "N/A"
                })
    
    # Create summary table
    print(f"\n📊 SUMMARY TABLE")
    print("=" * 120)
    
    # Create DataFrame for better formatting
    summary_df = pd.DataFrame(table_data)
    
    # Print formatted table
    print(f"{'Bank':<15} {'Category':<35} {'Total':<8} {'Neg':<8} {'Neg%':<8} {'Rating':<10} {'Sentiment':<10}")
    print("-" * 120)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Bank']:<15} {row['Category']:<35} {row['Total Reviews']:<8} {row['Negative Reviews']:<8} {row['Negative % of Total']:<8} {row['Average Rating']:<10} {row['Average Sentiment']:<10}")
    
    # Save to Excel
    summary_df.to_excel('three_banks_comprehensive_analysis.xlsx', index=False)
    print(f"\n💾 Comprehensive analysis saved to: three_banks_comprehensive_analysis.xlsx")
    
    return summary_df

if __name__ == "__main__":
    summary_df = create_comprehensive_analysis_table() 