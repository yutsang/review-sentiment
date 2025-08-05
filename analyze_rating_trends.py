#!/usr/bin/env python3
"""
Analyze rating trends for three banks and generate comprehensive Excel report
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_rating_trends():
    """Analyze rating trends for the three banks"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Filter for the three banks
    three_banks = ['WeLab Bank', 'Mox Bank', 'ZA Bank']
    df_filtered = df[df['bank'].isin(three_banks)]
    
    # Ensure year column exists
    if 'year' not in df_filtered.columns:
        df_filtered['year'] = pd.to_datetime(df_filtered['date']).dt.year
    
    print("📊 RATING TRENDS ANALYSIS FOR THREE BANKS")
    print("=" * 80)
    
    for bank in three_banks:
        bank_df = df_filtered[df_filtered['bank'] == bank]
        
        print(f"\n🏦 {bank.upper()}")
        print("-" * 50)
        
        # Overall rating trend by year
        yearly_ratings = bank_df.groupby('year')['rating'].agg(['mean', 'count']).round(2)
        
        print("Yearly Rating Trends:")
        for year, row in yearly_ratings.iterrows():
            print(f"  {year}: {row['mean']}/5.0 ({row['count']} reviews)")
        
        # Overall average
        overall_avg = bank_df['rating'].mean()
        print(f"  Overall Average: {overall_avg:.2f}/5.0")
        
        # Sentiment trend by year
        yearly_sentiment = bank_df.groupby('year')['ai_sentiment_score'].agg(['mean', 'count']).round(3)
        
        print("\nYearly Sentiment Trends:")
        for year, row in yearly_sentiment.iterrows():
            print(f"  {year}: {row['mean']} ({row['count']} reviews)")
        
        # Overall sentiment average
        overall_sentiment = bank_df['ai_sentiment_score'].mean()
        print(f"  Overall Sentiment: {overall_sentiment:.3f}")
        
        # Category performance
        print("\nCategory Performance:")
        category_performance = bank_df.groupby('ai_category')['rating'].agg(['mean', 'count']).round(2)
        for category, row in category_performance.iterrows():
            print(f"  {category}: {row['mean']}/5.0 ({row['count']} reviews)")

def create_comprehensive_excel():
    """Create comprehensive Excel report with all banks and categories"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Ensure year column exists
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
    
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
    
    # All banks
    banks = ['PAO Bank', 'Ant Bank', 'Airstar Bank', 'Fusion Bank', 'livi Bank', 'WeLab Bank', 'ZA Bank', 'Mox Bank']
    
    # Create comprehensive report
    report_data = []
    
    for bank in banks:
        bank_df = df[df['bank'] == bank]
        bank_total_reviews = len(bank_df)
        bank_avg_sentiment = bank_df['ai_sentiment_score'].mean()
        
        for category_key, category_name in categories.items():
            cat_df = bank_df[bank_df['ai_category'] == category_key]
            
            if len(cat_df) > 0:
                # Basic metrics
                total_reviews = len(cat_df)
                positive_reviews = len(cat_df[cat_df['ai_sentiment'] == 'positive'])
                negative_reviews = len(cat_df[cat_df['ai_sentiment'] == 'negative'])
                neutral_reviews = len(cat_df[cat_df['ai_sentiment'] == 'neutral'])
                
                # Sentiment metrics
                sentiment_score = cat_df['ai_sentiment_score'].mean()
                importance = total_reviews / bank_total_reviews if bank_total_reviews > 0 else 0
                
                # Overall sentiment
                if sentiment_score > 0.1:
                    overall_sentiment = 'Positive'
                elif sentiment_score < -0.1:
                    overall_sentiment = 'Negative'
                else:
                    overall_sentiment = 'Neutral'
                
                # Yearly sentiment scores
                yearly_sentiments = {}
                for year in range(2020, 2026):
                    year_df = cat_df[cat_df['year'] == year]
                    if len(year_df) > 0:
                        yearly_sentiments[year] = year_df['ai_sentiment_score'].mean()
                    else:
                        yearly_sentiments[year] = np.nan
                
                # Category average across all banks
                all_banks_cat_df = df[df['ai_category'] == category_key]
                category_avg = all_banks_cat_df['ai_sentiment_score'].mean()
                
                # Overall average
                overall_avg = df['ai_sentiment_score'].mean()
                
                row_data = {
                    'Bank': bank,
                    'Category': category_name,
                    'Total Review': total_reviews,
                    'Positive': positive_reviews,
                    'Negative': negative_reviews,
                    'Neutral': neutral_reviews,
                    'Overall Sentiment': overall_sentiment,
                    'Importance': importance,
                    'Sentiment Score': sentiment_score,
                    '2020': yearly_sentiments[2020],
                    '2021': yearly_sentiments[2021],
                    '2022': yearly_sentiments[2022],
                    '2023': yearly_sentiments[2023],
                    '2024': yearly_sentiments[2024],
                    '2025': yearly_sentiments[2025],
                    'Bank Average': bank_avg_sentiment,
                    'Category Average': category_avg,
                    'Overall Average': overall_avg
                }
                
                report_data.append(row_data)
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Round numeric columns
    numeric_columns = ['Importance', 'Sentiment Score', '2020', '2021', '2022', '2023', '2024', '2025', 
                      'Bank Average', 'Category Average', 'Overall Average']
    for col in numeric_columns:
        if col in report_df.columns:
            report_df[col] = report_df[col].round(6)
    
    # Save to Excel
    filename = f'comprehensive_bank_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    report_df.to_excel(filename, index=False)
    
    print(f"\n💾 Comprehensive Excel report saved to: {filename}")
    print(f"📊 Total rows: {len(report_df)} (Expected: 64 for 8 banks × 8 categories)")
    
    return report_df

if __name__ == "__main__":
    analyze_rating_trends()
    report_df = create_comprehensive_excel() 