#!/usr/bin/env python3
"""
Generate yearly sentiment table for each bank
"""

import pandas as pd
import numpy as np

def generate_yearly_sentiment_table():
    """Generate table showing average sentiment for each bank by year"""
    print("📊 GENERATING YEARLY SENTIMENT TABLE")
    print("=" * 50)
    
    # Load the final database
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    print(f"📊 Loaded {len(df)} reviews")
    
    # Filter for 2022 and 2025 only
    df_filtered = df[df['year'].isin([2022, 2025])].copy()
    
    print(f"📊 Reviews in 2022 and 2025: {len(df_filtered)}")
    
    # Group by bank and year, calculate average sentiment
    yearly_sentiment = df_filtered.groupby(['bank', 'year'])['ai_sentiment_score'].agg([
        'mean', 'count'
    ]).round(3)
    
    # Reset index to make it a table
    yearly_sentiment = yearly_sentiment.reset_index()
    
    # Pivot to create a table with years as columns
    pivot_table = yearly_sentiment.pivot(index='bank', columns='year', values='mean').round(3)
    count_table = yearly_sentiment.pivot(index='bank', columns='year', values='count')
    
    # Add count information
    for year in [2022, 2025]:
        if year in pivot_table.columns:
            pivot_table[f'{year}_count'] = count_table[year]
    
    # Fill NaN with "No data"
    pivot_table = pivot_table.fillna('No data')
    
    print(f"\n📈 YEARLY SENTIMENT AVERAGES")
    print("=" * 50)
    print("Sentiment scale: -1.0 (very negative) to +1.0 (very positive)")
    print("0.0 = neutral")
    print("=" * 50)
    
    # Display the table
    for bank in pivot_table.index:
        print(f"\n🏦 {bank}:")
        if 2022 in pivot_table.columns:
            sentiment_2022 = pivot_table.loc[bank, 2022]
            count_2022 = pivot_table.loc[bank, '2022_count']
            if sentiment_2022 != 'No data':
                print(f"   2022: {sentiment_2022} ({count_2022} reviews)")
            else:
                print(f"   2022: No data")
        
        if 2025 in pivot_table.columns:
            sentiment_2025 = pivot_table.loc[bank, 2025]
            count_2025 = pivot_table.loc[bank, '2025_count']
            if sentiment_2025 != 'No data':
                print(f"   2025: {sentiment_2025} ({count_2025} reviews)")
            else:
                print(f"   2025: No data")
    
    # Create a summary table
    print(f"\n📋 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Bank':<15} {'2022 Sentiment':<15} {'2022 Count':<12} {'2025 Sentiment':<15} {'2025 Count':<12}")
    print("=" * 80)
    
    for bank in pivot_table.index:
        sentiment_2022 = pivot_table.loc[bank, 2022] if 2022 in pivot_table.columns else 'No data'
        count_2022 = pivot_table.loc[bank, '2022_count'] if '2022_count' in pivot_table.columns else 'No data'
        sentiment_2025 = pivot_table.loc[bank, 2025] if 2025 in pivot_table.columns else 'No data'
        count_2025 = pivot_table.loc[bank, '2025_count'] if '2025_count' in pivot_table.columns else 'No data'
        
        print(f"{bank:<15} {str(sentiment_2022):<15} {str(count_2022):<12} {str(sentiment_2025):<15} {str(count_2025):<12}")
    
    # Save to CSV
    output_file = 'yearly_sentiment_table.csv'
    pivot_table.to_csv(output_file)
    print(f"\n✅ Table saved to: {output_file}")
    
    return pivot_table

if __name__ == "__main__":
    generate_yearly_sentiment_table() 