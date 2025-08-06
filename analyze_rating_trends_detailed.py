#!/usr/bin/env python3
"""
Detailed analysis of rating trends for three banks with actual review examples
"""

import pandas as pd
import numpy as np

def analyze_rating_trends_detailed():
    """Analyze detailed rating trends with actual review examples"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Ensure year column exists
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
    
    # Filter for the three banks
    three_banks = ['WeLab Bank', 'Mox Bank', 'ZA Bank']
    df_filtered = df[df['bank'].isin(three_banks)]
    
    print("📊 DETAILED RATING TRENDS ANALYSIS")
    print("=" * 80)
    
    analysis_results = {}
    
    for bank in three_banks:
        print(f"\n🏦 {bank.upper()}")
        print("=" * 80)
        
        bank_df = df_filtered[df_filtered['bank'] == bank]
        
        # Yearly analysis
        yearly_data = bank_df.groupby('year').agg({
            'rating': ['mean', 'count'],
            'ai_sentiment_score': 'mean',
            'content': lambda x: list(x.dropna())
        }).round(3)
        
        print("Yearly Rating Trends:")
        for year in sorted(bank_df['year'].unique()):
            year_data = bank_df[bank_df['year'] == year]
            avg_rating = year_data['rating'].mean()
            count = len(year_data)
            avg_sentiment = year_data['ai_sentiment_score'].mean()
            
            print(f"  {year}: {avg_rating:.2f}/5.0 ({count} reviews, sentiment: {avg_sentiment:.3f})")
        
        # Get specific year reviews for analysis
        analysis_results[bank] = {}
        
        # WeLab Bank: 2023-2025 gradual increase (1.5 -> 1.7 -> 1.8)
        if bank == 'WeLab Bank':
            print(f"\n🔍 WELAB BANK TREND ANALYSIS: 2023-2025 Gradual Increase")
            print("-" * 60)
            
            for year in [2023, 2024, 2025]:
                year_reviews = bank_df[bank_df['year'] == year]
                if len(year_reviews) > 0:
                    print(f"\n📅 {year} Reviews (Rating: {year_reviews['rating'].mean():.2f}/5.0):")
                    
                    # Get sample reviews
                    sample_reviews = year_reviews.sample(min(5, len(year_reviews)))
                    for _, review in sample_reviews.iterrows():
                        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                        rating = review['rating']
                        print(f"  Rating {rating}/5: \"{str(content)[:100]}...\"")
                    
                    analysis_results[bank][year] = {
                        'avg_rating': year_reviews['rating'].mean(),
                        'count': len(year_reviews),
                        'sample_reviews': sample_reviews[['rating', 'translated_content', 'content']].to_dict('records')
                    }
        
        # Mox Bank: 2022 sudden drop (3.2 -> 2.5), then increase 2023-2025 (2.5 -> 2.6 -> 2.9)
        elif bank == 'Mox Bank':
            print(f"\n🔍 MOX BANK TREND ANALYSIS: 2022 Drop & 2023-2025 Recovery")
            print("-" * 60)
            
            for year in [2022, 2023, 2024, 2025]:
                year_reviews = bank_df[bank_df['year'] == year]
                if len(year_reviews) > 0:
                    print(f"\n📅 {year} Reviews (Rating: {year_reviews['rating'].mean():.2f}/5.0):")
                    
                    # Get sample reviews
                    sample_reviews = year_reviews.sample(min(5, len(year_reviews)))
                    for _, review in sample_reviews.iterrows():
                        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                        rating = review['rating']
                        print(f"  Rating {rating}/5: \"{str(content)[:100]}...\"")
                    
                    analysis_results[bank][year] = {
                        'avg_rating': year_reviews['rating'].mean(),
                        'count': len(year_reviews),
                        'sample_reviews': sample_reviews[['rating', 'translated_content', 'content']].to_dict('records')
                    }
        
        # ZA Bank: 2022-2023 sudden increase (3.2 -> 3.8), then gradual decrease (3.5 -> 3.3)
        elif bank == 'ZA Bank':
            print(f"\n🔍 ZA BANK TREND ANALYSIS: 2022-2023 Increase & 2023-2025 Decrease")
            print("-" * 60)
            
            for year in [2022, 2023, 2024, 2025]:
                year_reviews = bank_df[bank_df['year'] == year]
                if len(year_reviews) > 0:
                    print(f"\n📅 {year} Reviews (Rating: {year_reviews['rating'].mean():.2f}/5.0):")
                    
                    # Get sample reviews
                    sample_reviews = year_reviews.sample(min(5, len(year_reviews)))
                    for _, review in sample_reviews.iterrows():
                        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                        rating = review['rating']
                        print(f"  Rating {rating}/5: \"{str(content)[:100]}...\"")
                    
                    analysis_results[bank][year] = {
                        'avg_rating': year_reviews['rating'].mean(),
                        'count': len(year_reviews),
                        'sample_reviews': sample_reviews[['rating', 'translated_content', 'content']].to_dict('records')
                    }
    
    return analysis_results

def write_trend_analysis_to_file(analysis_results):
    """Write detailed trend analysis to text file"""
    
    with open('rating_trends_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("📊 DETAILED RATING TRENDS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # WeLab Bank Analysis
        f.write("🏦 WELAB BANK: 2023-2025 GRADUAL INCREASE (1.5 → 1.7 → 1.8)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TREND SUMMARY:\n")
        f.write("WeLab Bank showed a gradual improvement in ratings from 2023 to 2025, ")
        f.write("increasing from 1.5 to 1.8. This suggests that while the bank was still ")
        f.write("performing poorly overall, there were incremental improvements in user experience.\n\n")
        
        f.write("POTENTIAL REASONS FOR IMPROVEMENT:\n")
        f.write("1. Technical fixes to major app issues\n")
        f.write("2. Improved onboarding process\n")
        f.write("3. Better customer service response\n")
        f.write("4. Reduced frequency of critical bugs\n")
        f.write("5. Enhanced security measures\n\n")
        
        f.write("REVIEW EXAMPLES BY YEAR:\n")
        if 'WeLab Bank' in analysis_results:
            for year in [2023, 2024, 2025]:
                if year in analysis_results['WeLab Bank']:
                    data = analysis_results['WeLab Bank'][year]
                    f.write(f"\n{year} (Rating: {data['avg_rating']:.2f}/5.0, {data['count']} reviews):\n")
                    for review in data['sample_reviews'][:3]:  # Show 3 examples
                        content = review['translated_content'] if review['translated_content'] else review['content']
                        f.write(f"  {review['rating']}/5: \"{str(content)[:150]}...\"\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Mox Bank Analysis
        f.write("🏦 MOX BANK: 2022 SUDDEN DROP (3.2 → 2.5) & 2023-2025 RECOVERY (2.5 → 2.6 → 2.9)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TREND SUMMARY:\n")
        f.write("Mox Bank experienced a significant drop in 2022 (3.2 to 2.5), followed by ")
        f.write("a gradual recovery from 2023 to 2025 (2.5 to 2.9). The 2022 drop suggests ")
        f.write("a major issue or update that negatively impacted user experience.\n\n")
        
        f.write("POTENTIAL REASONS FOR 2022 DROP:\n")
        f.write("1. Major app update with bugs\n")
        f.write("2. Security policy changes\n")
        f.write("3. Fee structure modifications\n")
        f.write("4. Technical infrastructure issues\n")
        f.write("5. Customer service problems\n\n")
        
        f.write("POTENTIAL REASONS FOR RECOVERY:\n")
        f.write("1. Bug fixes and app improvements\n")
        f.write("2. Better user interface updates\n")
        f.write("3. Enhanced security features\n")
        f.write("4. Improved customer support\n")
        f.write("5. Fee structure adjustments\n\n")
        
        f.write("REVIEW EXAMPLES BY YEAR:\n")
        if 'Mox Bank' in analysis_results:
            for year in [2022, 2023, 2024, 2025]:
                if year in analysis_results['Mox Bank']:
                    data = analysis_results['Mox Bank'][year]
                    f.write(f"\n{year} (Rating: {data['avg_rating']:.2f}/5.0, {data['count']} reviews):\n")
                    for review in data['sample_reviews'][:3]:  # Show 3 examples
                        content = review['translated_content'] if review['translated_content'] else review['content']
                        f.write(f"  {review['rating']}/5: \"{str(content)[:150]}...\"\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # ZA Bank Analysis
        f.write("🏦 ZA BANK: 2022-2023 SUDDEN INCREASE (3.2 → 3.8) & 2023-2025 GRADUAL DECREASE (3.5 → 3.3)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TREND SUMMARY:\n")
        f.write("ZA Bank showed a dramatic improvement from 2022 to 2023 (3.2 to 3.8), ")
        f.write("followed by a gradual decline from 2023 to 2025 (3.5 to 3.3). The 2023 ")
        f.write("peak suggests successful improvements that were not sustained long-term.\n\n")
        
        f.write("POTENTIAL REASONS FOR 2022-2023 INCREASE:\n")
        f.write("1. Major app redesign and improvements\n")
        f.write("2. Enhanced user interface and functionality\n")
        f.write("3. Better onboarding experience\n")
        f.write("4. Improved customer service\n")
        f.write("5. New features and capabilities\n\n")
        
        f.write("POTENTIAL REASONS FOR 2023-2025 DECREASE:\n")
        f.write("1. Initial excitement wearing off\n")
        f.write("2. New issues emerging over time\n")
        f.write("3. Competition from other banks\n")
        f.write("4. Service quality degradation\n")
        f.write("5. User expectations increasing\n\n")
        
        f.write("REVIEW EXAMPLES BY YEAR:\n")
        if 'ZA Bank' in analysis_results:
            for year in [2022, 2023, 2024, 2025]:
                if year in analysis_results['ZA Bank']:
                    data = analysis_results['ZA Bank'][year]
                    f.write(f"\n{year} (Rating: {data['avg_rating']:.2f}/5.0, {data['count']} reviews):\n")
                    for review in data['sample_reviews'][:3]:  # Show 3 examples
                        content = review['translated_content'] if review['translated_content'] else review['content']
                        f.write(f"  {review['rating']}/5: \"{str(content)[:150]}...\"\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("OVERALL ANALYSIS:\n")
        f.write("The rating trends show different patterns for each bank:\n")
        f.write("- WeLab Bank: Consistent but slow improvement from a very low base\n")
        f.write("- Mox Bank: Recovery from a major setback in 2022\n")
        f.write("- ZA Bank: Temporary success followed by gradual decline\n\n")
        
        f.write("These patterns suggest that virtual banks face different challenges:\n")
        f.write("1. WeLab Bank needs fundamental improvements to reach acceptable levels\n")
        f.write("2. Mox Bank needs to avoid major disruptions while maintaining quality\n")
        f.write("3. ZA Bank needs to sustain improvements rather than just temporary fixes\n")
    
    print(f"\n💾 Detailed analysis saved to: rating_trends_analysis.txt")

if __name__ == "__main__":
    analysis_results = analyze_rating_trends_detailed()
    write_trend_analysis_to_file(analysis_results) 