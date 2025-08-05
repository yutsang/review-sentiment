#!/usr/bin/env python3
"""
Analyze 4 specific categories for WeLab Bank and summarize underlying problems
"""

import pandas as pd
from collections import Counter

def analyze_welab_categories():
    """Analyze 4 categories for WeLab Bank"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Filter for WeLab Bank only
    welab_df = df[df['bank'] == 'WeLab Bank']
    
    # Define the 4 categories we want to analyze
    categories = {
        'onboarding_verification': 'Onboarding & Verification',
        'platform_performance': 'Platform Performance & Reliability',
        'accounts_interest': 'Accounts & Interest Terms',
        'interface_functionality': 'Interface & Functionality'
    }
    
    print("🏦 WELAB BANK - 4 CATEGORIES ANALYSIS")
    print("=" * 80)
    print(f"Total WeLab Bank reviews: {len(welab_df)}")
    
    category_summaries = {}
    
    for category_key, category_name in categories.items():
        # Filter reviews for this category
        cat_reviews = welab_df[welab_df['ai_category'] == category_key]
        
        if len(cat_reviews) > 0:
            # Get sentiment distribution
            sentiment_counts = cat_reviews['ai_sentiment'].value_counts()
            avg_rating = cat_reviews['rating'].mean()
            avg_sentiment = cat_reviews['ai_sentiment_score'].mean()
            
            print(f"\n📂 {category_name}:")
            print(f"   Reviews: {len(cat_reviews)}")
            print(f"   Average Rating: {avg_rating:.2f}/5.0")
            print(f"   Average Sentiment: {avg_sentiment:.3f}")
            print(f"   Sentiment Distribution: {dict(sentiment_counts)}")
            
            # Analyze problems by sentiment
            positive_reviews = cat_reviews[cat_reviews['ai_sentiment'] == 'positive']
            negative_reviews = cat_reviews[cat_reviews['ai_sentiment'] == 'negative']
            neutral_reviews = cat_reviews[cat_reviews['ai_sentiment'] == 'neutral']
            
            # Extract common themes from negative reviews
            negative_content = negative_reviews['translated_content'].fillna(negative_reviews['content']).tolist()
            positive_content = positive_reviews['translated_content'].fillna(positive_reviews['content']).tolist()
            
            # Common negative problems
            negative_problems = []
            positive_aspects = []
            
            # Analyze negative reviews for common problems
            for content in negative_content:
                if pd.notna(content) and content:
                    content_lower = str(content).lower()
                    
                    # Onboarding & Verification problems
                    if category_key == 'onboarding_verification':
                        if 'face id' in content_lower or 'face verification' in content_lower:
                            negative_problems.append("Face ID verification issues")
                        if 'id card' in content_lower or 'identity' in content_lower:
                            negative_problems.append("ID card recognition problems")
                        if 'phone number' in content_lower or 'verification' in content_lower:
                            negative_problems.append("Phone number verification issues")
                        if 'create account' in content_lower or 'sign up' in content_lower:
                            negative_problems.append("Account creation difficulties")
                    
                    # Platform Performance problems
                    elif category_key == 'platform_performance':
                        if 'crash' in content_lower or 'crashed' in content_lower:
                            negative_problems.append("App crashes frequently")
                        if 'bug' in content_lower or 'error' in content_lower:
                            negative_problems.append("System bugs and errors")
                        if 'slow' in content_lower or 'loading' in content_lower:
                            negative_problems.append("Slow performance and loading")
                        if 'update' in content_lower:
                            negative_problems.append("Update-related issues")
                    
                    # Accounts & Interest problems
                    elif category_key == 'accounts_interest':
                        if 'interest' in content_lower or 'rate' in content_lower:
                            negative_problems.append("Interest rate issues")
                        if 'account' in content_lower and ('lock' in content_lower or 'cancel' in content_lower):
                            negative_problems.append("Account locking/cancellation issues")
                        if 'loan' in content_lower:
                            negative_problems.append("Loan-related problems")
                    
                    # Interface & Functionality problems
                    elif category_key == 'interface_functionality':
                        if 'interface' in content_lower or 'ui' in content_lower:
                            negative_problems.append("Poor user interface")
                        if 'function' in content_lower or 'feature' in content_lower:
                            negative_problems.append("Missing or broken features")
                        if 'design' in content_lower or 'layout' in content_lower:
                            negative_problems.append("Poor design and layout")
                        if 'navigation' in content_lower or 'menu' in content_lower:
                            negative_problems.append("Navigation difficulties")
            
            # Count problems
            problem_counts = Counter(negative_problems)
            
            category_summaries[category_name] = {
                'total_reviews': len(cat_reviews),
                'positive_reviews': len(positive_reviews),
                'negative_reviews': len(negative_reviews),
                'neutral_reviews': len(neutral_reviews),
                'avg_rating': avg_rating,
                'avg_sentiment': avg_sentiment,
                'main_problems': problem_counts.most_common(5)
            }
            
            print(f"   Main Problems (from negative reviews):")
            for problem, count in problem_counts.most_common(5):
                print(f"     - {problem} ({count} mentions)")
            
        else:
            print(f"\n📂 {category_name}: No reviews")
            category_summaries[category_name] = {
                'total_reviews': 0,
                'positive_reviews': 0,
                'negative_reviews': 0,
                'neutral_reviews': 0,
                'avg_rating': 0,
                'avg_sentiment': 0,
                'main_problems': []
            }
    
    return category_summaries

def write_summary_to_file(category_summaries):
    """Write summary to text file"""
    
    with open('welab_bank_category_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("🏦 WELAB BANK - 4 CATEGORIES ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for category_name, data in category_summaries.items():
            f.write(f"📂 {category_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Reviews: {data['total_reviews']}\n")
            f.write(f"Positive Reviews: {data['positive_reviews']} ({(data['positive_reviews']/data['total_reviews']*100):.1f}%)\n" if data['total_reviews'] > 0 else "Positive Reviews: 0 (0.0%)\n")
            f.write(f"Negative Reviews: {data['negative_reviews']} ({(data['negative_reviews']/data['total_reviews']*100):.1f}%)\n" if data['total_reviews'] > 0 else "Negative Reviews: 0 (0.0%)\n")
            f.write(f"Neutral Reviews: {data['neutral_reviews']} ({(data['neutral_reviews']/data['total_reviews']*100):.1f}%)\n" if data['total_reviews'] > 0 else "Neutral Reviews: 0 (0.0%)\n")
            f.write(f"Average Rating: {data['avg_rating']:.2f}/5.0\n")
            f.write(f"Average Sentiment Score: {data['avg_sentiment']:.3f}\n\n")
            
            if data['main_problems']:
                f.write("UNDERLYING PROBLEMS (mainly from negative reviews):\n")
                for problem, count in data['main_problems']:
                    f.write(f"• {problem} ({count} mentions)\n")
            else:
                f.write("UNDERLYING PROBLEMS: None identified\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n💾 Analysis saved to: welab_bank_category_analysis.txt")

if __name__ == "__main__":
    category_summaries = analyze_welab_categories()
    write_summary_to_file(category_summaries) 