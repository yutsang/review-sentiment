#!/usr/bin/env python3
"""
Get 5 examples (10-20 words) for each of the 3 categories across all 3 banks
"""

import pandas as pd

def get_all_banks_examples():
    """Get 5 examples for each category across all banks"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Define categories
    categories = {
        'onboarding_verification': 'Onboarding & Verification',
        'platform_performance': 'Platform Performance & Reliability',
        'interface_functionality': 'Interface & Functionality'
    }
    
    # Three banks
    banks = ['WeLab Bank', 'Mox Bank', 'ZA Bank']
    
    print("🏦 ALL BANKS - 5 EXAMPLES PER CATEGORY (10-20 words)")
    print("=" * 80)
    
    for bank in banks:
        print(f"\n🏦 {bank.upper()}")
        print("=" * 80)
        
        bank_df = df[df['bank'] == bank]
        
        for category_key, category_name in categories.items():
            print(f"\n📂 {category_name}")
            print("-" * 50)
            
            # Filter for this category and negative reviews
            cat_reviews = bank_df[bank_df['ai_category'] == category_key]
            negative_reviews = cat_reviews[cat_reviews['ai_sentiment'] == 'negative']
            
            if len(negative_reviews) > 0:
                print(f"Total Reviews: {len(cat_reviews)} | Negative: {len(negative_reviews)} | Rating: {cat_reviews['rating'].mean():.2f}/5.0")
                print()
                
                # Get examples with 10-20 words
                examples = []
                
                for _, review in negative_reviews.iterrows():
                    content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                    rating = review['rating']
                    
                    if pd.notna(content) and content:
                        content_str = str(content).strip()
                        word_count = len(content_str.split())
                        
                        # Filter for 10-20 words (with some flexibility)
                        if 8 <= word_count <= 25:
                            examples.append({
                                'rating': rating,
                                'content': content_str,
                                'word_count': word_count
                            })
                
                # Sort by rating and get top 5
                examples.sort(key=lambda x: x['rating'])
                top_examples = examples[:5]
                
                print("📝 TOP 5 NEGATIVE REVIEW EXAMPLES:")
                for i, example in enumerate(top_examples, 1):
                    print(f"{i}. Rating {example['rating']}/5 ({example['word_count']} words):")
                    print(f"   \"{example['content']}\"")
                    print()
                
                # If we don't have enough examples, get more
                if len(top_examples) < 5:
                    remaining_examples = examples[5:10]
                    for i, example in enumerate(remaining_examples, len(top_examples) + 1):
                        if i <= 5:
                            print(f"{i}. Rating {example['rating']}/5 ({example['word_count']} words):")
                            print(f"   \"{example['content']}\"")
                            print()
            else:
                print("No negative reviews found")
            
            print()

if __name__ == "__main__":
    get_all_banks_examples() 