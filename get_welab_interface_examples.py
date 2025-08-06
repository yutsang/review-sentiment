#!/usr/bin/env python3
"""
Get detailed WeLab Bank Interface & Functionality review examples with 10-20 words
"""

import pandas as pd

def get_welab_interface_examples():
    """Get detailed WeLab Bank interface examples"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Filter for WeLab Bank Interface & Functionality negative reviews
    welab_df = df[df['bank'] == 'WeLab Bank']
    interface_reviews = welab_df[welab_df['ai_category'] == 'interface_functionality']
    negative_interface = interface_reviews[interface_reviews['ai_sentiment'] == 'negative']
    
    print("🏦 WELAB BANK - INTERFACE & FUNCTIONALITY DETAILED EXAMPLES")
    print("=" * 80)
    print(f"Total Interface Reviews: {len(interface_reviews)}")
    print(f"Negative Interface Reviews: {len(negative_interface)}")
    print(f"Average Rating: {interface_reviews['rating'].mean():.2f}/5.0")
    print()
    
    print("📝 DETAILED NEGATIVE REVIEW EXAMPLES (10-20 words):")
    print("-" * 80)
    
    # Get all negative interface reviews and format them properly
    examples = []
    
    for _, review in negative_interface.iterrows():
        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
        rating = review['rating']
        
        if pd.notna(content) and content:
            # Clean and format the content
            content_str = str(content).strip()
            
            # Count words
            word_count = len(content_str.split())
            
            # Only include reviews with 10-20 words or close to that range
            if 8 <= word_count <= 25:  # Allow some flexibility
                examples.append({
                    'rating': rating,
                    'content': content_str,
                    'word_count': word_count
                })
    
    # Sort by rating (lowest first) and show top examples
    examples.sort(key=lambda x: x['rating'])
    
    print("🔴 1-STAR REVIEWS:")
    print("-" * 40)
    one_star_examples = [ex for ex in examples if ex['rating'] == 1][:10]
    for i, example in enumerate(one_star_examples, 1):
        print(f"{i}. Rating {example['rating']}/5 ({example['word_count']} words):")
        print(f"   \"{example['content']}\"")
        print()
    
    print("🟡 2-STAR REVIEWS:")
    print("-" * 40)
    two_star_examples = [ex for ex in examples if ex['rating'] == 2][:5]
    for i, example in enumerate(two_star_examples, 1):
        print(f"{i}. Rating {example['rating']}/5 ({example['word_count']} words):")
        print(f"   \"{example['content']}\"")
        print()
    
    print("🟠 3-STAR REVIEWS:")
    print("-" * 40)
    three_star_examples = [ex for ex in examples if ex['rating'] == 3][:5]
    for i, example in enumerate(three_star_examples, 1):
        print(f"{i}. Rating {example['rating']}/5 ({example['word_count']} words):")
        print(f"   \"{example['content']}\"")
        print()
    
    # Also show some specific issue categories
    print("📊 ISSUE CATEGORIES WITH EXAMPLES:")
    print("-" * 40)
    
    # UI/Interface issues
    ui_issues = []
    for _, review in negative_interface.iterrows():
        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
        if pd.notna(content) and any(word in str(content).lower() for word in ['ui', 'interface', 'design', 'layout']):
            ui_issues.append(str(content)[:100] + "...")
    
    print("🎨 UI/Interface Issues:")
    for i, issue in enumerate(ui_issues[:5], 1):
        print(f"{i}. \"{issue}\"")
    
    print()
    
    # Functionality issues
    func_issues = []
    for _, review in negative_interface.iterrows():
        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
        if pd.notna(content) and any(word in str(content).lower() for word in ['function', 'feature', 'work', 'broken']):
            func_issues.append(str(content)[:100] + "...")
    
    print("⚙️ Functionality Issues:")
    for i, issue in enumerate(func_issues[:5], 1):
        print(f"{i}. \"{issue}\"")
    
    print()
    
    # Navigation issues
    nav_issues = []
    for _, review in negative_interface.iterrows():
        content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
        if pd.notna(content) and any(word in str(content).lower() for word in ['navigation', 'menu', 'click', 'tap']):
            nav_issues.append(str(content)[:100] + "...")
    
    print("🧭 Navigation Issues:")
    for i, issue in enumerate(nav_issues[:5], 1):
        print(f"{i}. \"{issue}\"")

if __name__ == "__main__":
    get_welab_interface_examples() 