#!/usr/bin/env python3
"""
Analyze major issues for onboarding, platform, and interface categories for three banks
"""

import pandas as pd
from collections import Counter

def analyze_major_issues():
    """Analyze major issues for onboarding, platform, and interface categories"""
    
    # Load the data
    df = pd.read_excel('output/combined_data_20250805_104554/FINAL_COMPREHENSIVE_DATABASE.xlsx')
    
    # Define categories to analyze
    categories = {
        'onboarding_verification': 'Onboarding & Verification',
        'platform_performance': 'Platform Performance & Reliability',
        'interface_functionality': 'Interface & Functionality'
    }
    
    # Three banks
    banks = ['WeLab Bank', 'Mox Bank', 'ZA Bank']
    
    print("🔍 MAJOR ISSUES ANALYSIS - THREE CATEGORIES")
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
                print(f"Total Reviews: {len(cat_reviews)}")
                print(f"Negative Reviews: {len(negative_reviews)}")
                print(f"Average Rating: {cat_reviews['rating'].mean():.2f}/5.0")
                
                # Extract major issues from negative reviews
                issues = []
                
                for _, review in negative_reviews.iterrows():
                    content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                    if pd.notna(content) and content:
                        content_lower = str(content).lower()
                        
                        # Onboarding & Verification issues
                        if category_key == 'onboarding_verification':
                            if 'face id' in content_lower or 'face verification' in content_lower:
                                issues.append("Face ID verification problems")
                            if 'id card' in content_lower or 'identity' in content_lower:
                                issues.append("ID card recognition issues")
                            if 'phone number' in content_lower or 'verification' in content_lower:
                                issues.append("Phone number verification issues")
                            if 'create account' in content_lower or 'sign up' in content_lower:
                                issues.append("Account creation difficulties")
                            if 'upload' in content_lower:
                                issues.append("Document upload problems")
                            if 'video' in content_lower:
                                issues.append("Video verification issues")
                        
                        # Platform Performance issues
                        elif category_key == 'platform_performance':
                            if 'crash' in content_lower or 'crashed' in content_lower:
                                issues.append("App crashes frequently")
                            if 'bug' in content_lower or 'error' in content_lower:
                                issues.append("System bugs and errors")
                            if 'slow' in content_lower or 'loading' in content_lower:
                                issues.append("Slow performance and loading")
                            if 'update' in content_lower:
                                issues.append("Update-related issues")
                            if 'connection' in content_lower or 'internet' in content_lower:
                                issues.append("Connection problems")
                            if 'compatible' in content_lower or 'android' in content_lower:
                                issues.append("Device compatibility issues")
                        
                        # Interface & Functionality issues
                        elif category_key == 'interface_functionality':
                            if 'interface' in content_lower or 'ui' in content_lower:
                                issues.append("Poor user interface")
                            if 'function' in content_lower or 'feature' in content_lower:
                                issues.append("Missing or broken features")
                            if 'design' in content_lower or 'layout' in content_lower:
                                issues.append("Poor design and layout")
                            if 'navigation' in content_lower or 'menu' in content_lower:
                                issues.append("Navigation difficulties")
                            if 'screen' in content_lower or 'blank' in content_lower:
                                issues.append("Screen display issues")
                            if 'click' in content_lower or 'tap' in content_lower:
                                issues.append("Touch/click response problems")
                
                # Count and display major issues
                issue_counts = Counter(issues)
                
                print("\nMAJOR ISSUES (from negative reviews):")
                for issue, count in issue_counts.most_common(5):
                    percentage = (count / len(negative_reviews)) * 100
                    print(f"  • {issue}: {count} mentions ({percentage:.1f}%)")
                
                # Show example reviews for top issues
                print("\nEXAMPLE REVIEWS FOR TOP ISSUES:")
                top_issue = issue_counts.most_common(1)[0][0] if issue_counts else "General complaints"
                
                # Get 3 example reviews mentioning the top issue
                examples = []
                for _, review in negative_reviews.iterrows():
                    content = review['translated_content'] if pd.notna(review['translated_content']) else review['content']
                    if pd.notna(content) and content:
                        content_lower = str(content).lower()
                        if any(keyword in content_lower for keyword in top_issue.lower().split()):
                            examples.append(str(content)[:150] + "...")
                            if len(examples) >= 3:
                                break
                
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. \"{example}\"")
                
            else:
                print("No negative reviews found")
            
            print()

if __name__ == "__main__":
    analyze_major_issues() 