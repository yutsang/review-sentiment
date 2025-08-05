#!/usr/bin/env python3
"""
Get actual Play Store review counts for 8 Hong Kong virtual banks
"""

import requests
import re
import time

def get_playstore_review_count(package_name):
    """Get Play Store review count"""
    try:
        url = f"https://play.google.com/store/apps/details?id={package_name}&hl=en&gl=HK"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Look for review count in the HTML
            review_pattern = r'"ratingCount":\s*(\d+)'
            rating_pattern = r'"ratingValue":\s*([\d.]+)'
            
            review_match = re.search(review_pattern, response.text)
            rating_match = re.search(rating_pattern, response.text)
            
            if review_match:
                review_count = int(review_match.group(1))
                rating = float(rating_match.group(1)) if rating_match else 0.0
                return review_count, rating
            else:
                return 0, 0.0
        else:
            return 0, 0.0
            
    except Exception as e:
        print(f"Error for {package_name}: {e}")
        return 0, 0.0

def get_all_playstore_reviews():
    """Get Play Store reviews for all 8 banks"""
    
    banks = {
        'ZA Bank': 'com.zabank.mobile',
        'WeLab Bank': 'com.welabbank.mobile', 
        'Fusion Bank': 'com.fusionbank.mobile',
        'Mox Bank': 'com.moxbank.mobile',
        'livi Bank': 'com.livibank.mobile',
        'PAO Bank': 'com.paobank.mobile',
        'Ant Bank': 'com.antbank.mobile',
        'Airstar Bank': 'com.airstarbank.mobile'
    }
    
    print("📊 PLAY STORE REVIEW COUNTS FOR 8 BANKS")
    print("=" * 60)
    print(f"{'Bank':<15} {'Package Name':<25} {'Reviews':<10} {'Rating':<10}")
    print("=" * 60)
    
    total_reviews = 0
    
    for bank, package in banks.items():
        print(f"Fetching {bank}...")
        review_count, rating = get_playstore_review_count(package)
        total_reviews += review_count
        
        print(f"{bank:<15} {package:<25} {review_count:<10} {rating:<10.1f}")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    print("=" * 60)
    print(f"Total Play Store Reviews: {total_reviews}")
    print(f"📅 Data retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    get_all_playstore_reviews() 