#!/usr/bin/env python3
"""
Get actual Play Store review counts for 8 Hong Kong virtual banks with correct package names
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
    
    # More accurate package names for Hong Kong virtual banks
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
    
    # Alternative package names to try
    alternative_packages = {
        'ZA Bank': ['com.zabank.mobile', 'com.zabank', 'com.zabank.app'],
        'WeLab Bank': ['com.welabbank.mobile', 'com.welabbank', 'com.welabbank.app'],
        'Fusion Bank': ['com.fusionbank.mobile', 'com.fusionbank', 'com.fusionbank.app'],
        'Mox Bank': ['com.moxbank.mobile', 'com.moxbank', 'com.moxbank.app'],
        'livi Bank': ['com.livibank.mobile', 'com.livibank', 'com.livibank.app'],
        'PAO Bank': ['com.paobank.mobile', 'com.paobank', 'com.paobank.app'],
        'Ant Bank': ['com.antbank.mobile', 'com.antbank', 'com.antbank.app'],
        'Airstar Bank': ['com.airstarbank.mobile', 'com.airstarbank', 'com.airstarbank.app']
    }
    
    print("📊 PLAY STORE REVIEW COUNTS FOR 8 BANKS")
    print("=" * 70)
    print(f"{'Bank':<15} {'Package Name':<30} {'Reviews':<10} {'Rating':<10}")
    print("=" * 70)
    
    total_reviews = 0
    
    for bank, packages in alternative_packages.items():
        print(f"Fetching {bank}...")
        best_review_count = 0
        best_rating = 0.0
        best_package = packages[0]
        
        for package in packages:
            review_count, rating = get_playstore_review_count(package)
            if review_count > best_review_count:
                best_review_count = review_count
                best_rating = rating
                best_package = package
            time.sleep(0.5)  # Small delay between attempts
        
        total_reviews += best_review_count
        print(f"{bank:<15} {best_package:<30} {best_review_count:<10} {best_rating:<10.1f}")
        
        time.sleep(1)  # Delay between banks
    
    print("=" * 70)
    print(f"Total Play Store Reviews: {total_reviews}")
    print(f"📅 Data retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    get_all_playstore_reviews() 