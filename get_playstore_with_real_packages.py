#!/usr/bin/env python3
"""
Get Play Store review counts using the actual package names from apps.json
"""

import requests
import re
import time
import json

def get_playstore_review_count(package_name):
    """Get Play Store review count for a specific package"""
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
            name_pattern = r'"name":\s*"([^"]+)"'
            
            review_match = re.search(review_pattern, response.text)
            rating_match = re.search(rating_pattern, response.text)
            name_match = re.search(name_pattern, response.text)
            
            if review_match:
                review_count = int(review_match.group(1))
                rating = float(rating_match.group(1)) if rating_match else 0.0
                app_name = name_match.group(1) if name_match else "Unknown"
                return review_count, rating, app_name
            else:
                return 0, 0.0, "No data"
        else:
            return 0, 0.0, "Error"
            
    except Exception as e:
        return 0, 0.0, f"Error: {e}"

def get_playstore_reviews_with_real_packages():
    """Get Play Store reviews using the actual package names from apps.json"""
    
    # Package names from apps.json
    banks = {
        'PAO Bank': 'com.paobank.mobilebank.personal',
        'Ant Bank': 'com.hk.ios.phone.portal',  # Note: enabled: false in config
        'Airstar Bank': 'com.airstarbank.mobilebanking',
        'Fusion Bank': 'com.fusionbank.vb',
        'livi Bank': 'com.livibank.hk',
        'WeLab Bank': 'welab.bank',
        'ZA Bank': 'com.zhongan.ibank',
        'Mox Bank': 'com.mox.app'
    }
    
    print("📊 PLAY STORE REVIEW COUNTS USING REAL PACKAGE NAMES")
    print("=" * 80)
    print(f"{'Bank':<15} {'Package Name':<35} {'App Name':<25} {'Reviews':<10} {'Rating':<10}")
    print("=" * 80)
    
    results = {}
    total_reviews = 0
    
    for bank, package in banks.items():
        print(f"Fetching {bank}...")
        review_count, rating, app_name = get_playstore_review_count(package)
        
        results[bank] = {
            'package': package,
            'name': app_name,
            'reviews': review_count,
            'rating': rating
        }
        
        total_reviews += review_count
        
        print(f"{bank:<15} {package:<35} {app_name:<25} {review_count:<10} {rating:<10.1f}")
        
        time.sleep(1)  # Delay between requests
    
    print("=" * 80)
    print(f"Total Play Store Reviews: {total_reviews}")
    print(f"📅 Data retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to JSON
    with open('playstore_real_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: playstore_real_results.json")
    
    return results

if __name__ == "__main__":
    get_playstore_reviews_with_real_packages() 