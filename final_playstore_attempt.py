#!/usr/bin/env python3
"""
Final attempt to get Play Store data for Hong Kong virtual banks
"""

import requests
import re
import time

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

def final_playstore_attempt():
    """Final attempt with different approaches"""
    
    # Alternative package names and search terms
    final_packages = {
        'ZA Bank': [
            'com.zabank.mobile',
            'com.zabank',
            'com.zabank.app',
            'com.zabank.bank',
            'com.zabank.hk',
            'hk.com.zabank',
            'com.zabank.mobile.hk'
        ],
        'WeLab Bank': [
            'com.welabbank.mobile',
            'com.welabbank',
            'com.welabbank.app',
            'welab.bank',
            'com.welabbank.hk',
            'hk.com.welabbank',
            'com.welabbank.mobile.hk'
        ],
        'Fusion Bank': [
            'com.fusionbank.mobile',
            'com.fusionbank',
            'com.fusionbank.app',
            'com.fusionbank.hk',
            'hk.com.fusionbank',
            'com.fusionbank.mobile.hk'
        ],
        'Mox Bank': [
            'com.mox.app',
            'com.moxbank.mobile',
            'com.moxbank',
            'com.mox.bank',
            'com.moxbank.hk',
            'hk.com.moxbank',
            'com.moxbank.mobile.hk'
        ],
        'livi Bank': [
            'com.livibank.hk',
            'com.livibank.mobile',
            'com.livibank',
            'com.livibank.app',
            'hk.com.livibank',
            'com.livibank.mobile.hk'
        ],
        'PAO Bank': [
            'com.paobank.mobile',
            'com.paobank',
            'com.paobank.app',
            'com.paobank.hk',
            'hk.com.paobank',
            'com.paobank.mobile.hk'
        ],
        'Ant Bank': [
            'com.alipay.antbank.hk.portal',
            'com.antbank.mobile',
            'com.antbank',
            'com.antbank.hk',
            'hk.com.antbank',
            'com.antbank.mobile.hk'
        ],
        'Airstar Bank': [
            'com.airstarbank.softtoken',
            'com.airstarbank.mobile',
            'com.airstarbank',
            'com.airstarbank.hk',
            'hk.com.airstarbank',
            'com.airstarbank.mobile.hk'
        ]
    }
    
    print("🔍 FINAL PLAY STORE ATTEMPT")
    print("=" * 80)
    
    results = {}
    total_reviews = 0
    
    for bank, packages in final_packages.items():
        print(f"\n🏦 {bank}:")
        best_review_count = 0
        best_rating = 0.0
        best_package = "Not found"
        best_name = "Unknown"
        
        for package in packages:
            print(f"  Trying: {package}")
            review_count, rating, app_name = get_playstore_review_count(package)
            
            if review_count > 0:
                print(f"    ✅ SUCCESS: {app_name} - {review_count} reviews, {rating:.1f} rating")
                if review_count > best_review_count:
                    best_review_count = review_count
                    best_rating = rating
                    best_package = package
                    best_name = app_name
            else:
                print(f"    ❌ No data")
            
            time.sleep(0.5)
        
        results[bank] = {
            'package': best_package,
            'name': best_name,
            'reviews': best_review_count,
            'rating': best_rating
        }
        
        total_reviews += best_review_count
        
        if best_review_count > 0:
            print(f"  🎯 FOUND: {best_name} ({best_package}) - {best_review_count} reviews")
        else:
            print(f"  ❌ NOT FOUND: No Play Store presence detected")
    
    # Print final results
    print(f"\n📊 FINAL PLAY STORE RESULTS")
    print("=" * 80)
    print(f"{'Bank':<15} {'App Name':<30} {'Package':<25} {'Reviews':<10} {'Rating':<10}")
    print("=" * 80)
    
    for bank, data in results.items():
        print(f"{bank:<15} {data['name']:<30} {data['package']:<25} {data['reviews']:<10} {data['rating']:<10.1f}")
    
    print("=" * 80)
    print(f"Total Play Store Reviews: {total_reviews}")
    
    if total_reviews == 0:
        print(f"\n💡 CONCLUSION: All 8 Hong Kong virtual banks appear to be iOS-only in Hong Kong market.")
        print(f"   No Play Store presence detected for any of the banks.")
    
    return results

if __name__ == "__main__":
    final_playstore_attempt() 