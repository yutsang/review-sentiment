#!/usr/bin/env python3
"""
Check specific packages that were found in the search
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

def check_specific_packages():
    """Check specific packages that were found"""
    
    # Packages found in the search that might be relevant
    packages_to_check = {
        'ZA Bank': ['com.zabank.mobile', 'com.zabank', 'com.zabank.app'],
        'WeLab Bank': ['com.welabbank.mobile', 'com.welabbank', 'welab.bank'],
        'Fusion Bank': ['com.fusionbank.mobile', 'com.fusionbank', 'com.fusionbank.app'],
        'Mox Bank': ['com.mox.app', 'com.moxbank.mobile', 'com.moxbank'],
        'livi Bank': ['com.livibank.hk', 'com.livibank.mobile', 'com.livibank'],
        'PAO Bank': ['com.paobank.mobile', 'com.paobank', 'com.paobank.app'],
        'Ant Bank': ['com.alipay.antbank.hk.portal', 'com.antbank.mobile', 'com.antbank'],
        'Airstar Bank': ['com.airstarbank.softtoken', 'com.airstarbank.mobile', 'com.airstarbank']
    }
    
    print("🔍 CHECKING SPECIFIC PACKAGES")
    print("=" * 80)
    
    results = {}
    
    for bank, packages in packages_to_check.items():
        print(f"\n🏦 {bank}:")
        best_review_count = 0
        best_rating = 0.0
        best_package = "Not found"
        best_name = "Unknown"
        
        for package in packages:
            print(f"  Checking: {package}")
            review_count, rating, app_name = get_playstore_review_count(package)
            
            if review_count > 0:
                print(f"    ✅ Found: {app_name} - {review_count} reviews, {rating:.1f} rating")
                if review_count > best_review_count:
                    best_review_count = review_count
                    best_rating = rating
                    best_package = package
                    best_name = app_name
            else:
                print(f"    ❌ No data")
            
            time.sleep(1)
        
        results[bank] = {
            'package': best_package,
            'name': best_name,
            'reviews': best_review_count,
            'rating': best_rating
        }
        
        print(f"  🎯 Best result: {best_name} ({best_package}) - {best_review_count} reviews")
    
    # Print final results
    print(f"\n📊 FINAL RESULTS")
    print("=" * 80)
    print(f"{'Bank':<15} {'App Name':<30} {'Package':<25} {'Reviews':<10} {'Rating':<10}")
    print("=" * 80)
    
    total_reviews = 0
    for bank, data in results.items():
        print(f"{bank:<15} {data['name']:<30} {data['package']:<25} {data['reviews']:<10} {data['rating']:<10.1f}")
        total_reviews += data['reviews']
    
    print("=" * 80)
    print(f"Total Play Store Reviews: {total_reviews}")
    
    return results

if __name__ == "__main__":
    check_specific_packages() 