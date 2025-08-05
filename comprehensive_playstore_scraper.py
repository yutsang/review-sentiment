#!/usr/bin/env python3
"""
Comprehensive Play Store scraper for Hong Kong virtual banks
"""

import requests
import re
import time
import json

def search_playstore_apps(query):
    """Search for apps on Play Store"""
    try:
        url = f"https://play.google.com/store/search?q={query}&c=apps&gl=HK"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Look for app links
            app_pattern = r'/store/apps/details\?id=([^&"]+)'
            apps = re.findall(app_pattern, response.text)
            return list(set(apps))  # Remove duplicates
        return []
    except:
        return []

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
    """Get Play Store reviews for all 8 banks with comprehensive search"""
    
    # Bank names and search terms
    banks = {
        'ZA Bank': ['ZA Bank', 'ZABank', 'ZABank Mobile'],
        'WeLab Bank': ['WeLab Bank', 'WeLabBank', 'WeLab Bank Mobile'],
        'Fusion Bank': ['Fusion Bank', 'FusionBank', 'Fusion Bank Mobile'],
        'Mox Bank': ['Mox Bank', 'MoxBank', 'Mox Bank Mobile'],
        'livi Bank': ['livi Bank', 'liviBank', 'livi Bank Mobile'],
        'PAO Bank': ['PAO Bank', 'PAOBank', 'PAO Bank Mobile'],
        'Ant Bank': ['Ant Bank', 'AntBank', 'Ant Bank Mobile'],
        'Airstar Bank': ['Airstar Bank', 'AirstarBank', 'Airstar Bank Mobile']
    }
    
    # Known package name variations
    known_packages = {
        'ZA Bank': ['com.zabank.mobile', 'com.zabank', 'com.zabank.app', 'com.zabank.bank'],
        'WeLab Bank': ['com.welabbank.mobile', 'com.welabbank', 'com.welabbank.app', 'com.welabbank.bank'],
        'Fusion Bank': ['com.fusionbank.mobile', 'com.fusionbank', 'com.fusionbank.app', 'com.fusionbank.bank'],
        'Mox Bank': ['com.moxbank.mobile', 'com.moxbank', 'com.moxbank.app', 'com.moxbank.bank'],
        'livi Bank': ['com.livibank.mobile', 'com.livibank', 'com.livibank.app', 'com.livibank.bank'],
        'PAO Bank': ['com.paobank.mobile', 'com.paobank', 'com.paobank.app', 'com.paobank.bank'],
        'Ant Bank': ['com.antbank.mobile', 'com.antbank', 'com.antbank.app', 'com.antbank.bank'],
        'Airstar Bank': ['com.airstarbank.mobile', 'com.airstarbank', 'com.airstarbank.app', 'com.airstarbank.bank']
    }
    
    print("🔍 COMPREHENSIVE PLAY STORE SEARCH")
    print("=" * 80)
    
    results = {}
    total_reviews = 0
    
    for bank, search_terms in banks.items():
        print(f"\n🏦 Searching for {bank}...")
        
        best_review_count = 0
        best_rating = 0.0
        best_package = "Not found"
        found_packages = []
        
        # Try known packages first
        for package in known_packages.get(bank, []):
            print(f"  Trying package: {package}")
            review_count, rating = get_playstore_review_count(package)
            if review_count > 0:
                found_packages.append((package, review_count, rating))
                if review_count > best_review_count:
                    best_review_count = review_count
                    best_rating = rating
                    best_package = package
            time.sleep(1)
        
        # Search for apps if no known packages worked
        if best_review_count == 0:
            for search_term in search_terms:
                print(f"  Searching for: {search_term}")
                found_apps = search_playstore_apps(search_term.replace(' ', '+'))
                
                for app_package in found_apps[:3]:  # Try first 3 results
                    print(f"    Found app: {app_package}")
                    review_count, rating = get_playstore_review_count(app_package)
                    if review_count > 0:
                        found_packages.append((app_package, review_count, rating))
                        if review_count > best_review_count:
                            best_review_count = review_count
                            best_rating = rating
                            best_package = app_package
                    time.sleep(1)
        
        results[bank] = {
            'package': best_package,
            'reviews': best_review_count,
            'rating': best_rating,
            'found_packages': found_packages
        }
        
        total_reviews += best_review_count
        
        print(f"  ✅ {bank}: {best_review_count} reviews, {best_rating:.1f} rating")
    
    # Print final results
    print(f"\n📊 FINAL RESULTS")
    print("=" * 80)
    print(f"{'Bank':<15} {'Package':<35} {'Reviews':<10} {'Rating':<10}")
    print("=" * 80)
    
    for bank, data in results.items():
        print(f"{bank:<15} {data['package']:<35} {data['reviews']:<10} {data['rating']:<10.1f}")
    
    print("=" * 80)
    print(f"Total Play Store Reviews: {total_reviews}")
    print(f"📅 Data retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to JSON
    with open('playstore_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: playstore_results.json")
    
    return results

if __name__ == "__main__":
    get_all_playstore_reviews() 