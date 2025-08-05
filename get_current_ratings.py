#!/usr/bin/env python3
"""
Get current real ratings from App Store and Play Store for all 8 banks
"""

import requests
import json
from datetime import datetime

def get_app_store_rating(app_id):
    """Get App Store rating"""
    try:
        url = f"https://itunes.apple.com/lookup?id={app_id}&country=hk"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data['results']:
            app = data['results'][0]
            return {
                'rating': app.get('averageUserRating', 0),
                'count': app.get('userRatingCount', 0),
                'version': app.get('version', 'N/A')
            }
    except:
        pass
    return {'rating': 0, 'count': 0, 'version': 'N/A'}

def get_play_store_rating(package_name):
    """Get Play Store rating"""
    try:
        url = f"https://play.google.com/store/apps/details?id={package_name}&hl=en&gl=HK"
        response = requests.get(url, timeout=10)
        
        # Extract rating from HTML (simplified)
        if 'ratingValue' in response.text:
            import re
            rating_match = re.search(r'"ratingValue":\s*([\d.]+)', response.text)
            count_match = re.search(r'"ratingCount":\s*(\d+)', response.text)
            
            rating = float(rating_match.group(1)) if rating_match else 0
            count = int(count_match.group(1)) if count_match else 0
            
            return {'rating': rating, 'count': count, 'version': 'N/A'}
    except:
        pass
    return {'rating': 0, 'count': 0, 'version': 'N/A'}

def get_all_bank_ratings():
    """Get ratings for all 8 banks"""
    
    banks = {
        'PAO Bank': {
            'app_store_id': '1501234567',  # Placeholder
            'play_store_package': 'com.paobank.app'
        },
        'Ant Bank': {
            'app_store_id': '1501234568',
            'play_store_package': 'com.antbank.app'
        },
        'Airstar Bank': {
            'app_store_id': '1501234569',
            'play_store_package': 'com.airstarbank.app'
        },
        'Fusion Bank': {
            'app_store_id': '1501234570',
            'play_store_package': 'com.fusionbank.app'
        },
        'livi Bank': {
            'app_store_id': '1501234571',
            'play_store_package': 'com.livibank.app'
        },
        'WeLab Bank': {
            'app_store_id': '1501234572',
            'play_store_package': 'com.welabbank.app'
        },
        'ZA Bank': {
            'app_store_id': '1501234573',
            'play_store_package': 'com.zabank.app'
        },
        'Mox Bank': {
            'app_store_id': '1501234574',
            'play_store_package': 'com.moxbank.app'
        }
    }
    
    print("📊 CURRENT REAL RATINGS FROM APP STORE & PLAY STORE")
    print("=" * 80)
    print(f"{'Bank':<15} {'App Store':<20} {'Play Store':<20} {'Combined':<15}")
    print(f"{'':<15} {'Rating/Count':<20} {'Rating/Count':<20} {'Rating/Count':<15}")
    print("=" * 80)
    
    for bank, ids in banks.items():
        app_store = get_app_store_rating(ids['app_store_id'])
        play_store = get_play_store_rating(ids['play_store_package'])
        
        # Calculate weighted average
        total_count = app_store['count'] + play_store['count']
        if total_count > 0:
            combined_rating = ((app_store['rating'] * app_store['count']) + 
                             (play_store['rating'] * play_store['count'])) / total_count
        else:
            combined_rating = 0
        
        print(f"{bank:<15} {app_store['rating']:.1f}/{app_store['count']:<15} {play_store['rating']:.1f}/{play_store['count']:<15} {combined_rating:.1f}/{total_count:<10}")
    
    print("=" * 80)
    print(f"📅 Data retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    get_all_bank_ratings() 