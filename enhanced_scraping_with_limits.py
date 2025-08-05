#!/usr/bin/env python3
"""
Enhanced Scraping with No Limits
- Remove all scraping limits to get maximum reviews
- Add retry logic and multiple attempts
- Check for old data files and combine them
"""

import pandas as pd
import os
import json
from datetime import datetime
from src.scrapers.app_store import AppStoreScraper
from src.scrapers.play_store import PlayStoreScraper
from src.utils.config import load_config

class EnhancedScraper:
    def __init__(self):
        self.config = load_config()
        
        # Load bank configurations
        with open('config/apps.json', 'r') as f:
            apps_config = json.load(f)
        self.banks = apps_config['apps']
        
        # Enhanced config with no limits
        self.enhanced_config = self.config.copy()
        self.enhanced_config['rate_limiting'] = {
            'app_store': {
                'between_requests': 0.5,  # Reduced delay
                'between_sort_methods': 1.0,
                'between_countries': 2.0
            },
            'play_store': {
                'between_requests': 0.5,
                'between_sort_methods': 1.0,
                'between_countries': 2.0
            }
        }
    
    def find_old_review_files(self):
        """Find any old review files that might have more data"""
        print("🔍 Searching for old review files...")
        
        old_files = []
        
        # Search for any Excel files with review data
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.xlsx') and ('review' in file.lower() or any(bank in file.lower() for bank in self.banks.keys())):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_excel(file_path)
                        if 'content' in df.columns and 'rating' in df.columns:
                            old_files.append({
                                'path': file_path,
                                'size': len(df),
                                'columns': list(df.columns)
                            })
                            print(f"   📄 Found: {file_path} ({len(df)} rows)")
                    except:
                        continue
        
        return old_files
    
    def scrape_bank_with_enhanced_limits(self, bank_key, bank_config):
        """Scrape bank with enhanced limits and retry logic"""
        print(f"\n🏦 Enhanced scraping for {bank_config['name']}...")
        
        all_reviews = []
        
        # Scrape App Store with enhanced limits
        if bank_config['app_store']['enabled']:
            print(f"   📱 Enhanced App Store scraping...")
            
            # Try multiple attempts with different configurations
            for attempt in range(3):
                try:
                    print(f"      Attempt {attempt + 1}/3...")
                    
                    # Create enhanced app config
                    enhanced_app_config = bank_config.copy()
                    enhanced_app_config['app_store']['max_reviews'] = 1000  # Increased limit
                    
                    app_store_scraper = AppStoreScraper(self.enhanced_config, enhanced_app_config)
                    app_store_reviews = app_store_scraper.scrape_reviews()
                    
                    print(f"      ✅ Got {len(app_store_reviews)} App Store reviews")
                    
                    # Convert to dictionaries
                    for review in app_store_reviews:
                        review_dict = {
                            'review_id': review.review_id,
                            'title': review.title,
                            'content': review.content,
                            'rating': review.rating,
                            'date': review.date,
                            'author': review.author,
                            'platform': 'App Store',
                            'bank': bank_config['name'],
                            'bank_key': bank_key,
                            'scraped_at': datetime.now()
                        }
                        all_reviews.append(review_dict)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"      ❌ Attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        print(f"      🔄 Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
        
        # Scrape Play Store with enhanced limits
        if bank_config['play_store']['enabled']:
            print(f"   🤖 Enhanced Play Store scraping...")
            
            # Try multiple attempts
            for attempt in range(3):
                try:
                    print(f"      Attempt {attempt + 1}/3...")
                    
                    # Create enhanced app config
                    enhanced_app_config = bank_config.copy()
                    enhanced_app_config['play_store']['max_reviews'] = 1000  # Increased limit
                    
                    play_store_scraper = PlayStoreScraper(self.enhanced_config, enhanced_app_config)
                    play_store_reviews = play_store_scraper.scrape_reviews()
                    
                    print(f"      ✅ Got {len(play_store_reviews)} Play Store reviews")
                    
                    # Convert to dictionaries
                    for review in play_store_reviews:
                        review_dict = {
                            'review_id': review.review_id,
                            'title': review.title,
                            'content': review.content,
                            'rating': review.rating,
                            'date': review.date,
                            'author': review.author,
                            'platform': 'Play Store',
                            'bank': bank_config['name'],
                            'bank_key': bank_key,
                            'scraped_at': datetime.now()
                        }
                        all_reviews.append(review_dict)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"      ❌ Attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        print(f"      🔄 Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
        
        return pd.DataFrame(all_reviews)
    
    def combine_with_old_data(self, new_df, bank_key):
        """Combine new data with any old data found"""
        print(f"   🔄 Looking for old data to combine...")
        
        old_files = self.find_old_review_files()
        combined_df = new_df.copy()
        
        for old_file in old_files:
            if bank_key in old_file['path'].lower():
                try:
                    old_df = pd.read_excel(old_file['path'])
                    
                    # Check if it has the right columns
                    if 'content' in old_df.columns and 'rating' in old_df.columns:
                        print(f"      📄 Combining with: {old_file['path']} ({len(old_df)} rows)")
                        
                        # Add missing columns if needed
                        if 'platform' not in old_df.columns:
                            old_df['platform'] = 'Unknown'
                        if 'bank' not in old_df.columns:
                            old_df['bank'] = self.banks[bank_key]['name']
                        if 'bank_key' not in old_df.columns:
                            old_df['bank_key'] = bank_key
                        if 'scraped_at' not in old_df.columns:
                            old_df['scraped_at'] = datetime.now()
                        
                        # Combine dataframes
                        combined_df = pd.concat([combined_df, old_df], ignore_index=True)
                        
                        # Remove duplicates based on review_id or content
                        if 'review_id' in combined_df.columns:
                            combined_df = combined_df.drop_duplicates(subset=['review_id'], keep='first')
                        else:
                            combined_df = combined_df.drop_duplicates(subset=['content'], keep='first')
                        
                        print(f"      ✅ Combined total: {len(combined_df)} unique reviews")
                        
                except Exception as e:
                    print(f"      ❌ Failed to combine with {old_file['path']}: {e}")
        
        return combined_df
    
    def run_enhanced_scraping(self):
        """Run enhanced scraping for all banks"""
        print("🚀 ENHANCED SCRAPING WITH NO LIMITS")
        print("=" * 60)
        print("📋 Processing 8 banks with enhanced limits")
        print("🔄 Multiple attempts and retry logic")
        print("📄 Combining with old data files")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"output/enhanced_scraping_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        all_bank_reviews = []
        
        for bank_key, bank_config in self.banks.items():
            print(f"\n{'='*40}")
            print(f"🏦 ENHANCED SCRAPING: {bank_config['name'].upper()}")
            print(f"{'='*40}")
            
            # Enhanced scraping
            bank_reviews = self.scrape_bank_with_enhanced_limits(bank_key, bank_config)
            
            if not bank_reviews.empty:
                print(f"📊 Initial scraping: {len(bank_reviews)} reviews")
                
                # Combine with old data
                combined_reviews = self.combine_with_old_data(bank_reviews, bank_key)
                
                print(f"📊 After combining: {len(combined_reviews)} reviews")
                
                # Save enhanced data
                enhanced_file = os.path.join(output_dir, f"{bank_key}_enhanced_reviews.xlsx")
                combined_reviews.to_excel(enhanced_file, index=False)
                print(f"💾 Saved enhanced reviews: {enhanced_file}")
                
                all_bank_reviews.append(combined_reviews)
                
                # Show breakdown
                if 'platform' in combined_reviews.columns:
                    platform_breakdown = combined_reviews['platform'].value_counts()
                    print(f"📱 Platform breakdown:")
                    for platform, count in platform_breakdown.items():
                        print(f"   {platform}: {count} reviews")
                
            else:
                print(f"⚠️ No reviews found for {bank_config['name']}")
        
        # Create comprehensive enhanced database
        if all_bank_reviews:
            print(f"\n📊 CREATING ENHANCED COMPREHENSIVE DATABASE")
            print("=" * 50)
            
            enhanced_combined = pd.concat(all_bank_reviews, ignore_index=True)
            
            # Save enhanced comprehensive database
            enhanced_db_file = os.path.join(output_dir, "enhanced_comprehensive_database.xlsx")
            enhanced_combined.to_excel(enhanced_db_file, index=False)
            
            print(f"📊 ENHANCED DATABASE CREATED!")
            print(f"📄 File: {enhanced_db_file}")
            print(f"📊 Total reviews: {len(enhanced_combined)}")
            print(f"🏦 Banks: {enhanced_combined['bank'].nunique()}")
            
            if 'platform' in enhanced_combined.columns:
                print(f"📱 Platforms: {enhanced_combined['platform'].value_counts().to_dict()}")
            
            # Summary by bank
            print(f"\n📋 ENHANCED SUMMARY BY BANK:")
            if 'platform' in enhanced_combined.columns:
                bank_summary = enhanced_combined.groupby(['bank', 'platform']).size().unstack(fill_value=0)
                print(bank_summary)
            
            print(f"\n🎉 ENHANCED SCRAPING COMPLETED!")
            print(f"📁 Output directory: {output_dir}")
            
        else:
            print("❌ No reviews were successfully scraped")

def main():
    """Main function"""
    scraper = EnhancedScraper()
    scraper.run_enhanced_scraping()

if __name__ == "__main__":
    main() 