#!/usr/bin/env python3
"""
Combine All Review Data
- Extract reviews from old analysis files
- Combine with new scraped data
- Remove duplicates and create comprehensive database
"""

import pandas as pd
import os
import json
from datetime import datetime

class ReviewDataCombiner:
    def __init__(self):
        # Load bank configurations
        with open('config/apps.json', 'r') as f:
            apps_config = json.load(f)
        self.banks = apps_config['apps']
        
        # Define old analysis files with more reviews
        self.old_analysis_files = {
            'welab_bank': [
                './analysis/WeLab_Bank_sentiment_analysis_IMPROVED.xlsx',
                './analysis/WeLab_Bank_sentiment_analysis.xlsx'
            ],
            'za_bank': [
                './analysis/ZA_Bank_sentiment_analysis_IMPROVED.xlsx'
            ],
            'mox_bank': [
                './analysis/Mox_Bank_sentiment_analysis_IMPROVED.xlsx'
            ]
        }
    
    def extract_reviews_from_old_file(self, file_path, bank_key):
        """Extract reviews from old analysis file"""
        try:
            df = pd.read_excel(file_path)
            
            # Check if it has the required columns
            required_cols = ['content', 'rating']
            if not all(col in df.columns for col in required_cols):
                print(f"   ⚠️ Missing required columns in {file_path}")
                return None
            
            # Create standardized review format
            reviews = []
            for _, row in df.iterrows():
                review = {
                    'review_id': row.get('review_id', f"old_{len(reviews)}"),
                    'title': row.get('title', ''),
                    'content': row['content'],
                    'rating': row['rating'],
                    'date': row.get('date', ''),
                    'author': row.get('author', ''),
                    'platform': row.get('platform', 'Unknown'),
                    'bank': self.banks[bank_key]['name'],
                    'bank_key': bank_key,
                    'source': 'old_analysis',
                    'scraped_at': datetime.now()
                }
                reviews.append(review)
            
            print(f"   ✅ Extracted {len(reviews)} reviews from {file_path}")
            return pd.DataFrame(reviews)
            
        except Exception as e:
            print(f"   ❌ Failed to extract from {file_path}: {e}")
            return None
    
    def load_new_scraped_data(self, bank_key):
        """Load newly scraped data"""
        new_files = [
            f'./output/{bank_key}_raw_reviews.xlsx',
            f'./output/{bank_key}_translated_reviews.xlsx',
            f'./output/enhanced_scraping_*/{bank_key}_enhanced_reviews.xlsx'
        ]
        
        new_data = []
        for file_pattern in new_files:
            if '*' in file_pattern:
                # Handle wildcard patterns
                import glob
                files = glob.glob(file_pattern)
                for file_path in files:
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_excel(file_path)
                            df['source'] = 'new_scraped'
                            new_data.append(df)
                            print(f"   ✅ Loaded {len(df)} reviews from {file_path}")
                        except Exception as e:
                            print(f"   ❌ Failed to load {file_path}: {e}")
            else:
                if os.path.exists(file_pattern):
                    try:
                        df = pd.read_excel(file_pattern)
                        df['source'] = 'new_scraped'
                        new_data.append(df)
                        print(f"   ✅ Loaded {len(df)} reviews from {file_pattern}")
                    except Exception as e:
                        print(f"   ❌ Failed to load {file_pattern}: {e}")
        
        if new_data:
            return pd.concat(new_data, ignore_index=True)
        return None
    
    def combine_bank_data(self, bank_key):
        """Combine all data for a single bank"""
        print(f"\n🏦 Combining data for {self.banks[bank_key]['name']}...")
        
        all_reviews = []
        
        # Load new scraped data
        new_data = self.load_new_scraped_data(bank_key)
        if new_data is not None:
            all_reviews.append(new_data)
            print(f"   📊 New scraped data: {len(new_data)} reviews")
        
        # Extract from old analysis files
        if bank_key in self.old_analysis_files:
            for old_file in self.old_analysis_files[bank_key]:
                if os.path.exists(old_file):
                    old_data = self.extract_reviews_from_old_file(old_file, bank_key)
                    if old_data is not None:
                        all_reviews.append(old_data)
                        print(f"   📊 Old analysis data: {len(old_data)} reviews")
        
        if not all_reviews:
            print(f"   ⚠️ No data found for {self.banks[bank_key]['name']}")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_reviews, ignore_index=True)
        
        # Remove duplicates
        print(f"   🔄 Removing duplicates...")
        initial_count = len(combined_df)
        
        # Remove duplicates based on review_id first, then content
        if 'review_id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['review_id'], keep='first')
        
        # Also remove duplicates based on content (for reviews without IDs)
        combined_df = combined_df.drop_duplicates(subset=['content'], keep='first')
        
        final_count = len(combined_df)
        removed_count = initial_count - final_count
        print(f"   ✅ Removed {removed_count} duplicates ({initial_count} → {final_count})")
        
        return combined_df
    
    def run_combination(self):
        """Run the complete data combination process"""
        print("🚀 COMBINING ALL REVIEW DATA")
        print("=" * 60)
        print("📋 Combining new scraped data with old analysis files")
        print("🔄 Removing duplicates and creating comprehensive database")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"output/combined_data_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        all_bank_reviews = []
        
        for bank_key, bank_config in self.banks.items():
            print(f"\n{'='*40}")
            print(f"🏦 COMBINING: {bank_config['name'].upper()}")
            print(f"{'='*40}")
            
            # Combine data for this bank
            combined_reviews = self.combine_bank_data(bank_key)
            
            if combined_reviews is not None and not combined_reviews.empty:
                # Save combined data for this bank
                combined_file = os.path.join(output_dir, f"{bank_key}_combined_reviews.xlsx")
                combined_reviews.to_excel(combined_file, index=False)
                print(f"💾 Saved combined reviews: {combined_file}")
                
                all_bank_reviews.append(combined_reviews)
                
                # Show breakdown
                if 'platform' in combined_reviews.columns:
                    platform_breakdown = combined_reviews['platform'].value_counts()
                    print(f"📱 Platform breakdown:")
                    for platform, count in platform_breakdown.items():
                        print(f"   {platform}: {count} reviews")
                
                if 'source' in combined_reviews.columns:
                    source_breakdown = combined_reviews['source'].value_counts()
                    print(f"📄 Source breakdown:")
                    for source, count in source_breakdown.items():
                        print(f"   {source}: {count} reviews")
                
            else:
                print(f"⚠️ No data to combine for {bank_config['name']}")
        
        # Create comprehensive combined database
        if all_bank_reviews:
            print(f"\n📊 CREATING COMPREHENSIVE COMBINED DATABASE")
            print("=" * 50)
            
            comprehensive_combined = pd.concat(all_bank_reviews, ignore_index=True)
            
            # Save comprehensive database
            comprehensive_file = os.path.join(output_dir, "comprehensive_combined_database.xlsx")
            comprehensive_combined.to_excel(comprehensive_file, index=False)
            
            print(f"📊 COMPREHENSIVE COMBINED DATABASE CREATED!")
            print(f"📄 File: {comprehensive_file}")
            print(f"📊 Total reviews: {len(comprehensive_combined)}")
            print(f"🏦 Banks: {comprehensive_combined['bank'].nunique()}")
            
            if 'platform' in comprehensive_combined.columns:
                print(f"📱 Platforms: {comprehensive_combined['platform'].value_counts().to_dict()}")
            
            if 'source' in comprehensive_combined.columns:
                print(f"📄 Sources: {comprehensive_combined['source'].value_counts().to_dict()}")
            
            # Summary by bank
            print(f"\n📋 COMBINED SUMMARY BY BANK:")
            if 'platform' in comprehensive_combined.columns:
                bank_summary = comprehensive_combined.groupby(['bank', 'platform']).size().unstack(fill_value=0)
                print(bank_summary)
            
            print(f"\n🎉 DATA COMBINATION COMPLETED!")
            print(f"📁 Output directory: {output_dir}")
            
        else:
            print("❌ No data was successfully combined")

def main():
    """Main function"""
    combiner = ReviewDataCombiner()
    combiner.run_combination()

if __name__ == "__main__":
    main() 