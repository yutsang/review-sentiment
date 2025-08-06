#!/usr/bin/env python3
"""
Fix comprehensive database by filling missing values from reference files
"""

import pandas as pd
import numpy as np
from datetime import datetime

def fix_comprehensive_database():
    """Fix the comprehensive database by filling missing values"""
    
    print("🔧 FIXING COMPREHENSIVE DATABASE")
    print("=" * 60)
    
    # Load the problematic file
    file_path = 'output/combined_data_20250805_104554/comprehensive_combined_database_FIXED.xlsx'
    df = pd.read_excel(file_path)
    
    print(f"Original file: {len(df)} rows")
    print(f"Missing values before fix:")
    print(df.isnull().sum())
    
    # Load reference files
    reference_files = {
        'mox_bank': 'output/combined_data_20250805_104554/mox_bank_combined_reviews.xlsx',
        'za_bank': 'output/combined_data_20250805_104554/za_bank_combined_reviews.xlsx',
        'welab_bank': 'output/combined_data_20250805_104554/welab_bank_combined_reviews.xlsx',
        'livi_bank': 'output/combined_data_20250805_104554/livi_bank_combined_reviews.xlsx',
        'fusion_bank': 'output/combined_data_20250805_104554/fusion_bank_combined_reviews.xlsx',
        'airstar_bank': 'output/combined_data_20250805_104554/airstar_bank_combined_reviews.xlsx',
        'ant_bank': 'output/combined_data_20250805_104554/ant_bank_combined_reviews.xlsx',
        'pao_bank': 'output/combined_data_20250805_104554/pao_bank_combined_reviews.xlsx'
    }
    
    # Create a reference dictionary
    reference_data = {}
    
    for bank_key, file_path in reference_files.items():
        try:
            ref_df = pd.read_excel(file_path)
            print(f"Loaded {bank_key}: {len(ref_df)} rows")
            
            # Create a mapping by review_id
            for _, row in ref_df.iterrows():
                review_id = str(row.get('review_id', ''))
                if review_id:
                    reference_data[review_id] = {
                        'title': row.get('title'),
                        'author': row.get('author'),
                        'platform': row.get('platform'),
                        'bank': row.get('bank'),
                        'bank_key': row.get('bank_key'),
                        'source': row.get('source'),
                        'scraped_at': row.get('scraped_at')
                    }
        except Exception as e:
            print(f"Error loading {bank_key}: {e}")
    
    print(f"Reference data loaded: {len(reference_data)} entries")
    
    # Fix missing values
    fixed_count = 0
    
    for idx, row in df.iterrows():
        review_id = str(row.get('review_id', ''))
        
        if review_id in reference_data:
            ref_data = reference_data[review_id]
            
            # Fix missing values
            if pd.isna(row['title']) and ref_data['title']:
                df.at[idx, 'title'] = ref_data['title']
                fixed_count += 1
            
            if pd.isna(row['author']) and ref_data['author']:
                df.at[idx, 'author'] = ref_data['author']
                fixed_count += 1
            
            if pd.isna(row['platform']) and ref_data['platform']:
                df.at[idx, 'platform'] = ref_data['platform']
                fixed_count += 1
            
            if pd.isna(row['bank']) and ref_data['bank']:
                df.at[idx, 'bank'] = ref_data['bank']
                fixed_count += 1
            
            if pd.isna(row['bank_key']) and ref_data['bank_key']:
                df.at[idx, 'bank_key'] = ref_data['bank_key']
                fixed_count += 1
            
            if pd.isna(row['source']) and ref_data['source']:
                df.at[idx, 'source'] = ref_data['source']
                fixed_count += 1
            
            if pd.isna(row['scraped_at']) and ref_data['scraped_at']:
                df.at[idx, 'scraped_at'] = ref_data['scraped_at']
                fixed_count += 1
    
    print(f"Fixed {fixed_count} missing values")
    
    # Fill remaining missing values with defaults
    print("\nFilling remaining missing values with defaults...")
    
    # Fill missing titles
    missing_titles = df['title'].isnull().sum()
    if missing_titles > 0:
        df['title'] = df['title'].fillna('No Title')
        print(f"Filled {missing_titles} missing titles")
    
    # Fill missing authors
    missing_authors = df['author'].isnull().sum()
    if missing_authors > 0:
        df['author'] = df['author'].fillna('Anonymous')
        print(f"Filled {missing_authors} missing authors")
    
    # Fill missing platforms
    missing_platforms = df['platform'].isnull().sum()
    if missing_platforms > 0:
        # Try to infer platform from bank data
        for idx, row in df.iterrows():
            if pd.isna(row['platform']) and not pd.isna(row['bank']):
                # Default to App Store for most banks
                df.at[idx, 'platform'] = 'App Store'
        print(f"Filled {missing_platforms} missing platforms")
    
    # Fill missing banks
    missing_banks = df['bank'].isnull().sum()
    if missing_banks > 0:
        # Try to infer bank from bank_key
        for idx, row in df.iterrows():
            if pd.isna(row['bank']) and not pd.isna(row['bank_key']):
                bank_mapping = {
                    'mox_bank': 'Mox Bank',
                    'za_bank': 'ZA Bank',
                    'welab_bank': 'WeLab Bank',
                    'livi_bank': 'livi Bank',
                    'fusion_bank': 'Fusion Bank',
                    'airstar_bank': 'Airstar Bank',
                    'ant_bank': 'Ant Bank',
                    'pao_bank': 'PAO Bank'
                }
                if row['bank_key'] in bank_mapping:
                    df.at[idx, 'bank'] = bank_mapping[row['bank_key']]
        print(f"Filled {missing_banks} missing banks")
    
    # Fill missing bank_keys
    missing_bank_keys = df['bank_key'].isnull().sum()
    if missing_bank_keys > 0:
        # Try to infer bank_key from bank
        for idx, row in df.iterrows():
            if pd.isna(row['bank_key']) and not pd.isna(row['bank']):
                bank_key_mapping = {
                    'Mox Bank': 'mox_bank',
                    'ZA Bank': 'za_bank',
                    'WeLab Bank': 'welab_bank',
                    'livi Bank': 'livi_bank',
                    'Fusion Bank': 'fusion_bank',
                    'Airstar Bank': 'airstar_bank',
                    'Ant Bank': 'ant_bank',
                    'PAO Bank': 'pao_bank'
                }
                if row['bank'] in bank_key_mapping:
                    df.at[idx, 'bank_key'] = bank_key_mapping[row['bank']]
        print(f"Filled {missing_bank_keys} missing bank_keys")
    
    # Fill missing sources
    missing_sources = df['source'].isnull().sum()
    if missing_sources > 0:
        df['source'] = df['source'].fillna('Unknown')
        print(f"Filled {missing_sources} missing sources")
    
    # Fill missing scraped_at
    missing_scraped_at = df['scraped_at'].isnull().sum()
    if missing_scraped_at > 0:
        df['scraped_at'] = df['scraped_at'].fillna(datetime.now())
        print(f"Filled {missing_scraped_at} missing scraped_at")
    
    # Fill missing analyzed_at
    missing_analyzed_at = df['analyzed_at'].isnull().sum()
    if missing_analyzed_at > 0:
        df['analyzed_at'] = df['analyzed_at'].fillna(datetime.now())
        print(f"Filled {missing_analyzed_at} missing analyzed_at")
    
    print(f"\nMissing values after fix:")
    print(df.isnull().sum())
    
    # Save the fixed file
    output_file = 'output/combined_data_20250805_104554/comprehensive_combined_database_FIXED_COMPLETE.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n✅ Fixed database saved to: {output_file}")
    
    # Verify the fix
    print(f"\n📊 VERIFICATION:")
    print(f"Total rows: {len(df)}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Banks present: {df['bank'].value_counts().to_dict()}")
    print(f"Platforms present: {df['platform'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    fixed_df = fix_comprehensive_database() 