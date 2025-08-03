#!/usr/bin/env python3

import os
import sys
import pandas as pd
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import functions from full_analysis
sys.path.append('.')
from full_analysis import (
    process_review_worker, create_analysis_charts, create_wordclouds, 
    generate_summary_report, output_phrase_frequencies
)

def run_analysis_for_bank(bank_key: str):
    """Run full analysis for a specific bank"""
    print(f"\n{'='*60}")
    print(f"🏦 Running analysis for {bank_key}")
    print(f"{'='*60}")
    
    # Check if reviews file exists
    reviews_file = f"output/{bank_key}_reviews.xlsx"
    if not os.path.exists(reviews_file):
        print(f"❌ Reviews file not found: {reviews_file}")
        print(f"   Please run scraping first for {bank_key}")
        return False
    
    try:
        # Load config
        with open('config/settings.json', 'r') as f:
            config = json.load(f)
        
        print(f"📊 Reading data from: {reviews_file}")
        df = pd.read_excel(reviews_file)
        print(f"📊 Processing {len(df)} reviews...")
        
        # Create analysis directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = f"output/analysis_{timestamp}"
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Initialize new columns for analysis
        df['translated_content'] = ""
        df['sentiment_score'] = 0.0
        df['positive_words'] = ""
        df['negative_words'] = ""
        df['problem_category'] = ""
        df['overall_sentiment'] = ""
        
        # Process reviews with multi-workers (8 workers)
        print("🔄 Processing reviews with 8 workers...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Prepare arguments for workers
            args_list = [(idx, row, config) for idx, row in df.iterrows()]
            
            # Submit all tasks
            future_to_idx = {executor.submit(process_review_worker, args): args[0] for args in args_list}
            
            # Process results with progress bar
            results = {}
            for future in tqdm(as_completed(future_to_idx), total=len(args_list), desc="Processing reviews"):
                result = future.result()
                results[result['idx']] = result
                
                # Update DataFrame with results
                idx = result['idx']
                df.at[idx, 'translated_content'] = result['translated_content']
                df.at[idx, 'sentiment_score'] = result['sentiment_score']
                df.at[idx, 'positive_words'] = result['positive_words']
                df.at[idx, 'negative_words'] = result['negative_words']
                df.at[idx, 'problem_category'] = result['problem_category']
                df.at[idx, 'overall_sentiment'] = result['overall_sentiment']
                
                if not result['success']:
                    print(f"  Error processing review {idx}: {result.get('error', 'Unknown error')}")
        
        # Save analyzed data
        analyzed_file = os.path.join(analysis_dir, f"{bank_key}_analyzed.xlsx")
        df.to_excel(analyzed_file, index=False)
        
        # Generate visualizations
        print("🔄 Creating analysis charts...")
        create_analysis_charts(df, bank_key, analysis_dir)
        
        # Create wordclouds
        print("🔄 Creating wordclouds...")
        create_wordclouds(df, bank_key, analysis_dir, config)
        
        # Generate summary report
        print("🔄 Generating summary report...")
        summary = generate_summary_report(df, bank_key)
        summary_file = os.path.join(analysis_dir, f"{bank_key}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Output phrase frequencies
        print("🔄 Outputting phrase frequencies...")
        output_phrase_frequencies(df, bank_key, analysis_dir)

        print(f"✅ Full analysis completed for {bank_key}! Results saved to: {analysis_dir}")
        print(f"📊 Sentiment distribution: {summary['overall_sentiment_distribution']}")
        print(f"📊 Average rating: {summary['average_rating']:.2f}")
        print(f"📊 Average sentiment score: {summary['average_sentiment_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed for {bank_key}: {e}")
        return False

def main():
    """Run analysis for all three banks"""
    banks = ["mox_bank", "za_bank", "welab_bank"]
    
    print("🚀 Starting full analysis for Mox, ZA, and WeLab banks...")
    
    success_count = 0
    for bank in banks:
        if run_analysis_for_bank(bank):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis completed for {success_count}/{len(banks)} banks")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 