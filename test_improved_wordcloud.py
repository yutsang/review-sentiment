#!/usr/bin/env python3
"""
Test script for improved wordcloud generation with DeepSeek optimization
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from full_analysis import create_wordclouds, load_config

def main():
    """Test the improved wordcloud generation for Mox Bank"""
    
    # Load configuration
    config = load_config()
    
    # Check if we have Mox Bank data
    mox_bank_file = "output/mox_bank_reviews.xlsx"
    
    if not os.path.exists(mox_bank_file):
        print(f"❌ Mox Bank reviews file not found: {mox_bank_file}")
        print("Please run the scraper first to get Mox Bank reviews.")
        return
    
    # Load Mox Bank reviews
    print(f"📊 Loading Mox Bank reviews from {mox_bank_file}")
    df = pd.read_excel(mox_bank_file)
    
    print(f"📈 Loaded {len(df)} reviews")
    print(f"📊 Columns: {df.columns.tolist()}")
    
    # Check if sentiment analysis is needed
    if 'overall_sentiment' not in df.columns:
        print("🔄 Adding sentiment analysis...")
        from full_analysis import analyze_sentiment_deepseek, translate_non_english_content
        
        # Add sentiment analysis
        sentiments = []
        translated_contents = []
        
        for idx, row in df.iterrows():
            content = str(row['content'])
            
            # Translate if needed
            if not content.isascii():
                translated_content = translate_non_english_content(content, config)
                translated_contents.append(translated_content)
            else:
                translated_contents.append(content)
            
            # Analyze sentiment
            sentiment_result = analyze_sentiment_deepseek(content, config)
            sentiments.append(sentiment_result.get('sentiment', 'neutral'))
            
            if (idx + 1) % 100 == 0:
                print(f"📊 Processed {idx + 1}/{len(df)} reviews")
        
        df['translated_content'] = translated_contents
        df['overall_sentiment'] = sentiments
        
        print(f"✅ Sentiment analysis completed")
    
    print(f"📊 Sentiment distribution:")
    print(df['overall_sentiment'].value_counts())
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Test wordcloud generation with DeepSeek optimization
    print("🔄 Testing improved wordcloud generation with DeepSeek optimization...")
    
    try:
        create_wordclouds(df, 'mox_bank', output_dir, config)
        print("✅ Wordcloud generation completed successfully!")
        
        # Check if files were created
        positive_file = os.path.join(output_dir, "mox_bank_positive_wordcloud.png")
        negative_file = os.path.join(output_dir, "mox_bank_negative_wordcloud.png")
        
        if os.path.exists(positive_file):
            print(f"✅ Positive wordcloud created: {positive_file}")
        else:
            print(f"❌ Positive wordcloud not created")
            
        if os.path.exists(negative_file):
            print(f"✅ Negative wordcloud created: {negative_file}")
        else:
            print(f"❌ Negative wordcloud not created")
            
    except Exception as e:
        print(f"❌ Error during wordcloud generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 