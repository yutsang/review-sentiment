#!/usr/bin/env python3
"""
Test script for Za Bank sentiment-aware wordcloud generation
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from full_analysis import create_wordclouds, load_config, analyze_sentiment_deepseek, translate_non_english_content

def main():
    """Test the improved wordcloud generation for Za Bank"""
    
    # Load configuration
    config = load_config()
    
    # Check if we have Za Bank data
    za_bank_file = "output/za_bank_reviews.xlsx"
    
    if not os.path.exists(za_bank_file):
        print(f"❌ Za Bank reviews file not found: {za_bank_file}")
        print("Please run the scraper first to get Za Bank reviews.")
        return
    
    # Load Za Bank reviews
    print(f"📊 Loading Za Bank reviews from {za_bank_file}")
    df = pd.read_excel(za_bank_file)
    
    print(f"📈 Loaded {len(df)} reviews")
    print(f"📊 Columns: {df.columns.tolist()}")
    
    # Check if sentiment analysis is needed
    if 'overall_sentiment' not in df.columns:
        print("🔄 Adding sentiment analysis...")
        
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
    output_dir = f"output/za_bank_improved_wordcloud_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Test wordcloud generation with sentiment awareness
    print("🔄 Testing improved sentiment-aware wordcloud generation...")
    
    try:
        create_wordclouds(df, 'za_bank', output_dir, config)
        print("✅ Wordcloud generation completed successfully!")
        
        # Check if files were created
        positive_file = os.path.join(output_dir, "za_bank_positive_wordcloud.png")
        negative_file = os.path.join(output_dir, "za_bank_negative_wordcloud.png")
        
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