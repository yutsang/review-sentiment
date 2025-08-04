#!/usr/bin/env python3
"""
Test script for sentiment-aware wordcloud generation
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from full_analysis import create_wordclouds, load_config

def create_sample_data():
    """Create sample data with known positive and negative reviews"""
    
    sample_reviews = [
        # Positive reviews
        {"content": "This app is amazing! Easy to use and very user friendly. The interface is intuitive and the app works perfectly. Customer service is helpful and responsive. Transfer money instantly and payment processed quickly. Face ID works perfectly and verification process smooth. No hidden fees and competitive rates.", "sentiment": "positive"},
        {"content": "Great app! App works well and loads quickly. Easy to navigate and simple to use. Account opening easy and login works every time. Very fast and reliable. Excellent experience overall.", "sentiment": "positive"},
        {"content": "Best banking app ever! App functions well and never crashes. Easy to set up and account setup easy. Transfer works and payment successful. Customer service great and support helpful.", "sentiment": "positive"},
        
        # Negative reviews
        {"content": "This app is terrible! App keeps crashing and not working. Cannot open account and login always fails. Very slow and difficult to use. Customer service unhelpful and support not responsive. Transfer money failed and payment failed. Face ID not working and verification process stuck. Hidden fees and too expensive.", "sentiment": "negative"},
        {"content": "Worst app ever! App freezes constantly and loads very slowly. Cannot access account and unable to login. Not user friendly and interface too confusing. Very bad experience and frustrating to use.", "sentiment": "negative"},
        {"content": "Awful app! App crashes every time and stops working. Account opening failed and verification failed. Poor customer service and no response from support. Transaction declined and payment rejected.", "sentiment": "negative"},
        
        # More positive reviews
        {"content": "Excellent app! App works perfectly and is very stable. Easy to use and user friendly interface. Quick login and fast transfer. Smooth experience and reliable service.", "sentiment": "positive"},
        {"content": "Fantastic banking app! App loads quickly and responds quickly. Easy to navigate and intuitive design. Secure login and verification works. Great customer service and helpful support.", "sentiment": "positive"},
        
        # More negative reviews
        {"content": "Horrible app! App refuses to open and force closed. Cannot scan id and verification stuck. Very confusing interface and difficult to navigate. Terrible experience and very frustrating.", "sentiment": "negative"},
        {"content": "Disappointing app! App update broke everything and new version worse. Cannot transfer money and payment processing error. Customer service useless and support not helpful.", "sentiment": "negative"},
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_reviews)
    df['translated_content'] = df['content']  # For simplicity
    df['overall_sentiment'] = df['sentiment']
    
    return df

def main():
    """Test the sentiment-aware wordcloud generation"""
    
    # Load configuration
    config = load_config()
    
    # Create sample data
    print("📊 Creating sample data with positive and negative reviews...")
    df = create_sample_data()
    
    print(f"📈 Created {len(df)} sample reviews")
    print(f"📊 Sentiment distribution:")
    print(df['overall_sentiment'].value_counts())
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/test_sentiment_wordcloud_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Test wordcloud generation with sentiment awareness
    print("🔄 Testing sentiment-aware wordcloud generation...")
    
    try:
        create_wordclouds(df, 'test_bank', output_dir, config)
        print("✅ Wordcloud generation completed successfully!")
        
        # Check if files were created
        positive_file = os.path.join(output_dir, "test_bank_positive_wordcloud.png")
        negative_file = os.path.join(output_dir, "test_bank_negative_wordcloud.png")
        
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