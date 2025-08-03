#!/usr/bin/env python3

import pandas as pd
import json
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from datetime import datetime

def call_deepseek_api(text: str, config: dict) -> str:
    """Call DeepSeek API to get AI-processed phrases for wordcloud"""
    try:
        deepseek_config = config.get('deepseek', {})
        
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            return ""
            
        base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = deepseek_config.get('model', 'deepseek-chat')
        max_tokens = deepseek_config.get('max_tokens', 1000)
        temperature = deepseek_config.get('temperature', 0.3)
        prompt_template = deepseek_config.get('wordcloud_prompt', '')
        
        # Truncate text if too long
        truncated_text = text[:2000] + "..." if len(text) > 2000 else text
        
        # Format prompt
        prompt = prompt_template.format(reviews=truncated_text)
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Clean up the response - extract only phrases
            phrases = content.replace('\n', ' ').replace('\r', ' ').strip()
            return phrases
        else:
            print(f"⚠️ DeepSeek API error: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"⚠️ DeepSeek API call failed: {e}")
        return ""

def preserve_phrases_for_wordcloud(text: str) -> str:
    """Preserve multi-word phrases for wordcloud with proper spacing"""
    words = text.split()
    processed_words = []
    
    i = 0
    while i < len(words):
        # Check if this word is part of a phrase (2-4 words max for readability)
        phrase_length = 1
        phrase_words = [words[i]]
        
        # Look ahead to find complete phrases (max 4 words for readability)
        for j in range(1, 4):  # Check up to 4 words
            if i + j < len(words):
                next_word = words[i + j]
                # If next word doesn't start with uppercase or is a common connector, it's part of phrase
                if (not next_word[0].isupper() and 
                    next_word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'has', 'have', 'had']):
                    phrase_words.append(next_word)
                    phrase_length += 1
                else:
                    break
            else:
                break
        
        # If we have a meaningful phrase (2-4 words), keep with spaces
        if phrase_length >= 2 and phrase_length <= 4:
            phrase = ' '.join(phrase_words)
            # Only add if it's a meaningful phrase (not too long)
            if len(phrase) <= 30:  # Limit phrase length
                processed_words.append(phrase)
            else:
                # If too long, just add the first word
                processed_words.append(words[i])
            i += phrase_length  # Skip the words we've already processed
        else:
            # Single word, keep as is
            processed_words.append(words[i])
            i += 1
    
    return ' '.join(processed_words)

def create_wordclouds(df: pd.DataFrame, app_key: str, output_dir: str, config: dict):
    """Create word clouds for positive and negative reviews with 2-5 word phrases"""
    try:
        # Set font to Arial
        plt.rcParams['font.family'] = 'Arial'
        
        # Positive reviews word cloud
        positive_reviews = df[df['overall_sentiment'] == 'positive']['translated_content'].dropna()
        if not positive_reviews.empty:
            positive_text = ' '.join(positive_reviews.astype(str))
            
            # Use DeepSeek to extract 2-5 word phrases
            ai_enhanced_text = call_deepseek_api(positive_text, config)
            if ai_enhanced_text and len(ai_enhanced_text.split()) > 10:
                print(f"🤖 Using AI-enhanced 2-5 word phrases for {app_key} positive reviews")
                print(f"📝 Sample phrases: {' '.join(ai_enhanced_text.split()[:20])}")
                wordcloud_text = ai_enhanced_text
            else:
                print(f"📝 Using fallback phrase extraction for {app_key} positive reviews")
                wordcloud_text = positive_text
            
            # Pre-process text to preserve phrases
            processed_text = preserve_phrases_for_wordcloud(wordcloud_text)
            print(f"📝 Processed phrases: {' '.join(processed_text.split()[:10])}")
            
            # Create positive wordcloud with Pacific Blue color (0, 184, 245)
            wordcloud_pos = WordCloud(
                width=800, height=800,  # Square size
                background_color='white',
                color_func=lambda *args, **kwargs: (0, 184, 245),  # Pacific Blue
                max_words=100,
                stopwords=None,
                min_font_size=10,
                max_font_size=80,
                prefer_horizontal=0.7,  # Allow more vertical text for phrases
                collocations=False  # Don't count word pairs as single words
            ).generate(processed_text)
            
            # Save positive wordcloud separately
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.axis('off')
            positive_wordcloud_file = os.path.join(output_dir, f"{app_key}_positive_wordcloud.png")
            plt.savefig(positive_wordcloud_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        
        # Negative reviews word cloud
        negative_reviews = df[df['overall_sentiment'] == 'negative']['translated_content'].dropna()
        if not negative_reviews.empty:
            negative_text = ' '.join(negative_reviews.astype(str))
            
            # Use DeepSeek to extract 2-5 word phrases
            ai_enhanced_text = call_deepseek_api(negative_text, config)
            if ai_enhanced_text and len(ai_enhanced_text.split()) > 10:
                print(f"🤖 Using AI-enhanced 2-5 word phrases for {app_key} negative reviews")
                print(f"📝 Sample phrases: {' '.join(ai_enhanced_text.split()[:20])}")
                wordcloud_text = ai_enhanced_text
            else:
                print(f"📝 Using fallback phrase extraction for {app_key} negative reviews")
                wordcloud_text = negative_text
            
            # Pre-process text to preserve phrases
            processed_text = preserve_phrases_for_wordcloud(wordcloud_text)
            print(f"📝 Processed phrases: {' '.join(processed_text.split()[:10])}")
            
            # Create negative wordcloud with Dark Blue color (12, 35, 60)
            wordcloud_neg = WordCloud(
                width=800, height=800,  # Square size
                background_color='white',
                color_func=lambda *args, **kwargs: (12, 35, 60),  # Dark Blue
                max_words=100,
                stopwords=None,
                min_font_size=10,
                max_font_size=80,
                prefer_horizontal=0.7,  # Allow more vertical text for phrases
                collocations=False  # Don't count word pairs as single words
            ).generate(processed_text)
            
            # Save negative wordcloud separately
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.axis('off')
            negative_wordcloud_file = os.path.join(output_dir, f"{app_key}_negative_wordcloud.png")
            plt.savefig(negative_wordcloud_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
                
    except Exception as e:
        print(f"⚠️ Could not generate word clouds: {e}")

def main():
    # Load config
    with open('config/settings.json', 'r') as f:
        config = json.load(f)
    
    # Read existing analysis file (use the analyzed version)
    file_path = "output/analysis_20250801_152428/welab_bank_analyzed.xlsx"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📊 Reading data from: {file_path}")
    df = pd.read_excel(file_path)
    print(f"📊 Processing {len(df)} reviews...")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Create analysis directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = f"output/analysis_{timestamp}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Check if sentiment columns exist
    if 'overall_sentiment' not in df.columns:
        print("⚠️ No overall_sentiment column found, creating default...")
        df['overall_sentiment'] = 'neutral'  # Default
    
    if 'translated_content' not in df.columns:
        print("⚠️ No translated_content column found, using content...")
        df['translated_content'] = df['content']  # Use original content
    
    print(f"📊 Sentiment distribution: {df['overall_sentiment'].value_counts().to_dict()}")
    print(f"📊 Sample content: {df['translated_content'].iloc[0][:100]}...")
    
    # Create wordclouds
    print("🔄 Creating wordclouds...")
    create_wordclouds(df, 'welab_bank', analysis_dir, config)
    
    print(f"✅ Analysis completed! Results saved to: {analysis_dir}")

if __name__ == "__main__":
    main() 