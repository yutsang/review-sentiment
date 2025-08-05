#!/usr/bin/env python3
"""
Comprehensive Review Processor
- Translate non-English content to English
- Perform AI sentiment analysis with 8 categories
- Process date columns
- Generate wordclouds with DeepSeek phrase merging
- Add sentiment scoring (-1 to 1 scale)
"""

import pandas as pd
import os
import asyncio
import aiohttp
import json
from datetime import datetime
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from src.utils.config import load_config

class ComprehensiveReviewProcessor:
    def __init__(self):
        self.config = load_config()
        self.deepseek_api_key = self.config.get('deepseek', {}).get('api_key')
        self.max_workers = 15
        
        # 8 Problem Categories
        self.categories = {
            'onboarding_verification': 'Onboarding & Verification',
            'platform_performance': 'Platform Performance & Reliability',
            'customer_service': 'Customer Service Quality & Accessibility',
            'transactions_payments': 'Transactions & Payments',
            'promotion_fees': 'Promotion & Fees',
            'interface_functionality': 'Interface & Functionality',
            'security_privacy': 'Security & Privacy',
            'accounts_interest': 'Accounts & Interest Terms'
        }
        
        # Load the combined database
        self.db_file = "output/combined_data_20250805_104554/comprehensive_combined_database.xlsx"
        
    def is_english(self, text):
        """Check if text is English"""
        if not text or not text.strip():
            return True
        
        # Simple heuristic: check if most characters are ASCII
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.8
    
    def clean_ai_response(self, text):
        """Clean AI response to remove system messages"""
        if not text:
            return text
        
        # Remove common AI system message patterns
        patterns_to_remove = [
            r'The translation of the given text to English is:\s*',
            r'This maintains the original meaning.*',
            r'If the text was already in English.*',
            r'Let me know if you\'d like any refinements.*',
            r'Here\'s the translation.*',
            r'Translation:.*',
            r'**.*?**',  # Remove bold markers
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned_text.strip()
    
    async def translate_text_with_ai(self, text, session):
        """Translate text to English using AI"""
        if not text or not text.strip() or self.is_english(text):
            return text
        
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"Translate the following text to English. If it's already in English, return it unchanged. Only return the translation, no explanations:\n\n{text}"
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    translated = result['choices'][0]['message']['content'].strip()
                    return self.clean_ai_response(translated)
                else:
                    return text
                    
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text
    
    async def analyze_text_with_ai(self, text, session):
        """Analyze text with AI for sentiment and category"""
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'category': 'platform_performance',
                'is_english': True
            }
        
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Analyze the following review text and provide:
            1. Sentiment: positive, negative, or neutral
            2. Sentiment score: -1.0 to +1.0 (negative to positive)
            3. Category: Choose ONE from these 8 categories:
               - onboarding_verification: Onboarding & Verification
               - platform_performance: Platform Performance & Reliability
               - customer_service: Customer Service Quality & Accessibility
               - transactions_payments: Transactions & Payments
               - promotion_fees: Promotion & Fees
               - interface_functionality: Interface & Functionality
               - security_privacy: Security & Privacy
               - accounts_interest: Accounts & Interest Terms
            4. Is English: true/false
            
            Review text: {text}
            
            Respond in JSON format:
            {{
                "sentiment": "positive/negative/neutral",
                "sentiment_score": -1.0 to 1.0,
                "category": "category_key",
                "is_english": true/false
            }}
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    
                    # Parse JSON response
                    try:
                        analysis = json.loads(content)
                        return analysis
                    except json.JSONDecodeError:
                        return self.parse_fallback_response(content)
                else:
                    return self.get_default_analysis(text)
                    
        except Exception as e:
            return self.get_default_analysis(text)
    
    def parse_fallback_response(self, content):
        """Parse AI response if JSON parsing fails"""
        try:
            sentiment = 'neutral'
            if 'positive' in content.lower():
                sentiment = 'positive'
            elif 'negative' in content.lower():
                sentiment = 'negative'
            
            category = 'platform_performance'
            for cat_key in self.categories.keys():
                if cat_key in content.lower():
                    category = cat_key
                    break
            
            return {
                'sentiment': sentiment,
                'sentiment_score': 0.0 if sentiment == 'neutral' else (0.5 if sentiment == 'positive' else -0.5),
                'category': category,
                'is_english': True
            }
        except:
            return self.get_default_analysis("")
    
    def get_default_analysis(self, text):
        """Get default analysis when AI fails"""
        return {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'category': 'platform_performance',
            'is_english': self.is_english(text)
        }
    
    def process_date_column(self, df):
        """Process date column to standardize format"""
        if 'date' in df.columns:
            print("📅 Processing date column...")
            
            # Convert to datetime with error handling
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Add year column
            df['year'] = df['date'].dt.year
            
            # Count valid dates
            valid_dates = df['date'].notna().sum()
            print(f"   ✅ Valid dates: {valid_dates}/{len(df)}")
        
        return df
    
    async def merge_phrases_with_ai(self, phrases, sentiment_type, session):
        """Merge similar phrases using AI"""
        if not phrases:
            return phrases
        
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Merge similar phrases from this {sentiment_type} phrase list. Keep only meaningful, complete phrases.
            Remove duplicates and merge similar meanings. Return only the merged phrases, one per line.
            
            Phrases:
            {chr(10).join(phrases)}
            
            Return only the merged phrases, no explanations:
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    merged_text = result['choices'][0]['message']['content'].strip()
                    
                    # Parse merged phrases
                    merged_phrases = [line.strip() for line in merged_text.split('\n') if line.strip()]
                    return merged_phrases
                else:
                    return phrases
                    
        except Exception as e:
            print(f"❌ Phrase merging error: {e}")
            return phrases
    
    def extract_phrases(self, text, min_length=3):
        """Extract meaningful phrases from text"""
        if not text:
            return []
        
        # Simple phrase extraction
        words = text.lower().split()
        phrases = []
        
        for i in range(len(words) - 1):
            for j in range(i + 2, min(i + 6, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                if len(phrase) >= min_length:
                    phrases.append(phrase)
        
        return phrases
    
    def create_wordcloud(self, phrases, title, output_file, sentiment_type):
        """Create wordcloud from phrases"""
        if not phrases:
            print(f"⚠️ No phrases for {sentiment_type} wordcloud")
            return
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Create wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            max_words=100,
            prefer_horizontal=0.7,
            min_font_size=10,
            colormap='viridis' if sentiment_type == 'positive' else 'Reds'
        ).generate_from_frequencies(phrase_counts)
        
        # Save wordcloud
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created wordcloud: {output_file}")
    
    async def process_reviews_comprehensive(self, reviews_df):
        """Process reviews comprehensively"""
        print(f"🔄 Processing {len(reviews_df)} reviews comprehensively...")
        
        async with aiohttp.ClientSession() as session:
            # Process in batches
            batch_size = 10
            processed_reviews = []
            
            for i in range(0, len(reviews_df), batch_size):
                batch = reviews_df.iloc[i:i+batch_size]
                batch_results = []
                
                for _, review in batch.iterrows():
                    processed_review = review.copy()
                    
                    # Get original content
                    content = review.get('content', '')
                    
                    # Translate if needed
                    if not self.is_english(content):
                        translated = await self.translate_text_with_ai(content, session)
                        processed_review['translated_content'] = translated
                    else:
                        processed_review['translated_content'] = content
                    
                    # Analyze with AI
                    analysis = await self.analyze_text_with_ai(processed_review['translated_content'], session)
                    
                    # Add analysis results
                    processed_review['ai_sentiment'] = analysis.get('sentiment', 'neutral')
                    processed_review['ai_sentiment_score'] = analysis.get('sentiment_score', 0.0)
                    processed_review['ai_category'] = analysis.get('category', 'platform_performance')
                    processed_review['ai_category_name'] = self.categories.get(analysis.get('category', 'platform_performance'), 'Platform Performance & Reliability')
                    processed_review['is_english'] = analysis.get('is_english', True)
                    processed_review['processed_at'] = datetime.now()
                    
                    batch_results.append(processed_review)
                
                processed_reviews.extend(batch_results)
                
                # Progress update
                if (i + batch_size) % 50 == 0:
                    print(f"   📝 Processed {min(i + batch_size, len(reviews_df))}/{len(reviews_df)} reviews")
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return pd.DataFrame(processed_reviews)
    
    async def generate_wordclouds_for_bank(self, bank_df, bank_name, session):
        """Generate wordclouds for a specific bank"""
        print(f"🎨 Generating wordclouds for {bank_name}...")
        
        # Create output directory
        output_dir = f"output/wordclouds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Separate positive and negative reviews
        positive_reviews = bank_df[bank_df['ai_sentiment'] == 'positive']
        negative_reviews = bank_df[bank_df['ai_sentiment'] == 'negative']
        
        # Process positive reviews
        if not positive_reviews.empty:
            print(f"   📈 Processing {len(positive_reviews)} positive reviews...")
            
            # Extract phrases
            all_positive_phrases = []
            for _, review in positive_reviews.iterrows():
                content = review.get('translated_content', review.get('content', ''))
                phrases = self.extract_phrases(content)
                all_positive_phrases.extend(phrases)
            
            # Merge phrases with AI
            merged_positive_phrases = await self.merge_phrases_with_ai(all_positive_phrases, 'positive', session)
            
            # Save phrases to text file
            positive_phrases_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_positive_phrases.txt")
            with open(positive_phrases_file, 'w', encoding='utf-8') as f:
                for phrase in merged_positive_phrases:
                    f.write(f"{phrase}\n")
            
            # Create wordcloud
            positive_wordcloud_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_positive_wordcloud.png")
            self.create_wordcloud(merged_positive_phrases, f"{bank_name} - Positive Reviews", positive_wordcloud_file, 'positive')
        
        # Process negative reviews
        if not negative_reviews.empty:
            print(f"   📉 Processing {len(negative_reviews)} negative reviews...")
            
            # Extract phrases
            all_negative_phrases = []
            for _, review in negative_reviews.iterrows():
                content = review.get('translated_content', review.get('content', ''))
                phrases = self.extract_phrases(content)
                all_negative_phrases.extend(phrases)
            
            # Merge phrases with AI
            merged_negative_phrases = await self.merge_phrases_with_ai(all_negative_phrases, 'negative', session)
            
            # Save phrases to text file
            negative_phrases_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_negative_phrases.txt")
            with open(negative_phrases_file, 'w', encoding='utf-8') as f:
                for phrase in merged_negative_phrases:
                    f.write(f"{phrase}\n")
            
            # Create wordcloud
            negative_wordcloud_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_negative_wordcloud.png")
            self.create_wordcloud(merged_negative_phrases, f"{bank_name} - Negative Reviews", negative_wordcloud_file, 'negative')
        
        return output_dir
    
    async def run_comprehensive_processing(self):
        """Run the complete comprehensive processing"""
        print("🚀 COMPREHENSIVE REVIEW PROCESSOR")
        print("=" * 60)
        print("🌐 Translating non-English content")
        print("🤖 AI sentiment analysis with 8 categories")
        print("📅 Processing date columns")
        print("🎨 Generating wordclouds with AI phrase merging")
        print(f"👥 Using {self.max_workers} workers")
        print("=" * 60)
        
        # Load the database
        if not os.path.exists(self.db_file):
            print(f"❌ Database file not found: {self.db_file}")
            return
        
        reviews_df = pd.read_excel(self.db_file)
        print(f"📊 Loaded {len(reviews_df)} reviews from database")
        
        # Process date column
        reviews_df = self.process_date_column(reviews_df)
        
        # Comprehensive processing
        print(f"\n🔄 COMPREHENSIVE PROCESSING")
        print("=" * 40)
        
        processed_reviews = await self.process_reviews_comprehensive(reviews_df)
        
        # Save processed data
        print(f"\n💾 SAVING PROCESSED DATA")
        print("=" * 40)
        
        processed_reviews.to_excel(self.db_file, index=False)
        print(f"✅ Saved processed data to: {self.db_file}")
        
        # Generate wordclouds for each bank
        print(f"\n🎨 GENERATING WORDCLOUDS")
        print("=" * 40)
        
        async with aiohttp.ClientSession() as session:
            for bank_name in processed_reviews['bank'].unique():
                bank_df = processed_reviews[processed_reviews['bank'] == bank_name]
                await self.generate_wordclouds_for_bank(bank_df, bank_name, session)
        
        # Generate summary statistics
        print(f"\n📊 PROCESSING SUMMARY")
        print("=" * 40)
        
        # Translation summary
        translated_count = processed_reviews['translated_content'].notna().sum()
        print(f"🌐 Translation Summary:")
        print(f"   Total reviews: {len(processed_reviews)}")
        print(f"   Translated reviews: {translated_count}")
        
        # Sentiment distribution
        sentiment_dist = processed_reviews['ai_sentiment'].value_counts()
        print(f"\n📈 Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(processed_reviews)) * 100
            print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Category distribution
        category_dist = processed_reviews['ai_category_name'].value_counts()
        print(f"\n📂 Category Distribution:")
        for category, count in category_dist.items():
            percentage = (count / len(processed_reviews)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Average sentiment scores
        print(f"\n📊 Average Sentiment Scores:")
        avg_sentiment = processed_reviews['ai_sentiment_score'].mean()
        print(f"   Overall average: {avg_sentiment:.3f}")
        
        bank_avg = processed_reviews.groupby('bank')['ai_sentiment_score'].mean()
        print(f"   By bank:")
        for bank, score in bank_avg.items():
            print(f"     {bank}: {score:.3f}")
        
        print(f"\n🎉 COMPREHENSIVE PROCESSING COMPLETED!")
        print(f"📄 Updated file: {self.db_file}")
        print(f"📊 Total reviews processed: {len(processed_reviews)}")
    
    def run(self):
        """Run the complete processing"""
        try:
            asyncio.run(self.run_comprehensive_processing())
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise

def main():
    """Main function"""
    processor = ComprehensiveReviewProcessor()
    processor.run()

if __name__ == "__main__":
    main() 