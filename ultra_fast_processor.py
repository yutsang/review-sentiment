#!/usr/bin/env python3
"""
Ultra Fast Review Processor with Multiple Workers
- Multiple workers for ALL tasks (translation, analysis, phrase merging)
- Proper date validation and processing
- Ultra-fast batch processing
- tqdm progress bars
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
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class UltraFastProcessor:
    def __init__(self):
        self.config = load_config()
        self.deepseek_api_key = self.config.get('deepseek', {}).get('api_key')
        self.max_workers = 20
        self.timeout = 15  # Reduced timeout for speed
        self.batch_size = 50  # Larger batches
        
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
            'The translation of the given text to English is:',
            'This maintains the original meaning',
            'If the text was already in English',
            'Let me know if you would like any refinements',
            'Here is the translation',
            'Translation:',
            '**',  # Remove bold markers
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = cleaned_text.replace(pattern, '')
        
        return cleaned_text.strip()
    
    def fix_date_column(self, df):
        """Fix date column with multiple validation methods"""
        print("📅 Fixing date column...")
        
        if 'date' not in df.columns:
            print("   ⚠️ No date column found")
            return df
        
        # Create a copy to work with
        df = df.copy()
        
        # Try multiple date parsing methods
        date_formats = [
            '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone
            '%Y-%m-%dT%H:%M:%S',    # ISO format without timezone
            '%Y-%m-%d %H:%M:%S',    # Standard format
            '%Y-%m-%d',             # Date only
            '%d/%m/%Y',             # DD/MM/YYYY
            '%m/%d/%Y',             # MM/DD/YYYY
        ]
        
        valid_dates = 0
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            date_value = row.get('date')
            if pd.isna(date_value) or date_value == '':
                continue
                
            # Convert to string if needed
            date_str = str(date_value)
            
            # Try different parsing methods
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = pd.to_datetime(date_str, format=fmt)
                    break
                except:
                    continue
            
            # If no format worked, try pandas automatic parsing
            if parsed_date is None:
                try:
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                except:
                    parsed_date = None
            
            # Update the dataframe
            if parsed_date is not None and pd.notna(parsed_date):
                df.at[idx, 'date'] = parsed_date
                valid_dates += 1
        
        # Add year column
        df['year'] = df['date'].dt.year if 'date' in df.columns else None
        
        print(f"   ✅ Valid dates: {valid_dates}/{total_rows} ({valid_dates/total_rows*100:.1f}%)")
        
        return df
    
    async def translate_batch_with_ai(self, texts, session):
        """Translate a batch of texts using AI with multiple workers"""
        if not texts:
            return texts
        
        # Create tasks for all texts
        tasks = []
        for text in texts:
            if not text or not text.strip() or self.is_english(text):
                tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Skip translation
            else:
                task = self.translate_single_text(text, session)
                tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        translated_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                translated_texts.append(texts[i])  # Keep original on error
            else:
                translated_texts.append(result)
        
        return translated_texts
    
    async def translate_single_text(self, text, session):
        """Translate a single text with timeout"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"Translate to English. If already English, return unchanged. Only translation, no explanations:\n\n{text}"
            
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    translated = result['choices'][0]['message']['content'].strip()
                    return self.clean_ai_response(translated)
                else:
                    return text
                    
        except Exception:
            return text
    
    async def analyze_batch_with_ai(self, texts, session):
        """Analyze a batch of texts using AI with multiple workers"""
        if not texts:
            return []
        
        # Create tasks for all texts
        tasks = []
        for text in texts:
            if not text or not text.strip():
                tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Skip analysis
            else:
                task = self.analyze_single_text(text, session)
                tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                analyses.append(self.get_default_analysis(texts[i]))
            else:
                analyses.append(result)
        
        return analyses
    
    async def analyze_single_text(self, text, session):
        """Analyze a single text with timeout"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Analyze this review and provide JSON response:
            {{
                "sentiment": "positive/negative/neutral",
                "sentiment_score": -1.0 to 1.0,
                "category": "onboarding_verification/platform_performance/customer_service/transactions_payments/promotion_fees/interface_functionality/security_privacy/accounts_interest",
                "is_english": true/false
            }}
            
            Review: {text}
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    
                    try:
                        analysis = json.loads(content)
                        return analysis
                    except json.JSONDecodeError:
                        return self.parse_fallback_response(content)
                else:
                    return self.get_default_analysis(text)
                    
        except Exception:
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
    
    async def merge_phrases_batch_with_ai(self, phrase_lists, sentiment_types, session):
        """Merge phrases for multiple banks using AI with multiple workers"""
        if not phrase_lists:
            return []
        
        # Create tasks for all phrase lists
        tasks = []
        for phrases, sentiment_type in zip(phrase_lists, sentiment_types):
            if not phrases:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            else:
                task = self.merge_single_phrase_list(phrases, sentiment_type, session)
                tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        merged_phrases = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                merged_phrases.append(phrase_lists[i] if i < len(phrase_lists) else [])
            else:
                merged_phrases.append(result)
        
        return merged_phrases
    
    async def merge_single_phrase_list(self, phrases, sentiment_type, session):
        """Merge a single phrase list with timeout"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            # Take first 20 phrases to avoid token limits
            sample_phrases = phrases[:20]
            
            prompt = f"""
            Merge similar {sentiment_type} phrases. Return only merged phrases, one per line:
            
            {chr(10).join(sample_phrases)}
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.post(url, headers=headers, json=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    merged_text = result['choices'][0]['message']['content'].strip()
                    
                    # Parse merged phrases
                    merged_phrases = [line.strip() for line in merged_text.split('\n') if line.strip()]
                    return merged_phrases
                else:
                    return phrases
                    
        except Exception:
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
    
    async def process_reviews_ultra_fast(self, reviews_df):
        """Process reviews ultra-fast with multiple workers"""
        print(f"🔄 Processing {len(reviews_df)} reviews ultra-fast...")
        
        # Fix date column first
        reviews_df = self.fix_date_column(reviews_df)
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.max_workers, limit_per_host=self.max_workers)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            processed_reviews = []
            
            # Process in large batches
            pbar = tqdm(total=len(reviews_df), desc="Processing reviews", unit="review")
            
            for i in range(0, len(reviews_df), self.batch_size):
                batch = reviews_df.iloc[i:i+self.batch_size]
                
                # Extract texts for batch processing
                texts = batch['content'].fillna('').tolist()
                
                # Translate batch
                translated_texts = await self.translate_batch_with_ai(texts, session)
                
                # Analyze batch
                analyses = await self.analyze_batch_with_ai(translated_texts, session)
                
                # Process results
                for j, (_, review) in enumerate(batch.iterrows()):
                    processed_review = review.copy()
                    
                    # Add translation
                    processed_review['translated_content'] = translated_texts[j]
                    
                    # Add analysis results
                    analysis = analyses[j]
                    processed_review['ai_sentiment'] = analysis.get('sentiment', 'neutral')
                    processed_review['ai_sentiment_score'] = analysis.get('sentiment_score', 0.0)
                    processed_review['ai_category'] = analysis.get('category', 'platform_performance')
                    processed_review['ai_category_name'] = self.categories.get(analysis.get('category', 'platform_performance'), 'Platform Performance & Reliability')
                    processed_review['is_english'] = analysis.get('is_english', True)
                    processed_review['processed_at'] = datetime.now()
                    
                    processed_reviews.append(processed_review)
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Minimal delay
                await asyncio.sleep(0.01)
            
            pbar.close()
        
        return pd.DataFrame(processed_reviews)
    
    async def generate_wordclouds_ultra_fast(self, processed_reviews, session):
        """Generate wordclouds ultra-fast with multiple workers"""
        print(f"🎨 Generating wordclouds ultra-fast...")
        
        # Create output directory
        output_dir = f"output/wordclouds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each bank
        for bank_name in tqdm(processed_reviews['bank'].unique(), desc="Generating wordclouds"):
            bank_df = processed_reviews[processed_reviews['bank'] == bank_name]
            
            # Separate positive and negative reviews
            positive_reviews = bank_df[bank_df['ai_sentiment'] == 'positive']
            negative_reviews = bank_df[bank_df['ai_sentiment'] == 'negative']
            
            # Extract phrases for both sentiment types
            positive_phrases = []
            negative_phrases = []
            
            for _, review in positive_reviews.iterrows():
                content = review.get('translated_content', review.get('content', ''))
                phrases = self.extract_phrases(content)
                positive_phrases.extend(phrases)
            
            for _, review in negative_reviews.iterrows():
                content = review.get('translated_content', review.get('content', ''))
                phrases = self.extract_phrases(content)
                negative_phrases.extend(phrases)
            
            # Merge phrases with AI (batch processing)
            phrase_lists = [positive_phrases, negative_phrases]
            sentiment_types = ['positive', 'negative']
            
            merged_phrases_list = await self.merge_phrases_batch_with_ai(phrase_lists, sentiment_types, session)
            
            # Save phrases and create wordclouds
            if merged_phrases_list[0]:  # Positive phrases
                positive_phrases_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_positive_phrases.txt")
                with open(positive_phrases_file, 'w', encoding='utf-8') as f:
                    for phrase in merged_phrases_list[0]:
                        f.write(f"{phrase}\n")
                
                positive_wordcloud_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_positive_wordcloud.png")
                self.create_wordcloud(merged_phrases_list[0], f"{bank_name} - Positive Reviews", positive_wordcloud_file, 'positive')
            
            if merged_phrases_list[1]:  # Negative phrases
                negative_phrases_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_negative_phrases.txt")
                with open(negative_phrases_file, 'w', encoding='utf-8') as f:
                    for phrase in merged_phrases_list[1]:
                        f.write(f"{phrase}\n")
                
                negative_wordcloud_file = os.path.join(output_dir, f"{bank_name.lower().replace(' ', '_')}_negative_wordcloud.png")
                self.create_wordcloud(merged_phrases_list[1], f"{bank_name} - Negative Reviews", negative_wordcloud_file, 'negative')
        
        return output_dir
    
    async def run_ultra_fast_processing(self):
        """Run the complete ultra-fast processing"""
        print("🚀 ULTRA FAST REVIEW PROCESSOR")
        print("=" * 60)
        print("🌐 Translating non-English content")
        print("🤖 AI sentiment analysis with 8 categories")
        print("📅 Processing date columns")
        print("🎨 Generating wordclouds with AI phrase merging")
        print(f"👥 Using {self.max_workers} workers")
        print(f"⏰ Timeout: {self.timeout} seconds")
        print(f"📦 Batch size: {self.batch_size}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load the database
        if not os.path.exists(self.db_file):
            print(f"❌ Database file not found: {self.db_file}")
            return
        
        reviews_df = pd.read_excel(self.db_file)
        print(f"📊 Loaded {len(reviews_df)} reviews from database")
        
        # Ultra-fast processing
        print(f"\n🔄 ULTRA-FAST PROCESSING")
        print("=" * 40)
        
        processed_reviews = await self.process_reviews_ultra_fast(reviews_df)
        
        # Save processed data
        print(f"\n💾 SAVING PROCESSED DATA")
        print("=" * 40)
        
        processed_reviews.to_excel(self.db_file, index=False)
        print(f"✅ Saved processed data to: {self.db_file}")
        
        # Generate wordclouds ultra-fast
        print(f"\n🎨 GENERATING WORDCLOUDS ULTRA-FAST")
        print("=" * 40)
        
        connector = aiohttp.TCPConnector(limit=self.max_workers, limit_per_host=self.max_workers)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            await self.generate_wordclouds_ultra_fast(processed_reviews, session)
        
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
        
        # Date summary
        if 'date' in processed_reviews.columns:
            valid_dates = processed_reviews['date'].notna().sum()
            print(f"\n📅 Date Summary:")
            print(f"   Valid dates: {valid_dates}/{len(processed_reviews)} ({valid_dates/len(processed_reviews)*100:.1f}%)")
        
        # Time summary
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n⏱️ Processing Time: {total_time:.2f} seconds")
        print(f"📊 Average time per review: {total_time/len(processed_reviews):.2f} seconds")
        
        print(f"\n🎉 ULTRA-FAST PROCESSING COMPLETED!")
        print(f"📄 Updated file: {self.db_file}")
        print(f"📊 Total reviews processed: {len(processed_reviews)}")
    
    def run(self):
        """Run the complete processing"""
        try:
            asyncio.run(self.run_ultra_fast_processing())
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise

def main():
    """Main function"""
    processor = UltraFastProcessor()
    processor.run()

if __name__ == "__main__":
    main() 