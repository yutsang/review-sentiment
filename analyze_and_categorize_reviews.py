#!/usr/bin/env python3
"""
Comprehensive Review Analysis and Categorization
- Check if all comments are in English
- Perform AI sentiment analysis using DeepSeek
- Categorize into 8 categories
- Use 15 workers for parallel processing
- Save to the same Excel files
"""

import pandas as pd
import os
import asyncio
import aiohttp
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from src.utils.config import load_config

class ReviewAnalyzer:
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
                        # Fallback parsing
                        return self.parse_fallback_response(content)
                else:
                    print(f"⚠️ AI analysis failed for: {text[:50]}...")
                    return self.get_default_analysis(text)
                    
        except Exception as e:
            print(f"❌ AI analysis error: {e}")
            return self.get_default_analysis(text)
    
    def parse_fallback_response(self, content):
        """Parse AI response if JSON parsing fails"""
        try:
            # Try to extract information from text response
            sentiment = 'neutral'
            if 'positive' in content.lower():
                sentiment = 'positive'
            elif 'negative' in content.lower():
                sentiment = 'negative'
            
            # Extract category
            category = 'platform_performance'  # default
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
    
    async def analyze_reviews_batch(self, reviews_df):
        """Analyze a batch of reviews using AI with 15 workers"""
        print(f"🤖 Analyzing {len(reviews_df)} reviews with AI (15 workers)...")
        
        # Create session for HTTP requests
        async with aiohttp.ClientSession() as session:
            # Process reviews in batches
            batch_size = 10
            analyzed_reviews = []
            
            for i in range(0, len(reviews_df), batch_size):
                batch = reviews_df.iloc[i:i+batch_size]
                batch_tasks = []
                
                for _, review in batch.iterrows():
                    # Analyze content
                    content = review.get('content', '')
                    analysis_task = self.analyze_text_with_ai(content, session)
                    batch_tasks.append(analysis_task)
                
                # Execute batch
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for j, (_, review) in enumerate(batch.iterrows()):
                    analyzed_review = review.copy()
                    
                    # Get analysis result
                    analysis_result = results[j]
                    if isinstance(analysis_result, Exception):
                        analysis_result = self.get_default_analysis(review.get('content', ''))
                    
                    # Add analysis results
                    analyzed_review['ai_sentiment'] = analysis_result.get('sentiment', 'neutral')
                    analyzed_review['ai_sentiment_score'] = analysis_result.get('sentiment_score', 0.0)
                    analyzed_review['ai_category'] = analysis_result.get('category', 'platform_performance')
                    analyzed_review['ai_category_name'] = self.categories.get(analysis_result.get('category', 'platform_performance'), 'Platform Performance & Reliability')
                    analyzed_review['is_english'] = analysis_result.get('is_english', True)
                    analyzed_review['analyzed_at'] = datetime.now()
                    
                    analyzed_reviews.append(analyzed_review)
                
                # Progress update
                if (i + batch_size) % 50 == 0:
                    print(f"   📝 Analyzed {min(i + batch_size, len(reviews_df))}/{len(reviews_df)} reviews")
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return pd.DataFrame(analyzed_reviews)
    
    def check_english_content(self, reviews_df):
        """Check if all content is in English"""
        print("🔍 Checking English content...")
        
        non_english_count = 0
        total_count = len(reviews_df)
        
        for _, review in reviews_df.iterrows():
            content = review.get('content', '')
            if not self.is_english(content):
                non_english_count += 1
        
        english_percentage = ((total_count - non_english_count) / total_count) * 100
        
        print(f"📊 English Content Analysis:")
        print(f"   Total reviews: {total_count}")
        print(f"   English reviews: {total_count - non_english_count}")
        print(f"   Non-English reviews: {non_english_count}")
        print(f"   English percentage: {english_percentage:.1f}%")
        
        return english_percentage
    
    async def process_all_reviews(self):
        """Process all reviews with AI analysis"""
        print("🚀 COMPREHENSIVE REVIEW ANALYSIS AND CATEGORIZATION")
        print("=" * 60)
        print("📋 Loading combined review database")
        print("🔍 Checking English content")
        print("🤖 Performing AI sentiment analysis")
        print("📂 Categorizing into 8 categories")
        print(f"👥 Using {self.max_workers} workers")
        print("=" * 60)
        
        # Load the combined database
        if not os.path.exists(self.db_file):
            print(f"❌ Database file not found: {self.db_file}")
            return
        
        reviews_df = pd.read_excel(self.db_file)
        print(f"📊 Loaded {len(reviews_df)} reviews from database")
        
        # Check English content
        english_percentage = self.check_english_content(reviews_df)
        
        # Perform AI analysis
        print(f"\n🤖 PERFORMING AI ANALYSIS")
        print("=" * 40)
        
        analyzed_reviews = await self.analyze_reviews_batch(reviews_df)
        
        # Save analyzed data back to the same file
        print(f"\n💾 SAVING ANALYZED DATA")
        print("=" * 40)
        
        analyzed_reviews.to_excel(self.db_file, index=False)
        print(f"✅ Saved analyzed data to: {self.db_file}")
        
        # Generate summary statistics
        print(f"\n📊 ANALYSIS SUMMARY")
        print("=" * 40)
        
        # Sentiment distribution
        sentiment_dist = analyzed_reviews['ai_sentiment'].value_counts()
        print(f"📈 Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(analyzed_reviews)) * 100
            print(f"   {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Category distribution
        category_dist = analyzed_reviews['ai_category_name'].value_counts()
        print(f"\n📂 Category Distribution:")
        for category, count in category_dist.items():
            percentage = (count / len(analyzed_reviews)) * 100
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Bank-wise summary
        print(f"\n🏦 Bank-wise Summary:")
        bank_summary = analyzed_reviews.groupby(['bank', 'ai_sentiment']).size().unstack(fill_value=0)
        print(bank_summary)
        
        # Platform-wise summary
        print(f"\n📱 Platform-wise Summary:")
        platform_summary = analyzed_reviews.groupby(['platform', 'ai_sentiment']).size().unstack(fill_value=0)
        print(platform_summary)
        
        # Average sentiment scores
        print(f"\n📊 Average Sentiment Scores:")
        avg_sentiment = analyzed_reviews['ai_sentiment_score'].mean()
        print(f"   Overall average: {avg_sentiment:.3f}")
        
        bank_avg = analyzed_reviews.groupby('bank')['ai_sentiment_score'].mean()
        print(f"   By bank:")
        for bank, score in bank_avg.items():
            print(f"     {bank}: {score:.3f}")
        
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print(f"📄 Updated file: {self.db_file}")
        print(f"📊 Total reviews analyzed: {len(analyzed_reviews)}")
        print(f"🔍 English content: {english_percentage:.1f}%")
    
    def run(self):
        """Run the complete analysis process"""
        try:
            asyncio.run(self.process_all_reviews())
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main function"""
    analyzer = ReviewAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main() 