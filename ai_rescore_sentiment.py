#!/usr/bin/env python3
"""
AI Rescore Sentiment with 15-20 workers using 0.2 level width from -1 to 1
"""

import pandas as pd
import asyncio
import aiohttp
import json
from datetime import datetime
from tqdm import tqdm
import re

class AISentimentRescorer:
    def __init__(self):
        self.max_workers = 20
        self.batch_size = 50
        self.timeout = 15
        self.db_file = 'output/combined_data_20250805_104554/comprehensive_combined_database_FIXED_COMPLETE.xlsx'
        self.output_file = 'output/combined_data_20250805_104554/comprehensive_combined_database_RESCORED.xlsx'
        
        # DeepSeek API configuration
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.api_key = "sk-013e00392a7d433b8d1d09d88bc0b62e"
        
        # Sentiment levels with 0.2 width
        self.sentiment_levels = {
            'very_negative': (-1.0, -0.8),
            'negative': (-0.8, -0.6),
            'somewhat_negative': (-0.6, -0.4),
            'slightly_negative': (-0.4, -0.2),
            'neutral': (-0.2, 0.2),
            'slightly_positive': (0.2, 0.4),
            'somewhat_positive': (0.4, 0.6),
            'positive': (0.6, 0.8),
            'very_positive': (0.8, 1.0)
        }
    
    def clean_ai_response(self, response_text):
        """Clean AI response and extract sentiment score"""
        try:
            # Remove system messages and extract JSON
            if '```json' in response_text:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            elif '```' in response_text:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            # Parse JSON
            data = json.loads(response_text.strip())
            
            # Extract sentiment score
            sentiment_score = data.get('sentiment_score', 0.0)
            
            # Ensure score is within -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_level': self.get_sentiment_level(sentiment_score),
                'confidence': data.get('confidence', 0.8)
            }
            
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_level': 'neutral',
                'confidence': 0.5
            }
    
    def get_sentiment_level(self, score):
        """Get sentiment level based on score"""
        for level, (min_score, max_score) in self.sentiment_levels.items():
            if min_score <= score < max_score:
                return level
        return 'neutral'
    
    async def analyze_sentiment_with_ai(self, content, session):
        """Analyze sentiment with AI using 0.2 level width"""
        try:
            prompt = f"""
Analyze the sentiment of this review and provide a precise sentiment score.

Review: "{content}"

Please provide a sentiment score from -1.0 to 1.0 with 0.2 level precision:
- Very Negative: -1.0 to -0.8
- Negative: -0.8 to -0.6  
- Somewhat Negative: -0.6 to -0.4
- Slightly Negative: -0.4 to -0.2
- Neutral: -0.2 to 0.2
- Slightly Positive: 0.2 to 0.4
- Somewhat Positive: 0.4 to 0.6
- Positive: 0.6 to 0.8
- Very Positive: 0.8 to 1.0

Respond in JSON format:
{{
    "sentiment_score": <score>,
    "confidence": <confidence 0-1>
}}
"""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a sentiment analysis expert. Provide precise sentiment scores with 0.2 level granularity."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            async with session.post(self.api_url, headers=headers, json=data, timeout=self.timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    return self.clean_ai_response(content)
                else:
                    print(f"API Error: {response.status}")
                    return {
                        'sentiment_score': 0.0,
                        'sentiment_level': 'neutral',
                        'confidence': 0.5
                    }
                    
        except asyncio.TimeoutError:
            print("Timeout in AI analysis")
            return {
                'sentiment_score': 0.0,
                'sentiment_level': 'neutral',
                'confidence': 0.5
            }
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_level': 'neutral',
                'confidence': 0.5
            }
    
    async def process_single_review(self, review, session):
        """Process a single review for sentiment rescoring"""
        processed_review = review.copy()
        
        # Get content to analyze
        content = review.get('translated_content', review.get('content', ''))
        
        if content and len(content.strip()) > 10:  # Only analyze if content exists and is substantial
            # Analyze with AI
            analysis = await self.analyze_sentiment_with_ai(content, session)
            
            # Update with new sentiment data
            processed_review['ai_sentiment_score_new'] = analysis['sentiment_score']
            processed_review['ai_sentiment_level_new'] = analysis['sentiment_level']
            processed_review['ai_confidence_new'] = analysis['confidence']
            
            # Map sentiment level to category
            level_to_category = {
                'very_negative': 'negative',
                'negative': 'negative',
                'somewhat_negative': 'negative',
                'slightly_negative': 'neutral',
                'neutral': 'neutral',
                'slightly_positive': 'neutral',
                'somewhat_positive': 'positive',
                'positive': 'positive',
                'very_positive': 'positive'
            }
            
            processed_review['ai_sentiment_new'] = level_to_category.get(analysis['sentiment_level'], 'neutral')
        else:
            # Use existing sentiment if no content
            processed_review['ai_sentiment_score_new'] = review.get('ai_sentiment_score', 0.0)
            processed_review['ai_sentiment_level_new'] = self.get_sentiment_level(review.get('ai_sentiment_score', 0.0))
            processed_review['ai_confidence_new'] = 0.5
            processed_review['ai_sentiment_new'] = review.get('ai_sentiment', 'neutral')
        
        processed_review['rescored_at'] = datetime.now()
        
        return processed_review
    
    async def rescore_sentiments(self, reviews_df):
        """Rescore sentiments for all reviews"""
        print(f"🔄 Rescoring sentiments for {len(reviews_df)} reviews...")
        print(f"Workers: {self.max_workers}, Batch size: {self.batch_size}, Timeout: {self.timeout}s")
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.max_workers, limit_per_host=self.max_workers)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        processed_reviews = []
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process in batches with progress bar
            with tqdm(total=len(reviews_df), desc="Rescoring sentiments") as pbar:
                for i in range(0, len(reviews_df), self.batch_size):
                    batch = reviews_df.iloc[i:i+self.batch_size]
                    batch_tasks = []
                    
                    # Create tasks for all reviews in batch
                    for _, review in batch.iterrows():
                        task = self.process_single_review(review, session)
                        batch_tasks.append(task)
                    
                    # Execute batch with timeout
                    try:
                        batch_results = await asyncio.wait_for(
                            asyncio.gather(*batch_tasks, return_exceptions=True),
                            timeout=self.timeout * 2
                        )
                        
                        # Process results
                        for result in batch_results:
                            if isinstance(result, Exception):
                                print(f"❌ Review processing error: {result}")
                                # Use default processing
                                processed_reviews.append(self.get_default_processed_review(review))
                            else:
                                processed_reviews.append(result)
                        
                        # Update progress bar
                        pbar.update(len(batch))
                        
                    except asyncio.TimeoutError:
                        print(f"⏰ Batch timeout at index {i}")
                        # Process remaining reviews with defaults
                        for _, review in batch.iterrows():
                            processed_reviews.append(self.get_default_processed_review(review))
                        pbar.update(len(batch))
                    
                    # Minimal delay
                    await asyncio.sleep(0.01)
            
            pbar.close()
        
        return pd.DataFrame(processed_reviews)
    
    def get_default_processed_review(self, review):
        """Get default processed review when AI fails"""
        processed_review = review.copy()
        
        processed_review['ai_sentiment_score_new'] = review.get('ai_sentiment_score', 0.0)
        processed_review['ai_sentiment_level_new'] = self.get_sentiment_level(review.get('ai_sentiment_score', 0.0))
        processed_review['ai_confidence_new'] = 0.5
        processed_review['ai_sentiment_new'] = review.get('ai_sentiment', 'neutral')
        processed_review['rescored_at'] = datetime.now()
        
        return processed_review
    
    async def run_rescoring(self):
        """Run the complete rescoring process"""
        print("🚀 AI SENTIMENT RESCORING WITH 0.2 LEVEL PRECISION")
        print("=" * 60)
        
        # Load the database
        print(f"📂 Loading database: {self.db_file}")
        df = pd.read_excel(self.db_file)
        print(f"✅ Loaded {len(df)} reviews")
        
        # Rescore sentiments
        rescored_df = await self.rescore_sentiments(df)
        
        # Save results
        print(f"\n💾 Saving rescored database...")
        rescored_df.to_excel(self.output_file, index=False)
        print(f"✅ Saved to: {self.output_file}")
        
        # Generate summary
        print(f"\n📊 RESCORING SUMMARY:")
        print(f"Total reviews processed: {len(rescored_df)}")
        print(f"New sentiment score range: {rescored_df['ai_sentiment_score_new'].min():.2f} to {rescored_df['ai_sentiment_score_new'].max():.2f}")
        
        # Show sentiment level distribution
        level_counts = rescored_df['ai_sentiment_level_new'].value_counts()
        print(f"\nSentiment Level Distribution:")
        for level, count in level_counts.items():
            percentage = (count / len(rescored_df)) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        # Show confidence statistics
        avg_confidence = rescored_df['ai_confidence_new'].mean()
        print(f"\nAverage confidence: {avg_confidence:.3f}")
        
        return rescored_df

async def main():
    """Main function"""
    rescorer = AISentimentRescorer()
    await rescorer.run_rescoring()

if __name__ == "__main__":
    asyncio.run(main()) 