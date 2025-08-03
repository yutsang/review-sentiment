#!/usr/bin/env python3
"""
Hong Kong Banking Apps Review Scraper and Sentiment Analyzer
Scrapes reviews from App Store and Google Play Store for Hong Kong banks
and performs comprehensive sentiment analysis with improved translation
"""
import argparse
import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import textwrap
import requests
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.config import (
    load_config, load_app_config, get_app_config,
    list_available_apps, ensure_directories
)
from src.utils.logger import setup_logger
from src.scrapers import AppStoreScraper, PlayStoreScraper
from src.exporters import CSVExporter, XLSXExporter
from src.models import Review

# Initialize translator for improved translation (disabled for now)
translator = None

def safe_translate(text: str, config: Dict, max_retries: int = None, delay: float = None) -> str:
    """Safely translate text with fallback handling"""
    # Handle NaN, None, or non-string values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    if not text or not text.strip():
        return text
    
    translation_config = config.get('translation', {})
    service = translation_config.get('service', 'none')
    
    # If translation is disabled, return original text
    if service == 'none':
        return text
    
    # Use DeepSeek for translation
    if service == 'deepseek':
        return translate_with_deepseek(text, config)
    
    # Use original googletrans logic for other services
    if pd.isna(text) or text == "" or text is None:
        return ""
    
    # Get config values
    max_retries = max_retries or config.get("translation", {}).get("max_retries", 3)
    delay = delay or config.get("translation", {}).get("delay", 1)
    chunk_size = config.get("translation", {}).get("chunk_size", 4000)
    
    # If text is already in English, return as is
    text_str = str(text)[:chunk_size]  # Limit text length
    
    # Simple check for English text
    if text_str.isascii() and len([c for c in text_str if c.isalpha()]) > len(text_str) * 0.7:
        return text_str
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)  # Rate limiting
            result = translator.translate(text_str, dest='en')
            return result.text if result and result.text else text_str
        except Exception as e:
            print(f"Translation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
    
    return text_str  # Return original if all attempts fail

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text using TextBlob"""
    if not text or pd.isna(text):
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'neutral',
            'sentiment_words': [],
            'positive_words': [],
            'negative_words': []
        }
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    
    # Categorize sentiment
    if polarity > 0.1:
        category = 'positive'
    elif polarity < -0.1:
        category = 'negative'
    else:
        category = 'neutral'
    
    # Extract sentiment words (simplified)
    words = text.lower().split()
    positive_words = [word for word in words if word in ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'fantastic', 'awesome', 'perfect', 'wonderful']]
    negative_words = [word for word in words if word in ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'useless', 'broken', 'fail', 'disappointing']]
    
    return {
        'sentiment_score': polarity,
        'sentiment_category': category,
        'sentiment_words': positive_words + negative_words,
        'positive_words': positive_words,
        'negative_words': negative_words
    }

def categorize_review(review_text: str, title_text: str = "") -> str:
    """Categorize review into issue types based on content"""
    text = f"{title_text} {review_text}".lower()
    
    categories = {
        'login_auth': ['login', 'password', 'authentication', 'biometric', 'fingerprint', 'face id', 'pin'],
        'performance': ['slow', 'crash', 'freeze', 'lag', 'loading', 'hang', 'stuck', 'glitch'],
        'ui_ux': ['interface', 'design', 'layout', 'navigation', 'menu', 'button', 'screen'],
        'payment': ['payment', 'transfer', 'transaction', 'fps', 'card', 'deposit', 'withdrawal'],
        'feature': ['feature', 'function', 'service', 'option', 'tool', 'update'],
        'customer_service': ['support', 'help', 'customer service', 'staff', 'representative'],
        'security': ['security', 'fraud', 'hack', 'safe', 'protection', 'privacy']
    }
    
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    
    return 'general'

def categorize_problem_simple(review_text: str) -> str:
    """Categorize review into one of 5 specific problem categories"""
    if not review_text:
        return "General"
    
    text = review_text.lower()
    
    # Account Opening Issues
    if any(word in text for word in ['open account', 'account opening', 'registration', 'sign up', 'signup', 'register']):
        return "Account Opening Issues"
    
    # Account Verification Issues
    if any(word in text for word in ['verify', 'verification', 'kyc', 'identity', 'document', 'proof']):
        return "Account Verification Issues"
    
    # App Operating Issues
    if any(word in text for word in ['app', 'login', 'crash', 'error', 'bug', 'slow', 'freeze', 'technical', 'system']):
        return "App Operating Issues"
    
    # Rewards Issues
    if any(word in text for word in ['reward', 'bonus', 'cashback', 'points', 'promotion', 'offer']):
        return "Rewards Issues"
    
    # Default to General
    return "General"

def create_overview_json(results: Dict[str, Dict], config: Dict) -> Dict[str, Any]:
    """Create comprehensive overview JSON with exact review numbers, ratings, and sentiment analysis"""
    overview = {
        "generated_at": datetime.now().isoformat(),
        "total_apps": len(results),
        "summary": {
            "total_reviews_all_apps": 0,
            "total_app_store_reviews": 0,
            "total_play_store_reviews": 0,
            "average_rating_across_apps": 0.0,
            "average_sentiment_across_apps": 0.0
        },
        "apps": {}
    }
    
    total_reviews_all = 0
    total_app_store_all = 0
    total_play_store_all = 0
    total_weighted_rating = 0
    total_sentiment_score = 0
    apps_with_sentiment = 0
    
    for app_key, app_data in results.items():
        app_config = get_app_config(app_key)
        app_name = app_config.get('name', app_key)
        
        app_store_reviews = app_data.get('app_store', [])
        play_store_reviews = app_data.get('play_store', [])
        
        app_store_rating = 0
        play_store_rating = 0
        app_store_sentiment = 0
        play_store_sentiment = 0
        
        if app_store_reviews:
            app_store_rating = sum(r.rating for r in app_store_reviews) / len(app_store_reviews)
            # Calculate sentiment if available
            sentiment_scores = [getattr(r, 'sentiment_score', 0) for r in app_store_reviews if hasattr(r, 'sentiment_score')]
            if sentiment_scores:
                app_store_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        if play_store_reviews:
            play_store_rating = sum(r.rating for r in play_store_reviews) / len(play_store_reviews)
            # Calculate sentiment if available
            sentiment_scores = [getattr(r, 'sentiment_score', 0) for r in play_store_reviews if hasattr(r, 'sentiment_score')]
            if sentiment_scores:
                play_store_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Calculate weighted rating and sentiment
        total_reviews = len(app_store_reviews) + len(play_store_reviews)
        weighted_rating = 0
        weighted_sentiment = 0
        
        if total_reviews > 0:
            weighted_rating = ((len(app_store_reviews) * app_store_rating) + 
                             (len(play_store_reviews) * play_store_rating)) / total_reviews
            weighted_sentiment = ((len(app_store_reviews) * app_store_sentiment) + 
                                (len(play_store_reviews) * play_store_sentiment)) / total_reviews
        
        # Update summary totals
        total_reviews_all += total_reviews
        total_app_store_all += len(app_store_reviews)
        total_play_store_all += len(play_store_reviews)
        total_weighted_rating += weighted_rating * total_reviews
        if weighted_sentiment > 0:
            total_sentiment_score += weighted_sentiment * total_reviews
            apps_with_sentiment += 1
        
        overview["apps"][app_key] = {
            "name": app_name,
            "app_store": {
                "reviews_count": len(app_store_reviews),
                "average_rating": round(app_store_rating, 2),
                "average_sentiment": round(app_store_sentiment, 3)
            },
            "play_store": {
                "reviews_count": len(play_store_reviews),
                "average_rating": round(play_store_rating, 2),
                "average_sentiment": round(play_store_sentiment, 3)
            },
            "total_reviews": total_reviews,
            "weighted_rating": round(weighted_rating, 2),
            "weighted_sentiment": round(weighted_sentiment, 3),
            "sentiment_category": "positive" if weighted_sentiment > 0.6 else "negative" if weighted_sentiment < 0.4 else "neutral"
        }
    
    # Calculate overall averages
    if total_reviews_all > 0:
        overview["summary"]["total_reviews_all_apps"] = total_reviews_all
        overview["summary"]["total_app_store_reviews"] = total_app_store_all
        overview["summary"]["total_play_store_reviews"] = total_play_store_all
        overview["summary"]["average_rating_across_apps"] = round(total_weighted_rating / total_reviews_all, 2)
        if apps_with_sentiment > 0:
            overview["summary"]["average_sentiment_across_apps"] = round(total_sentiment_score / total_reviews_all, 3)
    
    return overview

def call_deepseek_api(text: str, config: Dict) -> Optional[str]:
    """Call DeepSeek API to get AI-processed keywords for wordcloud"""
    try:
        deepseek_config = config.get('deepseek', {})
        
        if not deepseek_config.get('enabled', False):
            return None
            
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            print("⚠️ DeepSeek API key not configured. Skipping AI wordcloud processing.")
            return None
            
        base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = deepseek_config.get('model', 'deepseek-chat')
        max_tokens = deepseek_config.get('max_tokens', 1000)
        temperature = deepseek_config.get('temperature', 0.3)
        prompt_template = deepseek_config.get('wordcloud_prompt', '')
        
        # Truncate text if too long (keep first 5000 characters)
        truncated_text = text[:5000] + "..." if len(text) > 5000 else text
        
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
            # Clean up the response - extract only phrases, preserve spaces between words
            # Remove extra punctuation but keep spaces for multi-word phrases
            phrases = re.sub(r'[^\w\s]', ' ', content).strip()
            # Remove extra whitespace
            phrases = ' '.join(phrases.split())
            return phrases
        else:
            print(f"⚠️ DeepSeek API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"⚠️ DeepSeek API call failed: {e}")
        return None

def analyze_sentiment_deepseek(text: str, config: Dict) -> Dict[str, Any]:
    """Analyze sentiment using DeepSeek API"""
    try:
        deepseek_config = config.get('deepseek', {})
        
        if not deepseek_config.get('enabled', False):
            # Fallback to TextBlob if DeepSeek is disabled
            return analyze_sentiment(text)
            
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            return analyze_sentiment(text)
            
        base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = deepseek_config.get('model', 'deepseek-chat')
        max_tokens = deepseek_config.get('max_tokens', 1000)
        temperature = deepseek_config.get('temperature', 0.3)
        sentiment_prompt = deepseek_config.get('sentiment_prompt', 'Analyze the sentiment of the following text. Return only a JSON object with "sentiment" (positive/negative/neutral), "score" (0.0 to 1.0), and "confidence" (0.0 to 1.0). Text: {text}')
        
        # Format the prompt
        formatted_prompt = sentiment_prompt.format(review=text)
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': formatted_prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        response = requests.post(f'{base_url}/chat/completions', headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse JSON response
            try:
                import json
                sentiment_data = json.loads(content)
                return {
                    'sentiment_score': sentiment_data.get('score', 0.5),
                    'sentiment_category': sentiment_data.get('sentiment', 'neutral'),
                    'confidence': sentiment_data.get('confidence', 0.5)
                }
            except json.JSONDecodeError:
                # Fallback to TextBlob if JSON parsing fails
                return analyze_sentiment(text)
        else:
            print(f"⚠️ DeepSeek API error: {response.status_code}")
            return analyze_sentiment(text)
            
    except Exception as e:
        print(f"⚠️ DeepSeek sentiment analysis failed: {e}")
        return analyze_sentiment(text)

def translate_with_deepseek(text: str, config: Dict) -> str:
    """Translate text using DeepSeek API"""
    try:
        if not text or not text.strip():
            return text
            
        deepseek_config = config.get('deepseek', {})
        
        if not deepseek_config.get('enabled', False):
            return text
            
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            return text
            
        base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = deepseek_config.get('model', 'deepseek-chat')
        max_tokens = deepseek_config.get('max_tokens', 1000)
        temperature = deepseek_config.get('temperature', 0.3)
        
        # Translation prompt
        translation_prompt = f"Translate the following text to English. Keep the original meaning and tone. Return only the translated text, no explanations. Text: {text}"
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': translation_prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        response = requests.post(f'{base_url}/chat/completions', headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            return translated_text
        else:
            print(f"⚠️ DeepSeek translation error: {response.status_code}")
            return text
            
    except Exception as e:
        print(f"⚠️ DeepSeek translation failed: {e}")
        return text


class AppReviewScraper:
    """Hong Kong Banking Apps Review Scraper with Enhanced Analysis"""
    
    def __init__(self, config_path: str = "config/settings.json",
                 apps_config_path: str = "config/apps.json"):
        self.config = load_config(config_path)
        self.app_config = load_app_config(apps_config_path)
        
        # Setup logging
        self.logger = setup_logger(self.config)
        
        # Ensure output directories exist
        ensure_directories()
        
        # Initialize exporters
        self.csv_exporter = CSVExporter(self.config)
        self.xlsx_exporter = XLSXExporter(self.config)
    
        # Setup matplotlib for Chinese font support
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    def scrape_app(self, app_config: Dict) -> Dict[str, List[Review]]:
        """Scrape reviews for specified app"""
        app_name = app_config.get("name", app_config.get("app_key"))
        self.logger.info(f"🚀 Starting review scraping for: {app_name}")
        
            platforms = ["app_store", "play_store"]
        
        results = {}
        
        # Scrape App Store
        if "app_store" in platforms and app_config.get("app_store", {}).get("enabled", True):
            try:
                scraper = AppStoreScraper(self.config, app_config)
                reviews = scraper.scrape_reviews()
                if reviews:
                    results["app_store"] = reviews
                    stats = scraper.get_stats(reviews)
                    self.logger.info(f"🍎 App Store Results:")
                    self._log_platform_stats("🍎", stats)
                else:
                    self.logger.warning("🍎 No App Store reviews found")
            except Exception as e:
                self.logger.error(f"🍎 App Store scraping failed: {e}")
        
        # Scrape Play Store
        if "play_store" in platforms and app_config.get("play_store", {}).get("enabled", True):
            try:
                scraper = PlayStoreScraper(self.config, app_config)
                reviews = scraper.scrape_reviews()
                if reviews:
                    results["play_store"] = reviews
                    stats = scraper.get_stats(reviews)
                    self.logger.info(f"🤖 Play Store Results:")
                    self._log_platform_stats("🤖", stats)
                else:
                    self.logger.warning("🤖 No Play Store reviews found")
            except Exception as e:
                self.logger.error(f"🤖 Play Store scraping failed: {e}")
        
        return results
    
    def scrape_all_apps(self, platforms: List[str] = None) -> Dict[str, Dict[str, List[Review]]]:
        """Scrape all configured apps"""
        all_results = {}
        apps = list_available_apps(self.app_config)
        
        for app_key, app_name in apps.items():
            print(f"\n{'='*60}")
            print(f"🏦 Processing {app_name} ({app_key})")
            print(f"{'='*60}")
            
            try:
                results = self.scrape_app(self.app_config.get(app_key))
                if results:
                    all_results[app_key] = results
                    self.display_summary(app_key, results, None)
                else:
                    print(f"⚠️ No reviews found for {app_name}")
            except Exception as e:
                print(f"❌ Error processing {app_name}: {e}")
                self.logger.error(f"Error processing {app_key}: {e}")
        
        return all_results
    
    def export_results(self, app_key: str, results: Dict[str, List[Review]], 
                      timestamp_subfolder: str = None) -> str:
        """Export scraping results to output folder"""
        # Create output directory
        output_dir = self.config.get("output", {}).get("directory", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all reviews
        all_reviews = []
        for reviews in results.values():
            all_reviews.extend(reviews)
        
        if not all_reviews:
            self.logger.warning("No reviews to export")
            return ""
        
        # Generate filename for review files
        filename = f"{app_key}_reviews"
        
        # Export as XLSX to the output directory
        xlsx_file = self.xlsx_exporter.export(all_reviews, filename, output_dir)
        return xlsx_file if xlsx_file else ""
    
    def process_with_analysis(self, app_key: str, results: Dict[str, List[Review]], 
                            timestamp_subfolder: str = None) -> str:
        """Process reviews with comprehensive sentiment analysis"""
        # Export raw data first
        exported_file = self.export_results(app_key, results, timestamp_subfolder)
        
        if not exported_file or not os.path.exists(exported_file):
            return exported_file
        
        # Run comprehensive analysis
        try:
            analysis_results = self.run_comprehensive_analysis(exported_file, app_key, timestamp_subfolder)
            print(f"✅ Analysis completed! Results saved to: {analysis_results}")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            self.logger.error(f"Analysis failed for {app_key}: {e}")
        
        return exported_file
    
    def run_comprehensive_analysis(self, file_path: str, app_key: str, 
                                 timestamp_subfolder: str = None) -> str:
        """Run simplified analysis with tqdm progress bars"""
        print(f"\n🔬 Starting simplified analysis for {app_key}...")
        
        # Use provided timestamp or create new one
        if timestamp_subfolder:
            timestamp = timestamp_subfolder
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create analysis subfolder
        base_output_dir = self.config.get("output", {}).get("directory", "output")
        analysis_dir = os.path.join(base_output_dir, f"analysis_{timestamp}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Read the data
        df = pd.read_excel(file_path)
        print(f"📊 Processing {len(df)} reviews...")
        
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
            args_list = [(idx, row, self.config) for idx, row in df.iterrows()]
            
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
        
        # Save analyzed data to analysis subfolder
        analyzed_file = os.path.join(analysis_dir, f"{app_key}_analyzed.xlsx")
        df.to_excel(analyzed_file, index=False)
        
        # Generate visualizations in analysis subfolder
        self.create_analysis_charts(df, app_key, analysis_dir)
        
        # Create word cloud only for WeLab Bank
        deepseek_config = self.config.get('deepseek', {})
        wordcloud_enabled_for = deepseek_config.get('enable_wordcloud_only_for', [])
        if app_key in wordcloud_enabled_for:
            self.create_wordclouds(df, app_key, analysis_dir)
        else:
            print(f"📊 Skipping wordcloud generation for {app_key} (not in wordcloud_enabled_for list)")
        
        # Generate summary report in analysis subfolder
        summary = self.generate_summary_report(df, app_key)
        summary_file = os.path.join(analysis_dir, f"{app_key}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Simplified analysis completed for {app_key}")
        print(f"📁 Analysis files saved to: {analysis_dir}")
        return analyzed_file
    
    def create_analysis_charts(self, df: pd.DataFrame, app_key: str, output_dir: str):
        """Create analysis charts with configured colors"""
        colors = self.config.get("colors", {})
        sentiment_colors = colors.get("sentiment_colors", {
            "positive": "#2ca02c", "negative": "#d62728", "neutral": "#1f77b4"
        })
        
        # Set style
        plt.style.use('default')
        
        # Create sentiment distribution chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall sentiment distribution
        sentiment_counts = df['overall_sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       colors=[sentiment_colors.get(cat, '#999999') for cat in sentiment_counts.index],
                       autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        axes[0, 1].bar(rating_counts.index, rating_counts.values, 
                       color=colors.get("chart_palette", ['#1f77b4'])[0])
        axes[0, 1].set_title('Rating Distribution')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Count')
        
        # Problem category distribution
        category_counts = df['problem_category'].value_counts()
        axes[1, 0].barh(category_counts.index, category_counts.values,
                        color=colors.get("chart_palette", ['#1f77b4'] * len(category_counts))[0])
        axes[1, 0].set_title('Problem Categories')
        axes[1, 0].set_xlabel('Count')
        
        # Sentiment over time (if date available)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Remove timezone info to avoid warning
            df['date'] = df['date'].dt.tz_localize(None)
            monthly_sentiment = df.groupby([df['date'].dt.to_period('M'), 'overall_sentiment']).size().unstack(fill_value=0)
            monthly_sentiment.plot(kind='area', ax=axes[1, 1], 
                                 color=[sentiment_colors.get(col, '#999999') for col in monthly_sentiment.columns])
            axes[1, 1].set_title('Sentiment Trends Over Time')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Review Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'Date data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Sentiment Trends Over Time')
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f"{app_key}_analysis_charts.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        

    
    def create_wordclouds(self, df: pd.DataFrame, app_key: str, output_dir: str):
        """Create word clouds for positive and negative reviews with 2-5 word phrases"""
        try:
            # Set font to Arial
            plt.rcParams['font.family'] = 'Arial'
            
            # Positive reviews word cloud
            positive_reviews = df[df['overall_sentiment'] == 'positive']['translated_content'].dropna()
            if not positive_reviews.empty:
                positive_text = ' '.join(positive_reviews.astype(str))
                
                # Use DeepSeek to extract 2-5 word phrases
                ai_enhanced_text = call_deepseek_api(positive_text, self.config)
                if ai_enhanced_text and len(ai_enhanced_text.split()) > 10:  # Ensure we got meaningful phrases
                    print(f"🤖 Using AI-enhanced 2-5 word phrases for {app_key} positive reviews")
                    print(f"📝 Sample phrases: {' '.join(ai_enhanced_text.split()[:20])}")
                    wordcloud_text = ai_enhanced_text
                else:
                    print(f"📝 Using fallback phrase extraction for {app_key} positive reviews")
                    # Fallback: extract common banking phrases manually
                    wordcloud_text = self.extract_banking_phrases(positive_text)
                    print(f"📝 Fallback phrases: {' '.join(wordcloud_text.split()[:20])}")
                
                # Create positive wordcloud with Pacific Blue color (0, 184, 245)
                # Pre-process text to preserve phrases
                processed_text = self.preserve_phrases_for_wordcloud(wordcloud_text)
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
                ai_enhanced_text = call_deepseek_api(negative_text, self.config)
                if ai_enhanced_text and len(ai_enhanced_text.split()) > 10:  # Ensure we got meaningful phrases
                    print(f"🤖 Using AI-enhanced 2-5 word phrases for {app_key} negative reviews")
                    print(f"📝 Sample phrases: {' '.join(ai_enhanced_text.split()[:20])}")
                    wordcloud_text = ai_enhanced_text
                else:
                    print(f"📝 Using fallback phrase extraction for {app_key} negative reviews")
                    # Fallback: extract common banking phrases manually
                    wordcloud_text = self.extract_banking_phrases(negative_text)
                    print(f"📝 Fallback phrases: {' '.join(wordcloud_text.split()[:20])}")
                
                # Create negative wordcloud with Dark Blue color (12, 35, 60)
                # Pre-process text to preserve phrases
                processed_text = self.preserve_phrases_for_wordcloud(wordcloud_text)
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
    
    def generate_summary_report(self, df: pd.DataFrame, app_key: str) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        total_reviews = len(df)
        
        summary = {
            "app_key": app_key,
            "generated_at": datetime.now().isoformat(),
            "total_reviews": total_reviews,
            "overall_sentiment_distribution": df['overall_sentiment'].value_counts().to_dict(),
            "rating_distribution": df['rating'].value_counts().sort_index().to_dict(),
            "problem_category_distribution": df['problem_category'].value_counts().to_dict(),
            "average_rating": float(df['rating'].mean()) if 'rating' in df.columns else 0,
            "average_sentiment_score": float(df['sentiment_score'].mean()),
            "platform_distribution": df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
            "top_positive_words": df['positive_words'].str.split(', ').explode().value_counts().head(10).to_dict(),
            "top_negative_words": df['negative_words'].str.split(', ').explode().value_counts().head(10).to_dict(),
            "recommendations": self.generate_recommendations(df)
        }
        
        return summary
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Sentiment-based recommendations
        negative_ratio = len(df[df['overall_sentiment'] == 'negative']) / len(df)
        if negative_ratio > 0.3:
            recommendations.append("High negative sentiment detected. Focus on addressing user concerns.")
        
        # Problem category-based recommendations
        top_categories = df['problem_category'].value_counts().head(3)
        for category, count in top_categories.items():
            if category == 'App Operating Issues' and count > len(df) * 0.2:
                recommendations.append("App operating issues frequently mentioned. Consider app optimization.")
            elif category == 'Account Verification Issues' and count > len(df) * 0.15:
                recommendations.append("Account verification issues reported. Review KYC process.")
            elif category == 'Account Opening Issues' and count > len(df) * 0.15:
                recommendations.append("Account opening issues raised. Consider streamlining registration.")
        
        # Rating-based recommendations
        avg_rating = df['rating'].mean()
        if avg_rating < 3.0:
            recommendations.append("Low average rating. Urgent attention needed for app improvements.")
        elif avg_rating < 4.0:
            recommendations.append("Below-average rating. Focus on user satisfaction improvements.")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _log_platform_stats(self, platform_name: str, stats: Dict[str, Any]):
        """Log platform statistics"""
        if not stats:
            return
        
        total = stats.get("total_reviews", 0)
        avg_rating = stats.get("average_rating", 0)
        
        self.logger.info(f"  Total Reviews: {total:,}")
        self.logger.info(f"  Average Rating: {avg_rating:.2f}")
        
        # Rating distribution
        distribution = stats.get("rating_distribution", {})
        for rating in sorted(distribution.keys()):
            count = distribution[rating]
            percentage = (count / total * 100) if total > 0 else 0
            stars = "★" * rating + "☆" * (5 - rating)
            self.logger.info(f"    {stars} ({rating}): {count} reviews ({percentage:.1f}%)")
    
    def display_summary(self, app_key: str, results: Dict[str, List[Review]], 
                       exported_file: str):
        """Display final summary"""
        app_config = get_app_config(app_key)
        app_name = app_config.get("name", app_key) if app_config else app_key
        
        # Calculate totals
        app_store_count = len(results.get("app_store", []))
        play_store_count = len(results.get("play_store", []))
        total_reviews = app_store_count + play_store_count
        
        # Display summary
        print(f"\n🎯 {app_name.upper()} SCRAPING COMPLETED")
        print("=" * 60)
        print("📊 FINAL RESULTS:")
        if app_store_count > 0:
            print(f"   🍎 App Store: {app_store_count:,} reviews")
        if play_store_count > 0:
            print(f"   🤖 Play Store: {play_store_count:,} reviews")
        print(f"   📋 Total Reviews: {total_reviews:,}")
        
        if exported_file:
            print(f"\n📁 File Generated:")
            print(f"   📄 {exported_file}")
        
    def export_results_multi(self, app_keys: List[str], all_results: Dict[str, Dict[str, List[Review]]], 
                            all_reviews: List[Review], timestamp: str, base_filename: str) -> str:
        """Export results from multiple apps to single folder"""
        # Create output directory
        output_dir = os.path.join(self.config.get("output", {}).get("directory", "output"), timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to Excel
        filename = f"{base_filename}.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        # Create mapping of review to bank name
        review_to_bank = {}
        for app_key, results in all_results.items():
            for platform_reviews in results.values():
                for review in platform_reviews:
                    review_to_bank[review.review_id] = app_key.replace('_', ' ').title()
        
        # Create Excel with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Combined sheet
            if all_reviews:
                combined_df = pd.DataFrame([{
                    'Bank': review_to_bank.get(review.review_id, 'Unknown'),
                    'Platform': review.platform,
                    'Review ID': review.review_id,
                    'Author': review.author,
                    'Rating': review.rating,
                    'Review Text': review.content,
                    'Date': review.date,
                    'Version': review.version
                } for review in all_reviews])
                combined_df.to_excel(writer, sheet_name='All_Reviews', index=False)
            
            # Individual app sheets
            for app_key, results in all_results.items():
                app_reviews = []
                for platform_reviews in results.values():
                    app_reviews.extend(platform_reviews)
                
                if app_reviews:
                    app_df = pd.DataFrame([{
                        'Platform': review.platform,
                        'Review ID': review.review_id,
                        'Author': review.author,
                        'Rating': review.rating,
                        'Review Text': review.content,
                        'Date': review.date,
                        'Version': review.version
                    } for review in app_reviews])
                    
                    sheet_name = app_key.replace('_', ' ').title()[:31]  # Excel sheet name limit
                    app_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # If no data, create an empty sheet
            if not all_reviews:
                empty_df = pd.DataFrame({'Message': ['No reviews collected']})
                empty_df.to_excel(writer, sheet_name='Results', index=False)
        
        self.logger.info(f"📊 Multi-app XLSX exported: {filepath} ({len(all_reviews)} total reviews)")
        return filepath

    def process_with_analysis_multi(self, app_keys: List[str], all_results: Dict[str, Dict[str, List[Review]]], 
                                  all_reviews: List[Review], timestamp: str, base_filename: str) -> str:
        """Process multiple apps with combined analysis"""
        # Create output directory
        output_dir = os.path.join(self.config.get("output", {}).get("directory", "output"), timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create mapping of review to bank name
        review_to_bank = {}
        for app_key, results in all_results.items():
            bank_name = app_key.replace('_', ' ').title()
            for platform_reviews in results.values():
                for review in platform_reviews:
                    review_to_bank[review.review_id] = bank_name
        
        # Translate reviews
        print("🌐 Translating reviews...")
        translated_reviews = []
        for i, review in enumerate(all_reviews):
            if i % 50 == 0:
                print(f"   Translated {i}/{len(all_reviews)} reviews...")
            
            translated_content = safe_translate(review.content, self.config)
            bank_name = review_to_bank.get(review.review_id, 'Unknown')
            
            translated_reviews.append({
                'bank': bank_name,
                'platform': review.platform,
                'review_id': review.review_id,
                'author': review.author,
                'rating': review.rating,
                'original_content': review.content,
                'translated_content': translated_content,
                'date': review.date,
                'version': review.version
            })
        
        # Perform sentiment analysis
        print("🔍 Performing sentiment analysis...")
        for i, review_data in enumerate(translated_reviews):
            if i % 50 == 0:
                print(f"   Analyzed {i}/{len(translated_reviews)} reviews...")
            
            sentiment_result = analyze_sentiment(review_data['translated_content'])
            review_data.update(sentiment_result)
        
        # Create comprehensive Excel file
        filename = f"{base_filename}_analysis.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for app_key in app_keys:
                bank_name = app_key.replace('_', ' ').title()
                app_reviews = [r for r in translated_reviews if r['bank'] == bank_name]
                if app_reviews:
                    avg_rating = sum(r['rating'] for r in app_reviews) / len(app_reviews)
                    sentiment_dist = {}
                    for r in app_reviews:
                        cat = r.get('sentiment_category', 'neutral')
                        sentiment_dist[cat] = sentiment_dist.get(cat, 0) + 1
                    
                    summary_data.append({
                        'Bank': bank_name,
                        'Total Reviews': len(app_reviews),
                        'Average Rating': round(avg_rating, 2),
                        'Positive Sentiment': sentiment_dist.get('positive', 0),
                        'Negative Sentiment': sentiment_dist.get('negative', 0),
                        'Neutral Sentiment': sentiment_dist.get('neutral', 0)
                    })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # All reviews with analysis
            analysis_df = pd.DataFrame(translated_reviews)
            analysis_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
        
        # Generate visualizations
        print("🎨 Creating visualizations...")
        self.create_combined_visualizations(translated_reviews, output_dir, base_filename)
        
        self.logger.info(f"📊 Multi-app analysis complete: {filepath}")
        return filepath
    
    def create_overview_json(self, overview_data: Dict[str, Dict], timestamp: str, base_filename: str) -> str:
        """Create overview JSON with exact review counts and ratings"""
        output_dir = os.path.join(self.config.get("output", {}).get("directory", "output"), timestamp)
        
        # Calculate totals and weighted averages
        total_reviews = sum(data["total_reviews"] for data in overview_data.values())
        
        if total_reviews > 0:
            weighted_rating = sum(
                data["total_reviews"] * data["average_rating"] 
                for data in overview_data.values() 
                if data["total_reviews"] > 0
            ) / total_reviews
        else:
            weighted_rating = 0
        
        overview = {
            "analysis_timestamp": timestamp,
            "total_banks_analyzed": len(overview_data),
            "total_reviews_collected": total_reviews,
            "weighted_average_rating": round(weighted_rating, 2),
            "banks": overview_data,
            "platform_totals": {
                "app_store_total": sum(data["app_store_reviews"] for data in overview_data.values()),
                "play_store_total": sum(data["play_store_reviews"] for data in overview_data.values())
            }
        }
        
        filename = f"{base_filename}_overview.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(overview, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 Overview JSON created: {filepath}")
        return filepath

    def create_comprehensive_overview(self, timestamp: str) -> str:
        """Create comprehensive overview of ALL apps in config"""
        output_dir = os.path.join(self.config.get("output", {}).get("directory", "output"), timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        overview_file = os.path.join(output_dir, f"ALL_BANKS_OVERVIEW_{timestamp}.json")
        
        # Load all app configs
        app_config_data = load_app_config()
        all_apps = app_config_data.get("apps", {})
        
        comprehensive_overview = {
            "timestamp": timestamp,
            "total_banks": len(all_apps),
            "banks": {}
        }
        
        print(f"\n🔍 CREATING COMPREHENSIVE OVERVIEW FOR ALL {len(all_apps)} BANKS...")
        print("="*80)
        
        for app_key, app_config in all_apps.items():
            bank_name = app_config.get("name", app_key.replace("_", " ").title())
            print(f"\n📱 Checking {bank_name}...")
            
            bank_data = {
                "name": bank_name,
                "app_store": {
                    "enabled": app_config.get("app_store", {}).get("enabled", False),
                    "app_id": app_config.get("app_store", {}).get("app_id", ""),
                    "reviews_count": 0,
                    "total_available": 0,
                    "average_rating": 0
                },
                "play_store": {
                    "enabled": app_config.get("play_store", {}).get("enabled", False),
                    "package_name": app_config.get("play_store", {}).get("package_name", ""),
                    "reviews_count": 0,
                    "total_available": 0,
                    "average_rating": 0
                },
                "total_reviews": 0,
                "total_available_all": 0,
                "weighted_rating": 0
            }
            
            # Quick check Play Store
            if bank_data["play_store"]["enabled"]:
                try:
                    play_scraper = PlayStoreScraper(self.config, app_config)
                    play_reviews = play_scraper.scrape_reviews()
                    
                    # Get total available from app info
                    try:
                        from google_play_scraper import app
                        app_info = app(app_config.get("play_store", {}).get("package_name", ""), 
                                     country="hk", lang="en")
                        total_available_play = app_info.get('reviews', 0)
                        bank_data["play_store"]["total_available"] = total_available_play
                    except:
                        bank_data["play_store"]["total_available"] = 0
                    
                    if play_reviews:
                        bank_data["play_store"]["reviews_count"] = len(play_reviews)
                        ratings = [r.rating for r in play_reviews if r.rating and r.rating > 0]
                        if ratings:
                            bank_data["play_store"]["average_rating"] = round(sum(ratings) / len(ratings), 2)
                        print(f"  🤖 Play Store: {len(play_reviews)} scraped / {bank_data['play_store']['total_available']} total, avg rating: {bank_data['play_store']['average_rating']}")
                    else:
                        print(f"  🤖 Play Store: No reviews scraped / {bank_data['play_store']['total_available']} total available")
        except Exception as e:
                    print(f"  🤖 Play Store: Error - {str(e)[:50]}...")
            
            # Quick check App Store (with total available info)
            if bank_data["app_store"]["enabled"]:
                try:
                    app_scraper = AppStoreScraper(self.config, app_config)
                    app_reviews = app_scraper.scrape_reviews()
                    
                    # For App Store, we don't have easy access to total count via API
                    # But we can estimate based on what we tried to scrape
                    estimated_total = 0
                    
                    if app_reviews:
                        bank_data["app_store"]["reviews_count"] = len(app_reviews)
                        ratings = [r.rating for r in app_reviews if r.rating and r.rating > 0]
                        if ratings:
                            bank_data["app_store"]["average_rating"] = round(sum(ratings) / len(ratings), 2)
                        estimated_total = len(app_reviews)  # Conservative estimate
                        print(f"  🍎 App Store: {len(app_reviews)} scraped / ~{estimated_total}+ total, avg rating: {bank_data['app_store']['average_rating']}")
                    else:
                        print(f"  🍎 App Store: No reviews scraped (RSS may be disabled)")
                    
                    bank_data["app_store"]["total_available"] = estimated_total
                    
                except Exception as e:
                    print(f"  🍎 App Store: Error - {str(e)[:50]}...")
            
            # Calculate totals
            bank_data["total_reviews"] = bank_data["app_store"]["reviews_count"] + bank_data["play_store"]["reviews_count"]
            bank_data["total_available_all"] = bank_data["app_store"]["total_available"] + bank_data["play_store"]["total_available"]
            
            # Calculate weighted rating
            if bank_data["total_reviews"] > 0:
                app_weight = bank_data["app_store"]["reviews_count"]
                play_weight = bank_data["play_store"]["reviews_count"]
                
                weighted_sum = (bank_data["app_store"]["average_rating"] * app_weight + 
                              bank_data["play_store"]["average_rating"] * play_weight)
                bank_data["weighted_rating"] = round(weighted_sum / bank_data["total_reviews"], 2)
            
            comprehensive_overview["banks"][app_key] = bank_data
            
            scraping_efficiency = (bank_data["total_reviews"] / bank_data["total_available_all"] * 100) if bank_data["total_available_all"] > 0 else 0
            print(f"  📊 Scraped: {bank_data['total_reviews']} / {bank_data['total_available_all']} total ({scraping_efficiency:.1f}%), weighted rating: {bank_data['weighted_rating']}")
        
        # Save comprehensive overview
        with open(overview_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_overview, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Comprehensive overview saved: {overview_file}")
        
        # Display summary table
        self._display_comprehensive_summary(comprehensive_overview)
        
        return overview_file

    def create_combined_visualizations(self, reviews_data: List[Dict], output_dir: str, base_filename: str):
        """Create combined visualizations for all banks"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from wordcloud import WordCloud
        
        # Get colors from config
        colors = self.config.get("colors", {})
        chart_palette = colors.get("chart_palette", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        
        plt.style.use('default')
        
        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hong Kong Banking Apps Review Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Rating distribution by bank
        banks = {}
        for review in reviews_data:
            bank = review['bank']
            if bank not in banks:
                banks[bank] = []
            banks[bank].append(review['rating'])
        
        bank_names = list(banks.keys())
        avg_ratings = [sum(ratings)/len(ratings) for ratings in banks.values()]
        
        axes[0,0].bar(range(len(bank_names)), avg_ratings, color=chart_palette[:len(bank_names)])
        axes[0,0].set_title('Average Rating by Bank')
        axes[0,0].set_xticks(range(len(bank_names)))
        axes[0,0].set_xticklabels([name.replace('_', '\n') for name in bank_names], rotation=45)
        axes[0,0].set_ylabel('Average Rating')
        
        # 2. Sentiment distribution
        sentiment_counts = {}
        for review in reviews_data:
            sentiment = review.get('sentiment_category', 'neutral')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        sentiment_colors = colors.get("sentiment_colors", {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#1f77b4"})
        axes[0,1].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%',
                     colors=[sentiment_colors.get(k, "#999999") for k in sentiment_counts.keys()])
        axes[0,1].set_title('Overall Sentiment Distribution')
        
        # 3. Reviews by platform
        platform_counts = {}
        for review in reviews_data:
            platform = review['platform']
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        axes[0,2].pie(platform_counts.values(), labels=platform_counts.keys(), autopct='%1.1f%%')
        axes[0,2].set_title('Reviews by Platform')
        
        # 4. Rating distribution histogram
        all_ratings = [review['rating'] for review in reviews_data]
        rating_colors = colors.get("rating_colors", {})
        
        rating_counts = {}
        for rating in all_ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        bars = axes[1,0].bar(rating_counts.keys(), rating_counts.values())
        for i, bar in enumerate(bars):
            rating = list(rating_counts.keys())[i]
            bar.set_color(rating_colors.get(str(rating), chart_palette[i % len(chart_palette)]))
        
        axes[1,0].set_title('Rating Distribution (All Banks)')
        axes[1,0].set_xlabel('Rating')
        axes[1,0].set_ylabel('Count')
        
        # 5. Review volume by bank
        review_counts = {}
        for review in reviews_data:
            bank = review['bank']
            review_counts[bank] = review_counts.get(bank, 0) + 1
        
        axes[1,1].bar(range(len(review_counts)), review_counts.values(), color=chart_palette[:len(review_counts)])
        axes[1,1].set_title('Review Volume by Bank')
        axes[1,1].set_xticks(range(len(review_counts)))
        axes[1,1].set_xticklabels([name.replace('_', '\n') for name in review_counts.keys()], rotation=45)
        axes[1,1].set_ylabel('Number of Reviews')
        
        # 6. Sentiment by bank heatmap
        sentiment_by_bank = {}
        for bank in bank_names:
            sentiment_by_bank[bank] = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for review in reviews_data:
            bank = review['bank']
            sentiment = review.get('sentiment_category', 'neutral')
            sentiment_by_bank[bank][sentiment] += 1
        
        # Convert to percentages
        heatmap_data = []
        for bank in bank_names:
            total = sum(sentiment_by_bank[bank].values())
            if total > 0:
                row = [sentiment_by_bank[bank][sentiment] / total * 100 for sentiment in ['positive', 'neutral', 'negative']]
            else:
                row = [0, 0, 0]
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', xticklabels=['Positive', 'Neutral', 'Negative'],
                   yticklabels=[name.replace('_', ' ') for name in bank_names], ax=axes[1,2])
        axes[1,2].set_title('Sentiment Distribution by Bank (%)')
        
        plt.tight_layout()
        dashboard_path = os.path.join(output_dir, f"{base_filename}_dashboard.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create word clouds
        self.create_combined_wordclouds(reviews_data, output_dir, base_filename)

    def create_combined_wordclouds(self, reviews_data: List[Dict], output_dir: str, base_filename: str):
        """Create word clouds for combined analysis with AI-enhanced processing"""
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Set font to Arial
        plt.rcParams['font.family'] = 'Arial'
        
        # Separate by sentiment
        positive_text = " ".join([review['translated_content'] for review in reviews_data 
                                if review.get('sentiment_category') == 'positive'])
        negative_text = " ".join([review['translated_content'] for review in reviews_data 
                                if review.get('sentiment_category') == 'negative'])
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        if positive_text.strip():
            # Try AI-enhanced processing first
            ai_enhanced_text = call_deepseek_api(positive_text, self.config)
            if ai_enhanced_text:
                print(f"🤖 Using AI-enhanced keywords for combined positive reviews")
                wordcloud_text = ai_enhanced_text
            else:
                print(f"📝 Using standard text processing for combined positive reviews")
                wordcloud_text = positive_text
            
            wordcloud_pos = WordCloud(
                width=800, height=400, 
                background_color='white',
                colormap='Greens',
                font_path=None,  # Use default font
                max_words=100,
                stopwords=None,
                min_font_size=10,
                max_font_size=80
            ).generate(wordcloud_text)
            
            axes[0].imshow(wordcloud_pos, interpolation='bilinear')
            axes[0].set_title('Positive Reviews Word Cloud', fontsize=14, fontweight='bold', fontfamily='Arial')
            axes[0].axis('off')
        
        if negative_text.strip():
            # Try AI-enhanced processing first
            ai_enhanced_text = call_deepseek_api(negative_text, self.config)
            if ai_enhanced_text:
                print(f"🤖 Using AI-enhanced keywords for combined negative reviews")
                wordcloud_text = ai_enhanced_text
            else:
                print(f"📝 Using standard text processing for combined negative reviews")
                wordcloud_text = negative_text
            
            wordcloud_neg = WordCloud(
                width=800, height=400, 
                background_color='white',
                colormap='Reds',
                font_path=None,  # Use default font
                max_words=100,
                stopwords=None,
                min_font_size=10,
                max_font_size=80
            ).generate(wordcloud_text)
            
            axes[1].imshow(wordcloud_neg, interpolation='bilinear')
            axes[1].set_title('Negative Reviews Word Cloud', fontsize=14, fontweight='bold', fontfamily='Arial')
            axes[1].axis('off')
        
        plt.tight_layout()
        wordcloud_path = os.path.join(output_dir, f"{base_filename}_wordclouds.png")
        plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _display_comprehensive_summary(self, comprehensive_overview: Dict):
        """Display a summary table of the comprehensive overview"""
        print("\n📊 COMPREHENSIVE OVERVIEW SUMMARY:")
        print("="*140)
        print(f"{'Bank':<25} {'App Store':<15} {'Play Store':<15} {'Total Scraped':<15} {'Total Available':<15} {'Efficiency':<12} {'Rating':<8}")
        print("-"*140)
        
        total_scraped = 0
        total_available_all = 0
        total_weighted_sum = 0
        banks_with_reviews = 0
        
        for bank_key, bank_data in comprehensive_overview["banks"].items():
            bank_name = bank_data["name"][:20]  # Truncate long names
            app_store_scraped = bank_data["app_store"]["reviews_count"]
            app_store_total = bank_data["app_store"]["total_available"]
            play_store_scraped = bank_data["play_store"]["reviews_count"]
            play_store_total = bank_data["play_store"]["total_available"]
            
            bank_scraped = bank_data["total_reviews"]
            bank_available = bank_data["total_available_all"]
            weighted_rating = bank_data["weighted_rating"]
            
            efficiency = (bank_scraped / bank_available * 100) if bank_available > 0 else 0
            
            total_scraped += bank_scraped
            total_available_all += bank_available
            if bank_scraped > 0:
                total_weighted_sum += weighted_rating * bank_scraped
                banks_with_reviews += 1
            
            app_store_display = f"{app_store_scraped}/{app_store_total}" if app_store_total > 0 else f"{app_store_scraped}/~"
            play_store_display = f"{play_store_scraped}/{play_store_total}" if play_store_total > 0 else f"{play_store_scraped}/~"
            
            print(f"{bank_name:<25} {app_store_display:<15} {play_store_display:<15} {bank_scraped:<15} {bank_available:<15} {efficiency:<11.1f}% {weighted_rating:<8.2f}")
        
        # Calculate overall weighted average and efficiency
        overall_weighted_rating = total_weighted_sum / total_scraped if total_scraped > 0 else 0
        overall_efficiency = (total_scraped / total_available_all * 100) if total_available_all > 0 else 0
        
        print("-"*140)
        print(f"{'TOTAL (' + str(len(comprehensive_overview['banks'])) + ' banks)':<25} {'':<15} {'':<15} {total_scraped:<15} {total_available_all:<15} {overall_efficiency:<11.1f}% {overall_weighted_rating:<8.2f}")
        print("="*140)
        
        # Add totals to overview data
        comprehensive_overview["total_reviews_collected"] = total_scraped
        comprehensive_overview["total_reviews_available"] = total_available_all
        comprehensive_overview["scraping_efficiency"] = round(overall_efficiency, 1)
        comprehensive_overview["weighted_average_rating"] = round(overall_weighted_rating, 2)
        comprehensive_overview["banks_with_reviews"] = banks_with_reviews


def process_review_worker(args):
    """Worker function for processing individual reviews with multi-threading"""
    idx, row, config = args
    try:
        # Get content and title, handling NaN values
        content = row.get('content', '')
        title = row.get('title', '')
        
        # 1. Translate content using DeepSeek
        translated_content = translate_with_deepseek(content, config)
        
        # 2. Sentiment analysis
        analysis_text = f"{title} {translated_content}".strip()
        sentiment = analyze_sentiment_deepseek(analysis_text, config)
        
        # 3. Positive and negative words (simplified)
        blob = TextBlob(translated_content)
        # Extract words from assessments (assessments returns tuples with word, score, and other data)
        positive_words = []
        negative_words = []
        for assessment in blob.sentiment_assessments.assessments:
            if len(assessment) >= 2:
                word = assessment[0]
                score = assessment[1]
                # Handle case where word might be a list
                if isinstance(word, list):
                    word = ' '.join(word)
                if isinstance(word, str):
                    if score > 0:
                        positive_words.append(word)
                    elif score < 0:
                        negative_words.append(word)
        
        positive_words_str = ', '.join(positive_words[:5])  # Top 5 positive words
        negative_words_str = ', '.join(negative_words[:5])  # Top 5 negative words
        
        # 4. Problem categorization
        problem_category = categorize_problem_simple(translated_content)
        
        # 5. Overall sentiment (positive/negative)
        if sentiment['sentiment_score'] > 0.6:
            overall_sentiment = 'positive'
        elif sentiment['sentiment_score'] < 0.4:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'idx': idx,
            'translated_content': translated_content,
            'sentiment_score': sentiment['sentiment_score'],
            'positive_words': positive_words_str,
            'negative_words': negative_words_str,
            'problem_category': problem_category,
            'overall_sentiment': overall_sentiment,
            'success': True
        }
        
    except Exception as e:
        return {
            'idx': idx,
            'translated_content': '',
            'sentiment_score': 0.0,
            'positive_words': '',
            'negative_words': '',
            'problem_category': 'General',
            'overall_sentiment': 'neutral',
            'success': False,
            'error': str(e)
        }

    def extract_banking_phrases(self, text: str) -> str:
        """Extract common banking phrases from text as fallback"""
        import re
        
        # Common banking phrases to look for
        banking_phrases = [
            # Account related
            r'\baccount opening\b', r'\baccount verification\b', r'\baccount setup\b',
            r'\baccount access\b', r'\baccount management\b', r'\baccount security\b',
            
            # App related
            r'\bapp crashes\b', r'\bapp freezes\b', r'\bapp loading\b', r'\bapp updates\b',
            r'\bapp performance\b', r'\bapp interface\b', r'\bapp navigation\b',
            
            # Login/Auth related
            r'\blogin problems\b', r'\blogin issues\b', r'\bauthentication failed\b',
            r'\bpassword reset\b', r'\bsecurity verification\b', r'\bidentity verification\b',
            
            # Transaction related
            r'\btransfer money\b', r'\bpayment processing\b', r'\btransaction failed\b',
            r'\bpayment issues\b', r'\bmoney transfer\b', r'\btransaction processing\b',
            
            # Customer service
            r'\bcustomer service\b', r'\bcustomer support\b', r'\bsupport team\b',
            r'\bhelp desk\b', r'\bcustomer care\b',
            
            # Technical issues
            r'\bsystem error\b', r'\btechnical problems\b', r'\bserver issues\b',
            r'\bnetwork problems\b', r'\bconnection issues\b', r'\bdata loading\b',
            
            # User experience
            r'\buser interface\b', r'\buser experience\b', r'\beasy to use\b',
            r'\bdifficult to use\b', r'\bconfusing interface\b', r'\bintuitive design\b',
            
            # Banking features
            r'\bonline banking\b', r'\bmobile banking\b', r'\bdigital banking\b',
            r'\bbanking app\b', r'\bchecking account\b', r'\bsavings account\b',
            
            # Common issues
            r'\bvery slow\b', r'\btoo slow\b', r'\bnot working\b', r'\bdoes not work\b',
            r'\bkeep crashing\b', r'\bconstantly crashes\b', r'\bunable to access\b',
            r'\bcannot access\b', r'\bhard to use\b', r'\bdifficult to navigate\b'
        ]
        
        found_phrases = []
        text_lower = text.lower()
        
        for phrase_pattern in banking_phrases:
            matches = re.findall(phrase_pattern, text_lower)
            found_phrases.extend(matches)
        
        # Also extract common 2-3 word combinations
        words = text_lower.split()
        for i in range(len(words) - 1):
            # 2-word phrases
            phrase_2 = f"{words[i]} {words[i+1]}"
            if len(phrase_2.split()) == 2 and len(phrase_2) > 5:
                found_phrases.append(phrase_2)
            
            # 3-word phrases
            if i < len(words) - 2:
                phrase_3 = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(phrase_3.split()) == 3 and len(phrase_3) > 8:
                    found_phrases.append(phrase_3)
        
        # Remove duplicates and join
        unique_phrases = list(set(found_phrases))
        return ' '.join(unique_phrases)
    
    def preserve_phrases_for_wordcloud(self, text: str) -> str:
        """Preserve multi-word phrases for wordcloud by using underscores"""
        import re
        
        # Split text into words/phrases
        words = text.split()
        processed_words = []
        
        i = 0
        while i < len(words):
            # Check if this word is part of a phrase (2-5 words)
            phrase_length = 1
            phrase_words = [words[i]]
            
            # Look ahead to find complete phrases
            for j in range(1, 5):  # Check up to 5 words
                if i + j < len(words):
                    next_word = words[i + j]
                    # If next word doesn't start with uppercase or is a common connector, it's part of phrase
                    if (not next_word[0].isupper() and 
                        next_word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']):
                        phrase_words.append(next_word)
                        phrase_length += 1
                    else:
                        break
                else:
                    break
            
            # If we have a phrase (2+ words), join with underscores
            if phrase_length >= 2:
                phrase = '_'.join(phrase_words)
                processed_words.append(phrase)
                i += phrase_length  # Skip the words we've already processed
            else:
                # Single word, keep as is
                processed_words.append(words[i])
                i += 1
        
        return ' '.join(processed_words)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape and analyze app reviews from App Store and Play Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
Examples:
  python main.py welab_bank                    # Scrape and analyze WeLab Bank
  python main.py welab_bank za_bank mox_bank   # Scrape and analyze multiple banks
  python main.py --all                         # Scrape and analyze all banks
  python main.py --overview                    # Create overview of all banks
  python main.py --initial welab_bank          # Force re-scrape even if reviews exist
        """)
    )
    
    parser.add_argument('apps', nargs='*', 
                       help='App keys to scrape (e.g., welab_bank, za_bank)')
    parser.add_argument('--all', action='store_true',
                       help='Scrape all configured apps')
    parser.add_argument('--overview', action='store_true',
                       help='Create overview JSON for all apps')
    parser.add_argument('--initial', action='store_true',
                       help='Force re-scraping even if reviews already exist')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip analysis, only scrape reviews')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    scraper = AppReviewScraper()
    scraper.config = config
    
    # Determine which apps to process
    if args.all:
        app_config_data = load_app_config()
        app_keys = list(app_config_data.get("apps", {}).keys())
    elif args.overview:
        # Create comprehensive overview
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        overview_file = scraper.create_comprehensive_overview(timestamp)
        print(f"\n✅ Comprehensive overview created: {overview_file}")
            return
    elif not args.apps:
        parser.print_help()
        return
    else:
        app_keys = args.apps
    
    print(f"🚀 Starting review scraping and analysis for {len(app_keys)} app(s)...")
    print("="*80)
    
    # Create single timestamp for all analysis
    analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each app
    for app_key in app_keys:
        print(f"\n📱 Processing {app_key}...")
        
        # Check if reviews already exist (unless --initial is used)
        review_file = f"output/{app_key}_reviews.xlsx"
        if os.path.exists(review_file) and not args.initial:
            print(f"  ✅ Reviews already exist for {app_key}, skipping scraping...")
            print(f"  📄 Using existing file: {review_file}")
            
            # Run analysis on existing file
            if not args.no_analysis:
                try:
                    analysis_results = scraper.run_comprehensive_analysis(review_file, app_key, analysis_timestamp)
                    print(f"  ✅ Analysis completed! Results saved to: {analysis_results}")
                except Exception as e:
                    print(f"  ❌ Analysis failed: {e}")
        else:
            # Scrape new reviews
            try:
                app_config = get_app_config(app_key)
                if not app_config:
                    print(f"  ❌ App configuration not found for {app_key}")
                    continue
                
                print(f"  🔍 Scraping reviews for {app_config.get('name', app_key)}...")
                results = scraper.scrape_app(app_config)
                
                if not results or not any(results.values()):
                    print(f"  ⚠️ No reviews found for {app_key}")
                    continue
                
                # Export results (without timestamp in filename)
                exported_file = scraper.export_results(app_key, results)
                print(f"  📄 Reviews exported to: {exported_file}")
                
                # Run analysis if requested
                if not args.no_analysis:
                    try:
                        analysis_results = scraper.run_comprehensive_analysis(exported_file, app_key, analysis_timestamp)
                        print(f"  ✅ Analysis completed! Results saved to: {analysis_results}")
    except Exception as e:
                        print(f"  ❌ Analysis failed: {e}")
                
                # Log scraping time
                scraping_log = {
                    "app_key": app_key,
                    "scraped_at": datetime.now().isoformat(),
                    "app_store_reviews": len(results.get("app_store", [])),
                    "play_store_reviews": len(results.get("play_store", [])),
                    "total_reviews": sum(len(reviews) for reviews in results.values())
                }
                
                # Save scraping log
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{app_key}_scraping_log.json")
                with open(log_file, 'w') as f:
                    json.dump(scraping_log, f, indent=2)
                print(f"  📝 Scraping log saved to: {log_file}")
                
            except Exception as e:
                print(f"  ❌ Error processing {app_key}: {e}")
                continue
    
    print(f"\n✅ Processing completed for {len(app_keys)} app(s)!")


if __name__ == "__main__":
    main() 