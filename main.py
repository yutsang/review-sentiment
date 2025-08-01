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
from googletrans import Translator
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import textwrap

from src.utils.config import (
    load_config, load_app_config, get_app_config,
    list_available_apps, ensure_directories
)
from src.utils.logger import setup_logger
from src.scrapers import AppStoreScraper, PlayStoreScraper
from src.exporters import CSVExporter, XLSXExporter
from src.models import Review

# Initialize translator for improved translation
translator = Translator()

def safe_translate(text: str, config: Dict, max_retries: int = None, delay: float = None) -> str:
    """Safely translate text with improved retry logic and rate limiting"""
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

def create_overview_json(results: Dict[str, Dict], config: Dict) -> Dict[str, Any]:
    """Create comprehensive overview JSON with exact numbers and ratings"""
    overview = {
        "generated_at": datetime.now().isoformat(),
        "total_banks": len(results),
        "banks": {},
        "summary": {
            "total_reviews_all_banks": 0,
            "total_app_store_reviews": 0,
            "total_play_store_reviews": 0,
            "average_rating_across_banks": 0.0
        }
    }
    
    total_weighted_rating = 0
    total_reviews = 0
    
    for bank_key, bank_data in results.items():
        app_config = get_app_config(bank_key)
        bank_name = app_config.get("name", bank_key) if app_config else bank_key
        
        app_store_reviews = bank_data.get("app_store", [])
        play_store_reviews = bank_data.get("play_store", [])
        
        app_store_count = len(app_store_reviews)
        play_store_count = len(play_store_reviews)
        total_bank_reviews = app_store_count + play_store_count
        
        # Calculate average ratings
        app_store_ratings = [r.rating for r in app_store_reviews if hasattr(r, 'rating') and r.rating]
        play_store_ratings = [r.rating for r in play_store_reviews if hasattr(r, 'rating') and r.rating]
        
        app_store_avg = np.mean(app_store_ratings) if app_store_ratings else 0
        play_store_avg = np.mean(play_store_ratings) if play_store_ratings else 0
        
        # Weighted average rating
        if total_bank_reviews > 0:
            weighted_avg = (app_store_avg * app_store_count + play_store_avg * play_store_count) / total_bank_reviews
        else:
            weighted_avg = 0
        
        overview["banks"][bank_key] = {
            "name": bank_name,
            "total_reviews": total_bank_reviews,
            "app_store": {
                "review_count": app_store_count,
                "average_rating": round(app_store_avg, 2)
            },
            "play_store": {
                "review_count": play_store_count,
                "average_rating": round(play_store_avg, 2)
            },
            "weighted_rating": round(weighted_avg, 2)
        }
        
        # Update summary
        overview["summary"]["total_reviews_all_banks"] += total_bank_reviews
        overview["summary"]["total_app_store_reviews"] += app_store_count
        overview["summary"]["total_play_store_reviews"] += play_store_count
        
        if total_bank_reviews > 0:
            total_weighted_rating += weighted_avg * total_bank_reviews
            total_reviews += total_bank_reviews
    
    # Calculate overall average
    if total_reviews > 0:
        overview["summary"]["average_rating_across_banks"] = round(total_weighted_rating / total_reviews, 2)
    
    return overview


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
        """Export scraping results with timestamp subfolder structure"""
        timestamp = datetime.now().strftime(
            self.config.get("output", {}).get("timestamp_format", "%Y%m%d_%H%M%S")
        )
        
        # Create timestamp subfolder if enabled
        output_dir = self.config.get("output", {}).get("directory", "output")
        if self.config.get("output", {}).get("use_timestamp_subfolder", False):
            if timestamp_subfolder:
                output_dir = os.path.join(output_dir, timestamp_subfolder)
            else:
                output_dir = os.path.join(output_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
        
        # Combine all reviews
        all_reviews = []
        for reviews in results.values():
            all_reviews.extend(reviews)
        
        if not all_reviews:
            self.logger.warning("No reviews to export")
            return ""
        
        # Generate filename
        template = self.config.get("output", {}).get("combined_filename_template",
                                                     "{app_key}_reviews_{timestamp}")
        filename = template.format(app_key=app_key, timestamp=timestamp)
        
        # Export as XLSX to the subfolder
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
        """Run comprehensive sentiment analysis with improved translation"""
        print(f"\n🔬 Starting comprehensive analysis for {app_key}...")
        
        # Read the data
        df = pd.read_excel(file_path)
        print(f"📊 Processing {len(df)} reviews...")
        
        # Initialize new columns for analysis
        df['translated_content'] = ""
        df['translated_title'] = ""
        df['category'] = ""
        df['sentiment_score'] = 0.0
        df['sentiment_category'] = ""
        df['sentiment_words'] = ""
        df['positive_words'] = ""
        df['negative_words'] = ""
        
        # Process each review with improved translation
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"  Processing review {idx + 1}/{len(df)}...")
            
            try:
                # Translate content and title
                translated_content = safe_translate(row.get('content', ''), self.config)
                translated_title = safe_translate(row.get('title', ''), self.config)
                
                df.at[idx, 'translated_content'] = translated_content
                df.at[idx, 'translated_title'] = translated_title
                
                # Use translated text for analysis
                analysis_text = f"{translated_title} {translated_content}".strip()
                
                # Categorize
                category = categorize_review(translated_content, translated_title)
                df.at[idx, 'category'] = category
                
                # Sentiment analysis
                sentiment = analyze_sentiment(analysis_text)
                df.at[idx, 'sentiment_score'] = sentiment['sentiment_score']
                df.at[idx, 'sentiment_category'] = sentiment['sentiment_category']
                df.at[idx, 'sentiment_words'] = ', '.join(sentiment['sentiment_words'])
                df.at[idx, 'positive_words'] = ', '.join(sentiment['positive_words'])
                df.at[idx, 'negative_words'] = ', '.join(sentiment['negative_words'])
                
            except Exception as e:
                print(f"  Error processing review {idx}: {e}")
                continue
        
        # Save analyzed data
        output_dir = os.path.dirname(file_path)
        analyzed_file = os.path.join(output_dir, f"{app_key}_analyzed.xlsx")
        df.to_excel(analyzed_file, index=False)
        
        # Generate visualizations
        self.create_analysis_charts(df, app_key, output_dir)
        
        # Generate summary report
        summary = self.generate_summary_report(df, app_key)
        summary_file = os.path.join(output_dir, f"{app_key}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Comprehensive analysis completed for {app_key}")
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
        
        # Sentiment distribution
        sentiment_counts = df['sentiment_category'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       colors=[sentiment_colors.get(cat, '#999999') for cat in sentiment_counts.index],
                       autopct='%1.1f%%')
        axes[0, 0].set_title('Sentiment Distribution')
        
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        axes[0, 1].bar(rating_counts.index, rating_counts.values, 
                       color=colors.get("chart_palette", ['#1f77b4'])[0])
        axes[0, 1].set_title('Rating Distribution')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Count')
        
        # Category distribution
        category_counts = df['category'].value_counts()
        axes[1, 0].barh(category_counts.index, category_counts.values,
                        color=colors.get("chart_palette", ['#1f77b4'] * len(category_counts))[0])
        axes[1, 0].set_title('Review Categories')
        axes[1, 0].set_xlabel('Count')
        
        # Sentiment over time (if date available)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            monthly_sentiment = df.groupby([df['date'].dt.to_period('M'), 'sentiment_category']).size().unstack(fill_value=0)
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
        
        # Create word cloud for positive and negative reviews
        self.create_wordclouds(df, app_key, output_dir)
    
    def create_wordclouds(self, df: pd.DataFrame, app_key: str, output_dir: str):
        """Create word clouds for positive and negative reviews"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Positive reviews word cloud
            positive_reviews = df[df['sentiment_category'] == 'positive']['translated_content'].dropna()
            if not positive_reviews.empty:
                positive_text = ' '.join(positive_reviews.astype(str))
                wordcloud_pos = WordCloud(width=800, height=400, 
                                        background_color='white',
                                        colormap='Greens').generate(positive_text)
                axes[0].imshow(wordcloud_pos, interpolation='bilinear')
                axes[0].set_title('Positive Reviews Word Cloud', fontsize=16)
                axes[0].axis('off')
            
            # Negative reviews word cloud
            negative_reviews = df[df['sentiment_category'] == 'negative']['translated_content'].dropna()
            if not negative_reviews.empty:
                negative_text = ' '.join(negative_reviews.astype(str))
                wordcloud_neg = WordCloud(width=800, height=400, 
                                        background_color='white',
                                        colormap='Reds').generate(negative_text)
                axes[1].imshow(wordcloud_neg, interpolation='bilinear')
                axes[1].set_title('Negative Reviews Word Cloud', fontsize=16)
                axes[1].axis('off')
            
            plt.tight_layout()
            wordcloud_file = os.path.join(output_dir, f"{app_key}_wordclouds.png")
            plt.savefig(wordcloud_file, dpi=300, bbox_inches='tight')
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
            "sentiment_distribution": df['sentiment_category'].value_counts().to_dict(),
            "rating_distribution": df['rating'].value_counts().sort_index().to_dict(),
            "category_distribution": df['category'].value_counts().to_dict(),
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
        negative_ratio = len(df[df['sentiment_category'] == 'negative']) / len(df)
        if negative_ratio > 0.3:
            recommendations.append("High negative sentiment detected. Focus on addressing user concerns.")
        
        # Category-based recommendations
        top_categories = df['category'].value_counts().head(3)
        for category, count in top_categories.items():
            if category == 'performance' and count > len(df) * 0.2:
                recommendations.append("Performance issues frequently mentioned. Consider app optimization.")
            elif category == 'login_auth' and count > len(df) * 0.15:
                recommendations.append("Authentication issues reported. Review login process.")
            elif category == 'ui_ux' and count > len(df) * 0.15:
                recommendations.append("UI/UX concerns raised. Consider interface improvements.")
        
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
        """Create word clouds for combined analysis"""
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Separate by sentiment
        positive_text = " ".join([review['translated_content'] for review in reviews_data 
                                if review.get('sentiment_category') == 'positive'])
        negative_text = " ".join([review['translated_content'] for review in reviews_data 
                                if review.get('sentiment_category') == 'negative'])
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        if positive_text.strip():
            wordcloud_pos = WordCloud(width=800, height=400, background_color='white', 
                                     colormap='Greens').generate(positive_text)
            axes[0].imshow(wordcloud_pos, interpolation='bilinear')
            axes[0].set_title('Positive Reviews Word Cloud', fontsize=14, fontweight='bold')
            axes[0].axis('off')
        
        if negative_text.strip():
            wordcloud_neg = WordCloud(width=800, height=400, background_color='white', 
                                     colormap='Reds').generate(negative_text)
            axes[1].imshow(wordcloud_neg, interpolation='bilinear')
            axes[1].set_title('Negative Reviews Word Cloud', fontsize=14, fontweight='bold')
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hong Kong Banking Apps Review Scraper and Sentiment Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python main.py welab_bank                    # Analyze one bank
          python main.py mox_bank za_bank welab_bank   # Analyze multiple banks
          python main.py --all                         # Analyze all banks
          python main.py --list-apps                   # List available banks
          python main.py welab_bank --no-analysis     # Just scrape, no analysis
        ''')
    )
    
    parser.add_argument(
        'apps', 
        nargs='*',  # Accept multiple apps
        help='Bank app keys to analyze (space-separated for multiple)'
    )
    parser.add_argument('--overview', action='store_true', help='Create comprehensive overview of ALL banks in config')
    parser.add_argument('--all', action='store_true', help='Process all configured banks')
    parser.add_argument('--list-apps', action='store_true', help='List available bank apps')
    parser.add_argument('--no-analysis', action='store_true', help='Skip sentiment analysis')
    
    args = parser.parse_args()
    
    # Handle special options first
    if args.list_apps:
        list_available_apps()
        return
    
    if args.overview:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraper = AppReviewScraper()
        scraper.config = load_config() # Assuming load_config() is available or passed
        overview_file = scraper.create_comprehensive_overview(timestamp)
        print(f"\n✅ Comprehensive overview completed!")
        print(f"📄 Overview file: {overview_file}")
        return
        
    # Validate arguments
    if not args.all and not args.apps:
        parser.error("Please specify bank app keys or use --all")
        return
        
    try:
        # Load configuration
        config = load_config()
        app_config_data = load_app_config()
        
        # Create single timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine which apps to process
        if args.all:
            apps_to_process = list(app_config_data.get("apps", {}).keys())
            print(f"🚀 Processing all {len(apps_to_process)} banks...")
        else:
            apps_to_process = args.apps
            # Validate app keys
            available_apps = set(app_config_data.get("apps", {}).keys())
            invalid_apps = [app for app in apps_to_process if app not in available_apps]
            if invalid_apps:
                print(f"❌ Error: Unknown bank app keys: {', '.join(invalid_apps)}")
                print("Use --list-apps to see available banks")
                sys.exit(1)
            
            print(f"🚀 Processing {len(apps_to_process)} bank(s): {', '.join(apps_to_process)}")
        
        # Create main scraper instance
        scraper = AppReviewScraper()
        scraper.config = config
        
        # Collect all results
        all_results = {}
        all_reviews = []
        overview_data = {}
        
        for app_key in apps_to_process:
            print(f"\n{'='*60}")
            print(f"🏦 PROCESSING: {app_key.upper().replace('_', ' ')}")
            print(f"{'='*60}")
            
            app_config = get_app_config(app_key)
            if not app_config:
                print(f"❌ Error: Configuration not found for {app_key}")
                continue
            
            # Scrape reviews for this app
            results = scraper.scrape_app(app_config)
            
            if results:
                all_results[app_key] = results
                
                # Collect all reviews from both platforms
                app_reviews = []
                for platform_reviews in results.values():
                    app_reviews.extend(platform_reviews)
                
                all_reviews.extend(app_reviews)
                
                # Create overview entry
                overview_data[app_key] = {
                    "bank_name": app_config.get("name", app_key),
                    "app_store_reviews": len(results.get("app_store", [])),
                    "play_store_reviews": len(results.get("play_store", [])),
                    "total_reviews": len(app_reviews)
                }
                
                if app_reviews:
                    overview_data[app_key]["average_rating"] = sum(r.rating for r in app_reviews) / len(app_reviews)
                    
                    # Rating distribution
                    rating_dist = {}
                    for review in app_reviews:
                        rating_dist[review.rating] = rating_dist.get(review.rating, 0) + 1
                    overview_data[app_key]["rating_distribution"] = rating_dist
                else:
                    overview_data[app_key]["average_rating"] = 0
                    overview_data[app_key]["rating_distribution"] = {}
                
                print(f"✅ {app_key}: {len(app_reviews)} total reviews")
            else:
                print(f"❌ {app_key}: No reviews found")
                overview_data[app_key] = {
                    "bank_name": app_config.get("name", app_key),
                    "app_store_reviews": 0,
                    "play_store_reviews": 0,
                    "total_reviews": 0,
                    "average_rating": 0,
                    "rating_distribution": {}
                }
        
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING COMPLETE")
        print(f"{'='*60}")
        
        if all_reviews:
            # Create combined filename
            if len(apps_to_process) == 1:
                base_filename = f"{apps_to_process[0]}_complete_{timestamp}"
            elif args.all:
                base_filename = f"all_banks_complete_{timestamp}"
            else:
                base_filename = f"{'_'.join(apps_to_process)}_complete_{timestamp}"
            
            if not args.no_analysis:
                # Perform combined analysis
                print(f"🔍 Performing combined sentiment analysis for {len(all_reviews)} reviews...")
                exported_file = scraper.process_with_analysis_multi(apps_to_process, all_results, all_reviews, timestamp, base_filename)
            else:
                # Just export raw data
                exported_file = scraper.export_results_multi(apps_to_process, all_results, all_reviews, timestamp, base_filename)
            
            # Create overview JSON
            overview_file = scraper.create_overview_json(overview_data, timestamp, base_filename)
            
            # Display final summary
            total_app_store = sum(data["app_store_reviews"] for data in overview_data.values())
            total_play_store = sum(data["play_store_reviews"] for data in overview_data.values())
            total_reviews = sum(data["total_reviews"] for data in overview_data.values())
            
            print(f"\n🎯 FINAL RESULTS")
            print(f"{'='*50}")
            print(f"📱 Banks Processed: {len(apps_to_process)}")
            print(f"🍎 App Store Reviews: {total_app_store:,}")
            print(f"🤖 Play Store Reviews: {total_play_store:,}")
            print(f"📋 Total Reviews: {total_reviews:,}")
            print(f"\n📁 Files Generated:")
            print(f"   📄 {exported_file}")
            print(f"   📊 {overview_file}")
            
            if not args.no_analysis:
                print(f"\n🎨 Analysis includes:")
                print(f"   • Sentiment analysis with translation")
                print(f"   • Word clouds and visualizations") 
                print(f"   • Rating distributions and trends")
                print(f"   • Combined dashboard")
            
        else:
            print("❌ No reviews collected from any banks")
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 