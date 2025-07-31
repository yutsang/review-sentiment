#!/usr/bin/env python3
"""
Hong Kong Banking Apps Review Scraper
Scrapes reviews from Apple App Store and Google Play Store for Hong Kong banks
"""
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.utils.config import (
    load_config, load_app_config, get_app_config,
    list_available_apps, ensure_directories
)
from src.utils.logger import setup_logger
from src.scrapers import AppStoreScraper, PlayStoreScraper
from src.exporters import CSVExporter, XLSXExporter
from src.models import Review


class AppReviewScraper:
    """Hong Kong Banking Apps Review Scraper"""
    
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
    
    def scrape_app(self, app_key: str, platforms: List[str] = None) -> Dict[str, List[Review]]:
        """Scrape reviews for specified app"""
        app_config = get_app_config(app_key)
        if not app_config:
            raise ValueError(f"App '{app_key}' not found in configuration")
        
        app_name = app_config.get("name", app_key)
        self.logger.info(f"🚀 Starting review scraping for: {app_name}")
        
        if platforms is None:
            platforms = ["app_store", "play_store"]
        
        results = {}
        
        # Scrape App Store
        if "app_store" in platforms and app_config.get("app_store", {}).get("enabled", True):
            try:
                scraper = AppStoreScraper(self.config, app_config)
                reviews = scraper.scrape_reviews()
                if reviews:
                    results["app_store"] = reviews
                    stats = scraper.get_stats()
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
                    stats = scraper.get_stats()
                    self.logger.info(f"🤖 Play Store Results:")
                    self._log_platform_stats("🤖", stats)
                else:
                    self.logger.warning("🤖 No Play Store reviews found")
            except Exception as e:
                self.logger.error(f"🤖 Play Store scraping failed: {e}")
        
        return results
    
    def export_results(self, app_key: str, results: Dict[str, List[Review]]) -> str:
        """Export scraping results as combined XLSX file"""
        timestamp = datetime.now().strftime(
            self.config.get("output", {}).get("timestamp_format", "%Y%m%d_%H%M%S")
        )
        
        # Combine all reviews
        all_reviews = []
        for reviews in results.values():
            all_reviews.extend(reviews)
        
        if not all_reviews:
            self.logger.warning("No reviews to export")
            return ""
        
        # Generate combined filename
        combined_filename = self._generate_combined_filename(app_key, timestamp)
        
        # Export as XLSX
        xlsx_file = self.xlsx_exporter.export(all_reviews, combined_filename)
        return xlsx_file if xlsx_file else ""
    
    def _generate_combined_filename(self, app_key: str, timestamp: str) -> str:
        """Generate filename for combined export"""
        template = self.config.get("output", {}).get("combined_filename_template",
                                                     "{app_key}_combined_{timestamp}")
        return template.format(app_key=app_key, timestamp=timestamp)
    
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
        
        print(f"\n✅ Scraping completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hong Kong Banking Apps Review Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --app welab_bank      # Scrape WeLabBank reviews
  python main.py --app mox_bank        # Scrape Mox Bank reviews
  python main.py --app za_bank         # Scrape ZA Bank reviews
  python main.py --list-apps          # List available apps
        """
    )
    
    parser.add_argument("--app", type=str, help="App key to scrape (from config/apps.json)")
    parser.add_argument("--list-apps", action="store_true", help="List available apps")
    parser.add_argument("--no-app-store", action="store_true", help="Skip App Store scraping")
    parser.add_argument("--no-play-store", action="store_true", help="Skip Play Store scraping")
    parser.add_argument("--config", type=str, default="config/settings.json", 
                       help="Path to settings configuration file")
    parser.add_argument("--apps-config", type=str, default="config/apps.json",
                       help="Path to apps configuration file")
    
    args = parser.parse_args()
    
    try:
        # Initialize scraper
        scraper = AppReviewScraper(args.config, args.apps_config)
        
        # List apps if requested
        if args.list_apps:
            apps = list_available_apps(args.apps_config)
            print("\n📱 Available Apps:")
            print("=" * 40)
            for key, name in apps.items():
                print(f"  {key:<15} - {name}")
            print(f"\nUse: python main.py --app <key>")
            return
        
        # Validate app argument
        if not args.app:
            print("❌ Error: --app argument is required")
            print("Use --list-apps to see available apps")
            sys.exit(1)
        
        # Determine platforms to scrape
        platforms = []
        if not args.no_app_store:
            platforms.append("app_store")
        if not args.no_play_store:
            platforms.append("play_store")
        
        if not platforms:
            print("❌ Error: At least one platform must be enabled")
            sys.exit(1)
        
        # Scrape reviews
        results = scraper.scrape_app(args.app, platforms)
        
        # Export results as combined XLSX
        exported_file = scraper.export_results(args.app, results)
        
        # Display summary
        scraper.display_summary(args.app, results, exported_file)
        
    except KeyboardInterrupt:
        print("\n⚠️  Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 