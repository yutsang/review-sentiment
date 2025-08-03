"""
App Store scraper implementation
"""
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from .base import BaseScraper
from ..models.review import Review


class AppStoreScraper(BaseScraper):
    """App Store review scraper"""
    
    @property
    def platform_name(self) -> str:
        return "app_store"
    
    def __init__(self, config: Dict[str, Any], app_config: Dict[str, Any]):
        super().__init__(config, app_config)
        
        # Get App Store specific config
        self.app_store_config = app_config.get("app_store", {})
        self.app_id = self.app_store_config.get("app_id")
        
        if not self.app_id:
            raise ValueError("App Store app_id is required in app configuration")
        
        # Setup session
        self.session = requests.Session()
        self.base_url = self.config.get("app_store", {}).get("base_url", "https://itunes.apple.com")
        user_agent = self.config.get("app_store", {}).get("user_agent", 
                                   "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15")
        self.session.headers.update({'User-Agent': user_agent})
    
    def scrape_reviews(self) -> List[Review]:
        """Scrape all App Store reviews"""
        if not self.app_store_config.get("enabled", True):
            self.logger.info("App Store scraping disabled in configuration")
            return []
        
        self.logger.info(f"🍎 Starting App Store scraping for app ID: {self.app_id}")
        
        all_reviews = []
        countries = self.app_store_config.get("countries", ["us"])
        sort_methods = self.app_store_config.get("sort_methods", ["mostRecent"])
        
        for country in countries:
            self.logger.info(f"Scraping country: {country}")
            
            for sort_method in sort_methods:
                sort_name = sort_method if sort_method else "Default Sort"
                self.logger.info(f"Sorting by: {sort_name}")
                
                page_reviews = self._scrape_sort_method(country, sort_method)
                new_reviews = 0
                
                for review in page_reviews:
                    if self.add_review(review):
                        all_reviews.append(review)
                        new_reviews += 1
                
                self.logger.info(f"Added {new_reviews} new reviews for {sort_name}")
                self.rate_limit("between_sort_methods")
            
            self.rate_limit("between_countries")
        
        self.logger.info(f"🍎 Total unique App Store reviews collected: {len(all_reviews)}")
        return self.validate_reviews(all_reviews)
    
    def _scrape_sort_method(self, country: str, sort_method: str) -> List[Review]:
        """Scrape reviews for a specific sort method"""
        reviews = []
        
        # Remove max_pages limit - get all available pages
        page = 1
        while True:
            try:
                page_reviews = self._get_reviews_page(page, sort_method, country)
                
                if not page_reviews:
                    self.logger.debug(f"No reviews on page {page}")
                    break
                
                reviews.extend(page_reviews)
                self.logger.debug(f"Page {page}: {len(page_reviews)} reviews")
                
                # If we got fewer reviews than expected, we've reached the end
                if len(page_reviews) < 50:  # App Store typically returns 50 reviews per page
                    break
                
                page += 1
                self.rate_limit("between_requests")
                
            except Exception as e:
                self.logger.error(f"Error on page {page}: {e}")
                break
        
        return reviews
    
    def _get_reviews_page(self, page: int, sort_by: str, country: str) -> List[Review]:
        """Get reviews from a specific page"""
        reviews = []
        
        # Try different RSS URL formats
        rss_urls = [
            f"{self.base_url}/rss/customerreviews/page={page}/id={self.app_id}/sortby={sort_by}/xml",
            f"{self.base_url}/{country}/rss/customerreviews/page={page}/id={self.app_id}/sortby={sort_by}/xml",
            f"{self.base_url}/rss/customerreviews/id={self.app_id}/page={page}/sortby={sort_by}/xml",
            f"{self.base_url}/us/rss/customerreviews/page={page}/id={self.app_id}/sortby={sort_by}/xml"
        ]
        
        for rss_url in rss_urls:
            try:
                self.logger.debug(f"Trying RSS URL: {rss_url}")
                response = self.session.get(rss_url, timeout=30)
                
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                    
                    self.logger.debug(f"Found {len(entries)} total entries in RSS feed")
                    
                    # Skip the first entry which is usually app information
                    review_entries = entries[1:] if len(entries) > 1 else []
                    
                    if review_entries:
                        for entry in review_entries:
                            review = Review.from_app_store_entry(entry, "App Store")
                            if review:
                                reviews.append(review)
                        
                        self.logger.debug(f"RSS success: {rss_url} - {len(review_entries)} review entries")
                        break
                    else:
                        self.logger.debug(f"RSS feed has no review entries: {rss_url}")
                else:
                    self.logger.debug(f"RSS URL returned status {response.status_code}: {rss_url}")
                
            except Exception as e:
                self.logger.debug(f"RSS URL failed: {rss_url} - {e}")
                continue
        
        if not reviews:
            self.logger.warning(f"No reviews found for app ID {self.app_id} on page {page} with sort {sort_by}")
        
        return reviews 