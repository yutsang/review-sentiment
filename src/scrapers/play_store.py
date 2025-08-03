"""
Play Store scraper implementation
"""
from typing import List, Dict, Any
from .base import BaseScraper
from ..models.review import Review

# Import with error handling
try:
    from google_play_scraper import app as gp_app, reviews as gp_reviews, Sort
    GOOGLE_PLAY_AVAILABLE = True
except ImportError:
    GOOGLE_PLAY_AVAILABLE = False
    gp_app = gp_reviews = Sort = None


class PlayStoreScraper(BaseScraper):
    """Play Store review scraper"""
    
    @property
    def platform_name(self) -> str:
        return "play_store"
    
    def __init__(self, config: Dict[str, Any], app_config: Dict[str, Any]):
        super().__init__(config, app_config)
        
        # Get Play Store specific config
        self.play_store_config = app_config.get("play_store", {})
        self.package_name = self.play_store_config.get("package_name")
        
        if not self.package_name:
            raise ValueError("Play Store package_name is required in app configuration")
        
        # Get Play Store settings
        self.batch_size = self.config.get("play_store", {}).get("batch_size", 200)
        self.default_country = self.config.get("play_store", {}).get("default_country", "hk")
        self.languages = self.config.get("play_store", {}).get("languages", ["en"])
    
    def scrape_reviews(self) -> List[Review]:
        """Scrape all Play Store reviews"""
        if not GOOGLE_PLAY_AVAILABLE:
            self.logger.error("Google Play Scraper not available - install with: pip install google-play-scraper")
            return []
        
        if not self.play_store_config.get("enabled", True):
            self.logger.info("Play Store scraping disabled in configuration")
            return []
        
        self.logger.info(f"🤖 Starting Play Store scraping for package: {self.package_name}")
        
        # Verify app exists
        try:
            app_info = gp_app(self.package_name, lang='en', country=self.default_country)
            app_name = app_info.get('title', 'Unknown App')
            total_reviews = app_info.get('reviews', 0)
            
            self.logger.info(f"App: {app_name}")
            self.logger.info(f"Total reviews available: {total_reviews:,}")
            
        except Exception as e:
            self.logger.error(f"Failed to get app info: {e}")
            return []
        
        all_reviews = []
        countries = self.play_store_config.get("countries", [self.default_country])
        sort_methods = self._get_sort_methods()
        
        for country in countries:
            self.logger.info(f"Scraping country: {country}")
            
            for sort_method, sort_name in sort_methods:
                self.logger.info(f"Sorting by: {sort_name}")
                
                try:
                    sort_reviews = self._scrape_sort_method(
                        country, sort_method
                    )
                    
                    new_reviews = 0
                    for review in sort_reviews:
                        if self.add_review(review):
                            all_reviews.append(review)
                            new_reviews += 1
                    
                    self.logger.info(f"Added {new_reviews} new reviews for {sort_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error with sort {sort_name}: {e}")
                    continue
                
                self.rate_limit("between_sort_methods")
            
            self.rate_limit("between_countries")
        
        self.logger.info(f"🤖 Total unique Play Store reviews collected: {len(all_reviews)}")
        return self.validate_reviews(all_reviews)
    
    def _get_sort_methods(self) -> List[tuple]:
        """Get sort methods from configuration"""
        sort_config = self.play_store_config.get("sort_methods", ["NEWEST", "RATING"])
        sort_methods = []
        
        for sort_str in sort_config:
            try:
                sort_method = getattr(Sort, sort_str)
                sort_methods.append((sort_method, sort_str.replace("_", " ").title()))
            except AttributeError:
                self.logger.warning(f"Invalid sort method: {sort_str}")
        
        return sort_methods
    
    def _scrape_sort_method(self, country: str, sort_method) -> List[Review]:
        """Scrape reviews for a specific sort method"""
        reviews = []
        continuation_token = None
        total_fetched = 0
        
        # Remove max_reviews limit - get all available reviews
        while True:
            try:
                if continuation_token:
                    result, continuation_token = gp_reviews(
                        self.package_name,
                        continuation_token=continuation_token,
                        lang='en',
                        country=country,
                        sort=sort_method,
                        count=self.batch_size
                    )
                else:
                    result, continuation_token = gp_reviews(
                        self.package_name,
                        lang='en',
                        country=country,
                        sort=sort_method,
                        count=self.batch_size
                    )
                
                if not result:
                    self.logger.debug("No more reviews available")
                    break
                
                # Process reviews
                batch_reviews = []
                for review_data in result:
                    review = Review.from_play_store_data(review_data, "Play Store")
                    if review:
                        batch_reviews.append(review)
                
                reviews.extend(batch_reviews)
                total_fetched += len(result)
                
                self.logger.debug(f"Batch: {len(result)} reviews, total: {total_fetched}")
                
                if not continuation_token:
                    self.logger.debug("No continuation token, finished")
                    break
                
                self.rate_limit("between_requests")
                
            except Exception as e:
                self.logger.error(f"Error in batch: {e}")
                break
        
        return reviews 