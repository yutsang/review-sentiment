"""
Base scraper class with common functionality
"""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set
from ..models.review import Review
from ..utils.logger import get_logger


class BaseScraper(ABC):
    """Base class for all review scrapers"""
    
    def __init__(self, config: Dict[str, Any], app_config: Dict[str, Any]):
        self.config = config
        self.app_config = app_config
        self.logger = get_logger()
        self.unique_review_ids: Set[str] = set()
        
        # Get rate limiting config
        self.rate_limits = self.config.get("rate_limiting", {}).get(self.platform_name, {})
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Platform name for configuration lookup"""
        pass
    
    @abstractmethod
    def scrape_reviews(self) -> List[Review]:
        """Scrape reviews for the configured app"""
        pass
    
    def is_duplicate(self, review: Review) -> bool:
        """Check if review is a duplicate"""
        if not self.config.get("duplicate_detection", {}).get("enabled", True):
            return False
        
        return review.review_id in self.unique_review_ids
    
    def add_review(self, review: Review) -> bool:
        """Add review if not duplicate"""
        if self.is_duplicate(review):
            return False
        
        self.unique_review_ids.add(review.review_id)
        return True
    
    def rate_limit(self, delay_type: str = "between_requests") -> None:
        """Apply rate limiting"""
        delay = self.rate_limits.get(delay_type, 1.0)
        if delay > 0:
            time.sleep(delay)
    
    def validate_reviews(self, reviews: List[Review]) -> List[Review]:
        """Validate and filter reviews"""
        valid_reviews = []
        for review in reviews:
            if review.validate():
                valid_reviews.append(review)
            else:
                self.logger.warning(f"Invalid review data: {review.review_id}")
        
        return valid_reviews
    
    def get_stats(self, reviews: List[Review]) -> Dict[str, Any]:
        """Get review statistics"""
        if not reviews:
            return {}
        
        rating_counts = {}
        for review in reviews:
            rating_counts[review.rating] = rating_counts.get(review.rating, 0) + 1
        
        return {
            "total_reviews": len(reviews),
            "rating_distribution": rating_counts,
            "average_rating": sum(r.rating for r in reviews) / len(reviews),
            "platform": reviews[0].platform if reviews else "Unknown"
        } 