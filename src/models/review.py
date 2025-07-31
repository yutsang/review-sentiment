"""
Review data model for app store reviews
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class Review:
    """Review data structure for both App Store and Play Store reviews"""
    platform: str
    review_id: str
    title: str
    content: str
    rating: int
    author: str
    date: str
    version: str
    helpful_count: int = 0
    reply_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert review to dictionary"""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate review data"""
        if not self.platform or self.platform not in ['App Store', 'Play Store']:
            return False
        if not self.review_id:
            return False
        if not isinstance(self.rating, int) or self.rating < 1 or self.rating > 5:
            return False
        return True
    
    @classmethod
    def from_app_store_entry(cls, entry, platform_name: str = "App Store") -> Optional['Review']:
        """Create Review from App Store RSS entry"""
        try:
            title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
            content_elem = entry.find('.//{http://www.w3.org/2005/Atom}content')
            author_elem = entry.find('.//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
            updated_elem = entry.find('.//{http://www.w3.org/2005/Atom}updated')
            rating_elem = entry.find('.//{http://itunes.apple.com/rss}rating')
            version_elem = entry.find('.//{http://itunes.apple.com/rss}version')
            id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
            
            return cls(
                platform=platform_name,
                review_id=id_elem.text if id_elem is not None else "",
                title=title_elem.text if title_elem is not None else "",
                content=content_elem.text if content_elem is not None else "",
                rating=int(rating_elem.text) if rating_elem is not None else 0,
                author=author_elem.text if author_elem is not None else "Anonymous",
                date=updated_elem.text if updated_elem is not None else "",
                version=version_elem.text if version_elem is not None else ""
            )
        except Exception:
            return None
    
    @classmethod
    def from_play_store_data(cls, review_data: Dict[str, Any], platform_name: str = "Play Store") -> Optional['Review']:
        """Create Review from Play Store review data"""
        try:
            return cls(
                platform=platform_name,
                review_id=review_data.get('reviewId', ''),
                title="",  # Play Store doesn't have titles
                content=review_data.get('content', ''),
                rating=review_data.get('score', 0),
                author=review_data.get('userName', 'Anonymous'),
                date=review_data.get('at', '').isoformat() if review_data.get('at') else '',
                version=review_data.get('appVersion', ''),
                helpful_count=review_data.get('thumbsUpCount', 0),
                reply_count=1 if review_data.get('replyContent') else 0
            )
        except Exception:
            return None 