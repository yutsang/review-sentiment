"""
CSV exporter for review data
"""
import csv
from pathlib import Path
from typing import List, Dict, Any
from ..models.review import Review
from ..utils.logger import get_logger


class CSVExporter:
    """CSV file exporter for reviews"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Get output configuration
        self.output_config = config.get("output", {})
        self.encoding = self.output_config.get("encoding", "utf-8")
        self.output_dir = self.output_config.get("directory", "output")
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def export(self, reviews: List[Review], filename: str) -> str:
        """Export reviews to CSV file"""
        if not reviews:
            self.logger.warning("No reviews to export")
            return ""
        
        filepath = Path(self.output_dir) / f"{filename}.csv"
        
        try:
            with open(filepath, 'w', newline='', encoding=self.encoding) as f:
                writer = csv.writer(f)
                
                # Write header
                headers = [
                    'platform', 'review_id', 'title', 'content', 'rating',
                    'author', 'date', 'version', 'helpful_count', 'reply_count'
                ]
                writer.writerow(headers)
                
                # Write reviews
                for review in reviews:
                    writer.writerow([
                        review.platform,
                        review.review_id,
                        review.title,
                        review.content,
                        review.rating,
                        review.author,
                        review.date,
                        review.version,
                        review.helpful_count,
                        review.reply_count
                    ])
            
            self.logger.info(f"📄 CSV exported: {filepath} ({len(reviews)} reviews)")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
            return "" 