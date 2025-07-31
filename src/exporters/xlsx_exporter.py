"""
XLSX exporter for review data
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from ..models.review import Review
from ..utils.logger import get_logger

# Import with error handling
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


class XLSXExporter:
    """XLSX file exporter for reviews"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Get output configuration
        self.output_config = config.get("output", {})
        self.output_dir = self.output_config.get("directory", "output")
        
        # Get Excel configuration
        self.excel_config = config.get("excel", {})
        self.auto_adjust = self.excel_config.get("auto_adjust_columns", True)
        self.max_width = self.excel_config.get("max_column_width", 50)
        self.sheet_name = self.excel_config.get("sheet_name", "Reviews")
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def export(self, reviews: List[Review], filename: str) -> str:
        """Export reviews to XLSX file"""
        if not EXCEL_AVAILABLE:
            self.logger.warning("openpyxl not available - install with: pip install openpyxl")
            return ""
        
        if not reviews:
            self.logger.warning("No reviews to export")
            return ""
        
        filepath = Path(self.output_dir) / f"{filename}.xlsx"
        
        try:
            # Create DataFrame
            df = pd.DataFrame([review.to_dict() for review in reviews])
            
            # Export to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=self.sheet_name, index=False)
                
                # Auto-adjust column widths if enabled
                if self.auto_adjust:
                    self._adjust_column_widths(writer.book, writer.sheets[self.sheet_name])
            
            self.logger.info(f"📊 XLSX exported: {filepath} ({len(reviews)} reviews)")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting XLSX: {e}")
            return ""
    
    def _adjust_column_widths(self, workbook, worksheet):
        """Auto-adjust column widths"""
        try:
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                # Set width with maximum limit
                adjusted_width = min(max_length + 2, self.max_width)
                worksheet.column_dimensions[column_letter].width = adjusted_width
                
        except Exception as e:
            self.logger.debug(f"Error adjusting column widths: {e}") 