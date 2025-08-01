# Hong Kong Banking Apps Review Scraper and Sentiment Analyzer

A comprehensive tool for scraping app reviews from Apple App Store and Google Play Store for Hong Kong banking apps, with advanced sentiment analysis and visualization capabilities.

## Features

- **Multi-Platform Scraping**: Automated review collection from both App Store and Google Play Store
- **Comprehensive Bank Coverage**: Support for all major Hong Kong digital banks and traditional banks
- **Advanced Sentiment Analysis**: AI-powered sentiment classification with category detection
- **Improved Translation**: Enhanced Google Translate integration with retry logic and rate limiting
- **Professional Visualizations**: Charts, word clouds, and trend analysis with configurable colors
- **Timestamp Organization**: Organized output with timestamp-based subfolder structure
- **Comprehensive Reporting**: Detailed JSON summaries with actionable recommendations
- **Batch Processing**: Process single banks or all configured banks simultaneously
- **Overview Generation**: Consolidated statistics across all banks with exact review counts and ratings

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd review-sentiment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (first time only)
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

## Configuration

### Bank Apps Configuration (`config/apps.json`)
Configure the banking apps you want to analyze. Each app requires:
- App Store ID for iOS reviews
- Google Play package name for Android reviews
- Supported countries (usually ["hk"] for Hong Kong)
- Review sorting methods

### Global Settings (`config/settings.json`)
Customize the tool behavior:
- **Output Settings**: File formats, directory structure, timestamp formats
- **Color Configuration**: Chart colors, sentiment colors, rating colors
- **Translation Settings**: Retry logic, rate limiting, chunk sizes
- **Analysis Settings**: Enable/disable specific analysis features
- **Rate Limiting**: Configure delays between API requests

## Usage

### Basic Commands

```bash
# Process a specific bank
python main.py <bank_key>

# Process all configured banks
python main.py --all

# List available banks
python main.py --list-apps

# Process with custom options
python main.py <bank_key> --no-analysis
python main.py <bank_key> --no-play-store
python main.py --all --no-translation
```

### Examples

```bash
# Scrape and analyze reviews for a single bank
python main.py welab_bank

# Scrape all banks without sentiment analysis
python main.py --all --no-analysis

# App Store reviews only for a specific bank
python main.py mox_bank --no-play-store

# Scrape specific bank without translation (faster)
python main.py za_bank --no-translation
```

### Command Line Options

- `<bank_key>`: Bank identifier (required, or use --all)
- `--all`: Process all configured banks
- `--list-apps`: Show available bank configurations
- `--no-app-store`: Skip App Store scraping
- `--no-play-store`: Skip Google Play Store scraping
- `--no-analysis`: Skip sentiment analysis (scraping only)
- `--no-translation`: Skip translation step (faster processing)
- `--config <path>`: Custom settings configuration file
- `--apps-config <path>`: Custom apps configuration file

## Output Structure

All results are organized in timestamp-based subfolders under the `output/` directory:

```
output/
├── 20240130_143022/          # Timestamp folder
│   ├── bank1_reviews_20240130_143022.xlsx     # Raw review data
│   ├── bank1_analyzed.xlsx                    # Analyzed data with sentiment
│   ├── bank1_summary.json                     # Summary statistics
│   ├── bank1_analysis_charts.png              # Visualization charts
│   ├── bank1_wordclouds.png                   # Word clouds
│   ├── bank2_reviews_20240130_143022.xlsx
│   ├── ...
│   └── overview.json                          # Cross-bank overview (when using --all)
```

### File Types

1. **Raw Data** (`*_reviews_*.xlsx`): Original scraped reviews with metadata
2. **Analyzed Data** (`*_analyzed.xlsx`): Reviews with sentiment scores, categories, and translations
3. **Summary Report** (`*_summary.json`): Statistics, trends, and recommendations
4. **Charts** (`*_analysis_charts.png`): Sentiment distribution, rating trends, category breakdown
5. **Word Clouds** (`*_wordclouds.png`): Visual representation of positive/negative review content
6. **Overview** (`overview.json`): Comprehensive cross-bank comparison (with --all option)

## Analysis Features

### Sentiment Analysis
- **Polarity Scoring**: Numerical sentiment scores (-1 to +1)
- **Category Classification**: Positive, negative, neutral sentiment categories
- **Word Extraction**: Key positive and negative sentiment words
- **Rating Correlation**: Analysis of sentiment vs. star ratings

### Review Categorization
Automatic classification into categories:
- `login_auth`: Authentication and login issues
- `performance`: App speed, crashes, loading problems
- `ui_ux`: Interface design and user experience
- `payment`: Transaction and payment functionality
- `feature`: Feature requests and functionality
- `customer_service`: Support and service issues
- `security`: Security and privacy concerns
- `general`: Uncategorized reviews

### Improved Translation
- **Smart Detection**: Automatic English text detection to skip unnecessary translation
- **Retry Logic**: Exponential backoff for failed translation attempts
- **Rate Limiting**: Configurable delays to respect API limits
- **Chunk Processing**: Large text splitting for better translation accuracy
- **Error Handling**: Graceful fallback to original text on translation failure

### Visualization
- **Sentiment Distribution**: Pie charts showing sentiment breakdown
- **Rating Analysis**: Bar charts of star rating distribution
- **Category Trends**: Horizontal bar charts of issue categories
- **Time Series**: Sentiment trends over time (when date data available)
- **Word Clouds**: Visual representation of frequent positive/negative terms
- **Configurable Colors**: Customizable color schemes for all charts

## Performance and Rate Limiting

The tool includes intelligent rate limiting to respect API constraints:
- **App Store**: 1-3 second delays between requests
- **Google Play**: 2-5 second delays between requests
- **Translation API**: Configurable delays with exponential backoff
- **Batch Processing**: Efficient handling of multiple banks

## Error Handling

- **Network Issues**: Automatic retry with exponential backoff
- **API Limits**: Respectful rate limiting and graceful degradation
- **Translation Failures**: Fallback to original text
- **Data Processing**: Robust handling of malformed review data
- **File Operations**: Safe file handling with proper error reporting

## Troubleshooting

### Common Issues

1. **Translation Errors**: Reduce translation rate or use `--no-translation` for faster processing
2. **API Rate Limits**: Increase delay settings in `config/settings.json`
3. **Memory Issues**: Process banks individually instead of using `--all`
4. **Missing Dependencies**: Run `pip install -r requirements.txt` again

### Performance Tips

- Use `--no-translation` for faster processing if translation isn't needed
- Process banks individually for large datasets
- Adjust rate limiting settings based on your network/API limits
- Use `--no-analysis` for data collection only

## Data Privacy and Ethics

- **Respectful Scraping**: Built-in rate limiting to avoid overwhelming servers
- **Public Data Only**: Only collects publicly available review data
- **No Personal Information**: Does not collect or store personal user data
- **Research Purpose**: Intended for research and analysis purposes

## Contributing

When adding new banks or features:
1. Update `config/apps.json` with proper app store IDs and package names
2. Test with single bank before batch processing
3. Follow the existing code structure and documentation
4. Ensure proper error handling and rate limiting

## License

This project is intended for research and educational purposes. Please respect the terms of service of the platforms being scraped and use responsibly. 