# Hong Kong Banking Apps Review Scraper

A professional review scraper for Hong Kong banking applications from Apple App Store and Google Play Store.

## Features

- **Multi-Platform Scraping**: Apple App Store and Google Play Store
- **Multiple Sorting Methods**: Most Recent, Most Helpful, Most Favorable, Default Sort
- **Comprehensive Coverage**: Multiple countries and regions
- **Duplicate Prevention**: Intelligent deduplication across sorting methods
- **Professional Output**: Combined XLSX format with UTF-8 encoding
- **Rate Limiting**: Respectful scraping with configurable delays
- **Logging**: Detailed logging with rotation support

## Supported Banks

- **WeLabBank HK** - Hong Kong digital bank
- **Mox Bank** - Standard Chartered virtual bank  
- **ZA Bank** - ZhongAn virtual bank
- **HSBC HK** - HSBC Hong Kong mobile banking

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd appstore
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create output directories** (automatic on first run)
```bash
mkdir -p output logs
```

## Usage

### Basic Commands

```bash
# Scrape WeLabBank reviews
python main.py --app welab_bank

# Scrape Mox Bank reviews  
python main.py --app mox_bank

# Scrape ZA Bank reviews
python main.py --app za_bank

# List all available apps
python main.py --list-apps
```

### Advanced Options

```bash
# App Store only
python main.py --app welab_bank --no-play-store

# Play Store only
python main.py --app mox_bank --no-app-store
```

## Configuration

### Apps Configuration (`config/apps.json`)

Each app is configured with:
- **App Store ID** and **Play Store package name**
- **Supported countries** for regional reviews
- **Sorting methods** for comprehensive coverage
- **Rate limiting** settings

### Global Settings (`config/settings.json`)

- **Output format**: XLSX with UTF-8 encoding
- **Rate limiting**: Configurable delays between requests
- **Logging**: Level, file rotation, console output
- **Excel formatting**: Auto-adjust columns, sheet names

## Output

### File Format
- **Combined XLSX file**: `{app_key}_combined_{timestamp}.xlsx`
- **UTF-8 encoding** for international character support
- **Auto-adjusted columns** for readability

### Review Data Fields
- Platform (App Store/Play Store)
- Review ID, Title, Content
- Rating (1-5 stars)
- Author, Date, App Version
- Helpful Count, Reply Count

## Project Structure

```
appstore/
├── main.py                 # Main application entry point
├── config/
│   ├── apps.json          # App configurations
│   └── settings.json      # Global settings
├── src/
│   ├── models/            # Data models
│   ├── scrapers/          # Platform scrapers
│   ├── exporters/         # Output exporters
│   └── utils/             # Utilities
├── output/                # Generated review files
├── logs/                  # Application logs
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.7+
- requests
- pandas
- openpyxl
- google-play-scraper

## Rate Limiting

The scraper implements respectful rate limiting:
- **1-2 seconds** between requests
- **2-5 seconds** between sorting methods  
- **3 seconds** between countries
- Configurable delays in `settings.json`

## Legal Notice

This tool is for research and analysis purposes. Please ensure compliance with:
- Terms of Service for Apple App Store and Google Play Store
- Rate limiting and respectful scraping practices
- Data privacy regulations in your jurisdiction

## Support

For issues or questions, please check the logs in the `logs/` directory for detailed error information.

---

**Hong Kong Banking Apps Review Scraper** - Professional review data collection for market analysis. 