# Hong Kong Banking Apps Review Scraper & Sentiment Analyzer

Scrapes app reviews from App Store and Google Play, runs sentiment analysis, and produces summaries with customer-focused insights.

## Quick Start

```bash
./run.sh
```

One command: installs dependencies and processes all configured banks (scrape → sentiment analysis → charts → customer analysis → summary JSON).

## Single-Bank Usage

```bash
pip install -r requirements.txt
python main.py welab_bank
```

## Options

- `python main.py --all` — process all banks
- `python main.py welab_bank za_bank` — specific banks
- `python main.py --overview` — overview JSON only (no analysis)
- `python main.py welab_bank --no-analysis` — scrape only
- `python main.py welab_bank --initial` — force re-scrape

## Output

- `output/{app_key}_reviews.xlsx` — raw reviews
- `output/analysis_{timestamp}/` — analyzed data, charts, word clouds, summary JSON with:
  - Sentiment distribution
  - Problem categories (including Customer Service Issues)
  - Customer analysis metrics
  - Recommendations

## Config

- `config/apps.json` — bank app IDs and package names
- `config/settings.example.json` — template (copy to `config/settings.json` and add DeepSeek API key if using advanced sentiment)
