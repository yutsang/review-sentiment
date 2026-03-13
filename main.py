#!/usr/bin/env python3
"""
Hong Kong Banking Apps Review Scraper and Sentiment Analyzer
Scrapes reviews from App Store and Google Play Store and performs sentiment analysis.

Usage:
  python main.py welab_bank                  # Scrape and analyze single bank
  python main.py welab_bank za_bank          # Multiple banks
  python main.py --all                       # All configured banks
  python main.py --overview                  # Overview JSON for all banks
  python main.py welab_bank --initial        # Force re-scrape existing data
  python main.py welab_bank --no-analysis    # Scrape only, skip analysis
"""
import argparse
import json
import os
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from textblob import TextBlob
from tqdm import tqdm
from wordcloud import WordCloud

from src.exporters import XLSXExporter
from src.models import Review
from src.scrapers import AppStoreScraper, PlayStoreScraper
from src.utils.config import ensure_directories, get_app_config, load_app_config, load_config
from src.utils.logger import setup_logger


# ---------------------------------------------------------------------------
# DeepSeek helpers
# ---------------------------------------------------------------------------

def _call_deepseek(prompt: str, config: Dict) -> Optional[str]:
    """Call DeepSeek chat API and return raw content, or None on failure."""
    ds = config.get("deepseek", {})
    if not ds.get("enabled"):
        return None
    api_key = ds.get("api_key", "")
    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
        return None
    try:
        resp = requests.post(
            f"{ds.get('base_url', 'https://api.deepseek.com/v1')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": ds.get("model", "deepseek-chat"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": ds.get("max_tokens", 1000),
                "temperature": ds.get("temperature", 0.3),
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        print(f"DeepSeek API error {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"DeepSeek call failed: {e}")
    return None


def translate_with_deepseek(text: str, config: Dict) -> str:
    """Translate text to English via DeepSeek; returns original on failure."""
    if not text or not text.strip():
        return text
    result = _call_deepseek(
        f"Translate the following text to English. Return only the translated text, no explanations. Text: {text}",
        config,
    )
    return result.strip() if result else text


def analyze_sentiment_deepseek(text: str, config: Dict) -> Dict[str, Any]:
    """Analyze sentiment via DeepSeek API with TextBlob fallback.

    Returns dict with 'sentiment_score' (0–1) and 'sentiment_category'.
    """
    if not text:
        return {"sentiment_score": 0.5, "sentiment_category": "neutral"}

    ds = config.get("deepseek", {})
    if ds.get("enabled") and ds.get("api_key", "") not in ("", "YOUR_DEEPSEEK_API_KEY_HERE"):
        prompt = ds.get(
            "sentiment_prompt",
            "Analyze the sentiment of this banking app review. Return JSON with 'sentiment' "
            "(positive/negative/neutral), 'score' (0.0-1.0), 'confidence' (0.0-1.0). Review: {review}",
        ).format(review=text)
        raw = _call_deepseek(prompt, config)
        if raw:
            try:
                data = json.loads(raw)
                return {
                    "sentiment_score": float(data.get("score", 0.5)),
                    "sentiment_category": data.get("sentiment", "neutral"),
                }
            except (json.JSONDecodeError, ValueError):
                pass

    # TextBlob fallback: normalize polarity [-1, 1] → [0, 1]
    polarity = TextBlob(str(text)).sentiment.polarity
    score = (polarity + 1) / 2
    category = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
    return {"sentiment_score": score, "sentiment_category": category}


def categorize_problem(text: str) -> str:
    """Classify review into problem categories including customer-related issues."""
    if not text:
        return "General"
    t = text.lower()
    if any(w in t for w in ["customer service", "customer support", "support", "help", "客服", "contact", "complaint", "complained", "agent", "representative", "hotline", "phone", "email support", "poor service", "bad service"]):
        return "Customer Service Issues"
    if any(w in t for w in ["open account", "account opening", "registration", "sign up", "signup", "register"]):
        return "Account Opening Issues"
    if any(w in t for w in ["verify", "verification", "kyc", "identity", "document", "proof"]):
        return "Account Verification Issues"
    if any(w in t for w in ["app", "login", "crash", "error", "bug", "slow", "freeze", "technical", "system"]):
        return "App Operating Issues"
    if any(w in t for w in ["reward", "bonus", "cashback", "points", "promotion", "offer"]):
        return "Rewards Issues"
    return "General"


def get_wordcloud_phrases(text: str, config: Dict) -> Optional[str]:
    """Extract key 2–4 word phrases for a word cloud via DeepSeek."""
    template = config.get("deepseek", {}).get(
        "wordcloud_prompt",
        "Extract the most important 2-4 word phrases from these banking app reviews. "
        "Return only phrases separated by spaces. Reviews: {reviews}",
    )
    truncated = text[:5000] + ("..." if len(text) > 5000 else "")
    raw = _call_deepseek(template.format(reviews=truncated), config)
    if raw:
        cleaned = re.sub(r"[^\w\s]", " ", raw)
        return " ".join(cleaned.split())
    return None


# ---------------------------------------------------------------------------
# Multi-threaded review processing worker
# ---------------------------------------------------------------------------

def process_review_worker(args: tuple) -> Dict[str, Any]:
    """Worker for concurrent review translation, sentiment, and categorization."""
    idx, row, config = args
    try:
        content = str(row.get("content") or "")
        title = str(row.get("title") or "")

        translated = translate_with_deepseek(content, config)
        sentiment = analyze_sentiment_deepseek(f"{title} {translated}".strip(), config)

        pos_words, neg_words = [], []
        for assessment in TextBlob(translated).sentiment_assessments.assessments:
            word, score = assessment[0], assessment[1]
            if isinstance(word, list):
                word = " ".join(word)
            if isinstance(word, str):
                (pos_words if score > 0 else neg_words).append(word)

        score = sentiment["sentiment_score"]
        overall = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"

        return {
            "idx": idx,
            "translated_content": translated,
            "sentiment_score": score,
            "positive_words": ", ".join(pos_words[:5]),
            "negative_words": ", ".join(neg_words[:5]),
            "problem_category": categorize_problem(translated),
            "overall_sentiment": overall,
            "success": True,
        }
    except Exception as e:
        return {
            "idx": idx,
            "translated_content": "",
            "sentiment_score": 0.5,
            "positive_words": "",
            "negative_words": "",
            "problem_category": "General",
            "overall_sentiment": "neutral",
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main scraper / analyser class
# ---------------------------------------------------------------------------

class AppReviewScraper:
    """Scrapes App Store & Play Store reviews and runs sentiment analysis."""

    def __init__(
        self,
        config_path: str = "config/settings.json",
        apps_config_path: str = "config/apps.json",
    ):
        self.config = load_config(config_path)
        self.app_config = load_app_config(apps_config_path)
        self.logger = setup_logger(self.config)
        ensure_directories()
        self.xlsx_exporter = XLSXExporter(self.config)
        plt.rcParams.update({
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
        })

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape_app(self, app_cfg: Dict) -> Dict[str, List[Review]]:
        """Scrape both stores for one app; returns {'app_store': [...], 'play_store': [...]}."""
        app_name = app_cfg.get("name", "Unknown")
        self.logger.info(f"Starting scraping for: {app_name}")
        results: Dict[str, List[Review]] = {}

        for platform, scraper_cls in [("app_store", AppStoreScraper), ("play_store", PlayStoreScraper)]:
            if not app_cfg.get(platform, {}).get("enabled", True):
                continue
            try:
                scraper = scraper_cls(self.config, app_cfg)
                reviews = scraper.scrape_reviews()
                if reviews:
                    results[platform] = reviews
                    self.logger.info(f"{platform}: {len(reviews)} reviews collected")
                else:
                    self.logger.warning(f"{platform}: no reviews found")
            except Exception as e:
                self.logger.error(f"{platform} scraping failed for {app_name}: {e}")

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_reviews(self, app_key: str, results: Dict[str, List[Review]]) -> str:
        """Export all reviews to output/{app_key}_reviews.xlsx."""
        all_reviews = [r for platform_reviews in results.values() for r in platform_reviews]
        if not all_reviews:
            self.logger.warning("No reviews to export")
            return ""
        output_dir = self.config.get("output", {}).get("directory", "output")
        return self.xlsx_exporter.export(all_reviews, f"{app_key}_reviews", output_dir)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(self, file_path: str, app_key: str, timestamp: str) -> str:
        """Translate, score, categorise reviews; save charts, word clouds, and summary JSON."""
        print(f"\nStarting analysis for {app_key}...")
        df = pd.read_excel(file_path)
        print(f"Processing {len(df)} reviews with 8 workers...")

        # Initialise output columns
        for col in ["translated_content", "positive_words", "negative_words", "problem_category", "overall_sentiment"]:
            df[col] = ""
        df["sentiment_score"] = 0.0

        args_list = [(idx, row, self.config) for idx, row in df.iterrows()]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_review_worker, a): a[0] for a in args_list}
            for future in tqdm(as_completed(futures), total=len(args_list), desc="Analyzing"):
                r = future.result()
                for col in ["translated_content", "sentiment_score", "positive_words",
                            "negative_words", "problem_category", "overall_sentiment"]:
                    df.at[r["idx"], col] = r[col]

        output_dir = self.config.get("output", {}).get("directory", "output")
        analysis_dir = os.path.join(output_dir, f"analysis_{timestamp}")
        os.makedirs(analysis_dir, exist_ok=True)

        analyzed_file = os.path.join(analysis_dir, f"{app_key}_analyzed.xlsx")
        df.to_excel(analyzed_file, index=False)

        self._create_charts(df, app_key, analysis_dir)

        if app_key in self.config.get("deepseek", {}).get("enable_wordcloud_only_for", []):
            self._create_wordclouds(df, app_key, analysis_dir)

        summary = self._generate_summary(df, app_key)
        with open(os.path.join(analysis_dir, f"{app_key}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Analysis complete. Files saved to: {analysis_dir}")
        return analyzed_file

    def _create_charts(self, df: pd.DataFrame, app_key: str, output_dir: str) -> None:
        """Save a 2×2 analysis chart PNG."""
        colors = self.config.get("colors", {})
        sentiment_colors = colors.get("sentiment_colors", {
            "positive": "#2ca02c", "negative": "#d62728", "neutral": "#1f77b4",
        })
        palette = colors.get("chart_palette", ["#1f77b4"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Sentiment pie
        sc = df["overall_sentiment"].value_counts()
        axes[0, 0].pie(sc.values, labels=sc.index,
                       colors=[sentiment_colors.get(c, "#999") for c in sc.index],
                       autopct="%1.1f%%")
        axes[0, 0].set_title("Sentiment Distribution")

        # Rating bar
        rc = df["rating"].value_counts().sort_index()
        axes[0, 1].bar(rc.index, rc.values, color=palette[0])
        axes[0, 1].set(title="Rating Distribution", xlabel="Rating", ylabel="Count")

        # Category bar
        cc = df["problem_category"].value_counts()
        axes[1, 0].barh(cc.index, cc.values, color=palette[0])
        axes[1, 0].set(title="Problem Categories", xlabel="Count")

        # Sentiment trend over time
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
            monthly = (
                df.groupby([df["date"].dt.to_period("M"), "overall_sentiment"])
                .size()
                .unstack(fill_value=0)
            )
            monthly.plot(kind="area", ax=axes[1, 1],
                         color=[sentiment_colors.get(c, "#999") for c in monthly.columns])
            axes[1, 1].set(title="Sentiment Trends", xlabel="Month", ylabel="Reviews")
        else:
            axes[1, 1].text(0.5, 0.5, "Date data not available", ha="center", va="center",
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Sentiment Trends")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{app_key}_charts.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _create_wordclouds(self, df: pd.DataFrame, app_key: str, output_dir: str) -> None:
        """Create positive/negative word cloud PNGs using DeepSeek phrase extraction."""
        plt.rcParams["font.family"] = "Arial"
        for sentiment, color in [("positive", (0, 184, 245)), ("negative", (12, 35, 60))]:
            reviews = df[df["overall_sentiment"] == sentiment]["translated_content"].dropna()
            if reviews.empty:
                continue
            text = " ".join(reviews.astype(str))
            wc_text = get_wordcloud_phrases(text, self.config) or text
            wc = WordCloud(
                width=800, height=800, background_color="white",
                color_func=lambda *a, _color=color, **kw: _color,
                max_words=100, min_font_size=10, max_font_size=80,
                prefer_horizontal=0.7, collocations=False,
            ).generate(wc_text)
            plt.figure(figsize=(10, 10))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"{app_key}_{sentiment}_wordcloud.png"),
                dpi=300, bbox_inches="tight", facecolor="white",
            )
            plt.close()

    def _generate_summary(self, df: pd.DataFrame, app_key: str) -> Dict[str, Any]:
        """Build and return a summary dict with stats and recommendations."""
        recs = []
        if (df["overall_sentiment"] == "negative").mean() > 0.3:
            recs.append("High negative sentiment – focus on addressing user concerns.")
        avg = float(df["rating"].mean()) if "rating" in df.columns else 0
        if avg < 3.0:
            recs.append("Low average rating – urgent app improvements needed.")
        elif avg < 4.0:
            recs.append("Below-average rating – focus on user satisfaction.")
        for cat, count in df["problem_category"].value_counts().head(3).items():
            pct = count / len(df)
            if cat == "Customer Service Issues" and pct > 0.15:
                recs.append("Customer service complaints – improve support responsiveness and training.")
            elif cat == "App Operating Issues" and pct > 0.2:
                recs.append("Frequent app issues – consider performance optimisation.")
            elif cat == "Account Verification Issues" and pct > 0.15:
                recs.append("Verification issues reported – review KYC process.")
            elif cat == "Account Opening Issues" and pct > 0.15:
                recs.append("Account opening friction – streamline registration.")

        # Customer-related metrics
        customer_issues = df[df["problem_category"] == "Customer Service Issues"]
        customer_negative_pct = (
            (customer_issues["overall_sentiment"] == "negative").mean()
            if len(customer_issues) > 0 else 0.0
        )

        return {
            "app_key": app_key,
            "generated_at": datetime.now().isoformat(),
            "total_reviews": len(df),
            "average_rating": round(avg, 2),
            "average_sentiment_score": round(float(df["sentiment_score"].mean()), 3),
            "sentiment_distribution": df["overall_sentiment"].value_counts().to_dict(),
            "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
            "problem_categories": df["problem_category"].value_counts().to_dict(),
            "platform_distribution": df["platform"].value_counts().to_dict() if "platform" in df.columns else {},
            "customer_analysis": {
                "customer_service_reviews_count": len(customer_issues),
                "customer_service_reviews_pct": round(100 * len(customer_issues) / len(df), 2),
                "customer_service_negative_pct": round(100 * customer_negative_pct, 2),
            },
            "recommendations": recs[:5],
        }

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    def create_overview(self, timestamp: str) -> str:
        """Scrape metadata for every configured bank and save an overview JSON."""
        all_apps = load_app_config().get("apps", {})
        output_dir = os.path.join(self.config.get("output", {}).get("directory", "output"), timestamp)
        os.makedirs(output_dir, exist_ok=True)

        overview: Dict[str, Any] = {"timestamp": timestamp, "total_banks": len(all_apps), "banks": {}}

        for app_key, app_cfg in all_apps.items():
            bank: Dict[str, Any] = {"name": app_cfg.get("name", app_key), "app_store": {}, "play_store": {}}
            for platform, scraper_cls in [("app_store", AppStoreScraper), ("play_store", PlayStoreScraper)]:
                cfg = app_cfg.get(platform, {})
                bank[platform] = {"enabled": cfg.get("enabled", False), "reviews_count": 0, "average_rating": 0.0}
                if not cfg.get("enabled"):
                    continue
                try:
                    reviews = scraper_cls(self.config, app_cfg).scrape_reviews()
                    if reviews:
                        ratings = [r.rating for r in reviews if r.rating > 0]
                        bank[platform].update({
                            "reviews_count": len(reviews),
                            "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
                        })
                except Exception as e:
                    self.logger.error(f"Overview scrape failed – {app_key}/{platform}: {e}")
            overview["banks"][app_key] = bank

        filepath = os.path.join(output_dir, f"ALL_BANKS_OVERVIEW_{timestamp}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(overview, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Overview saved: {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape and analyse HK banking app reviews from App Store & Google Play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
Examples:
  python main.py welab_bank                  # Scrape and analyse WeLab Bank
  python main.py welab_bank za_bank          # Multiple banks
  python main.py --all                       # All configured banks
  python main.py --overview                  # Overview JSON (no analysis)
  python main.py welab_bank --initial        # Force re-scrape even if file exists
  python main.py welab_bank --no-analysis    # Scrape only, skip analysis
        """),
    )
    parser.add_argument("apps", nargs="*", help="App keys to process (e.g. welab_bank za_bank)")
    parser.add_argument("--all", action="store_true", help="Process all configured apps")
    parser.add_argument("--overview", action="store_true", help="Create overview JSON for all banks")
    parser.add_argument("--initial", action="store_true", help="Force re-scrape existing data")
    parser.add_argument("--no-analysis", action="store_true", help="Scrape only, skip sentiment analysis")
    args = parser.parse_args()

    scraper = AppReviewScraper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.overview:
        path = scraper.create_overview(timestamp)
        print(f"\nOverview saved to: {path}")
        return

    if args.all:
        app_keys = list(load_app_config().get("apps", {}).keys())
    elif args.apps:
        app_keys = args.apps
    else:
        parser.print_help()
        return

    print(f"Processing {len(app_keys)} app(s)…")

    for app_key in app_keys:
        print(f"\n{'='*60}\nProcessing: {app_key}\n{'='*60}")

        review_file = os.path.join("output", f"{app_key}_reviews.xlsx")

        if os.path.exists(review_file) and not args.initial:
            print(f"Existing reviews found: {review_file}")
        else:
            app_cfg = get_app_config(app_key)
            if not app_cfg:
                print(f"No config found for '{app_key}' – check config/apps.json")
                continue

            results = scraper.scrape_app(app_cfg)
            if not results:
                print(f"No reviews collected for {app_key}")
                continue

            review_file = scraper.export_reviews(app_key, results)
            if not review_file:
                print(f"Export failed for {app_key}")
                continue
            print(f"Reviews saved: {review_file}")

        if not args.no_analysis and os.path.exists(review_file):
            try:
                out = scraper.run_analysis(review_file, app_key, timestamp)
                print(f"Analysis saved: {out}")
            except Exception as e:
                print(f"Analysis failed for {app_key}: {e}")

    print(f"\nDone! Processed {len(app_keys)} app(s).")


if __name__ == "__main__":
    main()
