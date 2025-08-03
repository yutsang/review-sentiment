#!/usr/bin/env python3

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

def create_phrase_wordcloud(phrases_text: str, color_func, output_file: str):
    """Create wordcloud with proper phrase display using spaces"""
    
    # Split into phrases and count frequencies
    phrases = phrases_text.split()
    phrase_counts = Counter(phrases)
    
    # Convert underscores back to spaces for display
    display_phrases = {}
    for phrase, count in phrase_counts.items():
        # Replace underscores with spaces for display
        display_phrase = phrase.replace('_', ' ')
        display_phrases[display_phrase] = count
    
    # Create wordcloud with the display phrases
    wordcloud = WordCloud(
        width=800, height=800,
        background_color='white',
        color_func=color_func,
        max_words=50,
        stopwords=None,
        min_font_size=10,
        max_font_size=80,
        prefer_horizontal=0.7,
        collocations=False
    ).generate_from_frequencies(display_phrases)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Custom phrase wordcloud saved: {output_file}")
    print(f"📝 Sample phrases: {list(display_phrases.keys())[:10]}")

# Test with our phrases
test_phrases = "app_crashes login_problems Face_ID_verification poor_service customer_service account_opening transfer_money cannot_open money_locked bad_experience"

print("Creating custom phrase wordcloud...")
print(f"Input phrases: {test_phrases}")

create_phrase_wordcloud(
    test_phrases,
    lambda *args, **kwargs: (12, 35, 60),  # Dark Blue
    'custom_phrase_wordcloud.png'
) 