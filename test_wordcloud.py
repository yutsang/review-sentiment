#!/usr/bin/env python3

import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Test phrase text
test_phrases = "app_crashes customer_service account_opening login_problems transfer_money app_interface confusing_interface very_slow terrible_service"

print("Testing wordcloud with phrases...")
print(f"Input phrases: {test_phrases}")

# Create wordcloud
wordcloud = WordCloud(
    width=800, height=800,
    background_color='white',
    color_func=lambda *args, **kwargs: (12, 35, 60),  # Dark Blue
    max_words=100,
    stopwords=None,
    min_font_size=10,
    max_font_size=80,
    prefer_horizontal=0.7,
    collocations=False
).generate(test_phrases)

# Save test wordcloud
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('test_wordcloud.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Test wordcloud saved as test_wordcloud.png")
print("Check if phrases like 'app_crashes' appear as single items in the wordcloud") 