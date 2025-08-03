#!/usr/bin/env python3

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Test phrases with underscores (as processed by our system)
test_phrases = "app_crashes login_problems Face_ID_verification poor_service customer_service account_opening transfer_money cannot_open money_locked bad_experience"

print("Testing wordcloud with phrase display...")
print(f"Input phrases: {test_phrases}")

# Method 1: Use underscores but display with spaces
wordcloud = WordCloud(
    width=800, height=800,
    background_color='white',
    color_func=lambda *args, **kwargs: (12, 35, 60),  # Dark Blue
    max_words=50,
    stopwords=None,
    min_font_size=10,
    max_font_size=80,
    prefer_horizontal=0.7,
    collocations=False,
    regexp=r'[\w_]+'  # Allow underscores
).generate(test_phrases)

# Create figure
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('test_underscore_wordcloud.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Test wordcloud with underscores saved as test_underscore_wordcloud.png")

# Method 2: Create custom wordcloud with space replacement
# This is a more complex approach that would require custom wordcloud generation
print("📝 Note: The wordcloud will display underscores as part of the text.")
print("📝 To display spaces instead, we would need to modify the wordcloud generation process.")
print("📝 For now, the phrases are preserved as single entities but displayed with underscores.") 