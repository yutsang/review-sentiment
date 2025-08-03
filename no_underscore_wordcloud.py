#!/usr/bin/env python3

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

def extract_phrases_from_text(text: str) -> dict:
    """Extract meaningful phrases from text without using underscores"""
    # Common banking phrases to look for
    banking_phrases = [
        # Account related
        r'\baccount opening\b', r'\baccount verification\b', r'\baccount setup\b',
        r'\baccount access\b', r'\baccount management\b', r'\baccount security\b',
        
        # App related
        r'\bapp crashes\b', r'\bapp freezes\b', r'\bapp loading\b', r'\bapp updates\b',
        r'\bapp performance\b', r'\bapp interface\b', r'\bapp navigation\b',
        
        # Login/Auth related
        r'\blogin problems\b', r'\blogin issues\b', r'\bauthentication failed\b',
        r'\bpassword reset\b', r'\bsecurity verification\b', r'\bidentity verification\b',
        r'\bFace ID\b', r'\bFace ID verification\b',
        
        # Transaction related
        r'\btransfer money\b', r'\bpayment processing\b', r'\btransaction failed\b',
        r'\bpayment issues\b', r'\bmoney transfer\b', r'\btransaction processing\b',
        
        # Customer service
        r'\bcustomer service\b', r'\bcustomer support\b', r'\bsupport team\b',
        r'\bhelp desk\b', r'\bcustomer care\b',
        
        # Technical issues
        r'\bsystem error\b', r'\btechnical problems\b', r'\bserver issues\b',
        r'\bnetwork problems\b', r'\bconnection issues\b', r'\bdata loading\b',
        
        # User experience
        r'\buser interface\b', r'\buser experience\b', r'\beasy to use\b',
        r'\bdifficult to use\b', r'\bconfusing interface\b', r'\bintuitive design\b',
        
        # Banking features
        r'\bonline banking\b', r'\bmobile banking\b', r'\bdigital banking\b',
        r'\bbanking app\b', r'\bchecking account\b', r'\bsavings account\b',
        
        # Common issues
        r'\bvery slow\b', r'\btoo slow\b', r'\bnot working\b', r'\bdoes not work\b',
        r'\bkeep crashing\b', r'\bconstantly crashes\b', r'\bunable to access\b',
        r'\bcannot access\b', r'\bhard to use\b', r'\bdifficult to navigate\b',
        r'\bpoor service\b', r'\bbad experience\b', r'\bgood experience\b'
    ]
    
    text_lower = text.lower()
    phrase_counts = {}
    
    # Find exact phrase matches
    for phrase_pattern in banking_phrases:
        matches = re.findall(phrase_pattern, text_lower)
        for match in matches:
            if match in phrase_counts:
                phrase_counts[match] += 1
            else:
                phrase_counts[match] = 1
    
    # Also extract common 2-3 word combinations
    words = text_lower.split()
    for i in range(len(words) - 1):
        # 2-word phrases
        phrase_2 = f"{words[i]} {words[i+1]}"
        if len(phrase_2.split()) == 2 and len(phrase_2) > 5:
            if phrase_2 in phrase_counts:
                phrase_counts[phrase_2] += 1
            else:
                phrase_counts[phrase_2] = 1
        
        # 3-word phrases
        if i < len(words) - 2:
            phrase_3 = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase_3.split()) == 3 and len(phrase_3) > 8:
                if phrase_3 in phrase_counts:
                    phrase_counts[phrase_3] += 1
                else:
                    phrase_counts[phrase_3] = 1
    
    return phrase_counts

def create_phrase_wordcloud_no_underscore(phrases_text: str, color_func, output_file: str):
    """Create wordcloud with phrases using spaces, no underscores"""
    
    # Extract phrases directly from text
    phrase_counts = extract_phrases_from_text(phrases_text)
    
    # Filter out very low frequency phrases and very long phrases
    filtered_phrases = {}
    for phrase, count in phrase_counts.items():
        if count >= 2 and len(phrase) <= 30:  # Only phrases that appear at least twice and aren't too long
            filtered_phrases[phrase] = count
    
    print(f"📝 Found phrases: {list(filtered_phrases.keys())[:10]}")
    
    # Create wordcloud with the phrases
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
    ).generate_from_frequencies(filtered_phrases)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ No-underscore wordcloud saved: {output_file}")

# Test with sample text
test_text = """
The app crashes frequently and is very slow. Customer service is terrible. 
Account opening process takes too long. Login problems every day. 
Transfer money feature doesn't work properly. App interface is confusing.
Face ID verification process is poor. The bank has bad experience.
"""

print("Creating wordcloud without underscores...")
print(f"Input text: {test_text}")

create_phrase_wordcloud_no_underscore(
    test_text,
    lambda *args, **kwargs: (12, 35, 60),  # Dark Blue
    'no_underscore_wordcloud.png'
) 