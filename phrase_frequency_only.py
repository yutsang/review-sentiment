#!/usr/bin/env python3

import pandas as pd
import os
import re
from datetime import datetime

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
    
    # Stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'not', 'no', 'nor', 'neither', 'either', 'so', 'as', 'than', 'too', 'very', 'just',
        'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    }
    
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
    
    # Extract meaningful 2-3 word combinations (filtering out stop words)
    words = text_lower.split()
    for i in range(len(words) - 1):
        # Skip if current word is a stop word
        if words[i] in stop_words:
            continue
            
        # 2-word phrases
        if i + 1 < len(words) and words[i+1] not in stop_words:
            phrase_2 = f"{words[i]} {words[i+1]}"
            if len(phrase_2) > 5 and not any(word in stop_words for word in phrase_2.split()):
                if phrase_2 in phrase_counts:
                    phrase_counts[phrase_2] += 1
                else:
                    phrase_counts[phrase_2] = 1
        
        # 3-word phrases
        if i + 2 < len(words) and words[i+1] not in stop_words and words[i+2] not in stop_words:
            phrase_3 = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase_3) > 8 and not any(word in stop_words for word in phrase_3.split()):
                if phrase_3 in phrase_counts:
                    phrase_counts[phrase_3] += 1
                else:
                    phrase_counts[phrase_3] = 1
    
    return phrase_counts

def merge_similar_phrases(phrase_counts: dict) -> dict:
    """Merge similar phrases and combine their frequencies"""
    # Define phrase groups that should be merged
    phrase_groups = [
        # Quantity expressions
        ['a lot', 'lots of', 'a lot of', 'many', 'much'],
        ['very good', 'really good', 'quite good', 'pretty good'],
        ['very bad', 'really bad', 'quite bad', 'pretty bad'],
        ['very slow', 'really slow', 'quite slow', 'too slow'],
        ['very fast', 'really fast', 'quite fast'],
        ['very easy', 'really easy', 'quite easy'],
        ['very difficult', 'really difficult', 'quite difficult'],
        
        # App issues
        ['app crashes', 'app crash', 'crashes', 'crash'],
        ['app freezes', 'app freeze', 'freezes', 'freeze'],
        ['app loading', 'loading', 'loads slowly'],
        ['app updates', 'app update', 'updates', 'update'],
        
        # Login issues
        ['login problems', 'login problem', 'login issues', 'login issue'],
        ['authentication failed', 'auth failed', 'login failed'],
        ['password reset', 'reset password', 'password change'],
        
        # Account issues
        ['account opening', 'open account', 'account setup', 'setup account'],
        ['account verification', 'verify account', 'account verification process'],
        ['account access', 'access account', 'account login'],
        
        # Customer service
        ['customer service', 'customer support', 'support service'],
        ['poor service', 'bad service', 'terrible service'],
        ['good service', 'excellent service', 'great service'],
        
        # Transaction issues
        ['transfer money', 'money transfer', 'transfer funds'],
        ['payment processing', 'process payment', 'payment issues'],
        ['transaction failed', 'failed transaction', 'transaction error'],
        
        # User experience
        ['user interface', 'interface', 'ui'],
        ['user experience', 'experience', 'ux'],
        ['easy to use', 'easy use', 'simple to use', 'user friendly'],
        ['difficult to use', 'hard to use', 'complicated', 'confusing'],
        
        # Technical terms
        ['Face ID', 'face id', 'face recognition', 'biometric'],
        ['system error', 'system errors', 'error', 'errors'],
        ['network problems', 'network issues', 'connection problems'],
        ['data loading', 'loading data', 'data load'],
        
        # Banking terms
        ['online banking', 'mobile banking', 'digital banking', 'banking app'],
        ['checking account', 'current account', 'bank account'],
        ['savings account', 'deposit account'],
        
        # General issues
        ['not working', 'does not work', 'doesn\'t work', 'not work'],
        ['keep crashing', 'constantly crashes', 'always crashes'],
        ['unable to access', 'cannot access', 'can\'t access'],
        ['hard to use', 'difficult to use', 'not easy to use'],
        ['bad experience', 'poor experience', 'terrible experience'],
        ['good experience', 'great experience', 'excellent experience']
    ]
    
    # Create a mapping of phrases to their primary form
    phrase_mapping = {}
    for group in phrase_groups:
        primary = group[0]  # Use the first phrase as primary
        for phrase in group:
            phrase_mapping[phrase] = primary
    
    # Merge frequencies
    merged_counts = {}
    for phrase, count in phrase_counts.items():
        # Check if this phrase should be merged
        primary_phrase = phrase_mapping.get(phrase, phrase)
        
        if primary_phrase in merged_counts:
            merged_counts[primary_phrase] += count
        else:
            merged_counts[primary_phrase] = count
    
    return merged_counts

def get_top_phrases_with_frequency(text: str, top_n: int = 30) -> list:
    """Get top N phrases with their frequencies, merging similar phrases first"""
    # Extract phrases from text
    phrase_counts = extract_phrases_from_text(text)
    
    # Merge similar phrases
    merged_counts = merge_similar_phrases(phrase_counts)
    
    # Sort by frequency (descending) and get top N
    sorted_phrases = sorted(merged_counts.items(), key=lambda x: x[1], reverse=True)
    top_phrases = sorted_phrases[:top_n]
    
    return top_phrases

def output_phrase_frequencies_only(app_key: str):
    """Output top 30 phrases with frequency for positive and negative reviews"""
    
    # Read the reviews file
    reviews_file = f"output/{app_key}_reviews.xlsx"
    if not os.path.exists(reviews_file):
        print(f"❌ Reviews file not found: {reviews_file}")
        return
    
    print(f"📊 Reading data from: {reviews_file}")
    df = pd.read_excel(reviews_file)
    
    # Check if analysis columns exist
    if 'overall_sentiment' not in df.columns or 'translated_content' not in df.columns:
        print("❌ Analysis columns not found. Please run full analysis first.")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/phrase_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Positive reviews
    positive_reviews = df[df['overall_sentiment'] == 'positive']['translated_content'].dropna()
    if not positive_reviews.empty:
        positive_text = ' '.join(positive_reviews.astype(str))
        positive_phrases = get_top_phrases_with_frequency(positive_text, 30)
        
        # Save to file
        positive_file = os.path.join(output_dir, f"{app_key}_positive_phrases.txt")
        with open(positive_file, 'w', encoding='utf-8') as f:
            f.write(f"Top 30 Most Common Phrases in Positive Reviews for {app_key}\n")
            f.write("=" * 60 + "\n\n")
            for i, (phrase, count) in enumerate(positive_phrases, 1):
                f.write(f"{i:2d}. {phrase:<30} (frequency: {count})\n")
        
        print(f"✅ Positive phrases saved: {positive_file}")
        print("📝 Top 10 positive phrases:")
        for i, (phrase, count) in enumerate(positive_phrases[:10], 1):
            print(f"   {i:2d}. {phrase:<25} (frequency: {count})")
    
    # Negative reviews
    negative_reviews = df[df['overall_sentiment'] == 'negative']['translated_content'].dropna()
    if not negative_reviews.empty:
        negative_text = ' '.join(negative_reviews.astype(str))
        negative_phrases = get_top_phrases_with_frequency(negative_text, 30)
        
        # Save to file
        negative_file = os.path.join(output_dir, f"{app_key}_negative_phrases.txt")
        with open(negative_file, 'w', encoding='utf-8') as f:
            f.write(f"Top 30 Most Common Phrases in Negative Reviews for {app_key}\n")
            f.write("=" * 60 + "\n\n")
            for i, (phrase, count) in enumerate(negative_phrases, 1):
                f.write(f"{i:2d}. {phrase:<30} (frequency: {count})\n")
        
        print(f"✅ Negative phrases saved: {negative_file}")
        print("📝 Top 10 negative phrases:")
        for i, (phrase, count) in enumerate(negative_phrases[:10], 1):
            print(f"   {i:2d}. {phrase:<25} (frequency: {count})")
    
    print(f"✅ Phrase analysis completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    # You can change this to any bank key
    app_key = "welab_bank"
    output_phrase_frequencies_only(app_key) 