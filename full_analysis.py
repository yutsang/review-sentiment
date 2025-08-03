#!/usr/bin/env python3

import pandas as pd
import json
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def call_deepseek_api(text: str, config: dict) -> str:
    """Call DeepSeek API to get AI-processed phrases for wordcloud"""
    try:
        deepseek_config = config.get('deepseek', {})
        
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            return ""
            
        base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = deepseek_config.get('model', 'deepseek-chat')
        max_tokens = deepseek_config.get('max_tokens', 1000)
        temperature = deepseek_config.get('temperature', 0.3)
        prompt_template = deepseek_config.get('wordcloud_prompt', '')
        
        # Truncate text if too long
        truncated_text = text[:2000] + "..." if len(text) > 2000 else text
        
        # Format prompt
        prompt = prompt_template.format(reviews=truncated_text)
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Clean up the response - extract only phrases
            phrases = content.replace('\n', ' ').replace('\r', ' ').strip()
            return phrases
        else:
            print(f"⚠️ DeepSeek API error: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"⚠️ DeepSeek API call failed: {e}")
        return ""

def preserve_phrases_for_wordcloud(text: str) -> str:
    """Preserve multi-word phrases for wordcloud by using underscores to keep them together"""
    words = text.split()
    processed_words = []
    
    i = 0
    while i < len(words):
        # Check if this word is part of a phrase (2-4 words max for readability)
        phrase_length = 1
        phrase_words = [words[i]]
        
        # Look ahead to find complete phrases (max 4 words for readability)
        for j in range(1, 4):  # Check up to 4 words
            if i + j < len(words):
                next_word = words[i + j]
                # If next word doesn't start with uppercase or is a common connector, it's part of phrase
                if (not next_word[0].isupper() and 
                    next_word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'has', 'have', 'had']):
                    phrase_words.append(next_word)
                    phrase_length += 1
                else:
                    break
            else:
                break
        
        # If we have a meaningful phrase (2-4 words), join with underscores for wordcloud
        if phrase_length >= 2 and phrase_length <= 4:
            phrase = '_'.join(phrase_words)
            # Only add if it's a meaningful phrase (not too long)
            if len(phrase) <= 30:  # Limit phrase length
                processed_words.append(phrase)
            else:
                # If too long, just add the first word
                processed_words.append(words[i])
            i += phrase_length  # Skip the words we've already processed
        else:
            # Single word, keep as is
            processed_words.append(words[i])
            i += 1
    
    return ' '.join(processed_words)

def analyze_sentiment_deepseek(text: str, config: dict) -> dict:
    """Analyze sentiment using DeepSeek API"""
    try:
        deepseek_config = config.get('deepseek', {})
        
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            # Fallback to TextBlob
            blob = TextBlob(text)
            return {
                'sentiment_score': (blob.sentiment.polarity + 1) / 2,  # Convert to 0-1 scale
                'sentiment_category': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
            }
            
        base_url = deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = deepseek_config.get('model', 'deepseek-chat')
        max_tokens = deepseek_config.get('max_tokens', 1000)
        temperature = deepseek_config.get('temperature', 0.3)
        sentiment_prompt = deepseek_config.get('sentiment_prompt', 'Analyze the sentiment of the following text. Return only a JSON object with "sentiment" (positive/negative/neutral), "score" (0.0 to 1.0), and "confidence" (0.0 to 1.0). Text: {text}')
        
        # Format the prompt
        formatted_prompt = sentiment_prompt.format(review=text)
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': formatted_prompt}
            ],
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', content)
                if json_match:
                    sentiment_data = json.loads(json_match.group())
                    return {
                        'sentiment_score': float(sentiment_data.get('score', 0.5)),
                        'sentiment_category': sentiment_data.get('sentiment', 'neutral')
                    }
            except:
                pass
            
            # Fallback parsing
            if 'positive' in content.lower():
                return {'sentiment_score': 0.8, 'sentiment_category': 'positive'}
            elif 'negative' in content.lower():
                return {'sentiment_score': 0.2, 'sentiment_category': 'negative'}
            else:
                return {'sentiment_score': 0.5, 'sentiment_category': 'neutral'}
        else:
            # Fallback to TextBlob
            blob = TextBlob(text)
            return {
                'sentiment_score': (blob.sentiment.polarity + 1) / 2,
                'sentiment_category': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
            }
            
    except Exception as e:
        # Fallback to TextBlob
        blob = TextBlob(text)
        return {
            'sentiment_score': (blob.sentiment.polarity + 1) / 2,
            'sentiment_category': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
        }

def translate_non_english_content(text: str, config: dict) -> str:
    """Translate non-English content to English using DeepSeek API"""
    if not text or not isinstance(text, str):
        return text
    
    # Simple check for non-English content (Chinese characters, etc.)
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    if not chinese_pattern.search(text):
        return text  # No Chinese characters, assume English
    
    try:
        if not config.get('deepseek', {}).get('enabled', False):
            logger.warning("DeepSeek translation not enabled, skipping translation")
            return text
        
        api_key = config['deepseek']['api_key']
        base_url = config['deepseek']['base_url']
        model = config['deepseek']['model']
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': f'Translate the following text to English. If it\'s already in English, return it as is. Only return the translated text, no explanations: {text}'
                }
            ],
            'max_tokens': 500,
            'temperature': 0.1
        }
        
        response = requests.post(f'{base_url}/chat/completions', headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        translated_text = result['choices'][0]['message']['content'].strip()
        
        # Remove quotes if the API wrapped the response in quotes
        if translated_text.startswith('"') and translated_text.endswith('"'):
            translated_text = translated_text[1:-1]
        
        logger.debug(f"Translated: '{text[:50]}...' -> '{translated_text[:50]}...'")
        return translated_text
        
    except Exception as e:
        logger.warning(f"Translation failed for text: {text[:100]}... Error: {str(e)}")
        return text  # Return original text if translation fails

def categorize_problem_simple(review_text: str) -> str:
    """Categorize review into one of 5 specific problem categories"""
    if not review_text:
        return "General"
    
    text = review_text.lower()
    
    # Account Opening Issues
    if any(word in text for word in ['open account', 'account opening', 'registration', 'sign up', 'signup', 'register']):
        return "Account Opening Issues"
    
    # Account Verification Issues
    if any(word in text for word in ['verify', 'verification', 'kyc', 'identity', 'document', 'proof']):
        return "Account Verification Issues"
    
    # App Operating Issues
    if any(word in text for word in ['app', 'login', 'crash', 'error', 'bug', 'slow', 'freeze', 'technical', 'system']):
        return "App Operating Issues"
    
    # Rewards Issues
    if any(word in text for word in ['reward', 'bonus', 'cashback', 'points', 'promotion', 'offer']):
        return "Rewards Issues"
    
    # Default to General
    return "General"

def process_review_worker(args):
    """Worker function for processing individual reviews with multi-threading"""
    idx, row, config = args
    try:
        # Get content and title, handling NaN values
        content = row.get('content', '')
        title = row.get('title', '')
        
        # Translate non-English content before analysis
        translated_content = translate_non_english_content(content, config)
        translated_title = translate_non_english_content(title, config)
        
        # 1. Sentiment analysis using translated text
        analysis_text = f"{translated_title} {translated_content}".strip()
        sentiment = analyze_sentiment_deepseek(analysis_text, config)
        
        # 2. Positive and negative words (simplified)
        blob = TextBlob(translated_content)
        # Extract words from assessments
        positive_words = []
        negative_words = []
        for assessment in blob.sentiment_assessments.assessments:
            if len(assessment) >= 2:
                word = assessment[0]
                score = assessment[1]
                # Handle case where word might be a list
                if isinstance(word, list):
                    word = ' '.join(word)
                if isinstance(word, str):
                    if score > 0:
                        positive_words.append(word)
                    elif score < 0:
                        negative_words.append(word)
        
        positive_words_str = ', '.join(positive_words[:5])  # Top 5 positive words
        negative_words_str = ', '.join(negative_words[:5])  # Top 5 negative words
        
        # 3. Problem categorization
        problem_category = categorize_problem_simple(translated_content)
        
        # 4. Overall sentiment (positive/negative)
        if sentiment['sentiment_score'] > 0.6:
            overall_sentiment = 'positive'
        elif sentiment['sentiment_score'] < 0.4:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'idx': idx,
            'translated_content': translated_content,  # Use translated content
            'sentiment_score': sentiment['sentiment_score'],
            'positive_words': positive_words_str,
            'negative_words': negative_words_str,
            'problem_category': problem_category,
            'overall_sentiment': overall_sentiment,
            'success': True
        }
        
    except Exception as e:
        return {
            'idx': idx,
            'translated_content': '',
            'sentiment_score': 0.0,
            'positive_words': '',
            'negative_words': '',
            'problem_category': 'General',
            'overall_sentiment': 'neutral',
            'success': False,
            'error': str(e)
        }

def extract_phrases_from_text(text: str) -> dict:
    """Extract meaningful descriptive phrases from translated text with proper frequencies"""
    # More descriptive banking phrases that tell us what actually happened
    descriptive_phrases = [
        # Specific problems and experiences
        r'\bapp keeps crashing\b', r'\bapp crashes every time\b', r'\bapp crashes on startup\b',
        r'\bcannot open account\b', r'\baccount opening failed\b', r'\bverification process stuck\b',
        r'\blogin always fails\b', r'\bcannot access account\b', r'\bFace ID not working\b',
        r'\btransfer money failed\b', r'\bpayment processing error\b', r'\btransaction declined\b',
        r'\bcustomer service unhelpful\b', r'\bsupport team unresponsive\b', r'\bno response from support\b',
        r'\bapp freezes constantly\b', r'\bapp loads very slowly\b', r'\binterface too confusing\b',
        r'\bhard to navigate\b', r'\bnot user friendly\b', r'\btoo complicated to use\b',
        r'\bverification takes too long\b', r'\baccount setup difficult\b', r'\bregistration process slow\b',
        r'\bcredit card declined\b', r'\bpayment rejected\b', r'\binsufficient funds error\b',
        r'\bnetwork connection error\b', r'\bserver down\b', r'\bsystem maintenance\b',
        r'\bapp update broke\b', r'\bnew version worse\b', r'\bupdate caused problems\b',
        r'\bsecurity verification failed\b', r'\bidentity verification stuck\b', r'\bphoto verification failed\b',
        r'\bQR code not working\b', r'\bscanning failed\b', r'\bcard reader error\b',
        r'\boverdraft fee charged\b', r'\bhidden fees\b', r'\bunexpected charges\b',
        r'\binterest rate too high\b', r'\bcredit limit too low\b', r'\bminimum balance requirement\b',
        r'\bforeign transaction fee\b', r'\bcurrency conversion fee\b', r'\bexchange rate poor\b',
        
        # Positive experiences
        r'\bapp works perfectly\b', r'\beasy to set up\b', r'\bregistration completed quickly\b',
        r'\bverification process smooth\b', r'\blogin works every time\b', r'\bFace ID works perfectly\b',
        r'\btransfer money instantly\b', r'\bpayment processed quickly\b', r'\btransaction successful\b',
        r'\bcustomer service helpful\b', r'\bsupport team responsive\b', r'\bquick response from support\b',
        r'\bapp never crashes\b', r'\bapp loads quickly\b', r'\binterface intuitive\b',
        r'\beasy to navigate\b', r'\bvery user friendly\b', r'\bsimple to use\b',
        r'\bverification completed fast\b', r'\baccount setup easy\b', r'\bregistration process smooth\b',
        r'\bcredit card approved\b', r'\bpayment accepted\b', r'\bno hidden fees\b',
        r'\bcompetitive interest rate\b', r'\bgenerous credit limit\b', r'\bno minimum balance\b',
        r'\bno foreign transaction fee\b', r'\bfair exchange rate\b', r'\bgood currency conversion\b',
        
        # Specific banking features
        r'\bvirtual card works\b', r'\bphysical card received\b', r'\bcard activation easy\b',
        r'\bATM withdrawal works\b', r'\boverseas usage works\b', r'\bcontactless payment works\b',
        r'\bApple Pay integration\b', r'\bGoogle Pay works\b', r'\bSamsung Pay supported\b',
        r'\bautomatic bill payment\b', r'\brecurring payment setup\b', r'\bscheduled transfer works\b',
        r'\bstanding order created\b', r'\bdirect debit setup\b', r'\bautomatic savings\b',
        r'\bround up feature\b', r'\bsavings goal tracking\b', r'\bbudget management tools\b',
        r'\bspending analysis\b', r'\btransaction categorization\b', r'\bexpense tracking\b',
        r'\bmonthly statement\b', r'\btransaction history\b', r'\baccount balance check\b',
        r'\binterest earned\b', r'\bcashback rewards\b', r'\bpoints earned\b',
        r'\bpromotional offers\b', r'\bwelcome bonus\b', r'\breferral rewards\b'
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
    
    # Find exact descriptive phrase matches first
    for phrase_pattern in descriptive_phrases:
        matches = re.findall(phrase_pattern, text_lower)
        for match in matches:
            if match in phrase_counts:
                phrase_counts[match] += 1
            else:
                phrase_counts[match] = 1
    
    # Extract meaningful 3-6 word combinations from sentences (longer for more complete phrases)
    sentences = re.split(r'[.!?]+', text_lower)
    
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) < 3:
            continue
            
        # Extract 3-6 word phrases from each sentence for more complete phrases
        for start in range(len(words)):
            for length in range(3, 7):  # 3 to 6 words
                if start + length <= len(words):
                    phrase_words = words[start:start + length]
                    
                    # Skip if phrase contains too many stop words
                    stop_word_count = sum(1 for word in phrase_words if word in stop_words)
                    if stop_word_count > len(phrase_words) * 0.7:  # More than 70% stop words
                        continue
                    
                    # Skip if phrase starts AND ends with stop words
                    if phrase_words[0] in stop_words and phrase_words[-1] in stop_words:
                        continue
                    
                    phrase = ' '.join(phrase_words)
                    
                    # Filter out phrases that are too short or too long
                    if len(phrase) < 8 or len(phrase) > 80:
                        continue
                    
                    # Skip phrases that are just common word combinations
                    if phrase in ['i have been', 'i tried to', 'i can not', 'i will be', 'i want to', 'i need to']:
                        continue
                    
                    # Skip phrases that end with incomplete words
                    if len(phrase_words) > 3 and any(word in ['tried', 'can', 'will', 'have', 'had', 'was', 'were'] for word in phrase_words[:-1]):
                        continue
                    
                    # Skip incomplete phrases that start with common words
                    if phrase_words[0] in ['the', 'a', 'an', 'to', 'for', 'of', 'with', 'by']:
                        continue
                    
                    # Include phrases that contain meaningful words
                    meaningful_words = ['crashes', 'fails', 'works', 'broken', 'fixed', 'improved', 'worse', 'better', 
                                      'slow', 'fast', 'easy', 'hard', 'difficult', 'simple', 'complicated', 'confusing',
                                      'helpful', 'unhelpful', 'responsive', 'unresponsive', 'quick', 'slow', 'instant',
                                      'rejected', 'approved', 'declined', 'accepted', 'failed', 'successful', 'error',
                                      'problem', 'issue', 'bug', 'glitch', 'feature', 'function', 'service', 'support',
                                      'good', 'bad', 'great', 'terrible', 'excellent', 'awful', 'amazing', 'horrible',
                                      'smooth', 'rough', 'seamless', 'clunky', 'intuitive', 'confusing', 'clear', 'unclear',
                                      'app', 'bank', 'card', 'money', 'account', 'transfer', 'payment', 'login', 'user',
                                      'customer', 'interface', 'experience', 'process', 'verification', 'security',
                                      'virtual', 'online', 'mobile', 'digital', 'credit', 'debit', 'balance', 'fee',
                                      'interest', 'rate', 'limit', 'transaction', 'statement', 'history', 'reward',
                                      'cashback', 'bonus', 'promotion', 'offer', 'welcome', 'referral', 'id', 'hkid',
                                      'identity', 'photo', 'scan', 'document', 'verification', 'registration', 'setup']
                    
                    # Only include phrases that contain meaningful words
                    if any(meaningful_word in phrase for meaningful_word in meaningful_words):
                        if phrase in phrase_counts:
                            phrase_counts[phrase] += 1
                        else:
                            phrase_counts[phrase] = 1
    
    # Remove overlapping phrases but be more lenient
    final_phrases = {}
    phrases_list = sorted(phrase_counts.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)
    
    for phrase, count in phrases_list:
        # Check if this phrase is too similar to existing ones (more lenient)
        is_similar = False
        for existing_phrase in final_phrases.keys():
            # Check for exact substring matches
            if phrase in existing_phrase and phrase != existing_phrase:
                if final_phrases[existing_phrase] >= count * 1.5:  # Only skip if existing is 50% more frequent
                    is_similar = True
                    break
            
            # Check for high word overlap (more than 80% words in common)
            phrase_words = set(phrase.split())
            existing_words = set(existing_phrase.split())
            if len(phrase_words) > 0 and len(existing_words) > 0:
                overlap = len(phrase_words.intersection(existing_words))
                total_unique = len(phrase_words.union(existing_words))
                if overlap / total_unique > 0.8:
                    if final_phrases[existing_phrase] >= count * 1.5:  # Only skip if existing is 50% more frequent
                        is_similar = True
                        break
        
        if not is_similar:
            final_phrases[phrase] = count
    
    return final_phrases

def merge_similar_phrases(phrase_counts: dict) -> dict:
    """Merge similar phrases to get better frequency counts and reduce repetition"""
    if not phrase_counts:
        return {}
    
    # Create a copy to work with
    merged_phrases = phrase_counts.copy()
    
    # Define similar phrase patterns with more comprehensive coverage
    similar_patterns = [
        # ID card and scanning variations - more comprehensive
        (['id card', 'the id', 'hkid', 'hk id', 'identity card', 'identity verification', 'photo verification', 'my id card', 'invalid hk id', 'hkid is invalid', 'id photo taking'], 'id card verification'),
        (['stuck in scanning', 'stuck at id', 'scanning the id', 'scan the id', 'scan my id', 'id scanning', 'scanning failed', 'scanning the id card', 'stuck in scanning the id card', 'stuck in scanning the', 'in scanning the id card', 'in scanning the id', 'stuck in scanning the', 'scan the id card', 'scan my id card', 'unable to scan', 'past the hkid'], 'stuck in scanning id'),
        
        # Account variations - more comprehensive
        (['the account', 'my account', 'an account', 'account opening', 'open account', 'account setup', 'open an account', 'open the account', 'cannot open account', 'still cannot open account', 'close the account', 'closing my account', 'my account still', 'account and this', 'in my account'], 'account opening'),
        (['account balance', 'balance check', 'check balance', 'account balance check'], 'account balance'),
        
        # App variations - more comprehensive
        (['the app', 'this app', 'app crashes', 'app not working', 'app keeps crashing', 'use the app', 'open the app', 'close the app', 'reinstall the app', 'need to reinstall the app', 'cannot open the app', 'but the app', 'use this app', 'it only works'], 'app crashes'),
        (['app works', 'app working', 'app functions well', 'app works perfectly'], 'app works'),
        
        # Customer service variations - more comprehensive
        (['customer service', 'customer support', 'support team', 'help desk', 'customer care', 'service team', 'their customer service', 'customer service and'], 'customer service'),
        
        # Login variations - more comprehensive
        (['login problems', 'login issues', 'cannot login', 'login failed', 'login error', 'login always fails', 'log in', 'me to login', 'me to login again'], 'login problems'),
        (['cannot access', 'unable to access', 'access denied', 'access problems', 'cannot access account'], 'cannot access'),
        
        # Transfer variations
        (['transfer money', 'money transfer', 'send money', 'money sending', 'transfer money failed'], 'transfer money'),
        (['payment failed', 'transaction failed', 'payment error', 'transaction error', 'payment processing error'], 'payment failed'),
        
        # Service quality variations
        (['poor service', 'bad service', 'terrible service', 'awful service', 'customer service unhelpful'], 'poor service'),
        (['good service', 'excellent service', 'great service', 'amazing service', 'customer service helpful'], 'good service'),
        
        # Experience variations
        (['bad experience', 'terrible experience', 'awful experience', 'horrible experience'], 'bad experience'),
        (['good experience', 'excellent experience', 'great experience', 'amazing experience'], 'good experience'),
        
        # User interface variations - more comprehensive
        (['user friendly', 'easy to use', 'simple to use', 'intuitive interface', 'easy to navigate', 'very user friendly', 'is quite user friendly', 'quite user friendly'], 'user friendly'),
        (['not user friendly', 'difficult to use', 'hard to use', 'confusing interface', 'interface too confusing'], 'not user friendly'),
        
        # Speed variations
        (['very slow', 'too slow', 'extremely slow', 'painfully slow', 'app loads very slowly'], 'very slow'),
        (['very fast', 'super fast', 'extremely fast', 'lightning fast', 'app loads quickly'], 'very fast'),
        
        # Working variations - more comprehensive
        (['not working', 'doesn\'t work', 'does not work', 'stopped working', 'app is not working'], 'not working'),
        (['works well', 'work well', 'working well', 'functions well'], 'works well'),
        
        # Error variations - more comprehensive
        (['system error', 'error message', 'error occurred', 'technical error', 'no error message', 'error message communication'], 'system error'),
        
        # Card variations - more comprehensive
        (['credit card', 'credit cards', 'card payment'], 'credit card'),
        (['debit card', 'debit cards', 'bank card', 'my debit card', 'got my debit card'], 'debit card'),
        
        # Banking variations
        (['online banking', 'mobile banking', 'digital banking', 'internet banking'], 'online banking'),
        (['virtual bank', 'virtual banking', 'digital bank', 'best virtual bank', 'virtual bank with'], 'virtual bank'),
        
        # Fee variations
        (['monthly fee', 'monthly fees', 'monthly charge'], 'monthly fee'),
        (['transaction fee', 'transfer fee', 'payment fee'], 'transaction fee'),
        
        # Verification variations
        (['verification process', 'identity verification', 'security verification', 'verification process stuck'], 'verification process'),
        (['Face ID', 'Face ID verification', 'Face ID not working', 'Face ID works perfectly'], 'Face ID'),
        
        # Money variations
        (['my money', 'back my money', 'funds for days'], 'money issues'),
        
        # Sign up variations
        (['sign up', 'sign up experience', 'registration', 'account setup'], 'sign up experience'),
        
        # Failed variations
        (['still failed to', 'failed to', 'cannot', 'unable to'], 'failed to'),
        
        # Common expressions
        (['so far', 'so good', 'so far so good'], 'so far so good'),
        (['too much', 'too expensive', 'too costly'], 'too much'),
        (['not enough', 'insufficient', 'lack of'], 'not enough'),
        
        # Positive banking features
        (['easy to set up', 'account setup easy', 'registration completed quickly'], 'easy to set up'),
        (['transfer money instantly', 'payment processed quickly', 'transaction successful'], 'transfer money instantly'),
        (['app never crashes', 'app works perfectly', 'app loads quickly'], 'app works perfectly'),
        (['verification completed fast', 'verification process smooth'], 'verification process smooth'),
        (['no hidden fees', 'competitive interest rate', 'generous credit limit'], 'no hidden fees'),
        
        # Negative banking issues
        (['app keeps crashing', 'app crashes every time', 'app crashes on startup'], 'app keeps crashing'),
        (['verification takes too long', 'verification process stuck'], 'verification process stuck'),
        (['customer service unhelpful', 'support team unresponsive', 'no response from support'], 'customer service unhelpful'),
        (['app freezes constantly', 'app loads very slowly'], 'app freezes constantly'),
        (['hidden fees', 'unexpected charges', 'overdraft fee charged'], 'hidden fees'),
        
        # Incomplete phrases to remove
        (['to open', 'open an', 'the app', 'the account', 'the id', 'an account', 'my account'], None),  # These will be removed
        (['need to', 'to use', 'i can\'t', 'i cannot', 'unable to'], None),  # These will be removed
    ]
    
    # Merge similar phrases
    for similar_list, target_phrase in similar_patterns:
        if target_phrase is None:
            # Remove these incomplete phrases
            for phrase in similar_list:
                if phrase in merged_phrases:
                    del merged_phrases[phrase]
        else:
            total_count = 0
            phrases_to_remove = []
            
            for phrase in similar_list:
                if phrase in merged_phrases:
                    total_count += merged_phrases[phrase]
                    phrases_to_remove.append(phrase)
            
            if total_count > 0:
                # Add the target phrase with combined count
                merged_phrases[target_phrase] = total_count
                
                # Remove the individual phrases
                for phrase in phrases_to_remove:
                    if phrase != target_phrase:
                        del merged_phrases[phrase]
    
    return merged_phrases

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

def output_phrase_frequencies(df: pd.DataFrame, app_key: str, output_dir: str):
    """Output phrase frequencies from translated content"""
    # Get positive and negative reviews
    positive_reviews = df[df['overall_sentiment'] == 'positive']['translated_content'].dropna()
    negative_reviews = df[df['overall_sentiment'] == 'negative']['translated_content'].dropna()
    
    # Process positive reviews
    if not positive_reviews.empty:
        positive_text = ' '.join(positive_reviews.astype(str))
        positive_phrases = extract_phrases_from_text(positive_text)
        positive_phrases = merge_similar_phrases(positive_phrases)
        
        # Get top 30 phrases by frequency
        top_positive = sorted(positive_phrases.items(), key=lambda x: x[1], reverse=True)[:30]
        
        positive_file = os.path.join(output_dir, f"{app_key}_positive_phrases.txt")
        with open(positive_file, 'w', encoding='utf-8') as f:
            f.write(f"Top 30 Most Common Phrases in Positive Reviews for {app_key}\n")
            f.write("=" * 60 + "\n\n")
            for i, (phrase, count) in enumerate(top_positive, 1):
                f.write(f"{i:2d}. {phrase:<30} (frequency: {count})\n")
        print(f"✅ Positive phrases saved: {positive_file}")
        print("📝 Top 10 positive phrases:")
        for i, (phrase, count) in enumerate(top_positive[:10], 1):
            print(f"   {i:2d}. {phrase:<25} (frequency: {count})")
    
    # Process negative reviews
    if not negative_reviews.empty:
        negative_text = ' '.join(negative_reviews.astype(str))
        negative_phrases = extract_phrases_from_text(negative_text)
        negative_phrases = merge_similar_phrases(negative_phrases)
        
        # Get top 30 phrases by frequency
        top_negative = sorted(negative_phrases.items(), key=lambda x: x[1], reverse=True)[:30]
        
        negative_file = os.path.join(output_dir, f"{app_key}_negative_phrases.txt")
        with open(negative_file, 'w', encoding='utf-8') as f:
            f.write(f"Top 30 Most Common Phrases in Negative Reviews for {app_key}\n")
            f.write("=" * 60 + "\n\n")
            for i, (phrase, count) in enumerate(top_negative, 1):
                f.write(f"{i:2d}. {phrase:<30} (frequency: {count})\n")
        print(f"✅ Negative phrases saved: {negative_file}")
        print("📝 Top 10 negative phrases:")
        for i, (phrase, count) in enumerate(top_negative[:10], 1):
            print(f"   {i:2d}. {phrase:<25} (frequency: {count})")

def create_custom_wordcloud(text: str, color_func, output_file: str, title: str = ""):
    """Create wordcloud with phrases using spaces, no underscores"""
    
    # Extract phrases directly from text
    phrase_counts = extract_phrases_from_text(text)
    
    # Filter out very low frequency phrases and very long phrases
    filtered_phrases = {}
    for phrase, count in phrase_counts.items():
        if count >= 1 and len(phrase) <= 30:  # Only phrases that appear at least once and aren't too long
            filtered_phrases[phrase] = count
    
    print(f"📝 Found phrases: {list(filtered_phrases.keys())[:10]}")
    
    # Create wordcloud with the phrases
    wordcloud = WordCloud(
        width=800, height=800,
        background_color='white',
        color_func=color_func,
        max_words=100,
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
    
    # Save the wordcloud
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Wordcloud saved: {output_file}")

def create_wordclouds(df: pd.DataFrame, app_key: str, output_dir: str, config: dict):
    """Create word clouds for positive and negative reviews with 2-4 word phrases"""
    try:
        # Set font to Arial
        plt.rcParams['font.family'] = 'Arial'
        
        # Positive reviews word cloud
        positive_reviews = df[df['overall_sentiment'] == 'positive']['translated_content'].dropna()
        if not positive_reviews.empty:
            positive_text = ' '.join(positive_reviews.astype(str))
            
            # Use our improved phrase extraction instead of DeepSeek
            print(f"📝 Using improved phrase extraction for {app_key} positive reviews")
            wordcloud_text = positive_text
            
            # Create positive wordcloud with Pacific Blue color (0, 184, 245)
            positive_wordcloud_file = os.path.join(output_dir, f"{app_key}_positive_wordcloud.png")
            create_custom_wordcloud(
                wordcloud_text,
                lambda *args, **kwargs: (0, 184, 245),  # Pacific Blue
                positive_wordcloud_file,
                f"{app_key} Positive Reviews"
            )
        
        # Negative reviews word cloud
        negative_reviews = df[df['overall_sentiment'] == 'negative']['translated_content'].dropna()
        if not negative_reviews.empty:
            negative_text = ' '.join(negative_reviews.astype(str))
            
            # Use our improved phrase extraction instead of DeepSeek
            print(f"📝 Using improved phrase extraction for {app_key} negative reviews")
            wordcloud_text = negative_text
            
            # Create negative wordcloud with Dark Blue color (12, 35, 60)
            negative_wordcloud_file = os.path.join(output_dir, f"{app_key}_negative_wordcloud.png")
            create_custom_wordcloud(
                wordcloud_text,
                lambda *args, **kwargs: (12, 35, 60),  # Dark Blue
                negative_wordcloud_file,
                f"{app_key} Negative Reviews"
            )
            
    except Exception as e:
        print(f"❌ Error creating wordclouds for {app_key}: {e}")
        import traceback
        traceback.print_exc()

def create_analysis_charts(df: pd.DataFrame, app_key: str, output_dir: str):
    """Create analysis charts with configured colors"""
    try:
        # Get colors from config
        colors = {
            "chart_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "sentiment_colors": {
                "positive": "#2ca02c",
                "negative": "#d62728", 
                "neutral": "#1f77b4"
            }
        }
        
        # Set style
        plt.style.use('default')
        
        # Create sentiment distribution chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall sentiment distribution
        sentiment_counts = df['overall_sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       colors=[colors["sentiment_colors"].get(cat, '#999999') for cat in sentiment_counts.index],
                       autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        axes[0, 1].bar(rating_counts.index, rating_counts.values, 
                       color=colors["chart_palette"][0])
        axes[0, 1].set_title('Rating Distribution')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Count')
        
        # Problem category distribution
        category_counts = df['problem_category'].value_counts()
        axes[1, 0].barh(category_counts.index, category_counts.values,
                        color=colors["chart_palette"][0])
        axes[1, 0].set_title('Problem Categories')
        axes[1, 0].set_xlabel('Count')
        
        # Sentiment over time (if date available)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Remove timezone info to avoid warning
            df['date'] = df['date'].dt.tz_localize(None)
            monthly_sentiment = df.groupby([df['date'].dt.to_period('M'), 'overall_sentiment']).size().unstack(fill_value=0)
            monthly_sentiment.plot(kind='area', ax=axes[1, 1], 
                                 color=[colors["sentiment_colors"].get(col, '#999999') for col in monthly_sentiment.columns])
            axes[1, 1].set_title('Sentiment Trends Over Time')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Review Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'Date data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Sentiment Trends Over Time')
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f"{app_key}_analysis_charts.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not generate charts: {e}")

def generate_summary_report(df: pd.DataFrame, app_key: str) -> dict:
    """Generate comprehensive summary report"""
    total_reviews = len(df)
    
    summary = {
        "app_key": app_key,
        "generated_at": datetime.now().isoformat(),
        "total_reviews": total_reviews,
        "overall_sentiment_distribution": df['overall_sentiment'].value_counts().to_dict(),
        "rating_distribution": df['rating'].value_counts().sort_index().to_dict(),
        "problem_category_distribution": df['problem_category'].value_counts().to_dict(),
        "average_rating": float(df['rating'].mean()) if 'rating' in df.columns else 0,
        "average_sentiment_score": float(df['sentiment_score'].mean()),
        "platform_distribution": df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        "top_positive_words": df['positive_words'].str.split(', ').explode().value_counts().head(10).to_dict(),
        "top_negative_words": df['negative_words'].str.split(', ').explode().value_counts().head(10).to_dict(),
    }
    
    return summary

def main():
    # Load config
    with open('config/settings.json', 'r') as f:
        config = json.load(f)
    
    # Read existing reviews file
    file_path = "output/welab_bank_reviews.xlsx"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📊 Reading data from: {file_path}")
    df = pd.read_excel(file_path)
    print(f"📊 Processing {len(df)} reviews...")
    
    # Create analysis directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = f"output/analysis_{timestamp}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Initialize new columns for analysis
    df['translated_content'] = ""
    df['sentiment_score'] = 0.0
    df['positive_words'] = ""
    df['negative_words'] = ""
    df['problem_category'] = ""
    df['overall_sentiment'] = ""
    
    # Process reviews with multi-workers (8 workers)
    print("🔄 Processing reviews with 8 workers...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Prepare arguments for workers
        args_list = [(idx, row, config) for idx, row in df.iterrows()]
        
        # Submit all tasks
        future_to_idx = {executor.submit(process_review_worker, args): args[0] for args in args_list}
        
        # Process results with progress bar
        results = {}
        for future in tqdm(as_completed(future_to_idx), total=len(args_list), desc="Processing reviews"):
            result = future.result()
            results[result['idx']] = result
            
            # Update DataFrame with results
            idx = result['idx']
            df.at[idx, 'translated_content'] = result['translated_content']
            df.at[idx, 'sentiment_score'] = result['sentiment_score']
            df.at[idx, 'positive_words'] = result['positive_words']
            df.at[idx, 'negative_words'] = result['negative_words']
            df.at[idx, 'problem_category'] = result['problem_category']
            df.at[idx, 'overall_sentiment'] = result['overall_sentiment']
            
            if not result['success']:
                print(f"  Error processing review {idx}: {result.get('error', 'Unknown error')}")
    
    # Save analyzed data
    analyzed_file = os.path.join(analysis_dir, f"welab_bank_analyzed.xlsx")
    df.to_excel(analyzed_file, index=False)
    
    # Generate visualizations
    print("🔄 Creating analysis charts...")
    create_analysis_charts(df, 'welab_bank', analysis_dir)
    
    # Create wordclouds
    print("🔄 Creating wordclouds...")
    create_wordclouds(df, 'welab_bank', analysis_dir, config)
    
    # Generate summary report
    print("🔄 Generating summary report...")
    summary = generate_summary_report(df, 'welab_bank')
    summary_file = os.path.join(analysis_dir, f"welab_bank_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Output phrase frequencies
    print("🔄 Outputting phrase frequencies...")
    output_phrase_frequencies(df, 'welab_bank', analysis_dir)

    print(f"✅ Full analysis completed! Results saved to: {analysis_dir}")
    print(f"📊 Sentiment distribution: {summary['overall_sentiment_distribution']}")
    print(f"📊 Average rating: {summary['average_rating']:.2f}")
    print(f"📊 Average sentiment score: {summary['average_sentiment_score']:.3f}")

if __name__ == "__main__":
    main() 