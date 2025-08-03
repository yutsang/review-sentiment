#!/usr/bin/env python3

import json
import requests
from typing import Dict

def call_deepseek_api(text: str, config: Dict) -> str:
    """Test DeepSeek API phrase extraction"""
    try:
        deepseek_config = config.get('deepseek', {})
        
        api_key = deepseek_config.get('api_key')
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
            print("⚠️ DeepSeek API key not configured.")
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
            print(f"🤖 DeepSeek Response: {content}")
            return content
        else:
            print(f"⚠️ DeepSeek API error: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        print(f"⚠️ DeepSeek API call failed: {e}")
        return ""

# Load config
with open('config/settings.json', 'r') as f:
    config = json.load(f)

# Test text
test_text = """
The app crashes frequently and is very slow. Customer service is terrible. 
Account opening process takes too long. Login problems every day. 
Transfer money feature doesn't work properly. App interface is confusing.
"""

print("Testing phrase extraction...")
print(f"Input text: {test_text}")
print("-" * 50)

result = call_deepseek_api(test_text, config)
print(f"Result: {result}") 