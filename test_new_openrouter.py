#!/usr/bin/env python3
"""
Quick test for new OpenRouter API key
"""

import requests

def test_new_key(api_key):
    """Test a new OpenRouter API key"""
    
    print(f"🔍 Testing API Key: {api_key[:20]}...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try free models in order
    free_models = [
        "meta-llama/llama-2-70b-chat",      # Free tier
        "google/palm-2-chat-bison",          # Free tier
        "microsoft/dialo-gpt-medium",        # Free tier
        "mistralai/mistral-7b-instruct"      # Paid but try anyway
    ]
    
    for model in free_models:
        print(f"🔄 Trying model: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'Working'"
                }
            ],
            "max_tokens": 20,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"✅ Success with {model}! Response: {answer}")
                return True
            else:
                print(f"❌ {model} failed: {response.text[:100]}...")
                continue
                
        except Exception as e:
            print(f"❌ {model} error: {e}")
            continue
    
    return False
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            print(f"✅ Success! Response: {answer}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 OpenRouter API Key Test")
    print("=" * 40)
    
    # Get API key from user
    api_key = input("Enter your new OpenRouter API key: ").strip()
    
    if api_key:
        success = test_new_key(api_key)
        if success:
            print("\n🎉 API key is working!")
            print("✅ You can now use it in your HackRx API")
        else:
            print("\n❌ API key test failed")
            print("Check your OpenRouter account and credits")
    else:
        print("❌ No API key provided")
    
    print("\n" + "=" * 40)
