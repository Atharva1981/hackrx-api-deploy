#!/usr/bin/env python3
"""
Test script to verify OpenRouter API key and Mistral model access
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openrouter_connection():
    """Test OpenRouter API connection"""
    
    # Get API key from environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    print("🔍 Testing OpenRouter API Connection...")
    print("=" * 50)
    
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API Key found: {openrouter_key[:20]}...")
    
    # Test API call
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hackrx-api.railway.app",
        "X-Title": "HackRx Insurance API"
    }
    
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you confirm you're working? Please respond with 'Yes, I am working correctly.'"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    try:
        print("🔄 Making test API call to OpenRouter...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"✅ Success! Mistral response: {answer}")
                return True
            else:
                print("❌ No response content from Mistral")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Error details: {response.text}")
            
            if response.status_code == 401:
                print("🔑 Authentication failed - check your API key")
            elif response.status_code == 429:
                print("⏰ Rate limit exceeded")
            elif response.status_code == 400:
                print("📝 Bad request - check payload format")
            
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"🌐 Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_environment_setup():
    """Test environment variable setup"""
    print("\n🔧 Testing Environment Setup...")
    print("=" * 30)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found")
    
    # Check environment variables
    hackrx_key = os.getenv("HACKRX_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    print(f"HACKRX_API_KEY: {'✅ Set' if hackrx_key else '❌ Not set'}")
    print(f"OPENROUTER_API_KEY: {'✅ Set' if openrouter_key else '❌ Not set'}")
    
    if openrouter_key:
        print(f"OpenRouter Key preview: {openrouter_key[:20]}...")
    
    return bool(openrouter_key)

if __name__ == "__main__":
    print("🚀 OpenRouter API Test")
    print("=" * 50)
    
    # Test environment setup
    env_ok = test_environment_setup()
    
    if env_ok:
        # Test API connection
        api_ok = test_openrouter_connection()
        
        if api_ok:
            print("\n🎉 All tests passed! OpenRouter API is working correctly.")
            print("✅ Mistral model is accessible")
            print("✅ Your API key is valid")
        else:
            print("\n❌ API test failed. Check your OpenRouter API key.")
    else:
        print("\n❌ Environment setup failed. Check your .env file.")
    
    print("\n" + "=" * 50)
