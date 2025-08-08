#!/usr/bin/env python3
"""
Test script to verify OpenAI API key and GPT model access
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test OpenAI API connection"""
    
    # Get API key from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print("🔍 Testing OpenAI API Connection...")
    print("=" * 50)
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API Key found: {openai_key[:20]}...")
    
    # Test API call
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
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
        print("🔄 Making test API call to OpenAI...")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"✅ Success! GPT response: {answer}")
                return True
            else:
                print("❌ No response content from GPT")
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
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"HACKRX_API_KEY: {'✅ Set' if hackrx_key else '❌ Not set'}")
    print(f"OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")
    
    if openai_key:
        print(f"OpenAI Key preview: {openai_key[:20]}...")
    
    return bool(openai_key)

if __name__ == "__main__":
    print("🚀 OpenAI API Test")
    print("=" * 50)
    
    # Test environment setup
    env_ok = test_environment_setup()
    
    if env_ok:
        # Test API connection
        api_ok = test_openai_connection()
        
        if api_ok:
            print("\n🎉 All tests passed! OpenAI API is working correctly.")
            print("✅ GPT model is accessible")
            print("✅ Your API key is valid")
            print("✅ Your HackRx API will work with OpenAI as fallback")
        else:
            print("\n❌ API test failed. Check your OpenAI API key.")
    else:
        print("\n❌ Environment setup failed. Check your .env file.")
    
    print("\n" + "=" * 50)
