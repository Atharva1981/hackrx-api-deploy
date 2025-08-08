#!/usr/bin/env python3
"""
Check OpenRouter configuration and account status
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_openrouter_account():
    """Check OpenRouter account status and configuration"""
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    print("🔍 Checking OpenRouter Account Configuration...")
    print("=" * 50)
    
    if not openrouter_key:
        print("❌ OPENROUTER_API_KEY not found")
        return False
    
    print(f"✅ API Key found: {openrouter_key[:20]}...")
    
    # Check account status
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json"
    }
    
    try:
        print("🔄 Checking account status...")
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers=headers,
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            account_info = response.json()
            print("✅ Account is active!")
            print(f"📋 Account Info:")
            print(f"   - User ID: {account_info.get('user_id', 'N/A')}")
            print(f"   - Credits: {account_info.get('credits', 'N/A')}")
            print(f"   - Plan: {account_info.get('plan', 'N/A')}")
            return True
        else:
            print(f"❌ Account check failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking account: {e}")
        return False

def check_available_models():
    """Check which models are available"""
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    print("\n🔍 Checking Available Models...")
    print("=" * 30)
    
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            mistral_found = False
            
            print("📋 Available Models:")
            for model in models.get('data', []):
                model_id = model.get('id', '')
                if 'mistral' in model_id.lower():
                    print(f"   ✅ {model_id}")
                    mistral_found = True
                elif 'gpt' in model_id.lower() or 'claude' in model_id.lower():
                    print(f"   📝 {model_id}")
            
            if not mistral_found:
                print("   ⚠️  Mistral models not found in available models")
            
            return mistral_found
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def test_mistral_specific():
    """Test Mistral model specifically"""
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    print("\n🔍 Testing Mistral Model Access...")
    print("=" * 35)
    
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
                "content": "Test message - please respond with 'Working'"
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
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"✅ Mistral is working! Response: {answer}")
                return True
            else:
                print("❌ No response from Mistral")
                return False
        else:
            print(f"❌ Mistral test failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Mistral: {e}")
        return False

if __name__ == "__main__":
    print("🚀 OpenRouter Configuration Check")
    print("=" * 50)
    
    # Check account status
    account_ok = check_openrouter_account()
    
    if account_ok:
        # Check available models
        models_ok = check_available_models()
        
        # Test Mistral specifically
        mistral_ok = test_mistral_specific()
        
        if mistral_ok:
            print("\n🎉 All checks passed!")
            print("✅ OpenRouter account is active")
            print("✅ Mistral model is accessible")
            print("✅ Your API should work correctly")
        else:
            print("\n⚠️  Mistral model test failed")
            print("Check if Mistral is available in your plan")
    else:
        print("\n❌ Account check failed")
        print("Verify your OpenRouter API key and account status")
    
    print("\n" + "=" * 50)
