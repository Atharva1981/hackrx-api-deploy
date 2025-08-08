#!/usr/bin/env python3
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_webhook():
    """Test the webhook URL for your Railway deployment"""
    
    # Configuration - UPDATE THIS WITH YOUR RAILWAY URL
    RAILWAY_URL = "https://your-railway-app-name.railway.app"  # Replace with your actual Railway URL
    WEBHOOK_URL = f"{RAILWAY_URL}/hackrx/run"
    API_KEY = "8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21"
    
    print("🌐 Testing Your HackRx Webhook")
    print("=" * 50)
    print(f"Webhook URL: {WEBHOOK_URL}")
    print()
    
    # Test request
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    print("🔍 Testing Webhook Connection...")
    print("-" * 40)
    
    try:
        # Test 1: Health Check
        health_url = f"{RAILWAY_URL}/health"
        print(f"1. Health Check: {health_url}")
        
        health_response = requests.get(health_url, timeout=10)
        print(f"   Status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✅ Health: {health_data.get('status', 'unknown')}")
            print(f"   🤖 Mistral: {'✅ Connected' if health_data.get('mistral_connected') else '❌ Not Connected'}")
            print(f"   🔍 FAISS: {'✅ Connected' if health_data.get('faiss_connected') else '❌ Not Connected'}")
        else:
            print(f"   ❌ Health check failed: {health_response.text}")
        
        print()
        
        # Test 2: Main Webhook
        print(f"2. Main Webhook: {WEBHOOK_URL}")
        
        response = requests.post(WEBHOOK_URL, json=test_request, headers=headers, timeout=60)
        
        print(f"   Status: {response.status_code}")
        print(f"   Response Time: {response.elapsed.total_seconds():.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get('answers', [])
            
            print(f"   ✅ Success: {len(answers)} answers generated")
            print()
            print("📝 Generated Answers:")
            for i, answer in enumerate(answers):
                print(f"   Q{i+1}: {answer[:100]}...")
            
            print()
            print("🎉 Webhook is working perfectly!")
            
        elif response.status_code == 401:
            print("   ❌ Authentication failed - Check your API key")
        elif response.status_code == 404:
            print("   ❌ Endpoint not found - Check your Railway URL")
        elif response.status_code == 500:
            print("   ❌ Server error - Check your Railway deployment")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed - Check your Railway URL")
    except requests.exceptions.Timeout:
        print("   ❌ Request timeout - Server might be slow")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("📋 Webhook Configuration Summary:")
    print("-" * 40)
    print(f"URL: {WEBHOOK_URL}")
    print(f"Method: POST")
    print(f"Headers: Authorization: Bearer {API_KEY[:10]}...")
    print(f"Content-Type: application/json")
    print()
    print("🔧 To use this webhook:")
    print("1. Copy the URL above")
    print("2. Use POST method")
    print("3. Include the Authorization header")
    print("4. Send JSON with 'documents' and 'questions'")
    print()
    print("✅ Your webhook is ready for evaluation!")

def show_webhook_examples():
    """Show examples of how to use the webhook"""
    
    print("📚 Webhook Usage Examples")
    print("=" * 50)
    
    # Example 1: cURL
    print("1. cURL Example:")
    print("-" * 20)
    print("""curl -X POST "https://your-railway-app-name.railway.app/hackrx/run" \\
  -H "Authorization: Bearer 8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21" \\
  -H "Content-Type: application/json" \\
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'""")
    
    print()
    
    # Example 2: Python
    print("2. Python Example:")
    print("-" * 20)
    print("""import requests

url = "https://your-railway-app-name.railway.app/hackrx/run"
headers = {
    "Authorization": "Bearer 8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result)""")
    
    print()
    
    # Example 3: JavaScript
    print("3. JavaScript Example:")
    print("-" * 20)
    print("""fetch("https://your-railway-app-name.railway.app/hackrx/run", {
  method: "POST",
  headers: {
    "Authorization": "Bearer 8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    documents: "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    questions: [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  })
})
.then(response => response.json())
.then(data => console.log(data));""")

if __name__ == "__main__":
    test_webhook()
    print("\n" + "="*50 + "\n")
    show_webhook_examples() 