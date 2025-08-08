# 🚀 HackRx API Deployment Guide

## 📋 Essential Files for Webhook URL

Only these files are needed for your Railway deployment:

### ✅ **Core Files (Required):**
```
hackrx_faiss_api.py      # Main API application
requirements.txt         # Python dependencies
railway.json            # Railway deployment config
Procfile               # Railway process config
runtime.txt            # Python runtime version
.gitignore             # Git ignore rules
README.md              # Project documentation
```

### ✅ **Optional but Recommended:**
```
test_webhook.py         # Webhook testing script
```

## 🗑️ **Files to Exclude (Development/Testing):**
```
speed_test.py
fast_embedding_alternatives.py
embedding_comparison.py
dynamic_insurance_terms.py
insurance_coverage_test.py
performance_analysis.py
test_api.py
check_api_keys.py
test_api_key.py
```

## 🔧 **GitHub Commit Steps:**

### **1. Initialize Git (if not already done):**
```bash
git init
git add .
```

### **2. Commit Only Essential Files:**
```bash
# Add only the essential files
git add hackrx_faiss_api.py
git add requirements.txt
git add railway.json
git add Procfile
git add runtime.txt
git add .gitignore
git add README.md
git add test_webhook.py

# Commit
git commit -m "🚀 Deploy HackRx API with ultra-fast keyword hashing and Gemini integration"
```

### **3. Push to GitHub:**
```bash
git remote add origin https://github.com/yourusername/hackrx-api.git
git branch -M main
git push -u origin main
```

## 🌐 **Railway Deployment:**

### **1. Connect Railway to GitHub:**
1. Go to [railway.app](https://railway.app)
2. Create new project
3. Select "Deploy from GitHub repo"
4. Choose your repository

### **2. Set Environment Variables:**
In Railway dashboard, add these variables:
```
HACKRX_API_KEY=8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21
GEMINI_API_KEY=your-gemini-api-key-here
```

### **3. Deploy:**
Railway will automatically deploy your API.

## 🎯 **Get Your Webhook URL:**

After deployment, your webhook URL will be:
```
https://your-railway-app-name.railway.app/hackrx/run
```

## 🧪 **Test Your Webhook:**

1. Update the URL in `test_webhook.py`
2. Run: `python test_webhook.py`

## 📊 **What You Get:**

- **⚡ Ultra-fast search** (keyword hashing)
- **🤖 Smart answers** (Gemini AI)
- **🔒 Secure** (Bearer authentication)
- **📈 Scalable** (Railway infrastructure)
- **🎯 Accurate** (insurance-specific)

## ✅ **Ready for Evaluation:**

Your webhook will be ready for:
- **Platform evaluation**
- **Performance testing**
- **Accuracy assessment**
- **Real-world usage**

---

**🎉 Your HackRx API is now ready for deployment!** 