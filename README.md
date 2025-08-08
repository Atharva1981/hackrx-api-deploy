# HackRx API - Insurance Policy AI Query System

A FastAPI-based web service that accepts PDF URLs and questions, processes insurance policy documents using semantic search with FAISS, and generates accurate answers using Mistral 7B Instruct via OpenRouter.

## 🚀 Features

- **PDF Processing**: Downloads and extracts text from PDF documents
- **Ultra-Fast Search**: Uses FAISS with keyword-based hashing for lightning-fast retrieval
- **AI-Powered Answers**: Leverages Mistral 7B Instruct for intelligent responses
- **Authentication**: Bearer token-based API security
- **Production Ready**: HTTPS-enabled, scalable architecture
- **Robust Error Handling**: Comprehensive error handling and fallback mechanisms

## 📋 API Endpoints

### POST `/hackrx/run`
Main endpoint for processing insurance policy queries.

**Headers:**
```
Authorization: Bearer your-secret-api-key-here
Content-Type: application/json
Accept: application/json
```

**Request Body:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period is thirty days...",
    "The waiting period is thirty-six months..."
  ]
}
```

### GET `/health`
Health check endpoint.

### GET `/docs`
Interactive API documentation (Swagger UI).

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- Virtual environment

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd hackrx-api

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file:
```env
HACKRX_API_KEY=your-secret-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key
```

### Run Locally
```bash
uvicorn hackrx_faiss_api:app --host 127.0.0.1 --port 8000
```

### Test the API
```bash
python test_webhook.py
```

## 🚀 Deployment

### Railway (Recommended)
1. Push code to GitHub
2. Connect Railway to your GitHub repo
3. Set environment variables in Railway dashboard:
   - `HACKRX_API_KEY`: Your secret API key
   - `OPENROUTER_API_KEY`: Your OpenRouter API key

### Environment Variables for Railway
```
HACKRX_API_KEY=8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21
OPENROUTER_API_KEY=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
```

## 📦 Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **PyMuPDF**: PDF text extraction
- **FAISS**: Vector similarity search
- **OpenRouter**: Mistral 7B Instruct integration
- **Requests**: HTTP client
- **NumPy**: Numerical operations

## 🔐 Security

- Bearer token authentication
- Environment variable configuration
- Input validation and sanitization
- Rate limiting support

## 📊 Performance

- **Ultra-fast search**: Keyword-based hashing for instant retrieval
- **Response time**: < 10 seconds for most queries
- **Supports multiple questions** per request
- **Efficient text chunking** and processing
- **Optimized vector search** with hybrid approach

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions, please open an issue on GitHub.

## 🎯 Webhook URL

After deployment, your webhook URL will be:
```
https://your-railway-app-name.railway.app/hackrx/run
```

## 🔧 Testing

Use the provided `test_webhook.py` script to test your deployment:
```bash
python test_webhook.py
```

Update the `RAILWAY_URL` variable in the script with your actual Railway URL.
