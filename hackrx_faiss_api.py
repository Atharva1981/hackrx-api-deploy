#!/usr/bin/env python3
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import os
import uuid
import requests
from datetime import datetime
from pathlib import Path
import tempfile
import time
import numpy as np
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import all components
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv

# Set availability flags
PDF_AVAILABLE = True
AI_AVAILABLE = True

# Load environment variables
load_dotenv()

print("✅ All components loaded successfully!")

app = FastAPI(
    title="HackRx API with FAISS & Mistral",
    description="Insurance Policy AI Query System with PDF Processing, FAISS Vector Search, and Mistral 7B Integration",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Pydantic models
class QueryRequest(BaseModel):
    documents: str  # URL to PDF document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Configuration
API_KEY = os.getenv("HACKRX_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # New OpenAI option

# Validate required environment variables
if not API_KEY:
    print("⚠️  Warning: HACKRX_API_KEY not set. Using default for development.")
    API_KEY = "8b796ad826037b97ba28ae4cd36c4605bd9ed1464673ad5b0a3290a9867a9d21"

if not OPENROUTER_API_KEY:
    print("⚠️  Warning: OPENROUTER_API_KEY not set. AI features may not work.")
    OPENROUTER_API_KEY = "your-openrouter-api-key-here"

if not OPENAI_API_KEY:
    print("⚠️  Warning: OPENAI_API_KEY not set. OpenAI features may not work.")
    OPENAI_API_KEY = "your-openai-api-key-here"

# Global variables for AI components
faiss_index = None
document_chunks = []  # Store chunks in memory
document_embeddings = []  # Store embeddings in memory
document_cache = {}  # Cache for processed documents
embedding_cache = {}  # Cache for embeddings

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def initialize_ai_components():
    """Initialize FAISS and Mistral components"""
    global faiss_index
    
    if not AI_AVAILABLE:
        print("⚠️  AI components not available")
        return False
    
    try:
        # Initialize FAISS index with ultra-fast keyword hashing dimension
        dimension = 64  # Ultra-fast keyword hashing dimension
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Check if OpenRouter API key is provided
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip():
            print("✅ AI components initialized successfully!")
            return True
        else:
            print("⚠️  OPENROUTER_API_KEY not set. Mistral features will be disabled.")
            return False
        
    except Exception as e:
        print(f"❌ Error initializing AI components: {e}")
        return False

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and save to temporary file"""
    try:
        # Disable SSL verification for problematic URLs
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        
        # Clean up temporary file
        os.unlink(file_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        return text
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """Split text into semantically coherent chunks optimized for insurance documents"""
    if not text.strip():
        return []
    
    # Clean and normalize text
    text = text.replace('\r', '\n').replace('\t', ' ')
    lines = text.split('\n')
    
    chunks = []
    current_chunk = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new section (common in insurance documents)
        section_indicators = [
            'SECTION', 'CLAUSE', 'ARTICLE', 'PART', 'CHAPTER',
            'DEFINITIONS', 'EXCLUSIONS', 'COVERAGE', 'LIMITS',
            'CONDITIONS', 'ENDORSEMENTS', 'SCHEDULE', 'POLICY'
        ]
        
        is_section_start = any(indicator in line.upper() for indicator in section_indicators)
        
        # If adding this line would exceed chunk size, save current chunk
        if current_chunk and (len(current_chunk + line) > chunk_size or is_section_start):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks (likely headers or footers)
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate ultra-fast keyword-based embeddings for insurance documents"""
    if not texts:
        return []
    
    # Comprehensive insurance terminology for keyword hashing
    insurance_terms = {
        # Health Insurance
        'health': ['health', 'medical', 'hospital', 'doctor', 'treatment', 'surgery', 'medication', 'prescription'],
        'coverage': ['coverage', 'insured', 'policyholder', 'beneficiary', 'claim', 'premium', 'deductible'],
        'disease': ['disease', 'illness', 'condition', 'diagnosis', 'symptoms', 'chronic', 'acute'],
        'waiting': ['waiting', 'period', 'exclusion', 'pre-existing', 'grace', 'renewal'],
        
        # Life Insurance
        'life': ['life', 'death', 'mortality', 'survival', 'term', 'whole', 'universal'],
        'benefit': ['benefit', 'sum', 'assured', 'death', 'maturity', 'surrender'],
        
        # Motor Insurance
        'motor': ['motor', 'vehicle', 'car', 'auto', 'accident', 'collision', 'comprehensive'],
        'damage': ['damage', 'repair', 'replacement', 'liability', 'third', 'party'],
        
        # Property Insurance
        'property': ['property', 'building', 'house', 'fire', 'theft', 'burglary', 'natural'],
        'structure': ['structure', 'contents', 'belongings', 'furniture', 'appliances'],
        
        # Travel Insurance
        'travel': ['travel', 'trip', 'journey', 'overseas', 'international', 'domestic'],
        'emergency': ['emergency', 'evacuation', 'repatriation', 'medical', 'assistance'],
        
        # Liability Insurance
        'liability': ['liability', 'negligence', 'damages', 'compensation', 'legal'],
        'professional': ['professional', 'malpractice', 'errors', 'omissions'],
        
        # Marine Insurance
        'marine': ['marine', 'cargo', 'ship', 'vessel', 'freight', 'transit'],
        'shipping': ['shipping', 'transport', 'logistics', 'warehouse'],
        
        # Financial Insurance
        'financial': ['financial', 'credit', 'bond', 'guarantee', 'fidelity'],
        'investment': ['investment', 'fund', 'portfolio', 'market', 'risk'],
        
        # Policy Terms
        'policy': ['policy', 'contract', 'agreement', 'terms', 'conditions'],
        'exclusion': ['exclusion', 'limitation', 'restriction', 'exception'],
        'claim': ['claim', 'notification', 'settlement', 'investigation'],
        
        # Time-based Terms
        'time': ['time', 'period', 'duration', 'term', 'renewal', 'expiry'],
        'date': ['date', 'effective', 'commencement', 'termination'],
        
        # Medical Terms
        'medical': ['medical', 'hospitalization', 'surgery', 'diagnosis', 'treatment'],
        'medication': ['medication', 'drug', 'prescription', 'pharmacy'],
        
        # Coverage Types
        'coverage_type': ['individual', 'family', 'group', 'corporate', 'comprehensive'],
        'limit': ['limit', 'maximum', 'minimum', 'sub-limit', 'ceiling']
    }
    
    embeddings = []
    
    for text in texts:
        # Create a 64-dimensional vector based on keyword presence
        vector = [0.0] * 64
        
        text_lower = text.lower()
        
        # Hash keywords into vector positions
        for category, keywords in insurance_terms.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Simple hash function to map keyword to vector position
                    hash_val = hash(keyword) % 64
                    vector[hash_val] += 1.0
        
        # Normalize the vector
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x/magnitude for x in vector]
        
        embeddings.append(vector)
    
    return embeddings

def store_chunks_in_faiss(chunks: List[str], embeddings: List[List[float]], document_id: str):
    """Store document chunks and embeddings in FAISS index"""
    global faiss_index, document_chunks, document_embeddings
    
    if not chunks or not embeddings:
        return
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Add to FAISS index
    faiss_index.add(embeddings_array)
    
    # Store chunks and embeddings in memory
    start_idx = len(document_chunks)
    document_chunks.extend(chunks)
    document_embeddings.extend(embeddings)
    
    print(f"✅ Stored {len(chunks)} chunks for document {document_id}")

def search_relevant_chunks(question: str, top_k: int = 8) -> List[str]:
    """Search for relevant chunks using hybrid approach (FAISS + keyword matching)"""
    if not document_chunks or faiss_index is None:
        return []
    
    # Comprehensive insurance terms for keyword matching
    insurance_terms = {
        'health': ['health', 'medical', 'hospital', 'doctor', 'treatment', 'surgery', 'medication', 'prescription', 'diagnosis', 'symptoms'],
        'life': ['life', 'death', 'mortality', 'survival', 'term', 'whole', 'universal', 'benefit', 'sum assured'],
        'motor': ['motor', 'vehicle', 'car', 'auto', 'accident', 'collision', 'comprehensive', 'damage', 'repair'],
        'property': ['property', 'building', 'house', 'fire', 'theft', 'burglary', 'natural', 'disaster'],
        'travel': ['travel', 'trip', 'journey', 'overseas', 'international', 'domestic', 'emergency'],
        'liability': ['liability', 'negligence', 'damages', 'compensation', 'legal', 'professional'],
        'marine': ['marine', 'cargo', 'ship', 'vessel', 'freight', 'transit', 'shipping'],
        'financial': ['financial', 'credit', 'bond', 'guarantee', 'fidelity', 'investment'],
        'policy': ['policy', 'contract', 'agreement', 'terms', 'conditions', 'coverage'],
        'claim': ['claim', 'notification', 'settlement', 'investigation', 'benefit'],
        'exclusion': ['exclusion', 'limitation', 'restriction', 'exception', 'waiting', 'period'],
        'premium': ['premium', 'payment', 'grace', 'renewal', 'expiry', 'lapse'],
        'hospital': ['hospital', 'hospitalization', 'admission', 'discharge', 'room', 'icu'],
        'surgery': ['surgery', 'surgical', 'operation', 'procedure', 'anesthesia'],
        'medication': ['medication', 'drug', 'prescription', 'pharmacy', 'medicine'],
        'diagnosis': ['diagnosis', 'diagnostic', 'test', 'laboratory', 'pathology'],
        'treatment': ['treatment', 'therapy', 'rehabilitation', 'physiotherapy'],
        'emergency': ['emergency', 'urgent', 'critical', 'ambulance', 'evacuation'],
        'preventive': ['preventive', 'checkup', 'vaccination', 'screening', 'wellness'],
        'maternity': ['maternity', 'pregnancy', 'delivery', 'childbirth', 'antenatal'],
        'dental': ['dental', 'tooth', 'oral', 'dental surgery', 'orthodontics'],
        'ophthalmic': ['ophthalmic', 'eye', 'vision', 'glasses', 'contact lens', 'cataract'],
        'mental': ['mental', 'psychiatric', 'psychological', 'counseling', 'therapy'],
        'rehabilitation': ['rehabilitation', 'physiotherapy', 'occupational', 'speech therapy'],
        'prosthesis': ['prosthesis', 'artificial', 'limb', 'wheelchair', 'crutches'],
        'organ': ['organ', 'transplant', 'donor', 'recipient', 'tissue'],
        'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'yoga'],
        'alternative': ['alternative', 'complementary', 'traditional', 'herbal'],
        'daycare': ['daycare', 'day care', 'ambulatory', 'outpatient', 'same day'],
        'dialysis': ['dialysis', 'kidney', 'renal', 'hemodialysis', 'peritoneal'],
        'chemotherapy': ['chemotherapy', 'radiation', 'oncology', 'cancer', 'tumor'],
        'vaccination': ['vaccination', 'immunization', 'vaccine', 'inoculation'],
        'health_check': ['health check', 'checkup', 'screening', 'preventive', 'wellness'],
        'ncd': ['ncd', 'no claim', 'discount', 'bonus', 'loading'],
        'portability': ['portability', 'transfer', 'switch', 'migration'],
        'pre_existing': ['pre-existing', 'pre existing', 'existing', 'condition'],
        'grace_period': ['grace period', 'grace', 'payment', 'premium'],
        'waiting_period': ['waiting period', 'waiting', 'exclusion period'],
        'sub_limit': ['sub limit', 'sub-limit', 'limit', 'ceiling', 'maximum'],
        'room_rent': ['room rent', 'room', 'accommodation', 'boarding'],
        'icu': ['icu', 'intensive care', 'critical care', 'ccu'],
        'copay': ['copay', 'co-pay', 'co-payment', 'deductible', 'excess'],
        'network': ['network', 'provider', 'hospital', 'doctor', 'panel'],
        'cashless': ['cashless', 'cash less', 'direct settlement'],
        'reimbursement': ['reimbursement', 'reimburse', 'claim', 'settlement']
    }
    
    # Get question embedding
    question_embedding = get_embeddings([question])[0]
    question_embedding_array = np.array([question_embedding], dtype=np.float32)
    
    # FAISS search
    if faiss_index.ntotal > 0:
        scores, indices = faiss_index.search(question_embedding_array, min(top_k * 2, faiss_index.ntotal))
        faiss_results = [(document_chunks[i], scores[0][j]) for j, i in enumerate(indices[0]) if i < len(document_chunks)]
    else:
        faiss_results = []
    
    # Keyword matching
    question_lower = question.lower()
    keyword_matches = []
    
    for i, chunk in enumerate(document_chunks):
        chunk_lower = chunk.lower()
        score = 0
        
        for category, keywords in insurance_terms.items():
            for keyword in keywords:
                if keyword in question_lower and keyword in chunk_lower:
                    score += 2  # Higher weight for exact matches
                elif keyword in chunk_lower:
                    score += 1  # Lower weight for chunk relevance
        
        if score > 0:
            keyword_matches.append((chunk, score))
    
    # Combine and rank results
    all_results = {}
    
    # Add FAISS results
    for chunk, score in faiss_results:
        all_results[chunk] = all_results.get(chunk, 0) + score * 0.7  # Weight FAISS results
    
    # Add keyword results
    for chunk, score in keyword_matches:
        all_results[chunk] = all_results.get(chunk, 0) + score * 0.3  # Weight keyword results
    
    # Sort by score and return top results
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, score in sorted_results[:top_k]]
    
    return relevant_chunks

def generate_answer_with_openai(question: str, context: str) -> str:
    """Generate answer using OpenAI GPT models with optimized prompt for insurance accuracy"""
    if not AI_AVAILABLE or not OPENAI_API_KEY:
        return f"AI service not available. Please check if OPENAI_API_KEY is set in environment variables. Question: {question}"
    
    try:
        # Create an optimized prompt for insurance policy questions
        prompt = f"""You are an expert insurance policy analyst with deep knowledge of health insurance policies. Your task is to provide accurate, precise answers based solely on the provided policy document context.

POLICY DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based EXCLUSIVELY on the information provided in the policy document context above
2. If the specific information is not found in the context, respond with: "The provided policy document does not contain specific information about [exact topic]"
3. Be precise and include exact details, numbers, and terms from the policy when available
4. Use clear, professional language appropriate for insurance documentation
5. If you find relevant information, quote it accurately from the policy
6. Do not make assumptions or provide information not present in the context
7. If the context is insufficient, clearly state what specific information is missing
8. For numerical values (periods, amounts, percentages), be exact
9. For policy terms and conditions, be specific about requirements and limitations
10. Structure your answer logically with clear points

ANSWER:"""

        # OpenAI API call
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo-16k",  # Using 16k model which might have different pricing
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                # Ensure the answer is not too long but preserve important details
                if len(answer) > 1200:
                    # Truncate but keep complete sentences
                    sentences = answer.split('. ')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence + '. ') <= 1200:
                            truncated += sentence + '. '
                        else:
                            break
                    answer = truncated.strip()
                return answer
            else:
                return f"The provided policy document does not contain specific information about this question: {question}"
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            if response.status_code == 429:
                return f"Service temporarily unavailable due to rate limits. Please try again later. Question: {question}"
            elif response.status_code == 401:
                return f"API access denied. Please check your OpenAI API key configuration. Question: {question}"
            else:
                return f"Error generating answer: {error_msg}. Question: {question}"
            
    except requests.exceptions.Timeout:
        return f"Request timeout. The AI service is taking too long to respond. Question: {question}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}. Question: {question}"
    except Exception as e:
        return f"Error generating answer: {str(e)}. Question: {question}"

def generate_answer_with_mistral(question: str, context: str) -> str:
    """Generate answer using Mistral 7B Instruct via OpenRouter with optimized prompt for insurance accuracy"""
    if not AI_AVAILABLE or not OPENROUTER_API_KEY:
        return f"AI service not available. Please check if OPENROUTER_API_KEY is set in environment variables. Question: {question}"
    
    try:
        # Create an optimized prompt for insurance policy questions
        prompt = f"""You are an expert insurance policy analyst with deep knowledge of health insurance policies. Your task is to provide accurate, precise answers based solely on the provided policy document context.

POLICY DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based EXCLUSIVELY on the information provided in the policy document context above
2. If the specific information is not found in the context, respond with: "The provided policy document does not contain specific information about [exact topic]"
3. Be precise and include exact details, numbers, and terms from the policy when available
4. Use clear, professional language appropriate for insurance documentation
5. If you find relevant information, quote it accurately from the policy
6. Do not make assumptions or provide information not present in the context
7. If the context is insufficient, clearly state what specific information is missing
8. For numerical values (periods, amounts, percentages), be exact
9. For policy terms and conditions, be specific about requirements and limitations
10. Structure your answer logically with clear points

ANSWER:"""

        # OpenRouter API call to Mistral 7B Instruct
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hackrx-api.railway.app",
            "X-Title": "HackRx Insurance API"
        }
        
        payload = {
            "model": "mistralai/mistral-7b-instruct",  # Working model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                # Ensure the answer is not too long but preserve important details
                if len(answer) > 1200:
                    # Truncate but keep complete sentences
                    sentences = answer.split('. ')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence + '. ') <= 1200:
                            truncated += sentence + '. '
                        else:
                            break
                    answer = truncated.strip()
                return answer
            else:
                return f"The provided policy document does not contain specific information about this question: {question}"
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            if response.status_code == 429:
                return f"Service temporarily unavailable due to rate limits. Please try again later. Question: {question}"
            elif response.status_code == 401:
                return f"API access denied. Please check your OpenRouter API key configuration. Question: {question}"
            else:
                return f"Error generating answer: {error_msg}. Question: {question}"
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                # Ensure the answer is not too long but preserve important details
                if len(answer) > 1200:
                    # Truncate but keep complete sentences
                    sentences = answer.split('. ')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence + '. ') <= 1200:
                            truncated += sentence + '. '
                        else:
                            break
                    answer = truncated.strip()
                return answer
            else:
                return f"The provided policy document does not contain specific information about this question: {question}"
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            if response.status_code == 429:
                return f"Service temporarily unavailable due to rate limits. Please try again later. Question: {question}"
            elif response.status_code == 401:
                return f"API access denied. Please check your OpenRouter API key configuration. Question: {question}"
            else:
                return f"Error generating answer: {error_msg}. Question: {question}"
            
    except requests.exceptions.Timeout:
        return f"Request timeout. The AI service is taking too long to respond. Question: {question}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}. Question: {question}"
    except Exception as e:
        return f"Error generating answer: {str(e)}. Question: {question}"

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting HackRx API Server with FAISS & Mistral...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📄 Main Endpoint: POST http://localhost:8000/hackrx/run")
    print("🔐 Authentication: Bearer token required")
    print("=" * 60)
    initialize_ai_components()

@app.get("/")
async def root():
    return {
        "message": "HackRx API with FAISS & Mistral Integration",
        "version": "1.0.0",
        "status": "running",
        "ai_provider": "Mistral 7B Instruct (OpenRouter)",
        "vector_search": "FAISS",
        "endpoints": {
            "health": "/health",
            "main": "POST /hackrx/run"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_available": AI_AVAILABLE,
        "pdf_processing": PDF_AVAILABLE,
        "faiss_connected": faiss_index is not None,
        "mistral_connected": OPENROUTER_API_KEY is not None and OPENROUTER_API_KEY.strip() != "",
        "chunks_stored": len(document_chunks)
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Main endpoint for processing insurance policy queries"""
    start_time = time.time()
    
    try:
        # Check if document is already cached
        document_id = request.documents
        if document_id in document_cache:
            print(f"📄 Using cached document: {document_id}")
            chunks = document_cache[document_id]
        else:
            print(f"📄 Processing new document: {document_id}")
            
            # Download and extract text from PDF
            pdf_path = download_pdf_from_url(request.documents)
            text = extract_text_from_pdf(pdf_path)
            
            # Chunk the text
            chunks = chunk_text(text)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No meaningful content extracted from PDF")
            
            # Cache the chunks
            document_cache[document_id] = chunks
            print(f"✅ Extracted {len(chunks)} chunks from PDF")
        
        # Generate embeddings and store in FAISS
        embeddings = get_embeddings(chunks)
        store_chunks_in_faiss(chunks, embeddings, document_id)
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            print(f"🤔 Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Search for relevant chunks
            relevant_chunks = search_relevant_chunks(question)
            
            if not relevant_chunks:
                answers.append(f"The provided policy document does not contain specific information about this question: {question}")
                continue
            
            # Combine relevant chunks into context
            context = "\n\n".join(relevant_chunks)
            
            # Generate answer using AI (try Mistral first, then OpenAI as fallback)
            try:
                answer = generate_answer_with_mistral(question, context)
                # Check if Mistral failed and fallback to OpenAI
                if "API access denied" in answer or "AI service not available" in answer:
                    print(f"🔄 Mistral failed, trying OpenAI...")
                    answer = generate_answer_with_openai(question, context)
            except Exception as e:
                print(f"🔄 Mistral error, trying OpenAI: {e}")
                answer = generate_answer_with_openai(question, context)
            
            answers.append(answer)
            
            print(f"✅ Generated answer for question {i+1}")
        
        processing_time = time.time() - start_time
        print(f"🎉 Processed {len(request.questions)} questions in {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
