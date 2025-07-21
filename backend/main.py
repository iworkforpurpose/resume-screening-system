# --- Standard and Third-Party Libraries ---
import asyncio
import hashlib
import io
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Any

import PyPDF2
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Import the robust parser (now using Hugging Face)
from parser import parse_resume_with_llm

# --- Configuration & Initialization ---

# 1. Load environment variables from .env file
load_dotenv()

# 2. Configure logging
# This provides more structured logging than print() for better debugging in production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 3. Validate and load essential environment variables
REQUIRED_VARS = [
    "HF_API_KEY", "SUPABASE_URL", "SUPABASE_KEY", 
    "PINECONE_API_KEY", "OCR_API_KEY", "ALLOWED_ORIGINS"
]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        logger.critical(f"FATAL ERROR: Environment variable '{var}' is not set.")
        raise SystemExit(f"FATAL ERROR: Environment variable '{var}' is not set.")

# Hugging Face and OCR API Keys
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
OCR_API_KEY = os.environ.get("OCR_API_KEY")

# 4. Initialize External Service Connections
try:
    # Supabase (Database)
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    # Pinecone (Vector Search)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index("resumes")

    # Sentence Transformer Model (for creating embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.critical(f"Failed to initialize external services: {e}")
    raise

# --- Global Variables & In-Memory Storage ---
llm_cache = {}
upload_status = {}
CACHE_DURATION = timedelta(hours=24)
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- Pydantic Models ---
class RankRequest(BaseModel):
    job_description: str = Field(..., min_length=50, description="The job description to rank candidates against.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Resume Screener API",
    version="4.1.0",
    description="A robust API to upload, parse, and rank resumes against job descriptions."
)

# IMPORTANT: Dynamic CORS configuration for production
# Reads allowed origins from an environment variable
allowed_origins = os.environ.get("ALLOWED_ORIGINS").split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def get_text_from_pdf_fast(pdf_bytes: bytes) -> str:
    """Faster PDF text extraction, falling back to OCR if necessary."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages[:2]:  # Limit to first 2 pages for speed
            if page.extract_text():
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}. Text extraction will rely on OCR if needed.")
        text = ""

    if len(text.strip()) < 100:
        logger.info("Short text from PyPDF2, attempting OCR fallback.")
        try:
            response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': ('resume.pdf', pdf_bytes)},
                data={'apikey': OCR_API_KEY},
                timeout=30  # Add a timeout
            )
            response.raise_for_status()
            parsed_results = response.json().get('ParsedResults', [])
            if parsed_results and parsed_results[0].get('ParsedText'):
                text = parsed_results[0]['ParsedText']
        except requests.RequestException as e:
            logger.error(f"OCR request failed: {e}")
            return ""
    return text

def preprocess_text(text: str) -> str:
    """Cleans and standardizes text to improve matching accuracy."""
    text = re.sub(r'\s+', ' ', text.strip())
    # This is a simplified example; a real-world version might use a larger mapping
    replacements = {'js': 'JavaScript', 'ml': 'Machine Learning', 'ai': 'Artificial Intelligence'}
    for old, new in replacements.items():
        text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
    return text

def get_llm_analysis(job_description: str, candidate_summary: str) -> Dict[str, Any]:
    """Synchronous version of LLM analysis using Hugging Face."""
    prompt = f"""
    Analyze the candidate's fit for the job.

    **Job Description:** {job_description}
    **Candidate Summary:** {candidate_summary}

    Respond ONLY with valid JSON in this exact format:
    {{
        "match_score": [number from 0-100],
        "justification": "[50-word analysis of skills, experience, and gaps]"
    }}
    """
    try:
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250, "return_full_text": False}, "options": {"wait_for_model": True}}
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        response_text = result[0]['generated_text'] if isinstance(result, list) else str(result)
        # Clean up potential markdown code blocks
        response_text = re.sub(r'```json\n|\n```', '', response_text).strip()
        
        return json.loads(response_text)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"LLM analysis failed: {e}. Content: {response.text if 'response' in locals() else 'N/A'}")
        return {"match_score": 0, "justification": "AI analysis could not be performed due to an external service error."}

async def get_cached_llm_analysis(job_desc: str, candidate_summary: str, filename: str) -> Dict[str, Any]:
    """Async wrapper for LLM analysis with caching."""
    job_desc, candidate_summary = preprocess_text(job_desc), preprocess_text(candidate_summary)
    cache_key = hashlib.md5(f"{job_desc}:{candidate_summary}".encode()).hexdigest()

    if cache_key in llm_cache and datetime.now() - llm_cache[cache_key]['timestamp'] < CACHE_DURATION:
        logger.info(f"LLM cache hit for {filename}")
        return {**llm_cache[cache_key]['data'], "filename": filename, "cached": True}

    logger.info(f"LLM cache miss for {filename}. Querying API.")
    loop = asyncio.get_event_loop()
    analysis = await loop.run_in_executor(executor, get_llm_analysis, job_desc, candidate_summary)
    
    # Simple confidence and breakdown based on data quality
    confidence = 0.5 + (0.35 if len(candidate_summary) > 200 else 0.1)
    analysis['confidence'] = round(min(0.95, confidence), 2)
    analysis['filename'] = filename
    analysis['cached'] = False
    
    llm_cache[cache_key] = {'data': analysis, 'timestamp': datetime.now()}
    return analysis


async def process_single_resume(file_info: Dict, job_id: str):
    """Processes a single resume file: parses, validates, and stores it."""
    filename = file_info["filename"]
    try:
        if file_info.get("error"):
            raise ValueError(f"File read error: {file_info['error']}")
        if file_info["content_type"] != "application/pdf":
            raise ValueError("Invalid file type. Only PDF is supported.")
        
        raw_text = await asyncio.to_thread(get_text_from_pdf_fast, file_info["contents"])
        if len(raw_text.strip()) < 100:
            raise ValueError("Resume contains insufficient text for processing.")
        
        parsed_data = await asyncio.to_thread(parse_resume_with_llm, raw_text)
        if "error" in parsed_data:
            raise ValueError(f"AI parsing failed: {parsed_data['error']}")

        email = parsed_data.get("email", "").lower()
        if not email:
            raise ValueError("AI parser could not find a valid email address.")
        
        existing = supabase.table('resumes').select('id').eq('email', email).limit(1).execute()
        if existing.data:
            raise ValueError(f"A resume with the email '{email}' already exists.")

        db_response = supabase.table('resumes').insert({
            "filename": filename, "email": email, "extracted_data": parsed_data, "status": "processed"
        }).execute()
        new_id = db_response.data[0]['id']

        embedding_text = f"Name: {parsed_data.get('name')}. Experience: {parsed_data.get('experience')}. Skills: {', '.join(parsed_data.get('skills', []))}"
        embedding = await asyncio.to_thread(model.encode, embedding_text)
        pinecone_index.upsert(vectors=[(str(new_id), embedding.tolist())])
        
        return {"filename": filename, "status": "success", "resume_id": new_id}

    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")
        return {"filename": filename, "status": "error", "detail": str(e)}
    finally:
        # Update progress regardless of outcome
        if job_id in upload_status:
            status = upload_status[job_id]
            status["processed"] += 1
            if "success" in locals().get("result", {}).get("status", ""):
                 status["successful"] += 1
            else:
                 status["failed"] += 1


# --- API Endpoints ---
@app.get("/", summary="API Health Check")
def read_root():
    return {"status": "Resume Screener API is running."}

@app.post("/upload-and-process-resume/", summary="Upload and Process a Single Resume")
async def upload_and_process_resume(file: UploadFile = File(...)):
    """This endpoint is deprecated in favor of the batch upload endpoint."""
    raise HTTPException(status_code=400, detail="This endpoint is deprecated. Please use /batch-upload-and-process-resume/.")

@app.post("/batch-upload-and-process-resume/", summary="Upload Multiple Resumes for Processing")
async def batch_upload_and_process_resume(files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    file_data = []
    for file in files:
        try:
            file_data.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "contents": await file.read()
            })
        except Exception as e:
            file_data.append({"filename": file.filename, "error": str(e)})

    upload_status[job_id] = {
        "status": "processing", "total_files": len(files), "processed": 0,
        "successful": 0, "failed": 0, "results": [], "start_time": datetime.now().isoformat()
    }

    async def run_processing():
        results = await asyncio.gather(*(process_single_resume(f, job_id) for f in file_data))
        upload_status[job_id]["results"] = results
        upload_status[job_id]["status"] = "completed"
        upload_status[job_id]["end_time"] = datetime.now().isoformat()
    
    asyncio.create_task(run_processing())
    return {"job_id": job_id, "message": f"Upload started for {len(files)} files."}

@app.get("/upload-status/{job_id}", summary="Get Status of a Batch Upload Job")
async def get_upload_status(job_id: str):
    status = upload_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return status

@app.post("/rank-candidates/", summary="Rank All Candidates Against a Job Description")
async def rank_candidates(request: RankRequest):
    try:
        jd_embedding = model.encode(preprocess_text(request.job_description)).tolist()
        search_results = pinecone_index.query(vector=jd_embedding, top_k=20, include_metadata=False)
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        raise HTTPException(status_code=503, detail="Candidate search service is unavailable.")

    candidate_ids = [match['id'] for match in search_results.get('matches', [])]
    if not candidate_ids:
        return []

    db_response = supabase.table('resumes').select("id, filename, extracted_data").in_('id', candidate_ids).execute()
    candidates = db_response.data

    analysis_tasks = []
    for c in candidates:
        summary = f"Name: {c['extracted_data'].get('name', 'N/A')}. Experience: {c['extracted_data'].get('experience', 'N/A')}. Skills: {', '.join(c['extracted_data'].get('skills', []))}"
        analysis_tasks.append(get_cached_llm_analysis(request.job_description, summary, c['filename']))
    
    analysis_results = await asyncio.gather(*analysis_tasks)
    analysis_map = {res['filename']: res for res in analysis_results}

    def get_sort_key(candidate):
        """Defines the logic for smart ranking."""
        analysis = analysis_map.get(candidate['filename'], {})
        # Use match_score from AI if available, otherwise fall back to similarity score
        score = analysis.get('match_score', 0)
        confidence = analysis.get('confidence', 0.5)
        # Convert score to float for reliable sorting
        try:
            primary_score = float(score)
        except (ValueError, TypeError):
            primary_score = 0.0
        # Combine score and confidence for a more nuanced ranking
        return primary_score + (confidence * 0.1)

    # Combine data and sort
    final_candidates = []
    for cand in candidates:
        sim_score = next((match['score'] for match in search_results['matches'] if match['id'] == str(cand['id'])), 0)
        final_candidates.append({
            "candidate_data": cand,
            "ai_analysis": analysis_map.get(cand['filename'], {}),
            "similarity_score": round(sim_score, 4)
        })

    final_candidates.sort(key=get_sort_key, reverse=True)
    
    # Add final rank
    for i, candidate in enumerate(final_candidates):
        candidate['rank'] = i + 1
        
    return final_candidates

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)