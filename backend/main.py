import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import os
from dotenv import load_dotenv
import asyncio
import json
import requests
import re
from typing import List
import time
import random
import hashlib
from datetime import datetime, timedelta
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

# Import the robust parser (now using Hugging Face)
from parser import parse_resume_with_llm

# --- Libraries for storage, search, and ranking ---
from supabase import create_client, Client
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Hugging Face API
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# --- Initialize Connections ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index("resumes")

model = SentenceTransformer('all-MiniLM-L6-v2')
OCR_API_KEY = os.environ.get("OCR_API_KEY")

# Cache for LLM results
llm_cache = {}
CACHE_DURATION = timedelta(hours=24)  # Cache for 24 hours

# Background processing status
upload_status = {}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# --- FastAPI App Initialization ---
app = FastAPI(title="Resume Screener API", version="4.0.0") # Final Version
origins = ["http://localhost:3000", "http://localhost:3001"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- Helper Functions ---
def get_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            if page.extract_text(): text += page.extract_text() + "\n"
    except: text = ""
    if len(text) < 100:
        print("Falling back to OCR...")
        try:
            r = requests.post('https://api.ocr.space/parse/image', files={'filename': ('resume.pdf', pdf_bytes)}, data={'apikey': OCR_API_KEY})
            r.raise_for_status()
            text = r.json()['ParsedResults'][0]['ParsedText']
        except: return ""
    return text

def preprocess_text(text: str) -> str:
    """Clean and standardize text to handle typos and inconsistencies"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Standardize common abbreviations
    replacements = {
        'python': 'Python',
        'javascript': 'JavaScript',
        'js': 'JavaScript',
        'react': 'React',
        'node.js': 'Node.js',
        'nodejs': 'Node.js',
        'machine learning': 'Machine Learning',
        'ml': 'Machine Learning',
        'artificial intelligence': 'Artificial Intelligence',
        'ai': 'Artificial Intelligence',
        'data science': 'Data Science',
        'sql': 'SQL',
        'nosql': 'NoSQL',
        'mongodb': 'MongoDB',
        'postgresql': 'PostgreSQL',
        'mysql': 'MySQL',
        'aws': 'AWS',
        'azure': 'Azure',
        'gcp': 'GCP',
        'docker': 'Docker',
        'kubernetes': 'Kubernetes',
        'k8s': 'Kubernetes',
        'git': 'Git',
        'github': 'GitHub',
        'agile': 'Agile',
        'scrum': 'Scrum',
        'jira': 'Jira',
        'tableau': 'Tableau',
        'power bi': 'Power BI',
        'powerbi': 'Power BI',
        'excel': 'Excel',
        'word': 'Word',
        'powerpoint': 'PowerPoint',
        'ppt': 'PowerPoint'
    }
    
    for old, new in replacements.items():
        text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
    
    return text

def calculate_confidence_score(candidate_summary: str, job_description: str) -> float:
    """Calculate realistic confidence in the analysis based on data quality and completeness"""
    # Base confidence starts lower
    confidence = 0.3  # Start at 30%
    
    # Data completeness factors
    candidate_words = len(candidate_summary.split())
    job_words = len(job_description.split())
    
    # More data = higher confidence, but with diminishing returns
    if candidate_words > 50:
        confidence += 0.15
    elif candidate_words > 20:
        confidence += 0.1
    else:
        confidence += 0.05
    
    if job_words > 100:
        confidence += 0.1
    elif job_words > 50:
        confidence += 0.05
    
    # Quality indicators
    if '@' in candidate_summary:  # Has email
        confidence += 0.05
    if any(skill in candidate_summary.lower() for skill in ['python', 'java', 'javascript', 'sql', 'react', 'node']):
        confidence += 0.1
    if any(word in candidate_summary.lower() for word in ['experience', 'years', 'senior', 'junior', 'lead']):
        confidence += 0.1
    if any(word in candidate_summary.lower() for word in ['degree', 'university', 'college', 'bachelor', 'master']):
        confidence += 0.05
    
    # Penalties for missing data
    if 'unknown' in candidate_summary.lower():
        confidence -= 0.2
    if 'could not parse' in candidate_summary.lower():
        confidence -= 0.15
    
    # Cap confidence at realistic levels
    return max(0.1, min(0.85, confidence))  # Between 10% and 85%

def generate_score_breakdown(candidate_summary: str, job_description: str, match_score: int) -> dict:
    """Generate detailed breakdown of how the score was calculated"""
    # Extract key information
    candidate_lower = candidate_summary.lower()
    job_lower = job_description.lower()
    
    # Analyze skill matches
    common_skills = ['python', 'java', 'javascript', 'sql', 'react', 'node', 'machine learning', 'data science']
    skill_matches = []
    skill_gaps = []
    
    for skill in common_skills:
        if skill in job_lower:
            if skill in candidate_lower:
                skill_matches.append(skill.title())
            else:
                skill_gaps.append(skill.title())
    
    # Analyze experience level
    experience_indicators = ['years', 'experience', 'senior', 'junior', 'lead', 'manager']
    has_experience = any(indicator in candidate_lower for indicator in experience_indicators)
    
    # Generate breakdown
    breakdown = {
        "skill_matches": skill_matches,
        "skill_gaps": skill_gaps,
        "experience_level": "Experienced" if has_experience else "Entry-level",
        "data_quality": "Good" if len(candidate_summary) > 100 else "Limited",
        "confidence": calculate_confidence_score(candidate_summary, job_description)
    }
    
    return breakdown

def get_llm_analysis_async_hf(job_description: str, candidate_summary: str, filename: str) -> dict:
    """
    Uses Hugging Face Inference API to analyze candidate fit for a job.
    """
    prompt = f"""
    You are an expert recruitment analyst. Your task is to analyze a candidate's fit for a job.

    **Job Description:** {job_description}
    **Candidate Summary:** {candidate_summary}

    **Instructions:**
    1. Determine a match score from 0 to 100 (must be a number)
    2. Write a detailed justification (approximately 50 words) that includes:
       - Specific skills that match the job requirements
       - Relevant experience that aligns with the role
       - Any gaps or concerns that might affect performance
       - Overall assessment of fit for the position
    
    **IMPORTANT:** Respond ONLY with valid JSON in this exact format:
    {{
        "match_score": [number between 0-100],
        "justification": "[detailed 50-word analysis covering skills, experience, gaps, and overall fit]"
    }}
    """
    try:
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 400, "return_full_text": False},
            "options": {"wait_for_model": True}
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            response_content = result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            response_content = result['generated_text']
        else:
            response_content = str(result)
        try:
            analysis_content = json.loads(response_content)
        except json.JSONDecodeError:
            analysis_content = {
                "match_score": 50,
                "justification": response_content[:200] + "..." if len(response_content) > 200 else response_content
            }
        return analysis_content
    except Exception as e:
        return {"match_score": "Error", "justification": f"API Error: {e}", "filename": filename}

async def get_llm_analysis_async(job_description: str, candidate_summary: str, filename: str) -> dict:
    # Preprocess text
    job_description = preprocess_text(job_description)
    candidate_summary = preprocess_text(candidate_summary)
    
    # Create cache key
    cache_key = hashlib.md5(f"{job_description}:{candidate_summary}".encode()).hexdigest()
    
    # Check cache
    if cache_key in llm_cache:
        cached_result = llm_cache[cache_key]
        if datetime.now() - cached_result['timestamp'] < CACHE_DURATION:
            result = cached_result['data'].copy()
            result['filename'] = filename
            result['cached'] = True
            return result
    
    # Use Hugging Face for LLM analysis
    analysis_content = get_llm_analysis_async_hf(job_description, candidate_summary, filename)
    # Add score breakdown and confidence
    score_breakdown = generate_score_breakdown(candidate_summary, job_description, analysis_content.get('match_score', 50))
    analysis_content.update(score_breakdown)
    analysis_content['filename'] = filename
    analysis_content['cached'] = False
    # Cache the result
    llm_cache[cache_key] = {
        'data': analysis_content,
        'timestamp': datetime.now()
    }
    return analysis_content

async def analyze_all_candidates_async(job_desc, candidates):
    tasks = []
    for candidate in candidates:
        name = candidate.get('extracted_data', {}).get('name', '')
        exp_text = candidate.get('extracted_data', {}).get('experience', '') or ''
        skills_text = ', '.join(candidate.get('extracted_data', {}).get('skills', []))
        candidate_summary = f"Candidate Name: {name}. Experience: {exp_text}. Skills: {skills_text}"
        tasks.append(get_llm_analysis_async(job_desc, candidate_summary, candidate.get('filename')))
    return await asyncio.gather(*tasks)

def get_text_from_pdf_fast(pdf_bytes: bytes) -> str:
    """Faster PDF text extraction with minimal processing"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        # Only read first 2 pages for speed
        for page in reader.pages[:2]:
            if page.extract_text(): 
                text += page.extract_text() + "\n"
    except: 
        text = ""
    
    # Only use OCR if absolutely necessary
    if len(text) < 50:
        try:
            r = requests.post('https://api.ocr.space/parse/image', 
                            files={'filename': ('resume.pdf', pdf_bytes)}, 
                            data={'apikey': OCR_API_KEY})
            r.raise_for_status()
            text = r.json()['ParsedResults'][0]['ParsedText']
        except: 
            return ""
    return text

def parse_resume_with_llm_fast(resume_text: str) -> dict:
    """Simple, working resume parsing - just use the original parser"""
    return parse_resume_with_llm(resume_text)


async def process_single_resume_async(file_info: dict, job_id: str) -> dict:
    """Process a single resume asynchronously with progress updates"""
    try:
        if "error" in file_info:
            result = {"filename": file_info["filename"], "status": "error", "detail": f"File read error: {file_info['error']}"}
        elif file_info["content_type"] != "application/pdf":
            result = {"filename": file_info["filename"], "status": "error", "detail": "Invalid file type."}
        else:
            # Fast text extraction
            raw_text = await asyncio.get_event_loop().run_in_executor(
                executor, get_text_from_pdf_fast, file_info["contents"]
            )
            
            if not raw_text or len(raw_text) < 100:
                result = {"filename": file_info["filename"], "status": "error", "detail": "Resume contains insufficient text."}
            else:
                # Use the working LLM parsing
                parsed_data = await asyncio.get_event_loop().run_in_executor(
                    executor, parse_resume_with_llm_fast, raw_text
                )
                
                if "error" in parsed_data:
                    result = {"filename": file_info["filename"], "status": "error", "detail": f"AI parsing failed: {parsed_data['error']}"}
                else:
                    email = parsed_data.get("email")
                    if not email:
                        result = {"filename": file_info["filename"], "status": "error", "detail": "AI parser could not find a valid email address in the resume."}
                    else:
                        email = email.lower()
                        
                        # Check for duplicates
                        existing_resume = supabase.table('resumes').select('id').eq('email', email).execute()
                        if existing_resume.data:
                            result = {"filename": file_info["filename"], "status": "error", "detail": f"A resume with the email '{email}' already exists."}
                        else:
                            # Store in database
                            db_response = supabase.table('resumes').insert({
                                "filename": file_info["filename"],
                                "email": email,
                                "extracted_data": parsed_data,
                                "status": "processed"
                            }).execute()
                            new_resume_id = db_response.data[0]['id']
                            
                            # Fast embedding generation
                            embedding_text = f"Name: {parsed_data.get('name')}. Experience: {parsed_data.get('experience')}. Skills: {', '.join(parsed_data.get('skills', []))}"
                            embedding = await asyncio.get_event_loop().run_in_executor(
                                executor, lambda: model.encode(embedding_text).tolist()
                            )
                            pinecone_index.upsert(vectors=[(new_resume_id, embedding)])
                            
                            result = {"filename": file_info["filename"], "status": "success", "resume_id": new_resume_id}
        
        # Update progress immediately
        if job_id in upload_status:
            upload_status[job_id]["processed"] += 1
            upload_status[job_id]["results"].append(result)
            if result["status"] == "success":
                upload_status[job_id]["successful"] += 1
            else:
                upload_status[job_id]["failed"] += 1
        
        return result
        
    except Exception as e:
        result = {"filename": file_info["filename"], "status": "error", "detail": str(e)}
        
        # Update progress for errors too
        if job_id in upload_status:
            upload_status[job_id]["processed"] += 1
            upload_status[job_id]["results"].append(result)
            upload_status[job_id]["failed"] += 1
        
        return result


# --- API Endpoints ---

@app.post("/upload-and-process-resume/")
async def upload_and_process_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    contents = await file.read()
    raw_text = get_text_from_pdf(contents)
    if not raw_text or len(raw_text) < 100:
        raise HTTPException(status_code=400, detail="Resume contains insufficient text.")

    # --- NEW ROBUST LOGIC: PARSE FIRST ---
    parsed_data = parse_resume_with_llm_fast(raw_text)
    if "error" in parsed_data:
        raise HTTPException(status_code=500, detail=f"AI parsing failed: {parsed_data['error']}")

    email = parsed_data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="AI parser could not find a valid email address in the resume.")
    
    email = email.lower()

    # --- VALIDATE SECOND ---
    existing_resume = supabase.table('resumes').select('id').eq('email', email).execute()
    if existing_resume.data:
        raise HTTPException(status_code=409, detail=f"A resume with the email '{email}' already exists.")

    # If not a duplicate, proceed to store the data
    db_response = supabase.table('resumes').insert({
        "filename": file.filename,
        "email": email,
        "extracted_data": parsed_data,
        "status": "processed"
    }).execute()
    new_resume_id = db_response.data[0]['id']

    embedding_text = f"Name: {parsed_data.get('name')}. Experience: {parsed_data.get('experience')}. Skills: {', '.join(parsed_data.get('skills', []))}"
    embedding = model.encode(embedding_text).tolist()
    pinecone_index.upsert(vectors=[(new_resume_id, embedding)])

    return {"message": "Resume processed successfully", "resume_id": new_resume_id}

@app.post("/batch-upload-and-process-resume/")
async def batch_upload_and_process_resume(files: List[UploadFile] = File(...)):
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Read all file contents immediately before closing
    file_data = []
    for file in files:
        try:
            contents = await file.read()
            file_data.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "contents": contents
            })
        except Exception as e:
            file_data.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "error": str(e)
            })
    
    # Initialize status
    upload_status[job_id] = {
        "status": "processing",
        "total_files": len(files),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "results": [],
        "start_time": datetime.now()
    }
    
    # Process files in parallel with real-time updates
    async def process_all_files():
        try:
            # Process all files concurrently with progress updates
            tasks = [process_single_resume_async(file_info, job_id) for file_info in file_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Mark as complete
            upload_status[job_id]["status"] = "completed"
            upload_status[job_id]["end_time"] = datetime.now()
            
        except Exception as e:
            upload_status[job_id]["status"] = "error"
            upload_status[job_id]["error"] = str(e)
    
    # Start background processing
    asyncio.create_task(process_all_files())
    
    return {"job_id": job_id, "message": "Upload started", "total_files": len(files)}

@app.get("/upload-status/{job_id}")
async def get_upload_status(job_id: str):
    if job_id not in upload_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = upload_status[job_id]
    
    # Clean up old completed jobs (older than 1 hour)
    if status.get("status") == "completed" and "end_time" in status:
        if datetime.now() - status["end_time"] > timedelta(hours=1):
            del upload_status[job_id]
    
    return status

class RankRequest(BaseModel):
    job_description: str

@app.post("/rank-candidates/")
async def rank_candidates(request: RankRequest):
    # Preprocess job description
    job_description = preprocess_text(request.job_description)
    
    # Get embeddings and search
    jd_embedding = model.encode(job_description).tolist()
    search_results = pinecone_index.query(vector=jd_embedding, top_k=10)
    candidate_ids = [match['id'] for match in search_results['matches']]
    
    if not candidate_ids:
        return []
    
    # Get candidates from database
    db_response = supabase.table('resumes').select("*").in_('id', candidate_ids).execute()
    candidates_to_analyze = db_response.data
    
    # Create candidate summaries for analysis
    candidate_summaries = []
    for c in candidates_to_analyze:
        name = c.get('extracted_data', {}).get('name', '')
        exp_text = c.get('extracted_data', {}).get('experience', '') or ''
        skills_text = ', '.join(c.get('extracted_data', {}).get('skills', []))
        candidate_summary = f"Candidate Name: {name}. Experience: {exp_text}. Skills: {skills_text}"
        candidate_summaries.append(candidate_summary)
    
    # Analyze all candidates in parallel
    analysis_tasks = [
        get_llm_analysis_async(job_description, summary, c['filename'])
        for summary, c in zip(candidate_summaries, candidates_to_analyze)
    ]
    
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    
    # Process results
    analysis_map = {}
    for i, result in enumerate(analysis_results):
        if isinstance(result, Exception):
            # Fallback analysis for failed LLM calls
            c = candidates_to_analyze[i]
            analysis_map[c['filename']] = {
                "match_score": 50,
                "justification": "Analysis failed, using similarity score",
                "filename": c['filename'],
                "cached": False
            }
        else:
            analysis_map[result['filename']] = result

    ranked_candidates = []
    ordered_candidates = sorted(candidates_to_analyze, key=lambda c: candidate_ids.index(c['id']))
    
    for i, candidate in enumerate(ordered_candidates):
        similarity_score = next((match['score'] for match in search_results['matches'] if match['id'] == candidate['id']), 0)
        ranked_candidates.append({
            "rank": i + 1,
            "candidate_data": candidate,
            "ai_analysis": analysis_map.get(candidate['filename'], {}),
            "similarity_score": similarity_score
        })
    
    # Implement smart ranking logic
    def get_sort_key(candidate):
        match_score = candidate['ai_analysis'].get('match_score', 0)
        confidence = candidate['ai_analysis'].get('confidence', 0)
        
        try:
            match_score = float(match_score)
            confidence = float(confidence)
        except (ValueError, TypeError):
            match_score = 0.0
            confidence = 0.0
        
        # If match scores are close (within 5 points), rank by confidence
        # Otherwise, rank by match score
        return (match_score, confidence)
    
    # Sort by match score first, then by confidence (both descending)
    ranked_candidates.sort(key=get_sort_key, reverse=True)
    
    # Update ranks
    for i, candidate in enumerate(ranked_candidates):
        candidate['rank'] = i + 1
        
    return ranked_candidates

@app.get("/")
def read_root():
    return {"status": "Resume Screener API is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=300)