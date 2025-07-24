import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import os
from dotenv import load_dotenv
import asyncio
import json
import requests
import hashlib
import uuid

# Import the robust parser
from parser import parse_resume_with_llm

# --- Libraries for storage, search, and ranking ---
from supabase import create_client, Client
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# --- Initialize Connections ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index("resumes")

model = SentenceTransformer('./model')
OCR_API_KEY = os.environ.get("OCR_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(title="Resume Screener API", version="6.0.0") # Multi-Tenant Version
frontend_url = os.environ.get("FRONTEND_URL")
origins = ["http://localhost:3000", "http://localhost:3001"]
if frontend_url:
    origins.append(frontend_url)
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- In-memory stores for caching and job tracking ---
rank_cache = {}
upload_jobs = {}

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

async def get_llm_analysis_async(job_description: str, candidate_summary: str, filename: str) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
    prompt = f"""
    You are an expert recruitment analyst.
    **CRITICAL INSTRUCTIONS:**
    1. Analyze Fit: Based on the provided candidate summary and job description, determine a match score from 0 to 100.
    2. Score Breakdown:
       - List matching skills from the job description.
       - List non-matching skills.
    3. Personalize Justification: Write a 50-word justification for your score, quoting specific evidence.
    **Job Description:** {job_description}
    ---
    **Candidate Summary:** {candidate_summary}
    ---
    Provide your response in a strict JSON format with keys: "match_score", "skill_matches", "skill_gaps", and "justification".
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, max_tokens=300, response_format={"type": "json_object"}
            )
            
            analysis_content = json.loads(response.choices[0].message.content)
            analysis_content['filename'] = filename
            return analysis_content

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if attempt + 1 == max_retries:
                return {"match_score": "Error", "skill_matches": [], "skill_gaps": [], "justification": f"API Error after {max_retries} attempts: {e}", "filename": filename}
            await asyncio.sleep(1)

# --- UPDATED Background Processing Logic for Multi-Tenancy ---
def process_file_for_batch(file_contents: bytes, filename: str, job_id: str, session_id: str):
    job_status = upload_jobs[job_id]
    try:
        raw_text = get_text_from_pdf(file_contents)
        if not raw_text or len(raw_text) < 100:
            raise ValueError("Resume contains insufficient text.")

        parsed_data = parse_resume_with_llm(raw_text)
        if "error" in parsed_data:
            raise ValueError(f"AI parsing failed: {parsed_data['error']}")

        email = parsed_data.get("email")
        if not email:
            raise ValueError("AI parser could not find a valid email in the resume.")
        email = email.lower()

        # Check for duplicates within the same session to be safe, though email is globally unique
        existing_resume = supabase.table('resumes').select('id').eq('email', email).execute()
        if existing_resume.data:
            raise ValueError(f"A resume with the email '{email}' already exists.")

        # Save session_id to Supabase
        db_response = supabase.table('resumes').insert({
            "filename": filename,
            "email": email,
            "extracted_data": parsed_data,
            "status": "processed",
            "session_id": session_id
        }).execute()
        new_resume_id = db_response.data[0]['id']

        embedding_text = f"Name: {parsed_data.get('name')}. Experience: {parsed_data.get('experience', '')}. Skills: {', '.join(parsed_data.get('skills', []))}"
        embedding = model.encode(embedding_text).tolist()

        # Upsert to Pinecone with session_id in metadata
        pinecone_index.upsert(vectors=[(
            str(new_resume_id), 
            embedding, 
            {"session_id": session_id}
        )])

        job_status["results"].append({"filename": filename, "status": "success"})
        job_status["successful"] += 1

    except Exception as e:
        job_status["results"].append({"filename": filename, "status": "error", "detail": str(e)})
        job_status["failed"] += 1
    finally:
        job_status["processed"] += 1
        if job_status["processed"] == job_status["total_files"]:
            job_status["status"] = "completed"

# --- API Endpoints (UPDATED for Multi-Tenancy) ---

@app.post("/upload-and-process-resume/")
async def upload_and_process_resume(request: Request, file: UploadFile = File(...)):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    contents = await file.read()
    job_id = str(uuid.uuid4())
    upload_jobs[job_id] = {"status": "processing", "processed": 0, "total_files": 1, "successful": 0, "failed": 0, "results": []}
    
    # Pass session_id to the processing function
    process_file_for_batch(contents, file.filename, job_id, session_id)
    
    result = upload_jobs[job_id]["results"][0]
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["detail"])

    return {"message": "Resume processed successfully"}

@app.post("/batch-upload-and-process-resume/")
async def batch_upload_and_process_resume(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...)
):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")

    job_id = str(uuid.uuid4())
    upload_jobs[job_id] = {
        "status": "processing",
        "processed": 0,
        "total_files": len(files),
        "successful": 0,
        "failed": 0,
        "results": []
    }

    for file in files:
        contents = await file.read()
        # Pass session_id to the background task
        background_tasks.add_task(process_file_for_batch, contents, file.filename, job_id, session_id)

    return {"job_id": job_id, "total_files": len(files)}

@app.get("/upload-status/{job_id}")
async def get_upload_status(job_id: str):
    job = upload_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job

class RankRequest(BaseModel):
    job_description: str

@app.post("/rank-candidates/")
async def rank_candidates(request: Request, body: RankRequest):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")

    if not body.job_description:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    jd_embedding = model.encode(body.job_description).tolist()

    try:
        # Add the metadata filter to the Pinecone query
        query_response = pinecone_index.query(
            vector=jd_embedding,
            top_k=10,
            include_metadata=False,
            filter={"session_id": {"$eq": session_id}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying vector database: {e}")

    matches = query_response.get('matches', [])
    if not matches:
        return []

    candidate_scores = {match['id']: match['score'] for match in matches}
    resume_ids = list(candidate_scores.keys())

    try:
        db_response = supabase.table('resumes').select('*').in_('id', resume_ids).execute()
        if db_response.data is None:
             raise ValueError("Failed to fetch candidate data from the database.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching candidate data: {e}")

    top_candidates = []
    for candidate_record in db_response.data:
        candidate_id_str = str(candidate_record['id'])
        if candidate_id_str in candidate_scores:
            top_candidates.append({
                "candidate_data": candidate_record,
                "similarity_score": candidate_scores[candidate_id_str]
            })

    cache_key_str = body.job_description + '-' + ','.join(sorted(resume_ids))
    cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()

    if cache_key in rank_cache:
        analysis_results = rank_cache[cache_key]
    else:
        tasks = []
        for c in top_candidates:
            data = c['candidate_data']['extracted_data']
            summary = f"Name: {data.get('name')}. Experience: {data.get('experience', '')}. Skills: {', '.join(data.get('skills', []))}"
            tasks.append(get_llm_analysis_async(body.job_description, summary, c['candidate_data']['filename']))
        analysis_results = await asyncio.gather(*tasks)
        rank_cache[cache_key] = analysis_results
        
    analysis_map = {result['filename']: result for result in analysis_results}
    
    ranked_candidates = []
    for candidate in top_candidates:
        filename = candidate["candidate_data"]["filename"]
        analysis = analysis_map.get(filename, {})
        match_score = analysis.get('match_score', 0)
        
        ranked_candidates.append({
            "candidate_data": candidate["candidate_data"],
            "ai_analysis": analysis,
            "similarity_score": candidate["similarity_score"],
            "sort_score": match_score if isinstance(match_score, (int, float)) else 0
        })
    
    ranked_candidates.sort(key=lambda x: (x['sort_score'], x['similarity_score']), reverse=True)
    
    for i, candidate in enumerate(ranked_candidates):
        candidate['rank'] = i + 1
        del candidate['sort_score']

    return ranked_candidates

@app.get("/")
def read_root():
    return {"status": "Resume Screener API is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=300)