import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import os
import re
from dotenv import load_dotenv
import asyncio
import json
import requests
import hashlib
import uuid
import time
import concurrent.futures
from typing import Dict, List, Optional

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

model = SentenceTransformer('all-MiniLM-L6-v2')
OCR_API_KEY = os.environ.get("OCR_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(title="Resume Screener API", version="5.3.0") # Added JD Caching and Worker Optimization
origins = ["http://localhost:3000", "http://localhost:3001"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- In-memory stores for caching and job tracking ---
rank_cache = {}
upload_jobs = {}
jd_cache = {}  # Cache for predefined job descriptions

# Configuration for thread pool
MAX_WORKERS = 2  # Reduced from 4 to 2 for resource optimization

# --- Aggressive LLM Analysis Caching ---
# Cache structure: {(jd_id or jd_hash, candidate_id): analysis_result}
llm_analysis_cache = {}

# Helper to get cache key
import hashlib
def get_llm_cache_key(jd_id, job_description, candidate_id):
    if jd_id:
        return f"{jd_id}:{candidate_id}"
    else:
        jd_hash = hashlib.sha256(job_description.encode()).hexdigest()
        return f"{jd_hash}:{candidate_id}"

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

# --- UPDATED with Retry Logic, Error Handling, and Enhanced Prompt ---
async def get_llm_analysis_async(job_description: str, candidate_summary: str, filename: str) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    prompt = f"""
    You are an expert technical recruitment analyst with deep domain knowledge across multiple tech stacks.

    **CRITICAL INSTRUCTIONS:**
    1. Analyze Fit: Based on the provided candidate summary and job description, determine a match score from 0 to 100.

    2. Detailed Score Breakdown (40% of your evaluation):
       - List all matching technical skills from the job description with proficiency level estimation (Beginner/Intermediate/Expert).
       - List all non-matching required skills that are missing.
       - Evaluate years of relevant experience compared to requirements.
       - Assess education fit with requirements.

    3. Technical Role Alignment (30% of evaluation):
       - Evaluate how well the candidate's technical background aligns with the specific role.
       - Note any domain-specific experience that's relevant.

    4. Career Trajectory Analysis (20% of evaluation):
       - Assess if this role is a logical next step in the candidate's career path.
       - Evaluate growth potential in this position.

    5. Comprehensive Justification (10% of evaluation):
       - Write a 100-word justification for your score, citing specific evidence from both the resume and job description.
       - Be specific about strengths and potential areas for growth.

    **Job Description:** {job_description}
    ---
    **Candidate Summary:** {candidate_summary}
    ---

    IMPORTANT: Your response MUST be valid JSON format with no trailing commas, properly closed quotes, and balanced braces.

    Provide your response in a strict JSON format with the following keys:
    - "match_score": integer from 0-100
    - "skill_matches": array of objects with "skill" and "level" fields
    - "skill_gaps": array of strings
    - "experience_assessment": string
    - "education_fit": string
    - "role_alignment": string
    - "career_trajectory": string
    - "justification": string
    - "interview_recommendations": array of strings with suggested interview questions based on the candidate's background
    """

    start_time = time.time()
    max_retries = 3

    def fix_json_response(text):
        text = re.sub(r'```(?:json)?|```', '', text).strip()
        open_quotes = text.count('"')
        if open_quotes % 2 != 0:
            text += '"'
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        text = re.sub(r',\s*}', '}', text)
        return text

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            try:
                analysis_content = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt+1} for {filename}: {str(e)}")
                fixed_content = fix_json_response(content)
                try:
                    analysis_content = json.loads(fixed_content)
                    print(f"Successfully fixed malformed JSON for {filename}")
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        print(f"JSON parsing failed on attempt {attempt+1}, retrying...")
                        await asyncio.sleep(1)
                        continue
                    raise
            analysis_content['match_score'] = analysis_content.get('match_score', 50)
            analysis_content['skill_matches'] = analysis_content.get('skill_matches', [])
            analysis_content['skill_gaps'] = analysis_content.get('skill_gaps', [])
            analysis_content['justification'] = analysis_content.get('justification', "Analysis based on available information.")
            analysis_content['filename'] = filename
            analysis_content['processing_time'] = round(time.time() - start_time, 2)
            return analysis_content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if attempt + 1 == max_retries:
                return {
                    "match_score": 50,
                    "skill_matches": [],
                    "skill_gaps": [],
                    "justification": "Unable to complete analysis due to technical issues. The system encountered an error while analyzing this candidate. Please review the resume manually.",
                    "experience_assessment": "Assessment unavailable due to processing error.",
                    "education_fit": "Assessment unavailable due to processing error.",
                    "role_alignment": "Assessment unavailable due to processing error.",
                    "career_trajectory": "Assessment unavailable due to processing error.",
                    "interview_recommendations": ["Consider asking about their actual experience related to the job requirements"],
                    "filename": filename,
                    "processing_time": round(time.time() - start_time, 2),
                    "_error_details": str(e)
                }
            await asyncio.sleep(1)

# --- UNIFIED Background Processing Logic with Sequential Processing ---
def process_file_for_batch(file_contents: bytes, filename: str, job_id: str):
    """Process a single file in a batch job with comprehensive error handling"""
    try:
        start_time = time.time()

        # Track processing attempts with descriptive error messages
        processing_notes = []

        # Guard against invalid job ID
        if job_id not in upload_jobs:
            print(f"Warning: Job {job_id} not found in upload_jobs.")
            return

        # Step 1: Extract text from PDF
        try:
            raw_text = get_text_from_pdf(file_contents)
            if not raw_text or len(raw_text.strip()) < 50:  # Sanity check for extracted content
                raise ValueError("Insufficient text extracted from PDF")
        except Exception as extract_err:
            error_msg = f"PDF extraction error: {str(extract_err)}"
            processing_notes.append(error_msg)
            upload_jobs[job_id]["results"].append({
                "filename": filename,
                "status": "error",
                "detail": error_msg,
                "notes": "; ".join(processing_notes)
            })
            upload_jobs[job_id]["failed"] += 1
            upload_jobs[job_id]["processed"] += 1
            return

        # Step 2: Parse with LLM
        try:
            parsed_data = parse_resume_with_llm(raw_text)
            if parsed_data.get("error") or parsed_data.get("_error_details"):
                processing_notes.append(f"Parsing warning: {parsed_data.get('_error_details', 'Unknown parsing issue')}")
        except Exception as parse_err:
            error_msg = f"LLM parsing error: {str(parse_err)}"
            processing_notes.append(error_msg)
            upload_jobs[job_id]["results"].append({
                "filename": filename,
                "status": "error",
                "detail": error_msg,
                "notes": "; ".join(processing_notes)
            })
            upload_jobs[job_id]["failed"] += 1
            upload_jobs[job_id]["processed"] += 1
            return

        email = parsed_data.get("email")
        if not email:
            # Try to extract email with regex as fallback
            import re
            emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', raw_text)
            if emails:
                email = emails[0]
                parsed_data["email"] = email
                processing_notes.append(f"Email extracted via regex: {email}")
            else:
                # As a last resort, use the filename as an identifier
                email = f"unknown_{hash(filename)}@placeholder.com"
                parsed_data["email"] = email
                processing_notes.append(f"Using placeholder email: {email}")

        email = email.lower()

        # Step 3: Check for existing resume with this email
        try:
            existing_resume = supabase.table('resumes').select('id').eq('email', email).execute()

            if existing_resume.data:
                # If this is a duplicate, we'll add a suffix to the email instead of failing
                timestamp = int(time.time())
                original_email = email
                email = f"{email.split('@')[0]}+{timestamp}@{email.split('@')[1]}"
                parsed_data["email"] = email
                processing_notes.append(f"Modified duplicate email from {original_email} to {email}")
        except Exception as db_err:
            processing_notes.append(f"Database lookup warning: {str(db_err)}")
            # Continue with processing, assuming no duplicate

        # Step 4: Insert into database
        try:
            # Create database insert payload
            insert_payload = {
                "filename": filename,
                "email": email,
                "extracted_data": parsed_data,
                "status": "processed",
            }

            # Only add processing_notes if there were errors or warnings
            if processing_notes or "_error_details" in parsed_data:
                notes_text = "; ".join(processing_notes)
                if "_error_details" in parsed_data:
                    notes_text += f"; {parsed_data['_error_details']}"
                insert_payload["processing_notes"] = notes_text

            # Try insertion with processing_notes first
            try:
                db_response = supabase.table('resumes').insert(insert_payload).execute()
                new_resume_id = db_response.data[0]['id']
            except Exception as column_err:
                # If insertion fails due to column issues, try without processing_notes
                if "processing_notes" in insert_payload:
                    processing_notes.append("Removed processing_notes due to schema incompatibility")
                    del insert_payload["processing_notes"]
                    db_response = supabase.table('resumes').insert(insert_payload).execute()
                    new_resume_id = db_response.data[0]['id']
                else:
                    raise  # Re-raise if the error wasn't related to processing_notes
        except Exception as insert_err:
            error_msg = f"Database insertion failed: {str(insert_err)}"
            processing_notes.append(error_msg)
            upload_jobs[job_id]["results"].append({
                "filename": filename,
                "status": "error",
                "detail": error_msg,
                "notes": "; ".join(processing_notes)
            })
            upload_jobs[job_id]["failed"] += 1
            upload_jobs[job_id]["processed"] += 1
            return

        # Step 5: Generate embedding and store in Pinecone
        try:
            # Use both name and skills for embedding if available, otherwise use what we have
            name_part = f"Name: {parsed_data.get('name', 'Unknown')}. "
            exp_part = f"Experience: {parsed_data.get('experience', '')}. "

            # Handle different skill formats to maintain compatibility
            skills_part = ""
            if 'skills' in parsed_data and isinstance(parsed_data['skills'], list):
                skills_part = f"Skills: {', '.join(parsed_data['skills'])}. "
            elif 'technical_skills' in parsed_data and isinstance(parsed_data['technical_skills'], list):
                skills_part = f"Skills: {', '.join(parsed_data['technical_skills'])}. "

            embedding_text = name_part + exp_part + skills_part

            # Generate embedding using SentenceTransformer
            embedding = model.encode(embedding_text).tolist()

            # Store in Pinecone
            pinecone_index.upsert(
                vectors=[
                    {
                        'id': str(new_resume_id),
                        'values': embedding,
                        'metadata': {
                            'name': parsed_data.get('name'),
                            'email': email,
                            'resume_id': new_resume_id
                        }
                    }
                ]
            )
        except Exception as embedding_err:
            processing_notes.append(f"Embedding warning (resume still saved): {str(embedding_err)}")
            # We don't fail the entire process for embedding issues

        # Record the successful processing with processing time
        elapsed_time = round(time.time() - start_time, 2)
        status = "processed"

        if processing_notes:
            status = "processed_with_warnings"

        upload_jobs[job_id]["results"].append({
            "filename": filename,
            "status": status,
            "resume_id": new_resume_id,
            "processing_time": elapsed_time,
            "notes": "; ".join(processing_notes) if processing_notes else None
        })
        upload_jobs[job_id]["successful"] += 1
        upload_jobs[job_id]["processed"] += 1

        # Update job processing time statistics
        if "processing_times" not in upload_jobs[job_id]:
            upload_jobs[job_id]["processing_times"] = []
        upload_jobs[job_id]["processing_times"].append(elapsed_time)

    except Exception as e:
        # Catch any other exceptions we didn't anticipate
        error_msg = f"Unexpected error processing {filename}: {str(e)}"
        print(error_msg)

        if job_id in upload_jobs:
            upload_jobs[job_id]["results"].append({
                "filename": filename,
                "status": "error",
                "detail": error_msg
            })
            upload_jobs[job_id]["failed"] += 1
            upload_jobs[job_id]["processed"] += 1

# --- API Endpoints ---

class JobDescription(BaseModel):
    title: str
    description: str

class RankRequest(BaseModel):
    job_description: str
    jd_id: Optional[str] = None

# --- Precompute LLM analysis for predefined JDs and new resumes ---
async def precompute_llm_analysis_for_jd(jd_id, job_description):
    # Get all resumes
    db_response = supabase.table('resumes').select('*').execute()
    if not db_response.data:
        return
    for candidate_record in db_response.data:
        candidate_id = str(candidate_record['id'])
        data = candidate_record['extracted_data']
        summary = f"Name: {data.get('name')}. Experience: {data.get('experience', '')}. Education: {data.get('education', '')}. Skills: {', '.join(data.get('skills', []))}"
        cache_key = get_llm_cache_key(jd_id, job_description, candidate_id)
        if cache_key not in llm_analysis_cache:
            analysis = await get_llm_analysis_async(job_description, summary, candidate_record['filename'])
            llm_analysis_cache[cache_key] = analysis

# When a new JD is created, precompute for all resumes
@app.post("/job-descriptions/")
async def create_job_description(job_desc: JobDescription):
    """Create a new job description and store it"""
    jd_id = str(uuid.uuid4())

    # Store in database
    try:
        db_response = supabase.table('job_descriptions').insert({
            "id": jd_id,
            "title": job_desc.title,
            "description": job_desc.description
        }).execute()
        # Pre-cache the embedding
        jd_embedding = model.encode(job_desc.description).tolist()
        jd_cache[jd_id] = {
            "title": job_desc.title,
            "description": job_desc.description,
            "embedding": jd_embedding
        }
        # Precompute LLM analysis for all resumes (async)
        asyncio.create_task(precompute_llm_analysis_for_jd(jd_id, job_desc.description))
        return {"id": jd_id, "title": job_desc.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job description: {e}")

@app.get("/job-descriptions/")
async def list_job_descriptions():
    """List all stored job descriptions"""
    try:
        db_response = supabase.table('job_descriptions').select('id, title').execute()
        return db_response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job descriptions: {e}")

@app.get("/job-descriptions/{jd_id}")
async def get_job_description(jd_id: str):
    """Get a specific job description by ID"""
    # Check cache first
    if jd_id in jd_cache:
        return {
            "id": jd_id,
            "title": jd_cache[jd_id]["title"],
            "description": jd_cache[jd_id]["description"]
        }

    # If not in cache, fetch from database
    try:
        db_response = supabase.table('job_descriptions').select('*').eq('id', jd_id).execute()
        if not db_response.data:
            raise HTTPException(status_code=404, detail="Job description not found")

        # Add to cache
        jd_data = db_response.data[0]
        jd_embedding = model.encode(jd_data["description"]).tolist()
        jd_cache[jd_id] = {
            "title": jd_data["title"],
            "description": jd_data["description"],
            "embedding": jd_embedding
        }

        return jd_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching job description: {e}")

@app.post("/upload-and-process-resume/")
async def upload_and_process_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")

    contents = await file.read()
    job_id = str(uuid.uuid4())
    upload_jobs[job_id] = {
        "status": "processing",
        "processed": 0,
        "total_files": 1,
        "successful": 0,
        "failed": 0,
        "results": [],
        "start_time": time.time()
    }
    process_file_for_batch(contents, file.filename, job_id)

    result = upload_jobs[job_id]["results"][0]
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["detail"])

    return {
        "message": "Resume processed successfully",
        "processing_time": result.get("processing_time", 0)
    }

@app.post("/batch-upload-and-process-resume/")
async def batch_upload_and_process_resume(files: list[UploadFile] = File(...)):
    """Process multiple resumes with robust error handling"""
    job_id = str(uuid.uuid4())

    # Validate all files are PDFs
    non_pdfs = [f.filename for f in files if f.content_type != "application/pdf"]
    if non_pdfs:
        raise HTTPException(status_code=400, detail=f"Invalid file types: {', '.join(non_pdfs)}. Only PDFs are accepted.")

    # Initialize job status
    upload_jobs[job_id] = {
        "status": "processing",
        "processed": 0,
        "total_files": len(files),
        "successful": 0,
        "failed": 0,
        "results": [],
        "start_time": time.time()
    }

    # Read all files first to avoid file handle issues
    file_data = []

    try:
        for file in files:
            try:
                contents = await file.read()
                file_data.append((contents, file.filename))
            except Exception as e:
                # If we can't read a file, add it as a failed result but continue with others
                upload_jobs[job_id]["results"].append({
                    "filename": file.filename,
                    "status": "error",
                    "detail": f"File read error: {str(e)}"
                })
                upload_jobs[job_id]["failed"] += 1
                upload_jobs[job_id]["processed"] += 1
    except Exception as batch_e:
        # Handle catastrophic batch reading failure
        upload_jobs[job_id]["status"] = "failed"
        return {
            "job_id": job_id,
            "error": f"Failed to read batch files: {str(batch_e)}",
            "total_files": len(files)
        }

    # Update total files to reflect only the successfully read files
    upload_jobs[job_id]["total_files"] = len(file_data) + upload_jobs[job_id]["failed"]

    if not file_data:
        upload_jobs[job_id]["status"] = "completed"
        return {
            "job_id": job_id,
            "total_files": len(files),
            "message": "No valid files could be processed"
        }

    try:
        # Process files in parallel with limited workers and timeout handling
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks to the thread pool with timeout
            futures = []
            for contents, filename in file_data:
                # Submit with a timeout safety mechanism
                future = executor.submit(
                    process_file_for_batch,
                    contents,
                    filename,
                    job_id
                )
                futures.append(future)

            # Wait for all tasks to complete with timeout handling
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Just ensure the future completes
                    future.result()
                except concurrent.futures.TimeoutError:
                    print(f"A task in job {job_id} timed out")
                except Exception as task_ex:
                    print(f"Task exception in job {job_id}: {str(task_ex)}")
    except Exception as e:
        print(f"Executor exception in job {job_id}: {str(e)}")
        # Even if the executor fails, the job can continue

    # If all tasks completed but job status wasn't updated (edge case)
    if upload_jobs[job_id]["processed"] < upload_jobs[job_id]["total_files"]:
        print(f"Job {job_id} incomplete: {upload_jobs[job_id]['processed']}/{upload_jobs[job_id]['total_files']} processed")
        # Mark any unprocessed files as failed
        for contents, filename in file_data:
            if not any(r.get("filename") == filename for r in upload_jobs[job_id]["results"]):
                upload_jobs[job_id]["results"].append({
                    "filename": filename,
                    "status": "error",
                    "detail": "Processing timed out or was interrupted"
                })
                upload_jobs[job_id]["failed"] += 1
                upload_jobs[job_id]["processed"] += 1

    # Set job as completed if all files were processed
    if upload_jobs[job_id]["processed"] >= upload_jobs[job_id]["total_files"]:
        upload_jobs[job_id]["status"] = "completed"
        upload_jobs[job_id]["total_time"] = round(time.time() - upload_jobs[job_id]["start_time"], 2)

    return {
        "job_id": job_id,
        "total_files": len(files),
        "processed": upload_jobs[job_id]["processed"],
        "successful": upload_jobs[job_id]["successful"],
        "failed": upload_jobs[job_id]["failed"]
    }

@app.get("/upload-status/{job_id}")
async def get_upload_status(job_id: str):
    job = upload_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job

# --- Optimize /rank-candidates/ endpoint to use cache ---
@app.post("/rank-candidates/")
async def rank_candidates(body: RankRequest):
    """Rank candidates against a job description with performance tracking"""
    job_description = body.job_description
    jd_id = body.jd_id
    start_time = time.time()

    # If JD ID is provided, try to use cached JD
    if jd_id:
        if jd_id in jd_cache:
            job_description = jd_cache[jd_id]["description"]
            jd_embedding = jd_cache[jd_id]["embedding"]
        else:
            # Try to fetch and cache the JD
            try:
                db_response = supabase.table('job_descriptions').select('*').eq('id', jd_id).execute()
                if db_response.data:
                    jd_data = db_response.data[0]
                    job_description = jd_data["description"]
                    jd_embedding = model.encode(job_description).tolist()
                    jd_cache[jd_id] = {
                        "title": jd_data["title"],
                        "description": job_description,
                        "embedding": jd_embedding
                    }
                else:
                    # JD not found but ID was provided, return error
                    raise HTTPException(status_code=404, detail="Job description ID not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching job description: {e}")
    else:
        # No JD ID provided, use the provided job description text
        if not job_description:
            raise HTTPException(status_code=400, detail="Job description cannot be empty.")
        jd_embedding = model.encode(job_description).tolist()

    # Vector similarity search
    try:
        query_response = pinecone_index.query(
            vector=jd_embedding,
            top_k=10,
            include_metadata=False
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

    # Use JD ID for cache key if available, otherwise use description hash
    cache_key_base = jd_id if jd_id else job_description
    cache_key_str = cache_key_base + '-' + ','.join(sorted(resume_ids))
    cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()

    analysis_times = []
    cache_hit_flag = False
    analysis_results = []
    uncached_tasks = []
    uncached_indices = []
    # First, collect cached and uncached candidates
    for idx, candidate in enumerate(top_candidates):
        candidate_id = str(candidate['candidate_data']['id'])
        cache_key_llm = get_llm_cache_key(jd_id, job_description, candidate_id)
        if cache_key_llm in llm_analysis_cache:
            analysis = llm_analysis_cache[cache_key_llm]
            analysis["cached"] = True
            cache_hit_flag = True
            analysis_times.append(analysis.get("processing_time", 0))
            analysis_results.append(analysis)
        else:
            data = candidate['candidate_data']['extracted_data']
            summary = f"Name: {data.get('name')}. Experience: {data.get('experience', '')}. Education: {data.get('education', '')}. Skills: {', '.join(data.get('skills', []))}"
            uncached_tasks.append(get_llm_analysis_async(job_description, summary, candidate['candidate_data']['filename']))
            uncached_indices.append(idx)
    # Run all uncached LLM analysis in parallel
    if uncached_tasks:
        uncached_results = await asyncio.gather(*uncached_tasks)
        for i, result in enumerate(uncached_results):
            candidate = top_candidates[uncached_indices[i]]
            candidate_id = str(candidate['candidate_data']['id'])
            cache_key_llm = get_llm_cache_key(jd_id, job_description, candidate_id)
            llm_analysis_cache[cache_key_llm] = result
            analysis_times.append(result.get("processing_time", 0))
            analysis_results.insert(uncached_indices[i], result)

    # Create a map for easy lookup
    analysis_map = {result['filename']: result for result in analysis_results}

    # Calculate average processing time if available
    avg_analysis_time = sum(analysis_times) / len(analysis_times) if analysis_times else 0

    ranked_candidates = []
    for candidate in top_candidates:
        try:
            filename = candidate["candidate_data"]["filename"]
            analysis = analysis_map.get(filename, {})

            # Handle potential errors in match_score
            if isinstance(analysis.get('match_score'), (int, float)):
                match_score = analysis['match_score']
            elif isinstance(analysis.get('match_score'), str) and analysis['match_score'].isdigit():
                match_score = int(analysis['match_score'])
            else:
                # Default score for error cases
                match_score = 50

            # Add to ranked candidates with error handling for all fields
            ranked_candidates.append({
                "candidate_data": candidate["candidate_data"],
                "ai_analysis": analysis,
                "similarity_score": candidate["similarity_score"],
                "sort_score": match_score
            })
        except Exception as e:
            print(f"Error processing candidate ranking: {str(e)}")
            # Include the candidate with a default score rather than skipping
            ranked_candidates.append({
                "candidate_data": candidate["candidate_data"],
                "ai_analysis": {
                    "match_score": 50,
                    "justification": "Error occurred during candidate analysis.",
                    "skill_matches": [],
                    "skill_gaps": [],
                    "_error": str(e)
                },
                "similarity_score": candidate["similarity_score"],
                "sort_score": 50
            })

    # Sort by the AI match score (desc), then by similarity score (desc) to break ties
    ranked_candidates.sort(key=lambda x: (x['sort_score'], x['similarity_score']), reverse=True)

    for i, candidate in enumerate(ranked_candidates):
        candidate['rank'] = i + 1
        del candidate['sort_score']

    total_time = round(time.time() - start_time, 2)

    return {
        "candidates": ranked_candidates,
        "performance_metrics": {
            "total_processing_time": total_time,
            "avg_analysis_time": round(avg_analysis_time, 2),
            "candidates_count": len(ranked_candidates),
            "cache_hit": cache_hit_flag,
            "workers_used": min(MAX_WORKERS, len(top_candidates))
        }
    }

@app.get("/")
def read_root():
    return {"status": "Resume Screener API is running.", "version": "5.3.0"}

@app.get("/test-worker-performance")
async def test_worker_performance():
    """
    Test the performance of worker processing with the current configuration
    """
    return {
        "max_workers": MAX_WORKERS,
        "rank_cache_size": len(rank_cache),
        "jd_cache_size": len(jd_cache),
        "memory_usage": {
            "upload_jobs_count": len(upload_jobs),
        }
    }

@app.get("/processing-errors")
async def check_processing_errors():
    """
    Check for processing errors in recent upload jobs
    """
    # Get only the most recent 10 jobs
    recent_jobs = sorted(
        [job for job in upload_jobs.items()],
        key=lambda x: x[1].get("start_time", 0),
        reverse=True
    )[:10]

    error_summary = []

    for job_id, job in recent_jobs:
        job_errors = []
        for result in job.get("results", []):
            if result.get("status") == "error" or result.get("notes"):
                job_errors.append({
                    "filename": result.get("filename", "Unknown"),
                    "status": result.get("status", "unknown"),
                    "detail": result.get("detail", ""),
                    "notes": result.get("notes", "")
                })

        if job_errors:
            error_summary.append({
                "job_id": job_id,
                "start_time": job.get("start_time", 0),
                "total_files": job.get("total_files", 0),
                "successful": job.get("successful", 0),
                "failed": job.get("failed", 0),
                "errors": job_errors
            })

    return {
        "recent_error_count": sum(len(job.get("errors", [])) for job in error_summary),
        "jobs_with_errors": error_summary
    }

if __name__ == "__main__":
    # Load predefined JDs from the database at startup
    try:
        db_response = supabase.table('job_descriptions').select('*').execute()
        if db_response.data:
            for jd in db_response.data:
                jd_embedding = model.encode(jd["description"]).tolist()
                jd_cache[jd["id"]] = {
                    "title": jd["title"],
                    "description": jd["description"],
                    "embedding": jd_embedding
                }
            print(f"Loaded {len(jd_cache)} predefined job descriptions into cache")
    except Exception as e:
        print(f"Error loading predefined job descriptions: {e}")

    # Setup background task for cleanup
    @app.on_event("startup")
    async def setup_periodic_cleanup():
        """Setup background task for cleaning up old job data"""
        async def cleanup_old_jobs():
            """Remove old upload jobs to prevent memory issues"""
            while True:
                try:
                    # Wait for 1 hour between cleanups
                    await asyncio.sleep(3600)

                    # Current time
                    current_time = time.time()

                    # Find jobs older than 24 hours
                    jobs_to_delete = []
                    for job_id, job in upload_jobs.items():
                        # If job is over 24 hours old or doesn't have a start time
                        if not job.get("start_time") or (current_time - job.get("start_time", 0)) > 86400:
                            jobs_to_delete.append(job_id)

                    # Delete old jobs
                    for job_id in jobs_to_delete:
                        upload_jobs.pop(job_id, None)

                    print(f"Cleanup completed: removed {len(jobs_to_delete)} old upload jobs")
                except Exception as e:
                    print(f"Error in cleanup job: {str(e)}")

        # Start the background task
        asyncio.create_task(cleanup_old_jobs())

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=300)
