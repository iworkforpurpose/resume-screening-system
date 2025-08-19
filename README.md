# Automated Resume Screening and Candidate Ranking System

## Overview
This project is an end-to-end solution for automated resume screening and candidate ranking, designed to simulate how modern hiring tools prioritize candidates for recruiters. It leverages state-of-the-art NLP models to parse, embed, and rank resumes based on a given job description, providing explainable match scores and justifications for each candidate.

---

## Key Features

- **Batch Resume Upload:** Upload and process multiple PDF resumes at once.
- **LLM-Based Resume Parsing:** Extracts structured candidate data (skills, experience, education, etc.) from unstructured resumes using large language models.
- **Semantic Embedding & Vector Search:** Uses Sentence Transformers and Pinecone for fast, context-aware candidate retrieval.
- **Job Description Management:** Supports both manual input and predefined (cached) job descriptions for efficient, repeatable screening.
- **Detailed AI Analysis:** Each candidate is scored and analyzed by an LLM, with a comprehensive breakdown (skills, gaps, experience, education, role fit, career trajectory, justification, and interview recommendations).
- **Aggressive Caching:** Caches LLM analysis for (JD, candidate) pairs to minimize redundant computation and speed up repeated queries.
- **Resource-Efficient Batch Processing:** Optimized for low-resource servers (max 2 workers, sequential fallback).
- **Error Handling & Monitoring:** Robust error handling, metrics reporting, and endpoints for monitoring processing errors.
- **Modern Frontend:** Next.js/React UI for uploading, tracking, and viewing ranked candidates.

---

## Workflow

1. **Resume Upload:**  
   - Users upload one or more PDF resumes via the web interface.
   - Each resume is parsed using an LLM to extract structured candidate data.
   - Embeddings are generated for each candidate and stored in Pinecone for similarity search.

2. **Job Description Input:**  
   - Users can manually enter a job description or select from predefined/cached JDs.
   - The JD is embedded and used to query Pinecone for the most relevant candidates.

3. **Candidate Ranking:**  
   - The system retrieves the top matching candidates using semantic search.
   - For each candidate, an LLM generates a match score (0–100) and a detailed analysis (skills, gaps, experience, education, role fit, career trajectory, justification, and interview recommendations).
   - Results are cached for repeated queries with the same JD and candidate.

4. **Results Display:**  
   - The frontend displays ranked candidates, scores, justifications, and analysis breakdowns.
   - Performance metrics (total time, average analysis time, cache hits, etc.) are shown for transparency.

---

## ⚙️ Tech Stack & Architecture  

This project implements a **custom Retrieval-Augmented Generation (RAG) pipeline** for resume screening and candidate scoring.  
Unlike LangChain or LlamaIndex, the system uses lightweight, transparent glue code to integrate tools directly — keeping it simple, cost-efficient, and highly customizable.  

---

### 🔑 Core Components  

1. **Frontend (Next.js + TypeScript)**  
   - Handles resume upload and JD submission.  
   - Displays ranked candidates, analysis justifications, and performance metrics.  

2. **Backend (FastAPI + Uvicorn)**  
   - Provides REST APIs for uploading, processing, ranking, and polling job status.  
   - Orchestrates the resume parsing, embedding, and scoring pipeline.  

3. **Resume Parsing**  
   - **PyPDF2** → Extracts text from digital PDFs.  
   - **OCR.space API** → Fallback for scanned/image resumes.  
   - Ensures robust text extraction for all resume formats.  

4. **LLM Processing (OpenRouter + Mistral-7B)**  
   - Converts raw resume text into structured JSON fields (skills, experience, education, etc.).  
   - Repairs malformed JSON and falls back to regex extraction if needed.  
   - Performs candidate scoring: outputs  
     - `match_score` (0–100)  
     - skills fit/gap  
     - education fit  
     - career trajectory  
     - justification  
     - interview recommendation  

5. **Embeddings & Retrieval (RAG)**  
   - **SentenceTransformers (MiniLM)** → Generates vector embeddings for resumes and job descriptions.  
   - **Pinecone** → Vector database for semantic search. Retrieves top-K candidate resumes relevant to a JD.  
   - This is the **retrieval step** of the RAG pipeline.  

6. **Metadata & Persistence**  
   - **Supabase (Postgres)** → Stores structured candidate metadata, job descriptions, and processing states.  

7. **Ranking & Caching**  
   - After retrieval, each candidate is passed to the LLM for scoring + explanation.  
   - Results are cached for `(JD, candidate)` pairs to reduce latency and API costs.  
   - Parallel processing via **ThreadPoolExecutor** allows multiple resumes to be processed concurrently.  

---

### 🔄 Custom RAG Workflow  

1. **Retrieve**  
   - Embed the job description.  
   - Query Pinecone to retrieve semantically similar resumes.  

2. **Augment**  
   - Combine JD + candidate’s structured resume into an LLM prompt.  

3. **Generate**  
   - LLM outputs a structured analysis with a numeric match score and justification.  

This lightweight RAG implementation avoids heavy abstractions (e.g., LangChain), while retaining full control over prompts, embeddings, and caching.  

---

### ❓ Why not LangChain or LlamaIndex?  

- **Transparency** → Direct control over parsing, embedding, retrieval, and scoring.  
- **Performance** → Avoid overhead from generic chains and unused features.  
- **Cost-efficiency** → Custom caching prevents duplicate LLM calls.  
- **Flexibility** → Easy to swap models (change embedding models or LLM providers).  

---

## API Endpoints

- `POST /job-descriptions/` — Create a new job description
- `GET /job-descriptions/` — List all job descriptions
- `GET /job-descriptions/{jd_id}` — Retrieve a specific job description
- `POST /upload-and-process-resume/` — Upload and process a single resume
- `POST /batch-upload-and-process-resume/` — Upload and process multiple resumes
- `GET /upload-status/{job_id}` — Check the status of an upload job
- `POST /rank-candidates/` — Rank candidates for a job description (manual or predefined)
- `GET /test-worker-performance` — Check worker configuration and cache stats
- `GET /processing-errors` — View recent processing errors

---

## Setup & Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/iworkforpurpose/resume-screening-system.git
   cd resume-screening-system
   ```

2. **Backend Setup:**
   - Install Python dependencies:
     ```bash
     cd backend
     pip install -r requirements.txt
     ```
   - Set up environment variables in `backend/.env`:
     - `SUPABASE_URL`, `SUPABASE_KEY`, `PINECONE_API_KEY`, `OCR_API_KEY`, `OPENROUTER_API_KEY`
   - Run the backend:
     ```bash
     uvicorn main:app --reload
     ```

3. **Frontend Setup:**
   - Install Node.js dependencies:
     ```bash
     cd frontend/robust-frontend
     npm install
     ```
   - Run the frontend:
     ```bash
     npm run dev
     ```
   - Access the app at [http://localhost:3000]

---

## Performance

- **Batch Upload:** ~1 minute for 20+ resumes (on low-resource servers)
- **Ranking:** ~10 seconds for top 10 candidates (LLM + Pinecone)
- **Scalability:** Designed for 100+ resumes with minimal changes.

---

## Example Usage

1. Upload a batch of PDF resumes via the web interface.
2. Enter or select a job description (manual or predefined).
3. Click "Rank Candidates" to view the top matches, scores, justifications, and analysis.

---

For questions or issues, please open an issue on GitHub.
