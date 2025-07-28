# Automated Resume Screening and Candidate Ranking System

## Overview
This project is an end-to-end solution for automated resume screening and candidate ranking, designed to simulate how modern hiring tools prioritize candidates for recruiters. It leverages state-of-the-art NLP models to parse, embed, and rank resumes based on a given job description, providing explainable match scores and justifications for each candidate.

## Features
- **Batch Resume Upload:** Upload multiple resumes (PDF) at once with real-time progress tracking.
- **Robust Parsing:** Handles noisy/unstructured resume data, extracting name, skills, experience, and education using LLMs.
- **Semantic Embedding:** Uses Sentence Transformers (MiniLM) to generate embeddings for resumes and job descriptions.
- **Vector Search:** Stores embeddings in Pinecone for fast similarity search and retrieval.
- **LLM-Based Ranking:** Uses open-source LLMs (Mistral-7B, Llama 3 via Ollama) to generate match scores (0–100) and detailed justifications (~100 words) for each candidate.
- **Bias Mitigation:** Avoids over-weighting keywords by using context-aware LLM analysis.
- **Explainability:** Provides clear, recruiter-friendly explanations for each match score.
- **Fast Retrieval:** Returns top 10 candidates in under 20 seconds for 22+ resumes.
- **Modern Web Interface:** Built with Next.js (React), featuring upload, progress, and ranked results with score breakdowns.
- **Predefined Job Descriptions:** Support for storing and reusing job descriptions to reduce processing time.
- **Optimized Worker Configuration:** Balanced resource usage with configurable worker count.

## Tech Stack
- **Backend:** Python, FastAPI
- **NLP Models:**
  - Sentence Transformers (MiniLM, HuggingFace)
  - LLMs via Ollama (Llama 3, Mistral-7B)
- **Vector Database:** Pinecone
- **Database:** Supabase (Postgres)
- **Frontend:** Next.js (React, TypeScript)
- **PDF Parsing:** PyPDF2, OCR.space API (fallback)
- **Deployment:** (To be added: e.g., Vercel for frontend, Render/Heroku for backend)

## How It Works
1. **Upload Resumes:**
   - Upload PDFs in batch. Each file is parsed using an LLM to extract structured data (name, skills, experience, education).
   - Embeddings are generated for each resume and stored in Pinecone.
2. **Job Description Input:**
   - User enters a job description in the web interface.
   - The description is embedded and used to query Pinecone for the top 10 most similar resumes.
3. **Candidate Ranking:**
   - For each retrieved candidate, an LLM generates a match score (0–100) and a ~50-word justification.
   - Confidence is calculated based on data quality and completeness.
   - Candidates are ranked: primarily by match score, and by confidence if scores are tied.
4. **Results Display:**
   - The frontend displays ranked candidates, their scores, justifications, and a detailed skill/experience breakdown.

## Addressing the Assignment Requirements
- **Noisy/Unstructured Data:**
  - LLM-based parsing and preprocessing handle typos, inconsistent formatting, and missing fields.
- **Bias Mitigation:**
  - LLMs analyze context, not just keywords, and justifications explain the reasoning.
- **Fast Retrieval:**
  - Pinecone vector search and async processing ensure <1 min upload for 22 resumes and <20s ranking for 10+ candidates.
- **Explainability:**
  - Each candidate has a human-readable justification and a score breakdown for recruiters.

## New Features in v5.3.0

### 1. Predefined Job Descriptions with Caching
- Store and reuse job descriptions to reduce processing time
- Automatically cache JD embeddings for faster retrieval
- Use JD by ID instead of pasting the full text each time

### 2. Optimized Worker Configuration
- Reduced worker count from 4 to 2 for better resource utilization
- Works well on basic EC2 instances (1 CPU, 2GB RAM)
- Sequential processing with controlled parallelism

### 3. Enhanced Analysis Prompts
- More detailed candidate evaluation with proficiency levels
- Career trajectory and role alignment assessments
- Interview recommendations based on candidate background
- Processing time metrics for performance monitoring

### 4. Performance Testing

To test the performance of the new worker configuration:

1. Start the backend server:
   ```
   cd backend
   uvicorn main:app --reload
   ```

2. Use the test endpoint to check worker configuration:
   ```
   curl http://localhost:8000/test-worker-performance
   ```

3. Upload a batch of resumes and monitor processing times:
   - The system now reports processing time for each resume
   - Check the average processing time in the response

4. Compare times with different worker configurations:
   - Edit `MAX_WORKERS` in `main.py` to test with 1 or 2 workers
   - Default is now set to 2 workers for optimal balance

## API Endpoints

### Job Description Management
- `POST /job-descriptions/` - Create a new job description
- `GET /job-descriptions/` - List all job descriptions
- `GET /job-descriptions/{jd_id}` - Get a specific job description

### Resume Processing
- `POST /upload-and-process-resume/` - Upload and process a single resume
- `POST /batch-upload-and-process-resume/` - Upload and process multiple resumes
- `GET /upload-status/{job_id}` - Check the status of an upload job

### Candidate Ranking
- `POST /rank-candidates/` - Rank candidates against a job description (with optional JD ID)

### Testing
- `GET /test-worker-performance` - Check worker configuration and cache statistics

## Setup & Usage
1. **Clone the Repository:**
   ```bash
   git clone <this-repo-url>
   cd robust_screener
   ```
2. **Backend Setup:**
   - Install Python dependencies:
     ```bash
     cd backend
     pip install -r requirements.txt
     ```
   - Set up environment variables (`.env`):
     - `SUPABASE_URL`, `SUPABASE_KEY`, `PINECONE_API_KEY`, `OCR_API_KEY`
   - Start Ollama and pull the required models (e.g., `ollama pull llama3.2:3b`)
   - Run the backend:
     ```bash
     python main.py
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
   - Access the app at [http://localhost:3000](http://localhost:3000)

## Deployment
- **Frontend:** Deploy to Vercel or Netlify for instant preview.
- **Backend:** Deploy to Render, Heroku, or your preferred cloud provider.
- **Ollama/LLM:** Ensure the backend has access to the Ollama server and required models.
- **Supabase/Pinecone:** Use managed cloud services for production reliability.

## Evaluation
- **Upload Time:** ~1 minute for 22 resumes (batch upload, parallel processing)
- **Ranking Time:** ~20 seconds for top 10 candidates (LLM + Pinecone)
- **Scalability:** Designed to handle 100+ resumes with minimal changes.

## Example Usage
1. Upload a batch of resumes (PDFs) via the web interface.
2. Enter a job description (e.g., "Data Scientist, 3+ years experience, Python, TensorFlow, cloud platforms, strong communication skills").
3. Click "Rank Candidates" to view the top 10 matches, their scores, justifications, and skill breakdowns.

## Screenshots
![Candidate Ranking Screenshot](frontend/robust-frontend/public/screenshots/candidate-ranking.png)

## License
MIT

---

**For questions or issues, please open an issue on GitHub.**
