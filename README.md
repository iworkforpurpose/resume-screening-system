# AI Resume Screener 🤖📄

An intelligent resume screening platform that uses AI to analyze, parse, and rank candidate resumes against job descriptions. Built with FastAPI, Next.js, and powered by advanced language models.

## Features

- **Batch Resume Upload**: Upload multiple PDF resumes simultaneously
- **AI-Powered Parsing**: Extract structured data from resumes using LLM
- **Intelligent Ranking**: Rank candidates against job descriptions with match scores
- **Vector Search**: Semantic similarity matching using embeddings
- **Real-time Processing**: Background processing with live progress updates
- **Duplicate Detection**: Prevents duplicate resumes based on email addresses
- **Detailed Analytics**: Comprehensive scoring with skill gap analysis
- **Responsive UI**: Modern, intuitive interface built with React/Next.js

## Architecture

### Backend (FastAPI)
- **Resume Processing**: PDF text extraction with OCR fallback
- **AI Analysis**: OpenRouter integration for LLM-powered resume parsing and ranking
- **Vector Search**: Pinecone for semantic similarity matching
- **Database**: Supabase for structured data storage
- **Caching**: In-memory caching for improved performance

### Frontend (Next.js)
- **React Components**: Modern UI with TypeScript
- **Real-time Updates**: Progress tracking for batch uploads
- **Responsive Design**: Mobile-friendly interface
- **State Management**: React hooks for application state

### External Services
- **OpenRouter**: LLM API for resume analysis
- **Pinecone**: Vector database for similarity search
- **Supabase**: PostgreSQL database and authentication
- **OCR.space**: Fallback OCR service for poor-quality PDFs

## Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**
- **Pinecone account**
- **Supabase project**
- **OpenRouter API key**
- **OCR.space API key** (optional, for OCR fallback)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-resume-screener
```

### 2. Backend Setup

#### Navigate to Backend Directory
```bash
cd backend
```

#### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Environment Variables
Create a `.env` file in the backend directory:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key

# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key

# OCR Service (Optional)
OCR_API_KEY=your_ocr_space_api_key
```

#### Database Setup
1. Create a Supabase project
2. Create a table named `resumes` with the following schema:
```sql
CREATE TABLE resumes (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    extracted_data JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'processed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Pinecone Setup
1. Create a Pinecone index named `resumes`
2. Use dimension: `384` (for all-MiniLM-L6-v2 model)
3. Use cosine similarity metric

### 3. Frontend Setup

#### Navigate to Frontend Directory
```bash
cd ../frontend
```

#### Install Dependencies
```bash
npm install
# or
yarn install
```

#### Environment Variables
Create a `.env.local` file in the frontend directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Start Backend Server
```bash
cd backend
python main.py
```
The API will be available at `http://localhost:8000`

### Start Frontend Development Server
```bash
cd frontend
npm run dev
# or
yarn dev
```
The web application will be available at `http://localhost:3000`

## 📖 API Documentation

### Core Endpoints

#### Upload Single Resume
```http
POST /upload-and-process-resume/
Content-Type: multipart/form-data

file: PDF file
```

#### Batch Upload Resumes
```http
POST /batch-upload-and-process-resume/
Content-Type: multipart/form-data

files: Multiple PDF files
```

#### Check Upload Status
```http
GET /upload-status/{job_id}
```

#### Rank Candidates
```http
POST /rank-candidates/
Content-Type: application/json

{
  "job_description": "Job requirements and description..."
}
```

### Debug Endpoints
```http
GET /debug/database-stats     # Database statistics
GET /debug/pinecone-stats     # Vector database stats  
GET /debug/test-query         # Test Pinecone query
```

## Configuration

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `mistralai/mistral-7b-instruct` via OpenRouter
- **Vector Search**: Top-k=10 results

### Performance Settings
- **Thread Pool**: 4 workers for CPU-intensive tasks
- **Cache Duration**: 24 hours for LLM results
- **Retry Logic**: 3 attempts for failed API calls
- **Timeout**: 300 seconds for keep-alive

## How It Works

### Resume Processing Pipeline
1. **PDF Upload**: User uploads PDF resume(s)
2. **Text Extraction**: PyPDF2 extracts text, OCR fallback if needed
3. **AI Parsing**: LLM extracts structured data (name, skills, experience)
4. **Validation**: Email extraction and duplicate checking
5. **Storage**: Save to Supabase database
6. **Embedding**: Generate vector embedding and store in Pinecone

### Candidate Ranking Pipeline
1. **Job Description Input**: User provides job requirements
2. **Vector Search**: Find similar candidates using embeddings
3. **Data Retrieval**: Fetch candidate details from database
4. **AI Analysis**: LLM analyzes candidate-job fit
5. **Scoring**: Generate match scores and skill gap analysis
6. **Ranking**: Sort by AI match score and similarity

## Testing

### Backend Testing
```bash
cd backend
pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
# or
yarn test
```

## Data Flow

```
PDF Upload → Text Extraction → AI Parsing → Database Storage
                                    ↓
                            Vector Embedding → Pinecone Storage
                                    ↓
Job Description → Vector Search → Candidate Retrieval → AI Ranking
```

## Troubleshooting

### Common Issues

#### Backend Issues
- **Pinecone Connection**: Verify API key and index name
- **Supabase Connection**: Check URL and API key
- **OpenRouter Limits**: Monitor API usage and rate limits
- **OCR Failures**: Ensure OCR.space API key is valid

#### Frontend Issues
- **CORS Errors**: Verify backend CORS configuration
- **API Timeouts**: Increase timeout settings for large uploads
- **State Issues**: Clear browser cache and localStorage

#### Performance Issues
- **Slow Processing**: Reduce batch size or increase worker threads
- **Memory Usage**: Monitor embedding model memory consumption
- **Database Queries**: Optimize Supabase queries and indexing

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

- **API Keys**: Store in environment variables, never commit to version control
- **File Validation**: Only accept PDF files, validate file size
- **Rate Limiting**: Implement API rate limiting for production
- **Data Privacy**: Ensure compliance with data protection regulations
- **CORS**: Configure appropriate CORS policies

## Deployment

### Backend Deployment (Docker)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment (Vercel/Netlify)
```bash
npm run build
npm start
```

## Monitoring

### Key Metrics
- **Upload Success Rate**: Monitor failed uploads
- **Processing Time**: Track resume processing duration
- **API Response Times**: Monitor endpoint performance
- **Error Rates**: Track and alert on errors
- **Database Performance**: Monitor query execution times

### Logging
- **Structured Logging**: Use JSON format for production
- **Error Tracking**: Implement error tracking service
- **Performance Monitoring**: Use APM tools

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Write tests for new features
- Update documentation for API changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

**Created by**: Vighnesh Nama

### Technologies Used
- **Backend**: FastAPI, Python, uvicorn
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **AI/ML**: OpenRouter, Sentence Transformers, Pinecone
- **Database**: Supabase (PostgreSQL)
- **Additional**: PyPDF2, OCR.space, UUID

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review API documentation
3. Create an issue in the repository
4. Contact the development team

---

**Happy Screening! **
