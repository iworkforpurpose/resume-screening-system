"use client";

import { useState, FormEvent, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

// --- TypeScript Interfaces ---
interface ParsedData {
  name: string;
  skills: string[];
  experience: string;
  education: string;
}

interface CandidateData {
  id: string;
  filename: string;
  extracted_data: ParsedData;
  status: string;
  created_at: string;
}

interface AiAnalysis {
  match_score: number | string;
  justification: string;
  skill_matches?: string[];
  skill_gaps?: string[];
  cached?: boolean;
}

interface RankedCandidate {
  rank: number;
  candidate_data: CandidateData;
  ai_analysis: AiAnalysis;
  similarity_score: number;
}

// --- Session ID logic ---
function getSessionId() {
  if (typeof window !== 'undefined') {
    let sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = uuidv4();
      localStorage.setItem('session_id', sessionId);
    }
    return sessionId || '';
  }
  return '';
}

// --- Main Component ---
export default function HomePage() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [failedUploads, setFailedUploads] = useState<{name: string, reason: string}[]>([]);
  const [uploadProgress, setUploadProgress] = useState<{jobId: string | null, status: string, processed: number, total: number, successful: number, failed: number}>({
    jobId: null,
    status: '',
    processed: 0,
    total: 0,
    successful: 0,
    failed: 0
  });

  const [jobDescription, setJobDescription] = useState('');
  const [isRanking, setIsRanking] = useState(false);
  const [rankedCandidates, setRankedCandidates] = useState<RankedCandidate[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Poll upload status
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (uploadProgress.jobId && uploadProgress.status === 'processing') {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:8000/upload-status/${uploadProgress.jobId}`);
          if (response.ok) {
            const status = await response.json();
            setUploadProgress({
              jobId: uploadProgress.jobId,
              status: status.status,
              processed: status.processed,
              total: status.total_files,
              successful: status.successful,
              failed: status.failed
            });
            
            if (status.status === 'completed') {
              setIsUploading(false);
              setUploadMessage(`${status.successful} of ${status.total_files} resumes processed successfully!`);
              
              if (status.failed > 0) {
                const failedDetails = status.results
                  .filter((r: any) => r.status === 'error')
                  .map((r: any) => ({ name: r.filename, reason: r.detail }));
                setFailedUploads(failedDetails);
              }
              
              // Clear progress after 5 seconds
              setTimeout(() => {
                setUploadProgress({ jobId: null, status: '', processed: 0, total: 0, successful: 0, failed: 0 });
              }, 5000);
            }
          }
        } catch (err) {
          console.error('Error polling upload status:', err);
        }
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [uploadProgress.jobId, uploadProgress.status]);

  const handleUpload = async (event: FormEvent) => {
    event.preventDefault();
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setError(null);
    setUploadMessage(null);
    setFailedUploads([]);

    // Batch upload all files in one request
    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append('files', file);
    });

    const sessionId = getSessionId();

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await fetch(`${apiUrl}/batch-upload-and-process-resume/`, {
        method: 'POST',
        headers: { 'X-Session-Id': sessionId },
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An unknown error occurred.');
      }
      const data = await response.json();
      
      // Start tracking progress
      setUploadProgress({
        jobId: data.job_id,
        status: 'processing',
        processed: 0,
        total: data.total_files,
        successful: 0,
        failed: 0
      });
      
      setUploadMessage(`Upload started! Processing ${data.total_files} files in the background...`);
      
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
      setIsUploading(false);
    } finally {
      const fileInput = document.getElementById('file-upload-input') as HTMLInputElement;
      if(fileInput) fileInput.value = "";
      setFiles(null);
    }
  };

  const handleRank = async (event: FormEvent) => {
    event.preventDefault();
    if (!jobDescription.trim()) {
      setError("Please enter a job description.");
      return;
    }

    setIsRanking(true);
    setError(null);
    setRankedCandidates([]);

    const sessionId = getSessionId();

    try {
      const response = await fetch('http://localhost:8000/rank-candidates/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': sessionId },
        body: JSON.stringify({ job_description: jobDescription }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An error occurred during ranking.');
      }

      const data: RankedCandidate[] = await response.json();
      setRankedCandidates(data);

    } catch (err: any) {
      if (err.name === 'TypeError') {
        setError("Ranking failed. This is often due to a server timeout. Please ensure the backend server is running with an increased timeout.");
      } else {
        setError(err.message);
      }
    } finally {
      setIsRanking(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#f8fafc] to-[#e0e7ef] text-gray-900">
      {/* Header Bar */}
      <header className="w-full flex items-center justify-between px-6 py-4 bg-white/80 shadow-sm sticky top-0 z-10">
        <div className="flex items-center gap-2">
          {/* Removed logo image */}
          <span className="font-bold text-lg tracking-tight">AI Resume Screener</span>
          <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">AI-Powered Recruitment</span>
        </div>
        <div className="text-xs text-gray-500 font-semibold">Smart Hiring Platform <span className="ml-1 text-gray-400">Powered by advanced AI</span></div>
      </header>

      {/* Hero Section */}
      <section className="max-w-4xl mx-auto text-center py-12 md:py-20">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
          The <span className="text-blue-600">Smartest Way</span> to Screen Resumes
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Transform your hiring process with AI-powered resume analysis. Find the perfect candidates faster than ever before.
        </p>
        <div className="flex flex-col md:flex-row gap-4 justify-center">
          <div className="flex-1 bg-white rounded-xl shadow p-6 flex flex-col items-center">
            <div className="bg-blue-100 text-blue-600 rounded-full p-2 mb-2"><svg width="24" height="24" fill="none" stroke="currentColor"><path d="M13 2v8h8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 12a9 9 0 0 1 9-9h1v8a1 1 0 0 0 1 1h8v1a9 9 0 1 1-9-9z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg></div>
            <div className="font-semibold text-lg mb-1">Lightning Fast</div>
            <div className="text-gray-500 text-sm text-center">Upload and analyze dozens of resumes in under a minute</div>
          </div>
          <div className="flex-1 bg-white rounded-xl shadow p-6 flex flex-col items-center">
            <div className="bg-blue-100 text-blue-600 rounded-full p-2 mb-2"><svg width="24" height="24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10" strokeWidth="2"/><circle cx="12" cy="12" r="4" strokeWidth="2"/></svg></div>
            <div className="font-semibold text-lg mb-1">Precise Matching</div>
            <div className="text-gray-500 text-sm text-center">AI-powered skill and experience matching</div>
          </div>
          <div className="flex-1 bg-white rounded-xl shadow p-6 flex flex-col items-center">
            <div className="bg-blue-100 text-blue-600 rounded-full p-2 mb-2"><svg width="24" height="24" fill="none" stroke="currentColor"><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeWidth="2"/><rect x="7" y="9" width="10" height="8" rx="2" strokeWidth="2"/><path d="M12 3v6" strokeWidth="2"/><path d="M8 7h8" strokeWidth="2"/></svg></div>
            <div className="font-semibold text-lg mb-1">Detailed Analytics</div>
            <div className="text-gray-500 text-sm text-center">Comprehensive candidate scoring and insights</div>
          </div>
        </div>
      </section>

      {/* Main Content: Upload & Rank */}
      <section className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 pb-16">
        {/* Upload Resumes */}
        <div className="bg-white rounded-2xl shadow-lg p-8 flex flex-col gap-6">
          <h2 className="text-xl font-bold mb-2">Upload Resumes</h2>
          <p className="text-gray-500 text-sm mb-4">Upload candidate resumes for AI-powered analysis</p>
          <form onSubmit={handleUpload} className="flex flex-col gap-4">
            <label htmlFor="file-upload-input" className="border-2 border-dashed border-blue-200 rounded-lg p-8 flex flex-col items-center justify-center cursor-pointer hover:bg-blue-50 transition">
              <span className="text-blue-600 text-3xl mb-2">📄</span>
              <span className="font-medium">Drop your resumes here</span>
              <input
                id="file-upload-input"
                type="file"
                multiple
                accept="application/pdf"
                className="hidden"
                onChange={e => setFiles(e.target.files)}
              />
            </label>
            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition disabled:opacity-50"
              disabled={isUploading || !files || files.length === 0}
            >
              {isUploading ? 'Uploading...' : 'Upload file(s)'}
            </button>
            {uploadMessage && <div className="text-green-600 text-sm font-medium">{uploadMessage}</div>}
            {error && <div className="text-red-600 text-sm font-medium">{error}</div>}
            {failedUploads.length > 0 && (
              <div className="bg-red-50 border border-red-200 text-red-700 rounded p-2 text-xs mt-2">
                <div>Failed uploads:</div>
                <ul className="list-disc ml-4">
                  {failedUploads.map((f, i) => (
                    <li key={i}>{f.name}: {f.reason}</li>
                  ))}
                </ul>
              </div>
            )}
            {uploadProgress.jobId && uploadProgress.status === 'processing' && (
              <div className="w-full bg-blue-100 rounded-full h-2 mt-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${(uploadProgress.processed / uploadProgress.total) * 100}%` }}
                ></div>
              </div>
            )}
            {uploadProgress.jobId && (
              <div className="text-xs text-gray-500 mt-1">
                {uploadProgress.processed} of {uploadProgress.total} files processed
              </div>
            )}
          </form>
        </div>

        {/* Rank Candidates */}
        <div className="bg-white rounded-2xl shadow-lg p-8 flex flex-col gap-6">
          <h2 className="text-xl font-bold mb-2">Rank Candidates</h2>
          <p className="text-gray-500 text-sm mb-4">AI-powered candidate analysis and ranking</p>
          <form onSubmit={handleRank} className="flex flex-col gap-4">
            <label className="font-medium">Job Description & Requirements</label>
            <textarea
              className="border border-gray-300 rounded-lg p-3 min-h-[100px] focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Enter the job description, required skills, qualifications, and any specific requirements for this position..."
              value={jobDescription}
              onChange={e => setJobDescription(e.target.value)}
              disabled={isRanking}
            />
            <button
              type="submit"
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition disabled:opacity-50"
              disabled={isRanking || !jobDescription.trim()}
            >
              {isRanking ? 'Ranking...' : 'Rank Candidates'}
            </button>
            {error && <div className="text-red-600 text-sm font-medium">{error}</div>}
          </form>
        </div>
      </section>

      {/* Ranked Candidates Section */}
      {rankedCandidates.length > 0 && (
        <section className="max-w-6xl mx-auto pb-16">
          <h2 className="text-2xl font-bold mb-6 text-center">Top Candidate Matches</h2>
          <div className="flex flex-col gap-8">
            {rankedCandidates.map((candidate, idx) => (
              <div key={idx} className="bg-white rounded-2xl shadow-lg p-8 flex flex-col gap-4">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                  <div className="text-xl font-bold text-blue-700">{candidate.candidate_data.extracted_data.name || 'Unknown'}</div>
                  <div className="flex items-center gap-4">
                    <div className="text-lg font-semibold text-gray-700">{candidate.ai_analysis.match_score}/100</div>
                  </div>
                </div>
                <div className="text-gray-700 mb-2">{candidate.ai_analysis.justification}</div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="font-semibold mb-1">Skill Breakdown:</div>
                  <div className="flex flex-wrap gap-2 mb-1">
                    {candidate.ai_analysis.skill_matches && candidate.ai_analysis.skill_matches.length > 0 && (
                      <span className="text-green-700 bg-green-100 rounded px-2 py-0.5 text-xs">✓ Matching Skills: {candidate.ai_analysis.skill_matches.join(', ')}</span>
                    )}
                    {candidate.ai_analysis.skill_gaps && candidate.ai_analysis.skill_gaps.length > 0 && (
                      <span className="text-red-700 bg-red-100 rounded px-2 py-0.5 text-xs">✗ Missing Skills: {candidate.ai_analysis.skill_gaps.join(', ')}</span>
                    )}
                  </div>
                </div>
                <div className="text-xs text-gray-400 mt-2">Source: {candidate.candidate_data.filename} | Similarity: {candidate.similarity_score ? (candidate.similarity_score * 100).toFixed(2) + '%' : 'N/A'}</div>
              </div>
            ))}
          </div>
        </section>
      )}
      {/* Footer */}
      <footer className="w-full flex justify-center items-center py-6 mt-8">
        <span className="text-xs text-gray-400">Made by Vighnesh Nama</span>
      </footer>
    </main>
  );
}
