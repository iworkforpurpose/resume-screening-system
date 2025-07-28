"use client";

import { useState, FormEvent, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

// --- TypeScript Interfaces ---
interface ParsedData {
  name: string;
  skills?: string[];
  technical_skills?: string[];
  soft_skills?: string[];
  domain_knowledge?: string[];
  experience: string;
  education: string;
  years_of_experience?: string;
  phone?: string;
  location?: string;
  linkedin_url?: string;
  certifications?: string[];
  projects?: string[];
}

interface CandidateData {
  id: string;
  filename: string;
  extracted_data: ParsedData;
  status: string;
  created_at: string;
}

interface SkillMatch {
  skill: string;
  level: string;
}

interface AiAnalysis {
  match_score: number | string;
  justification: string;
  skill_matches?: SkillMatch[] | string[];
  skill_gaps?: string[];
  experience_assessment?: string;
  education_fit?: string;
  role_alignment?: string;
  career_trajectory?: string;
  interview_recommendations?: string[];
  processing_time?: number;
  cached?: boolean;
}

interface RankedCandidate {
  rank: number;
  candidate_data: CandidateData;
  ai_analysis: AiAnalysis;
  similarity_score: number;
}

interface JobDescription {
  id: string;
  title: string;
  description: string;
}

interface RankingResponse {
  candidates: RankedCandidate[];
  performance_metrics: {
    total_processing_time: number;
    avg_analysis_time: number;
    candidates_count: number;
    cache_hit: boolean;
    workers_used: number;
  }
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
  const [uploadProgress, setUploadProgress] = useState<{jobId: string | null, status: string, processed: number, total: number, successful: number, failed: number, totalTime?: number}>({
    jobId: null,
    status: '',
    processed: 0,
    total: 0,
    successful: 0,
    failed: 0
  });

  const [jobDescription, setJobDescription] = useState('');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobDescriptions, setJobDescriptions] = useState<JobDescription[]>([]);
  const [isLoadingJds, setIsLoadingJds] = useState(false);
  const [isCreatingJd, setIsCreatingJd] = useState(false);
  const [newJdTitle, setNewJdTitle] = useState('');
  const [showCreateJdForm, setShowCreateJdForm] = useState(false);
  
  const [isRanking, setIsRanking] = useState(false);
  const [rankedCandidates, setRankedCandidates] = useState<RankedCandidate[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<{
    total_processing_time: number;
    avg_analysis_time: number;
    candidates_count: number;
    cache_hit: boolean;
    workers_used: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch job descriptions on page load
  useEffect(() => {
    fetchJobDescriptions();
  }, []);

  // Fetch job descriptions from the backend
  const fetchJobDescriptions = async () => {
    setIsLoadingJds(true);
    try {
      const response = await fetch('http://localhost:8000/job-descriptions/');
      if (!response.ok) {
        throw new Error('Failed to fetch job descriptions');
      }
      const data = await response.json();
      setJobDescriptions(data);
    } catch (err) {
      console.error('Error fetching job descriptions:', err);
    } finally {
      setIsLoadingJds(false);
    }
  };

  // Create a new job description
  const handleCreateJobDescription = async (e: FormEvent) => {
    e.preventDefault();
    if (!jobDescription.trim() || !newJdTitle.trim()) {
      setError("Please enter both a title and description for the job.");
      return;
    }

    setIsCreatingJd(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/job-descriptions/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          title: newJdTitle,
          description: jobDescription 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create job description');
      }

      const newJd = await response.json();
      setJobDescriptions([...jobDescriptions, newJd]);
      setSelectedJobId(newJd.id);
      setNewJdTitle('');
      setShowCreateJdForm(false);
      setUploadMessage("Job description saved successfully!");
      
      // Hide message after 3 seconds
      setTimeout(() => {
        setUploadMessage(null);
      }, 3000);
      
    } catch (err: any) {
      setError(err.message || 'An error occurred while creating the job description');
    } finally {
      setIsCreatingJd(false);
    }
  };

  // Handle job description selection
  const handleSelectJobDescription = async (id: string) => {
    setSelectedJobId(id);
    setJobDescription('');
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:8000/job-descriptions/${id}`);
      if (!response.ok) {
        throw new Error('Failed to fetch job description');
      }
      const data = await response.json();
      setJobDescription(data.description);
    } catch (err: any) {
      setError(err.message || 'An error occurred while fetching the job description');
      setSelectedJobId(null);
    }
  };

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
              failed: status.failed,
              totalTime: status.total_time
            });
            
            if (status.status === 'completed') {
              setIsUploading(false);
              setUploadMessage(`${status.successful} of ${status.total_files} resumes processed successfully${status.total_time ? ` in ${status.total_time} seconds` : ''}!`);
              
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
      const response = await fetch('http://localhost:8000/batch-upload-and-process-resume/', {
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
    if (!jobDescription.trim() && !selectedJobId) {
      setError("Please enter a job description or select a saved one.");
      return;
    }

    setIsRanking(true);
    setError(null);
    setRankedCandidates([]);
    setPerformanceMetrics(null);

    const sessionId = getSessionId();

    try {
      const response = await fetch('http://localhost:8000/rank-candidates/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': sessionId },
        body: JSON.stringify({ 
          job_description: jobDescription,
          jd_id: selectedJobId
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An error occurred during ranking.');
      }

      const data: RankingResponse = await response.json();
      setRankedCandidates(data.candidates);
      setPerformanceMetrics(data.performance_metrics);

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
              <div className="text-xs text-gray-500 mt-1 flex justify-between">
                <span>{uploadProgress.processed} of {uploadProgress.total} files processed</span>
                {uploadProgress.totalTime && (
                  <span>Total time: {uploadProgress.totalTime}s</span>
                )}
              </div>
            )}
            {uploadProgress.jobId && uploadProgress.status === 'completed' && uploadProgress.successful > 0 && (
              <div className="bg-green-50 border border-green-100 rounded p-2 text-xs mt-2">
                <div className="font-medium text-green-700">Processing complete!</div>
                <div className="text-green-600">
                  Successfully processed {uploadProgress.successful} of {uploadProgress.total} files
                  {uploadProgress.totalTime && ` in ${uploadProgress.totalTime} seconds`}.
                </div>
              </div>
            )}
          </form>
        </div>

        {/* Rank Candidates */}
        <div className="bg-white rounded-2xl shadow-lg p-8 flex flex-col gap-6">
          <h2 className="text-xl font-bold mb-2">Rank Candidates</h2>
          <p className="text-gray-500 text-sm mb-4">AI-powered candidate analysis and ranking</p>
          
          {/* Job Description Selection */}
          <div className="mb-4">
            <label className="font-medium">Saved Job Descriptions</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {isLoadingJds ? (
                <div className="text-sm text-gray-500">Loading job descriptions...</div>
              ) : jobDescriptions.length > 0 ? (
                jobDescriptions.map(jd => (
                  <button
                    key={jd.id}
                    onClick={() => handleSelectJobDescription(jd.id)}
                    className={`px-3 py-1 text-sm rounded-full transition ${
                      selectedJobId === jd.id 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-blue-100 text-blue-800 hover:bg-blue-200'
                    }`}
                  >
                    {jd.title}
                  </button>
                ))
              ) : (
                <div className="text-sm text-gray-500">No saved job descriptions</div>
              )}
              <button 
                onClick={() => {
                  setShowCreateJdForm(!showCreateJdForm);
                  setSelectedJobId(null);
                }}
                className="px-3 py-1 text-sm rounded-full bg-gray-100 text-gray-800 hover:bg-gray-200 transition"
              >
                {showCreateJdForm ? 'Cancel' : '+ New Job'}
              </button>
            </div>
          </div>
          
          {showCreateJdForm && (
            <div className="bg-blue-50 p-4 rounded-lg mb-4">
              <h3 className="font-medium mb-2">Save New Job Description</h3>
              <form onSubmit={handleCreateJobDescription} className="flex flex-col gap-3">
                <input
                  type="text"
                  placeholder="Job Title (e.g. Senior Software Engineer)"
                  className="border border-gray-300 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
                  value={newJdTitle}
                  onChange={e => setNewJdTitle(e.target.value)}
                  disabled={isCreatingJd}
                />
                <button
                  type="submit"
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-1 rounded-lg transition disabled:opacity-50"
                  disabled={isCreatingJd || !jobDescription.trim() || !newJdTitle.trim()}
                >
                  {isCreatingJd ? 'Saving...' : 'Save Job Description'}
                </button>
              </form>
            </div>
          )}
          
          <form onSubmit={handleRank} className="flex flex-col gap-4">
            <label className="font-medium">Job Description & Requirements</label>
            <textarea
              className="border border-gray-300 rounded-lg p-3 min-h-[100px] focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Enter the job description, required skills, qualifications, and any specific requirements for this position..."
              value={jobDescription}
              onChange={e => {
                setJobDescription(e.target.value);
                setSelectedJobId(null);
              }}
              disabled={isRanking}
            />
            <button
              type="submit"
              className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition disabled:opacity-50"
              disabled={isRanking || (!jobDescription.trim() && !selectedJobId)}
            >
              {isRanking ? 'Ranking...' : 'Rank Candidates'}
            </button>
            {error && <div className="text-red-600 text-sm font-medium">{error}</div>}
          </form>
        </div>
      </section>

      {/* Ranked Candidates Section */}
      {rankedCandidates && rankedCandidates.length > 0 && (
        <section className="max-w-6xl mx-auto pb-16">
          <h2 className="text-2xl font-bold mb-6 text-center">Top Candidate Matches</h2>
          
          {/* Performance Metrics */}
          {performanceMetrics && (
            <div className="bg-gray-50 p-4 rounded-lg mb-6">
              <h3 className="font-semibold mb-2">Processing Metrics</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                <div className="bg-white p-2 rounded shadow">
                  <div className="text-gray-500">Total Time</div>
                  <div className="font-medium">{performanceMetrics.total_processing_time}s</div>
                </div>
                <div className="bg-white p-2 rounded shadow">
                  <div className="text-gray-500">Avg Analysis</div>
                  <div className="font-medium">{performanceMetrics.avg_analysis_time}s</div>
                </div>
                <div className="bg-white p-2 rounded shadow">
                  <div className="text-gray-500">Candidates</div>
                  <div className="font-medium">{performanceMetrics.candidates_count}</div>
                </div>
                <div className="bg-white p-2 rounded shadow">
                  <div className="text-gray-500">Workers</div>
                  <div className="font-medium">{performanceMetrics.workers_used}</div>
                </div>
                <div className="bg-white p-2 rounded shadow">
                  <div className="text-gray-500">Cache Hit</div>
                  <div className="font-medium">{performanceMetrics.cache_hit ? 'Yes' : 'No'}</div>
                </div>
              </div>
            </div>
          )}
          
          <div className="flex flex-col gap-8">
            {rankedCandidates.map((candidate, idx) => (
              <div key={idx} className="bg-white rounded-2xl shadow-lg p-8 flex flex-col gap-4">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                  <div className="text-xl font-bold text-blue-700">{candidate.candidate_data.extracted_data.name || 'Unknown'}</div>
                  <div className="flex items-center gap-4">
                    <div className="px-3 py-1 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
                      Rank #{candidate.rank}
                    </div>
                    <div className="text-lg font-semibold text-gray-700">Match Score: {candidate.ai_analysis.match_score}/100</div>
                  </div>
                </div>
                
                {/* Main content */}
                <div className="text-gray-700 mb-2 bg-blue-50 p-4 rounded-lg">{candidate.ai_analysis.justification}</div>
                
                {/* Analysis Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                  {/* Skill matches with levels */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="font-semibold mb-1">Skill Matches:</div>
                    <div className="flex flex-wrap gap-2 mb-1">
                      {Array.isArray(candidate.ai_analysis.skill_matches) && candidate.ai_analysis.skill_matches.length > 0 && (
                        <div className="grid grid-cols-1 gap-1 w-full">
                          {candidate.ai_analysis.skill_matches.map((skill, i) => {
                            if (typeof skill === 'string') {
                              return (
                                <span key={i} className="text-green-700 bg-green-50 rounded px-2 py-1 text-xs">
                                  ✓ {skill}
                                </span>
                              );
                            } else {
                              return (
                                <span key={i} className="text-green-700 bg-green-50 rounded px-2 py-1 text-xs flex justify-between">
                                  <span>✓ {skill.skill}</span>
                                  <span className={
                                    skill.level === 'Expert' ? 'text-green-700' : 
                                    skill.level === 'Intermediate' ? 'text-blue-700' : 
                                    'text-gray-700'
                                  }>
                                    {skill.level}
                                  </span>
                                </span>
                              );
                            }
                          })}
                        </div>
                      )}
                      {(!candidate.ai_analysis.skill_matches || candidate.ai_analysis.skill_matches.length === 0) && 
                        <span className="text-gray-500 text-xs">No matching skills found</span>
                      }
                    </div>
                  </div>
                  
                  {/* Skill gaps */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="font-semibold mb-1">Skill Gaps:</div>
                    <div className="flex flex-wrap gap-2 mb-1">
                      {candidate.ai_analysis.skill_gaps && candidate.ai_analysis.skill_gaps.length > 0 && (
                        <div className="grid grid-cols-1 gap-1 w-full">
                          {candidate.ai_analysis.skill_gaps.map((skill, i) => (
                            <span key={i} className="text-red-700 bg-red-50 rounded px-2 py-1 text-xs">
                              ✗ {skill}
                            </span>
                          ))}
                        </div>
                      )}
                      {(!candidate.ai_analysis.skill_gaps || candidate.ai_analysis.skill_gaps.length === 0) && 
                        <span className="text-gray-500 text-xs">No skill gaps identified</span>
                      }
                    </div>
                  </div>
                  
                  {/* Experience assessment */}
                  {candidate.ai_analysis.experience_assessment && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="font-semibold mb-1">Experience Assessment:</div>
                      <div className="text-sm text-gray-700">{candidate.ai_analysis.experience_assessment}</div>
                    </div>
                  )}
                  
                  {/* Education fit */}
                  {candidate.ai_analysis.education_fit && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="font-semibold mb-1">Education Fit:</div>
                      <div className="text-sm text-gray-700">{candidate.ai_analysis.education_fit}</div>
                    </div>
                  )}
                  
                  {/* Role alignment */}
                  {candidate.ai_analysis.role_alignment && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="font-semibold mb-1">Role Alignment:</div>
                      <div className="text-sm text-gray-700">{candidate.ai_analysis.role_alignment}</div>
                    </div>
                  )}
                  
                  {/* Career trajectory */}
                  {candidate.ai_analysis.career_trajectory && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="font-semibold mb-1">Career Trajectory:</div>
                      <div className="text-sm text-gray-700">{candidate.ai_analysis.career_trajectory}</div>
                    </div>
                  )}
                </div>
                
                {/* Interview recommendations */}
                {candidate.ai_analysis.interview_recommendations && candidate.ai_analysis.interview_recommendations.length > 0 && (
                  <div className="bg-amber-50 rounded-lg p-4 mt-2">
                    <div className="font-semibold mb-1">Suggested Interview Questions:</div>
                    <ul className="list-disc pl-5 text-sm">
                      {candidate.ai_analysis.interview_recommendations.map((question, i) => (
                        <li key={i} className="text-gray-700">{question}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Metadata */}
                <div className="flex flex-wrap justify-between text-xs text-gray-400 mt-4">
                  <div>Source: {candidate.candidate_data.filename}</div>
                  <div>Similarity: {candidate.similarity_score ? (candidate.similarity_score * 100).toFixed(2) + '%' : 'N/A'}</div>
                  {candidate.ai_analysis.processing_time && <div>Processing time: {candidate.ai_analysis.processing_time}s</div>}
                </div>
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
