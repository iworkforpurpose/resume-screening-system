import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import random
import ollama
import hashlib

# Load environment variables from a .env file
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/mistral-7b-instruct"

def parse_resume_with_llm(resume_text: str) -> dict:
    """
    Uses a Large Language Model to parse raw resume text into a structured JSON object,
    including the candidate's email address.
    """
    prompt = f"""
    You are an expert resume parser. Extract information from this resume and return ONLY valid JSON.

    **Resume Text:**
    {resume_text[:2000]}

    **Instructions:**
    1. Find the person's full name (look at the top/header of the resume)
    2. Find their email address (search for @ symbol)
    3. Extract skills (technical and soft skills)
    4. Summarize work experience
    5. Summarize education

    **IMPORTANT:** Respond ONLY with valid JSON in this exact format:
    {{
        "name": "[full name]",
        "email": "[email address or 'no-email-found@placeholder.com']",
        "skills": ["skill1", "skill2", "skill3"],
        "experience": "[work experience summary]",
        "education": "[education summary]"
    }}

    Be thorough and extract all available information. If you cannot find information, use appropriate defaults but always return valid JSON.
    """
    try:
        response = ollama.chat(model='llama3.2:3b', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ], options={
            'temperature': 0,
            'num_predict': 400
        })
        
        response_content = response['message']['content']
        # Try to parse JSON from the response
        try:
            analysis_content = json.loads(response_content)
            # Ensure email is never empty and is unique if placeholder
            if not analysis_content.get("email") or analysis_content.get("email") == "" or analysis_content.get("email") == "no-email-found@placeholder.com":
                # Generate unique placeholder email based on content hash
                content_hash = hashlib.md5(resume_text.encode()).hexdigest()[:8]
                analysis_content["email"] = f"no-email-{content_hash}@placeholder.com"
            return analysis_content
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic structure with unique email
            content_hash = hashlib.md5(resume_text.encode()).hexdigest()[:8]
            analysis_content = {
                "name": "Unknown",
                "email": f"no-email-{content_hash}@placeholder.com",
                "skills": [],
                "experience": "Could not parse experience",
                "education": "Could not parse education"
            }
            return analysis_content

    except Exception as e:
        print(f"An error occurred during LLM parsing: {e}")
        return {"error": str(e)}