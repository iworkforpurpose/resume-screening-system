import os
import json
import requests
from dotenv import load_dotenv
import hashlib

# Load environment variables from a .env file
load_dotenv()

HF_API_KEY = os.environ.get("HF_API_KEY")  # <-- Add your Hugging Face API key to your .env
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")  # You can change the model here

def parse_resume_with_llm(resume_text: str) -> dict:
    """
    Uses Hugging Face Inference API to parse raw resume text into a structured JSON object,
    including the candidate's email address.
    """
    prompt = f"""
    You are an expert resume parser. Extract information from this resume and return ONLY valid JSON.

    **Resume Text:**
    {resume_text[:2000]}

    **Instructions:**
    1. Find the person's full name (look at the top/header of the resume)
    2. Find their email address (look for @ symbol)
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
        # Hugging Face Inference API call
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
        # The result is a list of dicts with 'generated_text'
        if isinstance(result, list) and 'generated_text' in result[0]:
            response_content = result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            response_content = result['generated_text']
        else:
            response_content = str(result)
        # Try to parse JSON from the response
        try:
            analysis_content = json.loads(response_content)
            # Ensure email is never empty and is unique if placeholder
            if not analysis_content.get("email") or analysis_content.get("email") == "" or analysis_content.get("email") == "no-email-found@placeholder.com":
                content_hash = hashlib.md5(resume_text.encode()).hexdigest()[:8]
                analysis_content["email"] = f"no-email-{content_hash}@placeholder.com"
            return analysis_content
        except json.JSONDecodeError:
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