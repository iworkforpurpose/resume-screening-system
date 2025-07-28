import os
import json
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/mistral-7b-instruct"
MAX_RETRIES = 2  # Number of retries for JSON parsing issues

def fix_malformed_json(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues that can occur with LLM responses
    with enhanced error recovery for complex edge cases
    """
    # Remove any markdown code block markers
    json_str = re.sub(r'```(?:json)?|```', '', json_str).strip()
    
    # Try to identify and extract a JSON block if it's surrounded by text
    json_pattern = r'(\{[\s\S]*\})'
    json_matches = re.search(json_pattern, json_str)
    if json_matches:
        potential_json = json_matches.group(1)
        if len(potential_json) > 100:  # Only use if it's substantial enough to be valid JSON
            json_str = potential_json

    # Fix unclosed quotes - count quotes and ensure they're balanced
    open_quotes = json_str.count('"')
    if open_quotes % 2 != 0:
        json_str += '"'
    
    # Try to fix missing closing braces/brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    # Fix trailing commas before closing braces or brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Fix missing quotes around keys
    json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
    
    # Fix missing commas between key-value pairs
    json_str = re.sub(r'("(?:\\.|[^"\\])*")\s*:\s*("(?:\\.|[^"\\])*"|\d+(?:\.\d+)?|\{(?:[^{}]|\{[^{}]*\})*\}|\[(?:[^\[\]]|\[[^\[\]]*\])*\]|true|false|null)\s*([a-zA-Z_])', r'\1: \2, \3', json_str)
    
    # Fix boolean values and null values that might not be lowercase
    json_str = re.sub(r'"(True|False|TRUE|FALSE)"', lambda m: m.group(1).lower(), json_str)
    json_str = re.sub(r'"(None|NULL|Null)"', 'null', json_str)
    
    # Fix single quotes used instead of double quotes (carefully)
    if (json_str.count("'") > json_str.count('"') * 2) and (json_str.count('"') < 10):
        # This suggests single quotes are being used predominantly
        # First, escape any existing double quotes
        json_str = json_str.replace('\\"', '__ESCAPED_DOUBLE_QUOTE__')
        json_str = json_str.replace('"', '\\"')
        # Now replace single quotes with double quotes (but not escaped ones)
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        json_str = re.sub(r"\\'", "'", json_str)
        # Restore escaped double quotes
        json_str = json_str.replace('__ESCAPED_DOUBLE_QUOTE__', '\\"')
    
    return json_str

def extract_fallback_json(text: str) -> dict:
    """
    Last resort fallback extraction for when JSON parsing completely fails.
    Tries to extract key fields using regex patterns.
    """
    result = {
        "_error_details": "Fallback extraction used",
        "parsing_time": 0  # Will be updated by caller
    }
    
    # Try to extract name
    name_patterns = [
        r"[\"']name[\"']\s*:\s*[\"']([^\"']+)[\"']",
        r"Name:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:Resume|CV)"
    ]
    for pattern in name_patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            result["name"] = matches.group(1).strip()
            break
    
    # Try to extract email
    email_matches = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    if email_matches:
        result["email"] = email_matches[0]
    
    # Try to extract skills (look for a list of technical terms)
    skill_sections = re.findall(r'(?:Technical Skills|Skills|Technologies|Programming Languages)[:\s]+((?:[A-Za-z0-9+#]+(?:, | |, |\n)?)+)', text)
    if skill_sections:
        # Take the longest skill section as it's likely the most complete
        longest_section = max(skill_sections, key=len)
        # Split by common delimiters
        skills = re.findall(r'[A-Za-z0-9+#]+', longest_section)
        result["skills"] = [s for s in skills if len(s) > 1]  # Filter out single characters
        result["technical_skills"] = result["skills"].copy()
    
    # Basic experience extraction (find years of experience)
    exp_matches = re.search(r'(\d+)(?:\+)?\s+years?\s+(?:of\s+)?experience', text, re.IGNORECASE)
    if exp_matches:
        result["years_of_experience"] = exp_matches.group(1)
        result["experience"] = f"{exp_matches.group(1)}+ years of experience"
    
    # Basic education extraction
    edu_patterns = [
        r'(?:Bachelor|Master|PhD|Doctorate|BSc|BA|MS|MSc|MBA)[\s\w]*(?:in|of)[\s\w]+(?:from|at)\s+([\w\s]+)',
        r'([\w\s]+University|College)',
    ]
    for pattern in edu_patterns:
        edu_matches = re.search(pattern, text, re.IGNORECASE)
        if edu_matches:
            result["education"] = f"Degree from {edu_matches.group(1).strip()}"
            break
    
    return result

def parse_resume_with_llm(resume_text: str) -> dict:
    """
    Uses a Large Language Model to parse raw resume text into a structured JSON object,
    including the candidate's email address. Includes robust retry and fallback logic for JSON parsing issues.
    """
    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY is not set. Please check your .env file."}

    client = OpenAI(
      base_url=OPENROUTER_API_BASE,
      api_key=OPENROUTER_API_KEY,
    )

    start_time = time.time()
    
    # --- ORIGINAL PROMPT: More detailed, pre-optimization ---
    prompt = f"""
You are an expert technical recruiter. Carefully analyze the following resume and extract all relevant information as a structured JSON object.

Resume:
---
{resume_text}
---

Instructions:
- Extract the following fields: name, email, phone, location, technical_skills, soft_skills, domain_knowledge, years_of_experience, experience, education, certifications, projects.
- For each field, provide the most accurate and complete information possible based on the resume content.
- If a field is missing or not found, set its value to null.
- Do not infer or guess information that is not present in the resume.
- Output only valid JSON with exactly these keys and no extra commentary.
- Ensure the JSON is parseable and does not contain trailing commas or comments.

Example format:
{{
  "name": "...",
  "email": "...",
  "phone": "...",
  "location": "...",
  "technical_skills": [...],
  "soft_skills": [...],
  "domain_knowledge": [...],
  "years_of_experience": ..., 
  "experience": "...",
  "education": "...",
  "certifications": [...],
  "projects": [...]
}}
"""

    for retry in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # First try parsing as is
            try:
                analysis_content = json.loads(content)
            except json.JSONDecodeError as e:
                # If that fails, try to fix common issues
                fixed_content = fix_malformed_json(content)
                try:
                    analysis_content = json.loads(fixed_content)
                    print(f"Successfully fixed malformed JSON on retry {retry}")
                except json.JSONDecodeError:
                    # If still failing and we have retries left, try again
                    if retry < MAX_RETRIES:
                        print(f"JSON parsing failed on attempt {retry+1}, retrying...")
                        continue
                    else:
                        # Last resort: try fallback extraction
                        print("Using fallback extraction for JSON parsing")
                        analysis_content = extract_fallback_json(content)
            
            # Add metadata
            analysis_content['parsing_time'] = round(time.time() - start_time, 2)
            
            # Ensure we have at least a fallback for the most critical fields
            if not analysis_content.get('name'):
                analysis_content['name'] = "Unknown"
            if not analysis_content.get('email'):
                # Try to extract email with regex as fallback
                emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', resume_text)
                analysis_content['email'] = emails[0] if emails else None
                
            # Add some legacy fields for backward compatibility
            if 'technical_skills' in analysis_content and not 'skills' in analysis_content:
                tech_skills = analysis_content.get('technical_skills', [])
                if isinstance(tech_skills, list):
                    analysis_content['skills'] = tech_skills
                
            return analysis_content

        except Exception as e:
            print(f"An error occurred during LLM parsing (attempt {retry+1}): {e}")
            if retry == MAX_RETRIES:  # If this was our last retry
                # Try fallback extraction as absolute last resort
                fallback_result = extract_fallback_json(resume_text)
                fallback_result['parsing_time'] = round(time.time() - start_time, 2)
                fallback_result['error'] = str(e)
                fallback_result['_error_details'] = f"LLM parsing failed after {MAX_RETRIES} retries: {str(e)}"
                return fallback_result