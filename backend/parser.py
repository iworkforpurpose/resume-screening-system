import os
import json
from openai import OpenAI
from dotenv import load_dotenv

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
    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY is not set. Please check your .env file."}

    client = OpenAI(
      base_url=OPENROUTER_API_BASE,
      api_key=OPENROUTER_API_KEY,
    )

    # --- UPDATED PROMPT: Now asks for the email address ---
    prompt = f"""
    You are an expert resume parsing system. Your task is to analyze the provided resume text
    and extract the key information in a structured JSON format.

    **Resume Text:**
    ---
    {resume_text}
    ---

    **Instructions:**
    1.  Extract the full name of the candidate.
    2.  Extract the candidate's primary email address.
    3.  Extract the candidate's skills into a list of strings.
    4.  Summarize the candidate's work experience.
    5.  Summarize the candidate's education.
    6.  If a field is not present, return null for its value.

    Provide your response as a single, valid JSON object with the following keys:
    "name", "email", "skills", "experience", "education".
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        analysis_content = json.loads(response.choices[0].message.content)
        return analysis_content

    except Exception as e:
        print(f"An error occurred during LLM parsing: {e}")
        return {"error": str(e)}