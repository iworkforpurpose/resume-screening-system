# Use a modern, slim Python version as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker layer caching
# Paths are relative to the project root (the build context)
COPY ./backend/requirements.txt /code/requirements.txt
COPY ./backend/constraints.txt /code/constraints.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt -c /code/constraints.txt

# --- CORRECTED MODEL DOWNLOAD STEP ---
# Set an environment variable for the cache directory and then run the download.
# This ensures the model is downloaded directly to the specified folder.
ENV SENTENCE_TRANSFORMERS_HOME=/code/model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of your application code LAST
# Only this layer will be rebuilt when you change your Python code
COPY ./backend/ /code/

# Command to run your FastAPI application, using the PORT environment variable
# provided by the deployment platform (with a fallback to 8000 for local use)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
