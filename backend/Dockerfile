# Use a modern, slim Python version as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first to leverage Docker layer caching
COPY ./backend/requirements.txt /code/requirements.txt
COPY ./backend/constraints.txt /code/constraints.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt -c /code/constraints.txt

# --- FINAL MODEL DOWNLOAD STEP ---
# Explicitly download and save the model to a relative 'model' directory
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('model')"

# Copy the rest of your application code LAST
COPY ./backend/ /code/

# Command to run your FastAPI application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
