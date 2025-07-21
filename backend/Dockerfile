# Use a modern, slim Python version as the base image
FROM python:3.10-slim

# Set the working directory inside the container to /code
WORKDIR /code

# Copy your requirements and constraints files
COPY ./requirements.txt /code/requirements.txt
COPY ./constraints.txt /code/constraints.txt

# Install dependencies using both files
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt -c /code/constraints.txt

# --- NEW STEP ---
# Download and save the model to a local directory named 'model'
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model')"

# Copy the rest of your application code into the container
COPY . /code

# Command to run your FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]