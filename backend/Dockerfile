# Use a modern, slim Python version as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first. The paths are now relative to the Dockerfile's location.
COPY ./requirements.txt /code/requirements.txt
COPY ./constraints.txt /code/constraints.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt -c /code/constraints.txt

# Explicitly download and save the model to a relative 'model' directory
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-Mini-LM-L6-v2').save('model')"

# Copy the rest of the application code from the current directory (the backend folder)
COPY . /code/

# Command to run your FastAPI application, using the PORT environment variable
# provided by the deployment platform (with a fallback to 8080 for Google Cloud)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
