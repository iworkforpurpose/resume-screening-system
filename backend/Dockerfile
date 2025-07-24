# Use a modern, slim Python version as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy requirements first. The paths are now relative to the Dockerfile's location.
COPY ./requirements.txt /code/requirements.txt
COPY ./constraints.txt /code/constraints.txt

# --- NEW: Accept the Hugging Face token as a build secret ---
# This allows us to securely use the token during the build
RUN --mount=type=secret,id=huggingface_token pip config set global.extra-index-url https://$(cat /run/secrets/huggingface_token)@huggingface.co/simple

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt -c /code/constraints.txt

# --- UPDATED: Use the token to download the model ---
RUN --mount=type=secret,id=huggingface_token python -c "from huggingface_hub import login; import os; login(token=os.environ.get('HUGGING_FACE_HUB_TOKEN', open('/run/secrets/huggingface_token').read())); from sentence_transformers import SentenceTransformer; SentenceTransformer('all-Mini-LM-L6-v2').save('model')"

# Copy the rest of the application code from the current directory (the backend folder)
COPY . /code/

# Command to run your FastAPI application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
