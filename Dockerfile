# Use the official Python image as a base
FROM python:3.11-slim
# Install necessary build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

# Copy the entire application
COPY . .

# Expose the port the app runs on # For FastAPI
EXPOSE 8000  
# or # For Streamlit
EXPOSE 8501  

# Command to run the application
#CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

