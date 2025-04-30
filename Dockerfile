# Use Python 3.10 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make sure critical directories exists
RUN mkdir -p models
RUN mkdir -p cached_images

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port for Streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_classifier_app.py", "--server.port=8501", "--server.address=0.0.0.0"]