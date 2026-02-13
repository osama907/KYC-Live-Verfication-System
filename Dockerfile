# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Install system dependencies (Mandatory for OpenCV and Tesseract)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the uploads directory with correct permissions
RUN mkdir -p static/uploads && chmod 777 static/uploads

# Hostinger/Standard VPS usually uses Port 8000
EXPOSE 8000

# Start the application using Uvicorn
# Ensure 'main' matches your filename (if your file is main1.py, change to main1:app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]