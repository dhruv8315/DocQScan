# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first (better caching)
COPY requirement.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

# Copy entire project
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "src/app.py"]