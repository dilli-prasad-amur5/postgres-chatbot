# Base image for Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install system dependencies for psycopg2 and others
RUN apt-get update && apt-get install -y libpq-dev gcc

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
