# Base image for Python
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Streamlit app folder
COPY ./streamlit_app /app

# Copy the backend folder
COPY ./backend /app/backend

# Copy the pdf_processing folder
COPY ./pdf_processing /app/pdf_processing

# Copy the database folder
COPY ./database /app/database

# Copy the requirements.txt file
COPY ./requirements.txt /app/requirements.txt

# Copy the .env file into the container
COPY .env /app/.env

# Install system dependencies for psycopg2 and others
RUN apt-get update && apt-get install -y libpq-dev gcc

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
