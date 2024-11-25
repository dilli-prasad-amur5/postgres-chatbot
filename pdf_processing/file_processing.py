import fitz
from typing import List
from openai import OpenAI
import os
import logging
from backend.utils import setup_logging

# Configure logging for this file
setup_logging("pdf_processing.log")

# Initialize OpenAI client using OpenAI API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_pdf(binary_content: bytes) -> str:
    """
    Extract text from a binary PDF content and return text data.
    """
    logging.info("Starting PDF text extraction from binary content...")
    text = ""
    try:
        with open("temp.pdf", "wb") as temp_file:
            temp_file.write(binary_content)

        with fitz.open("temp.pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        logging.info("Text extraction complete.")
    except Exception as e:
        logging.error(f"Error during text extraction: {e}")
        raise
    return text


def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into smaller chunks for processing.
    """
    logging.info("Starting text chunking...")
    try:
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        logging.info(f"Text chunking complete. Total chunks created: {len(chunks)}")
    except Exception as e:
        logging.error(f"Error during text chunking: {e}")
        raise
    return chunks


def get_embeddings(text: str) -> List[float]:
    """
    Generate embeddings using OpenAI embedding model.
    """
    logging.info("Generating embeddings using OpenAI API...")
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        logging.info("Embeddings generated successfully.")
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise
