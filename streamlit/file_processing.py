import fitz 
from typing import List
from openai import OpenAI
import os

# Initialize OpenAI client using openai Api key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(binary_content: bytes) -> str:
    """extract text from a binary PDF content and retrun text data """
    print("Extracting text from PDF binary content...")
    text = ""
    with open("temp.pdf", "wb") as temp_file:
        temp_file.write(binary_content)

    with fitz.open("temp.pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    
    return text

def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """ Split text into smaller chunks for processing. """
    print("Splitting text into chunks...")
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def get_embeddings(text: str) -> List[float]:
    """ Generate embeddings using OpenAI embedding model"""
    print("Generating embeddings...")
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding
