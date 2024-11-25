from database.database import fetch_table_data, insert_embeddings
from pdf_processing.file_processing import (
    extract_text_from_pdf,
    split_text_into_chunks,
    get_embeddings,
)
from backend.utils import setup_logging

import os
import logging
from openai import OpenAI

# Configure logging for this file
setup_logging("backend.log")

# Initialize OpenAI client using API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_first_pdf():
    """
    Fetch the first PDF from the source database, process it into embeddings,
    and store the embeddings in the target database.
    """
    SOURCE_DB_CONFIG = {
        "dbname": os.getenv("SOURCE_DB_NAME"),
        "user": os.getenv("SOURCE_DB_USER"),
        "password": os.getenv("SOURCE_DB_PASSWORD"),
        "host": os.getenv("SOURCE_DB_HOST"),
        "port": int(os.getenv("SOURCE_DB_PORT", 5432)),
    }

    # Fetch only the first PDF data from the source database
    query = "SELECT id, content, file_name FROM papers LIMIT 1;"
    pdf_data = fetch_table_data(query, db_config=SOURCE_DB_CONFIG)

    if pdf_data is None or pdf_data.empty:
        logging.warning("No PDF data found in the source database.")
        return "No PDF data found."

    logging.info("Found 1 PDF entry. Starting processing...")
    row = pdf_data.iloc[0]  # Get the first row
    pdf_id = row["id"]
    file_name = row["file_name"]
    pdf_content = row["content"]

    logging.info(f"Processing PDF: {file_name} (ID: {pdf_id})")

    try:
        # Extract text from binary content
        text = extract_text_from_pdf(pdf_content)
        if not text.strip():
            logging.warning(f"No text found in PDF: {file_name}")
            return "No text found in the PDF."

        # Split text into chunks
        chunks = split_text_into_chunks(text)

        # Process each chunk and store embeddings
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i + 1}/{len(chunks)} for PDF ID {pdf_id}...")
            embedding = get_embeddings(chunk)
            description = f"Chunk {i + 1} of {file_name}"
            insert_embeddings(embedding, chunk, description, file_name)

        logging.info("PDF processing and embedding insertion complete.")
        return "PDF processed and embeddings inserted."

    except Exception as e:
        logging.error(f"Error processing PDF {file_name} (ID: {pdf_id}): {e}")
        return f"Error processing PDF: {e}"


def process_all_pdfs():
    """Fetch all PDFs from the source database, process each into embeddings,
    and store the embeddings in the target database.
    """
    SOURCE_DB_CONFIG = {
        "dbname": os.getenv("SOURCE_DB_NAME"),
        "user": os.getenv("SOURCE_DB_USER"),
        "password": os.getenv("SOURCE_DB_PASSWORD"),
        "host": os.getenv("SOURCE_DB_HOST"),
        "port": int(os.getenv("SOURCE_DB_PORT", 5432)),
    }

    # Fetch all PDF data
    query = "SELECT id, content, file_name FROM papers;"
    pdf_data = fetch_table_data(query, db_config=SOURCE_DB_CONFIG)

    if pdf_data is None or pdf_data.empty:
        logging.warning("No PDF data found in the source database.")
        return "No PDF data found."

    logging.info(f"Found {len(pdf_data)} PDF entries. Starting processing...")

    for index, row in pdf_data.iterrows():
        pdf_id = row["id"]
        file_name = row["file_name"]
        pdf_content = row["content"]

        logging.info(f"Processing PDF: {file_name} (ID: {pdf_id})")

        try:
            # Extract text from binary content
            text = extract_text_from_pdf(pdf_content)
            if not text.strip():
                logging.warning(f"No text found in PDF: {file_name} (ID: {pdf_id})")
                continue

            # Split text into chunks
            chunks = split_text_into_chunks(text)

            # Process each chunk and store these embeddings in database
            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i + 1}/{len(chunks)} for PDF ID {pdf_id}...")
                embedding = get_embeddings(chunk)
                description = f"Chunk {i + 1} of {file_name}"
                insert_embeddings(embedding, chunk, description, file_name)

            logging.info(f"PDF processing and embedding insertion complete for {file_name} (ID: {pdf_id}).")

        except Exception as e:
            logging.error(f"Error processing PDF {file_name} (ID: {pdf_id}): {e}")

    logging.info("All PDFs have been processed.")
    return "All PDFs processed and embeddings inserted."


def prompt_template(context, query):
    return f"""
You are an intelligent assistant. Your task is to generate responses based strictly on the context provided. Use the information accurately without guessing or fabricating details. If the context does not contain enough information to answer the question, explicitly state: 
"The provided context does not contain enough information to answer this question."

### Context:
{context}

### Question:
{query}

### Response:
"""


def generate_response(query: str, contexts: list) -> str:
    """
    Generate a response based on the query and retrieved contexts.
    """
    if not contexts:
        logging.warning("No contexts provided for response generation.")
        return "I'm sorry, I couldn't find relevant contexts to answer your query."

    combined_context = "\n\n".join(
        [f"Context {i+1}: {ctx['Context']}" for i, ctx in enumerate(contexts)]
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_template(combined_context, query)},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "sorry, couldn't generate a response at this time."
