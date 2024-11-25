import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import logging
from backend.utils import setup_logging, normalize_vector, sanitize_text
from pdf_processing.file_processing import get_embeddings

# Configure logging for this file
setup_logging("database.log")

# Load environment variables
load_dotenv()

# Documents database configurations
SOURCE_DB_CONFIG = {
    "dbname": os.getenv("SOURCE_DB_NAME"),
    "user": os.getenv("SOURCE_DB_USER"),
    "password": os.getenv("SOURCE_DB_PASSWORD"),
    "host": os.getenv("SOURCE_DB_HOST"),
    "port": int(os.getenv("SOURCE_DB_PORT", 5432)),
}

# Vector database configurations
TARGET_DB_CONFIG = {
    "dbname": os.getenv("TARGET_DB_NAME"),
    "user": os.getenv("TARGET_DB_USER"),
    "password": os.getenv("TARGET_DB_PASSWORD"),
    "host": os.getenv("TARGET_DB_HOST"),
    "port": int(os.getenv("TARGET_DB_PORT", 5432)),
}


def connect_to_db(db_config):
    """ Establish a connection to a PostgreSQL database. """
    try:
        conn = psycopg2.connect(**db_config)
        logging.info(f"Connected to database: {db_config['dbname']} at {db_config['host']}")
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return None


def fetch_table_data(query, db_config):
    """ Execute a SQL query and fetch data from the source database."""
    conn = connect_to_db(db_config)
    logging.info("Database config details", db_config)
    if conn:
        try:
            df = pd.read_sql(query, conn)
            logging.info(f"Data fetched successfully: {len(df)} rows.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
        finally:
            conn.close()
            logging.info("Database connection closed.")
    return None


def insert_embeddings(embedding, context, description, source):
    """ Insert embeddings and associated metadata into the target database. """
    
    try:
        conn = connect_to_db(TARGET_DB_CONFIG)  # Connect to the target database
        cursor = conn.cursor()

        normalized_embedding = normalize_vector(embedding)
        sanitized_context = sanitize_text(context)

        query = """
        INSERT INTO embeddings (vector, context, description, source)
        VALUES (%s, %s, %s, %s);
        """
        cursor.execute(query, (normalized_embedding, sanitized_context, description, source))
        conn.commit()
        logging.info(f"Inserted embedding for: {description}")
    except Exception as e:
        logging.error(f"Error inserting embedding: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        logging.info("Database connection to target closed.")


def retrieve_relevant_data_cosine(query: str, top_k: int = 2):
    """ Retrieve the most relevant documents based on cosine similarity query embedding. """
    try:
        conn = psycopg2.connect(**TARGET_DB_CONFIG)
        cursor = conn.cursor()

        # Generate embedding for the query
        query_embedding = get_embeddings(query)
        if not query_embedding:
            logging.error("Failed to generate embedding for the query.")
            return []

        query_embedding_str = ','.join(map(str, query_embedding))

        # Query for top-k similar embeddings using cosine similarity
        similarity_query = f"""
        SELECT id, description, context, source,
               1 - (vector <=> '[{query_embedding_str}]'::vector) AS cosine_similarity
        FROM embeddings
        ORDER BY cosine_similarity DESC
        LIMIT {top_k};
        """
        logging.info("Executing similarity query...")
        cursor.execute(similarity_query)
        results = cursor.fetchall()

        formatted_results = [
            {
                "ID": row[0],
                "Description": row[1],
                "Context": row[2],
                "Source": row[3],
                "Similarity Score": row[4],
            }
            for row in results
        ]

        return formatted_results

    except Exception as e:
        logging.error(f"Error querying vector database: {e}")
        return []

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
