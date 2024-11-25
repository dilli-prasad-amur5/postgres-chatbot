-- Check if the database exists and create it only if it doesn't
DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'vector_db'
    ) THEN
        PERFORM dblink_exec('dbname=postgres user=user password=password', 
                            'CREATE DATABASE vector_db');
    END IF;
END
$$;

-- Connect to the database
\c vector_db

-- Create the pgvector extension (if not exists)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the embeddings table (if not exists)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    vector VECTOR(1536),
    context TEXT,
    description TEXT,
    source TEXT
);
