# **PDF Vector Search Engine with LLM Integration**

## **Overview**
This project is a chatbot designed to generate responses to user queries by leveraging documents stored in a PostgreSQL database. The documents are converted into vector embeddings, which are stored using Docker containers. Alternatively, the vector database can be stored directly in the PostgreSQL server to retrieve relevant text without the need to convert the documents into vectors every time the Docker container is run.

The system processes PDF documents, extracts their text, converts the text into vector embeddings, and stores them in a PostgreSQL database with the `pgvector` extension. Retrieval of relevant documents is performed using cosine similarity search techniques.

The project features a **Streamlit-based web interface**, allowing users to interact with a GPT-3 model to receive responses to queries based on the retrieved content.

For this implementation, only the first document is processed and converted into embeddings to minimize costs associated with using the OpenAI API for embedding generation. If you wish to process all documents, modify the `app.py` script to use the `process_all_pdfs()` function instead of `process_first_pdf()`.

---

## **Key Features**
- **Text Extraction**: Extracts text from PDF files using the `PyMuPDF` library.
- **Text Chunking & Embedding**: Splits extracted text into manageable chunks and converts them into vector embeddings using OpenAI's embedding model (`text-embedding-ada-002`).
- **Database Storage**: Stores embeddings in a PostgreSQL database with `pgvector` support for efficient similarity-based searches.
- **Query Response**: Utilizes `gpt-3.5-turbo` to generate responses to user queries based on the retrieved content.
- **User Interface**: Offers a Streamlit web application for seamless interaction with the LLM.

## **Environment Configuration**

The project uses a `.env` file to store configuration details, such as database credentials and API keys. The `.env` file is ignored by Git and an example configuration file, `example.env`, is provided. Copy its contents into your `.env` file and fill in the necessary details.

## **Run the Application**

1. **Using Docker (Recommended)**:
   Run the following docker command from root directory to build and start the services:
   ```bash
   docker-compose up --build

Then, open your browser and navigate to http://localhost:8501 to access the Streamlit app.

## **Folder Structure**
The project directory is organized as follows:

```plaintext
postgres-chatbot/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
├── database/
│   ├── __init__.py
│   ├── database.py
│   ├── Dockerfile
│   ├── init.sql
├── pdf_processing/
│   ├── __init__.py
│   ├── file_processing.py
├── streamlit_app/
│   ├── __init__.py
│   ├── app.py
│   ├── Dockerfile
├── .gitignore
├── docker-compose.yml
├── example.env
├── README.md
├── requirements.txt