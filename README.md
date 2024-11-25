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


## **Folder Structure**
The project directory is organized as follows:

```plaintext
project/
├── postgres/
│   ├── Dockerfile
│   ├── init.sql
├── streamlit/
│   ├── __init__.py
│   ├── .env
│   ├── app.py
│   ├── database.py
│   ├── Dockerfile
│   ├── file_processing.py
│   ├── main.py
│   ├── requirements.txt
│   ├── utils.py
├── .env
├── docker-compose.yml
├── README.md
