import streamlit as st
from backend.main import process_first_pdf, generate_response, process_all_pdfs
from database.database import retrieve_relevant_data_cosine

# Streamlit app configuration
st.title("Chatmodel for Postgres Database")

# Automatically process the first PDF in the database
st.header("Initializing Database...")
processing_message = process_first_pdf()
# processing_message = process_all_pdfs()    # This will process all the pdf files availables in the database
st.write(processing_message)

# Query Input and Retrieval
st.header("Ask Your Question")
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        st.write("Fetching the response...")
        # Retrieve relevant contexts
        results = retrieve_relevant_data_cosine(query)
        
        if results:
            # Generate response using LLM
            response = generate_response(query, results)
            st.header("Output:")
            st.write(response)
        else:
            st.write("No relevant documents found to answer your query.")
    else:
        st.write("Please enter a query.")
