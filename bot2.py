import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader

os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided below.
Be concise and accurate.

<context>
{context}

Question: {input}
""")

st.markdown("""
    <style>
        /* Page and header */
        .main {
            padding: 2rem 3rem 3rem 3rem;
            max-width: 900px;
            margin: auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1 {
            text-align: center;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.1rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }

        /* Input */
        .stTextInput>div>div>input {
            font-size: 1rem !important;
            padding: 10px !important;
            border-radius: 8px !important;
            border: 1.5px solid #ccc !important;
            transition: border-color 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #3498db !important;
            outline: none !important;
            box-shadow: 0 0 5px #3498db !important;
        }

        /* Response box */
        .response-box {
            background-color: #f9fbfd;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1.5px solid #d1d9e6;
            font-size: 1.05rem;
            line-height: 1.5;
            color: #34495e;
            margin-top: 1rem;
            white-space: pre-wrap;
        }

        /* Sidebar */
        .css-1d391kg {  /* sidebar padding */
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .sidebar .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .sidebar h2 {
            color: #34495e;
            font-weight: 600;
        }
        .sidebar button {
            width: 100%;
            background-color: #3498db;
            border: none;
            padding: 0.6rem 0;
            border-radius: 8px;
            font-weight: 600;
            color: white;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 1rem;
            transition: background-color 0.3s ease;
        }
        .sidebar button:hover {
            background-color: #2980b9;
        }

        /* Footer */
        footer {
            text-align: center;
            margin-top: 3rem;
            font-size: 0.9rem;
            color: #95a5a6;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title("AI PDF Chatbot")
    st.markdown('<p class="subtitle">Ask questions based on the content of your PDF documents.</p>', unsafe_allow_html=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

uploaded_files = st.file_uploader(
    "Upload PDF files", accept_multiple_files=True, type=["pdf"]
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Saved {len(uploaded_files)} file(s) to '{DATA_DIR}' folder. Please embed documents in the sidebar.")

with st.sidebar:
    st.header("Document setup")
    st.write("Upload your PDFs above, then click below to embed the documents.")
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents..."):
            try:
                st.session_state.embeddings = GPT4AllEmbeddings()
                loader = PyPDFDirectoryLoader(DATA_DIR)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)
                st.success("Vector database is ready!")
            except Exception as e:
                st.error(f"Failed to embed documents: {e}")

st.subheader("Ask a question")
user_prompt = st.text_input(
    "Enter your question:", 
    placeholder="e.g., What are the main conclusions of the report?"
)

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please embed documents using the sidebar first.")
    else:
        with st.spinner("Generating response..."):
            try:
                doc_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                chain = create_retrieval_chain(retriever, doc_chain)

                start = time.process_time()
                response = chain.invoke({'input': user_prompt})
                end = time.process_time()

                answer = response.get("answer", "No answer found.")

                st.markdown("Answer:")
                st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)
                st.caption(f"Response time: {end - start:.2f} seconds")

            except Exception as e:
                st.error(f"Error generating response: {e}")

st.markdown("""
<footer>
    © 2025 Shruti Rana
</footer>
""", unsafe_allow_html=True)
