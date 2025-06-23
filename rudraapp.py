import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure the API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key is missing. Please check your .env file.")

# Load PDFs from backend (hidden from the user)
def load_backend_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store from the text chunks
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load FAISS index
def load_faiss_index():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return new_db
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

# Get conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a financial advisor specializing in personal financial planning, investment strategies, and tax management.
    If the question is related to finance, answer it to the best of your knowledge even if no specific context is provided.
    If the context is available, use the context to give the most accurate and detailed answer.
    If the answer cannot be found in the provided context, respond with "The answer is not available in the provided context."
    If its still regarding finance like how should you allocate money then answer it to the best of your ability .
    You can ask clarifying questions if more details are needed to provide accurate advice.
    Ask any neccesary questions before you answer to give the best possible answer. 
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and generate response
def user_input(user_question):
    new_db = load_faiss_index()
    if new_db:
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        # Append user question and response to chat history
        st.session_state.chat_history.append({"user": user_question, "FINLY": response["output_text"]})
    else:
        st.error("Could not load FAISS index.")

# Main function to set up the Streamlit app
def main():
    st.set_page_config(page_title="📈💰📊   Financial Planning Chatbot    📈💰📊💁")
    st.header("📈💰📊   Financial Planning Chatbot    📈💰📊💁")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Process PDFs in the backend
    pdf_files = [
        "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs\\1.pdf",
        "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs\\2.pdf",
        "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs\\3.pdf"
    ]  # Paths to backend PDFs
    raw_text = load_backend_pdfs(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    create_vector_store(text_chunks)

    # User interaction
    user_question = st.text_input("Ask a financial question")
    if user_question:
        user_input(user_question)

    # Display chat history
    st.subheader("Chat History")
    for entry in st.session_state.chat_history:
        st.write(f"**You:** {entry['user']}")
        st.write(f"**Bot:** {entry['bot']}")

    # Clear history button
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()