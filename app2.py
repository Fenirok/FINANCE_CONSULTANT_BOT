import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Google API key is missing. Ensure it is set in the environment variables.")
genai.configure(api_key=api_key)

# Directory containing your pre-uploaded PDF files
PDF_DIRECTORY = "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs"

# Load PDFs from directory
def load_preuploaded_pdfs(pdf_directory):
    """Loads PDF files from a predefined directory."""
    pdf_files = []
    for file in os.listdir(pdf_directory):
        if file.endswith('.pdf'):
            pdf_files.append(os.path.join(pdf_directory, file))
    return pdf_files

# Extract text from PDF files
def get_pdf_text(pdf_files):
    """Extract text from the provided list of PDF files."""
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception as e:
            st.warning(f"Could not read {pdf}: {e}")
    return text

# Split extracted text into chunks
def get_text_chunks(text, chunk_size=2000, chunk_overlap=200):
    """Split the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store for similarity search
@st.cache_resource
def get_vector_store(text_chunks):
    """Create a vector store for similarity search."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Configure conversational AI chain
def get_conversational_chain():
    prompt_template = """
    You are a financial advisor specializing in personal financial planning, investment strategies, and tax management.
    If the question is related to finance, provide the most accurate and detailed answer based on your knowledge even if no specific context is provided.
    If context is available from the documents, use it to provide a more tailored answer.
    If the answer cannot be found in the provided context, answer based on your expertise in finance.
    You can ask clarifying questions if more details are needed to provide accurate advice.
    Ask any necessary questions before you answer to give the best possible answer.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and return AI-generated responses
def user_input(user_question):
    """Handle user input and return AI-generated responses."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chain = get_conversational_chain()

    try:
        # Load the FAISS index
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        docs = []  # Continue without document context

    try:
        # If documents are retrieved, use them; otherwise, answer from general knowledge
        if docs:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        else:
            response = chain({"context": "", "question": user_question}, return_only_outputs=True)
        response_text = response["output_text"]

        # Append the user question and response to chat history in session state
        st.session_state.chat_history.append({"user": user_question, "FINLY": response_text})
    except Exception as e:
        st.error(f"Error in conversational chain: {e}")

# Main Application Layout
def main():
    st.set_page_config(page_title="AI-Powered Savings Assistant", layout="wide")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar Navigation
    with st.sidebar:
        st.title("FINLY")
        option = st.selectbox("Select a Page", ["Home Dashboard", "Set Savings Goals", "Spending Analysis", "Chat with FINLY"])

    # Home Dashboard
    if option == "Home Dashboard":
        st.title("AI-Powered Savings Assistant")
        st.header("Hello, John Doe!")
        st.subheader("Overview")
        st.write("Total Savings: $1,200")
        st.write("Total Spending: $800")
        st.write("Monthly Savings: $400")

        # Savings Goal Progress Bar
        st.subheader("Savings Goal Progress")
        progress = 60  # Example progress percentage
        st.progress(progress)

    # Set Savings Goals Page
    if option == "Set Savings Goals":
        st.title("Set Your Savings Goal")
        goal_amount = st.number_input("Enter your savings goal amount ($):", min_value=0)
        target_date = st.date_input("Select your target date:")

        # Button to Save Goal
        if st.button("Save Goal"):
            st.success(f"Your savings goal of ${goal_amount} by {target_date} has been saved!")

        # Display Progress (Placeholder)
        st.subheader("Current Progress")
        progress = 40  # Example progress percentage
        st.progress(progress)

    # Spending Analysis Page
    elif option == "Spending Analysis":
        st.title("Spending Analysis")

        # Sample Data for Analysis
        data = pd.DataFrame({
            'category': ['Groceries', 'Entertainment', 'Transportation', 'Rent', 'Utilities'],
            'amount': [300, 150, 100, 600, 150]
        })

        # Pie Chart for Category-wise Spending
        st.subheader("Category-wise Spending")
        fig1 = px.pie(data, names='category', values='amount', title='Spending Breakdown by Category')
        st.plotly_chart(fig1)

        # Sample Time Series Data
        time_series_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=12, freq='M'),
            'spending': [400, 350, 500, 450, 600, 550, 700, 650, 800, 750, 900, 850]
        })

        # Line Chart for Spending Over Time
        st.subheader("Spending Over Time")
        try:
            fig2 = px.line(time_series_data, x='date', y='spending', title='Spending Over the Last Year')
            st.plotly_chart(fig2)
        except Exception as e:
            st.error(f"An error occurred while generating the line chart: {e}")

    # Chat with FINLY Page
    elif option == "Chat with FINLY":
        st.title("Chat with FINLY")
        user_question = st.text_input("Ask your Question")
        if user_question:
            user_input(user_question)

        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False

        if st.button("Submit & Process"):
            pdf_files = load_preuploaded_pdfs(PDF_DIRECTORY)
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state.pdf_processed = True

        # Display chat history
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['user']}")
            st.write(f"**FINLY:** {chat['FINLY']}")

        # Clear chat history button
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()
