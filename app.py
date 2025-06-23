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

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key is missing. Please check your .env file.")

# Function Definitions (No Changes Needed)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()  # Fixed undefined 'page_text'
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_faiss_index():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return new_db
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return None
    
def get_conversational_chain():
    prompt_template = """
    You are a financial advisor specializing in personal financial planning, investment strategies, and tax management.
    If the question is related to finance, answer it to the best of your knowledge even if no specific context is provided.
    If the context is available, use the context to give the most accurate and detailed answer.
    If the answer cannot be found in the provided context, respond with "The answer is not available in the provided context."
    If its still regarding finance like how should you allocate money then answer it to the best of your ability .
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

def user_input(user_question):
    new_db = load_faiss_index()  # Load the FAISS index
    if not new_db:  # Check if the index is loaded correctly
        return

    # Perform similarity search
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    try:
        # Execute the chain with the retrieved documents and the user's question
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        output_text = response.get("output_text", "No output text available")
        st.write("Reply: ", output_text)

        # Update the chat history
        st.session_state.chat_history.append({"user": user_question, "bot": output_text})
    except Exception as e:
        st.error(f"Error in conversational chain: {e}")

# Main Application Layout
def main():
    st.set_page_config(page_title="AI-Powered Savings Assistant", layout="wide")
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("FINLY")
        option = st.selectbox("Select a Page", ["Home Dashboard", "Set Savings Goals", "Spending Analysis", "Chat with FINLY"])
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
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
    elif option == "Set Savings Goals":
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
        import pandas as pd
        import plotly.express as px

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
        fig2 = px.line(time_series_data, x='date', y='spending', title='Spending Over the Last Year')
        st.plotly_chart(fig2)
        
    # Chat with PDF Page
    elif option == "Chat with FINLY":
        st.title("Chat with FINLY")

        # Define the pdf_files variable before use
        pdf_files = [
            "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs\\1.pdf",
            "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs\\2.pdf",
            "C:\\Coding\\devnexus\\NewProjectFileReader\\backend_docs\\3.pdf"
        ]

        # Process the PDFs and create the vector store
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

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
