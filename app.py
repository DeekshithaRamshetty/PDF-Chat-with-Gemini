import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import tempfile

# Configuration
GOOGLE_API_KEY = "AIzaSyCnBkxXJsmpc5xV26u4chlcilfjZmp4mNI"  # Replace with your actual key
CHAT_MODEL = "gemini-2.5-flash-preview-04-17"  # Gemini chat model
EMBEDDING_MODEL = "models/embedding-001"

# Streamlit UI Setup
st.set_page_config(page_title="PDF Chat with Gemini", page_icon="📄")
st.title("📄 PDF Chat with Gemini")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar with reset button
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file and not st.session_state.vector_store:
    with st.spinner("Processing PDF..."):
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Process PDF
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            st.session_state.vector_store = FAISS.from_documents(
                chunks,
                embedding=GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=GOOGLE_API_KEY
                )
            )
            st.session_state.messages.append({"role": "assistant", "content": "PDF is ready! Ask me anything about it."})
            st.success("PDF processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the PDF (or say 'quit' to exit)..."):
    # Define exit phrases
    exit_phrases = ["quit", "exit", "thank you", "thanks", "ok thank you", "ok thanks", "that's all"]
    
    # Check for exit phrases
    if prompt.lower().strip() in exit_phrases:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "You're welcome! Click 'Start New Conversation' to begin again."})
        st.rerun()
    
    # Check if PDF is uploaded
    elif not st.session_state.vector_store:
        st.warning("Please upload a PDF first")
    
    # Process actual questions
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=CHAT_MODEL,
                        google_api_key=GOOGLE_API_KEY,
                        temperature=0.3
                    )
                    
                    prompt_template = ChatPromptTemplate.from_template(
                        """Answer based on this context:
                        <context>
                        {context}
                        </context>
                        Question: {input}"""
                    )
                    
                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                    
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)