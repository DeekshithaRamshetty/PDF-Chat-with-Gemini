import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document  # Add this import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import tempfile

# Configuration
GOOGLE_API_KEY = " # "
CHAT_MODEL = "gemini-2.5-flash-preview-04-17"  # Use the latest Gemini model
EMBEDDING_MODEL = "models/embedding-001"

# Streamlit UI
st.set_page_config(page_title="MultiDoc Chat", page_icon="📄")
st.title("📄 Document Chat with Gemini")

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    
    # Refresh conversation button
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.rerun()
    
    st.divider()
    
    # Supported file types
    st.header("📁 Supported Files")
    st.markdown("""
    - **PDF** (.pdf)
    - **Word** (.docx)
    - **PowerPoint** (.pptx)
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Fixed PPT Processor
def process_ppt(file_path):
    try:
        loader = UnstructuredPowerPointLoader(file_path)
        raw_data = loader.load()
        
        # Convert to LangChain Document format
        documents = []
        for item in raw_data:
            if isinstance(item, dict):
                # Handle dictionary output
                documents.append(Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {})
                ))
            elif hasattr(item, 'page_content'):
                # Handle Document objects
                documents.append(Document(
                    page_content=item.page_content,
                    metadata=getattr(item, 'metadata', {})
                ))
        return documents
    except Exception as e:
        st.error(f"PPT Processing Error: {str(e)}")
        return None

# Unified Document Processing
def process_document(file_path, file_extension):
    try:
        if file_extension == "pdf":
            return PyPDFLoader(file_path).load()
        elif file_extension == "pptx":
            return process_ppt(file_path)
        elif file_extension == "docx":
            return Docx2txtLoader(file_path).load()
    except Exception as e:
        st.error(f"Error loading {file_extension.upper()}: {str(e)}")
        return None

# File Upload
uploaded_file = st.file_uploader(
    "Upload Document (PDF, PPTX, DOCX)",
    type=["pdf", "pptx", "docx"]
)

if uploaded_file and not st.session_state.vector_store:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            pages = process_document(tmp_path, file_extension)
            
            if pages:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                )
                chunks = text_splitter.split_documents(pages)
                
                # Store in vector database
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=GOOGLE_API_KEY
                )
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.success("Document processed and stored in vector database!")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Ready to answer questions about your {file_extension.upper()}!"
                })
            
        except Exception as e:
            st.error(f"Failed: {str(e)}")
        finally:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)

# Chat Interface - Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Input
if st.session_state.vector_store:
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # RAG Process
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # Initialize LLM
                    llm = ChatGoogleGenerativeAI(
                        model=CHAT_MODEL,
                        google_api_key=GOOGLE_API_KEY,
                        temperature=0.3
                    )
                    
                    # Similarity search from vector DB
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    
                    # Structured prompt template for RAG
                    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the "
                        "answer concise.\n\n"
                        "Context: {context}\n\n"
                        "Question: {input}\n\n"
                        "Answer:"
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}")
                    ])
                    
                    # Create RAG chain
                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    # Get response
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    
                    # Add assistant response to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    # Show retrieved context (optional)
                    with st.expander("View Retrieved Context"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.page_content[:300] + "...")
                            st.write("---")
                            
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
else:
    st.info("Please upload a document to start chatting!") 
