from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Configure Gemini with your direct API key
GOOGLE_API_KEY = "AIzaSyCnBkxXJsmpc5xV26u4chlcilfjZmp4mNI"  # ⚠️ Replace with your actual key

def load_and_index_pdf(pdf_path):
    """Load PDF, split text, and create searchable vector store"""
    # 2. Load PDF file
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # Extract all pages
    
    # 3. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Optimal for Gemini context
        chunk_overlap=200  # Maintains context between chunks
    )
    chunks = text_splitter.split_documents(pages)
    
    # 4. Convert text to vectors and store locally
    vector_store = FAISS.from_documents(
        chunks,
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    )
    return vector_store

def answer_question(vector_store, question):
    """Process questions using Gemini and the PDF content"""
    # 5. Initialize Gemini-Pro model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3  # Balance creativity vs accuracy
    )
    
    # 6. Define the QA prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer strictly based on the context:
        <context>
        {context}
        </context>
        Question: {input}"""
    )
    
    # 7. Chain components together
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Top 3 relevant chunks
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 8. Execute the chain
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

if __name__ == "__main__":
    # 9. Configure your PDF path
    pdf_path = "Dee.pdf"  # 📂 Replace with your PDF filename
    
    # 10. Process the PDF
    print("Indexing PDF...")
    vector_store = load_and_index_pdf(pdf_path)
    
    # 11. Interactive Q&A loop
    print("Ready! Ask about your PDF (type 'quit' to exit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "quit":
            break
        answer = answer_question(vector_store, question)
        print(f"\nAnswer: {answer}")