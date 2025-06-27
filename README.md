# PDF Chat with Gemini

A Streamlit application for querying PDFs , PPT, DOCX using Google's Gemini AI.

## Features
- PDF text extraction and vectorization
- PPT Slide text extraction
- DOCX Content extraction
- Natural language questioning
- Conversation history
- Multi-page document support

## Setup
1. Clone repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.streamlit/secrets.toml` with your API key:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```
5. Run locally:
   ```bash
   streamlit run app.py
   ```

\
