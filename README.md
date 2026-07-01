# RAG Chat Bot (LangChain + Streamlit)

A simple session-based website that answers questions about a document you upload.
It supports PDF, DOCX, TXT, and CSV files.

## How it works

1. Upload a document.
2. Click **Process document**.
3. The app reads the file and splits it into text chunks.
4. The chunks are embedded with Google GenAI embeddings and stored in an in-memory FAISS index.
5. Ask questions in the chat box.
6. The app retrieves relevant chunks, sends them to a Google GenAI chat model, and shows the answer with source snippets.

Uploaded content and the FAISS index are session-only. They are cleared when you restart the app or click **Clear chat and document**.

## Setup

Open this folder in VS Code, then run:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Google API key, or paste it into the sidebar when the app runs:

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="..."

# macOS/Linux
export GOOGLE_API_KEY="..."
```

## Run

```bash
streamlit run app.py
```

The app opens at:

```text
http://localhost:8501
```

## Usage

1. Enter your Google API key if it is not already set as an environment variable.
2. Upload a PDF, DOCX, TXT, or CSV file.
3. Click **Process document**.
4. Ask questions about the uploaded document.
5. Open **Sources** under an answer to inspect the retrieved snippets.

## Notes

- This version uses Google GenAI via `langchain-google-genai`.
- It does not include login, a database, or persistent document storage.
- For scanned PDFs, the PDF must contain selectable text. OCR is not included.
- Very large files may take longer to index.
