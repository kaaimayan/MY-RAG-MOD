"""
RAG Chat Bot - Streamlit + LangChain

Upload a PDF, DOCX, TXT, or CSV file and ask questions about its content.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "csv"]
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}


class DocumentProcessingError(Exception):
    """Raised when the uploaded document cannot be loaded or indexed."""


def extract_response_text(content: Any) -> str:
    """Return displayable text from LangChain message content parts."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "\n".join(part.strip() for part in text_parts if part.strip())

    return str(content).strip()


class DocumentQA:
    """Small session-only RAG helper compatible with current LangChain packages."""

    def __init__(self, vectorstore: FAISS, llm: BaseChatModel, retrieval_count: int) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.retrieval_count = retrieval_count

    def ask(self, question: str, chat_history: list[dict[str, Any]]) -> dict[str, Any]:
        source_documents = self.vectorstore.similarity_search(
            question,
            k=self.retrieval_count,
        )
        context = "\n\n".join(
            f"Source {index}:\n{document.page_content}"
            for index, document in enumerate(source_documents, start=1)
        )
        prior_messages = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in chat_history[-6:]
            if message.get("content")
        )

        system_prompt = (
            "You are a helpful RAG assistant. Answer only using the provided document "
            "context. If the answer is not in the context, say that you could not find "
            "enough information in the uploaded document. Keep the answer clear and concise."
        )
        user_prompt = (
            f"Document context:\n{context or 'No relevant context found.'}\n\n"
            f"Recent chat history:\n{prior_messages or 'No previous messages.'}\n\n"
            f"Question: {question}"
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        return {
            "answer": extract_response_text(response.content),
            "source_documents": source_documents,
        }


st.set_page_config(page_title="RAG Chat Bot", page_icon=":page_facing_up:", layout="wide")
st.title("RAG Chat Bot")
st.caption("Upload a document, index it for this session, then ask questions about it.")


def reset_chat() -> None:
    """Clear all session-only document and chat state."""
    for key in ("chat_history", "chain", "file_name", "chunk_count"):
        st.session_state.pop(key, None)


def load_documents(uploaded_file: Any) -> list[Document]:
    """Save an uploaded file temporarily and load it with the matching parser."""
    suffix = Path(uploaded_file.name).suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)

    if loader_cls is None:
        supported = ", ".join(SUPPORTED_FILE_TYPES).upper()
        raise DocumentProcessingError(
            f"Unsupported file type '{suffix or 'unknown'}'. Please upload one of: {supported}."
        )

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        if suffix == ".txt":
            loader = loader_cls(temp_path, autodetect_encoding=True)
        else:
            loader = loader_cls(temp_path)

        documents = loader.load()
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            "The file could not be read. Check that it is not password-protected, corrupted, or empty."
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not documents:
        raise DocumentProcessingError("No readable text was found in this file.")

    return documents


def build_chain(
    uploaded_file: Any,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_count: int,
) -> tuple[DocumentQA, int]:
    """Load, split, embed, index, and build a session-only RAG helper."""
    documents = load_documents(uploaded_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]

    if not chunks:
        raise DocumentProcessingError(
            "The document was read, but no searchable text chunks could be created."
        )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=api_key,
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise DocumentProcessingError(
            "Google embeddings failed. Check your API key, billing status, and internet connection."
        ) from exc

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
        api_key=api_key,
    )
    chain = DocumentQA(vectorstore=vectorstore, llm=llm, retrieval_count=retrieval_count)
    return chain, len(chunks)


def format_source_label(index: int, source_document: Document) -> str:
    """Create a compact source label from document metadata."""
    metadata = source_document.metadata or {}
    source = Path(str(metadata.get("source", "uploaded document"))).name
    page = metadata.get("page")

    if page is not None:
        return f"Source {index}: {source}, page {int(page) + 1}"
    return f"Source {index}: {source}"


with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Used only for this session. It is not stored by the app.",
    ).strip()

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=SUPPORTED_FILE_TYPES,
        help="Supported formats: PDF, DOCX, TXT, CSV",
    )

    st.divider()
    st.subheader("Retrieval settings")
    chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    retrieval_count = st.slider("Chunks to retrieve", 1, 10, 4)

    process_btn = st.button("Process document", type="primary", use_container_width=True)
    clear_btn = st.button("Clear chat and document", use_container_width=True)

    if clear_btn:
        reset_chat()
        st.rerun()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your Google API key.")
    elif uploaded_file is None:
        st.sidebar.error("Please upload a PDF, DOCX, TXT, or CSV file first.")
    else:
        with st.spinner("Reading, chunking, embedding, and indexing your document..."):
            try:
                chain, chunk_count = build_chain(
                    uploaded_file=uploaded_file,
                    api_key=api_key,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    retrieval_count=retrieval_count,
                )
            except DocumentProcessingError as exc:
                st.sidebar.error(str(exc))
            except Exception as exc:
                st.sidebar.error(f"Unexpected processing error: {exc}")
            else:
                st.session_state.chain = chain
                st.session_state.chat_history = []
                st.session_state.file_name = uploaded_file.name
                st.session_state.chunk_count = chunk_count
                st.sidebar.success(f"Indexed {chunk_count} chunks from {uploaded_file.name}.")


if "chain" not in st.session_state:
    st.info("Upload a document in the sidebar, then click **Process document** to begin.")
else:
    st.success(
        f"Ready: {st.session_state.file_name} indexed into "
        f"{st.session_state.chunk_count} chunks."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['label']}**")
                        st.text(source["content"])

    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and writing an answer..."):
                try:
                    result = st.session_state.chain.ask(
                        question,
                        st.session_state.chat_history[:-1],
                    )
                    answer = result.get("answer", "").strip()
                    source_documents = result.get("source_documents", [])

                    if not answer:
                        answer = "I could not find enough information in the document to answer that."

                    sources = [
                        {
                            "label": format_source_label(index, source_document),
                            "content": source_document.page_content[:700],
                        }
                        for index, source_document in enumerate(source_documents, start=1)
                    ]

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.markdown(f"**{source['label']}**")
                                st.text(source["content"])

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(
                        "The answer could not be generated. Check your API key, "
                        f"network connection, and model access. Details: {exc}"
                    )
