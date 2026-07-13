"""
RAG Ingestion Pipeline — Run this script once to build the vector store.

Processes:
1. All PDFs in data/ → text chunks → embedded → stored in ChromaDB (pdf_knowledge)
2. colleges.json → natural language text per college-branch → embedded → stored in ChromaDB (college_cutoffs)

Usage:
    python ingest.py

Re-run whenever source PDFs or colleges.json changes.
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import json
import shutil

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CATEGORY_CODES,
    CHROMA_PATH,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    COLLEGE_COLLECTION,
    COLLEGES_JSON,
    DATA_DIR,
    EMBEDDING_MODEL,
    PDF_COLLECTION,
    get_category_name,
)


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Extract text from a PDF file, returning a list of {page, text} dicts."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, source_file: str, page_number: int = None) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    result = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source_file": source_file,
            "chunk_index": i,
        }
        if page_number is not None:
            metadata["page_number"] = page_number
        result.append({"text": chunk, "metadata": metadata})

    return result


def process_pdfs() -> list[dict]:
    """Process all PDFs in the data directory into text chunks."""
    all_chunks = []
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])

    print(f"\n📄 Processing {len(pdf_files)} PDF files...")

    for pdf_file in pdf_files:
        pdf_path = DATA_DIR / pdf_file
        pages = extract_pdf_text(str(pdf_path))
        file_chunks = []

        for page_data in pages:
            chunks = chunk_text(
                page_data["text"],
                source_file=pdf_file,
                page_number=page_data["page"],
            )
            file_chunks.extend(chunks)

        all_chunks.extend(file_chunks)
        print(f"   ✅ {pdf_file}: {len(pages)} pages → {len(file_chunks)} chunks")

    print(f"   📊 Total PDF chunks: {len(all_chunks)}")
    return all_chunks


def format_cutoffs(cutoff_data: dict, round_name: str) -> str:
    """Format cutoff data for a single round into readable text."""
    lines = []
    for code, percentile in cutoff_data.items():
        name = get_category_name(code)
        if name != code:
            lines.append(f"  {code} ({name}): {percentile}")
        else:
            lines.append(f"  {code}: {percentile}")
    return f"{round_name}: " + ", ".join(
        f"{code}: {percentile}" for code, percentile in cutoff_data.items()
    )


def process_colleges() -> list[dict]:
    """Convert colleges.json into text chunks for vector storage."""
    all_chunks = []

    with open(COLLEGES_JSON, "r", encoding="utf-8") as f:
        colleges = json.load(f)

    print(f"\n🏫 Processing {len(colleges)} colleges...")

    for college_name, college_data in colleges.items():
        college_code = college_data.get("collegeCode", "")
        region = college_data.get("region", "")
        district = college_data.get("district", "")
        coords = college_data.get("coords", {})

        for branch_name, branch_data in college_data.get("offerings", {}).items():
            branch_code = branch_data.get("branchCode", "")

            # Build a natural language description of this college-branch combo
            text_parts = [
                f"College: {college_name}",
                f"College Code: {college_code}",
                f"Branch: {branch_name}",
                f"Branch Code: {branch_code}",
                f"Region: {region}",
                f"District: {district}",
            ]

            # Add cutoff data for each round
            for round_key, round_label in [
                ("cutoffsR1", "CAP Round 1"),
                ("cutoffsR2", "CAP Round 2"),
                ("cutoffsR3", "CAP Round 3"),
                ("cutoffsR4", "CAP Round 4"),
            ]:
                if round_key in branch_data:
                    cutoff_entries = []
                    for code, percentile in branch_data[round_key].items():
                        cutoff_entries.append(f"{code}: {percentile}")
                    text_parts.append(
                        f"{round_label} Cutoffs: {', '.join(cutoff_entries)}"
                    )

            text = "\n".join(text_parts)

            # If the text is too long, it will be a single chunk anyway
            # Most college-branch entries fit within one chunk
            metadata = {
                "source_file": "colleges.json",
                "college_name": college_name,
                "branch_name": branch_name,
                "college_code": college_code,
                "branch_code": branch_code,
                "region": region,
                "district": district,
            }

            all_chunks.append({"text": text, "metadata": metadata})

    print(f"   📊 Total college-branch chunks: {len(all_chunks)}")
    return all_chunks


def build_vector_store(pdf_chunks: list[dict], college_chunks: list[dict]):
    """Embed all chunks and store in ChromaDB."""
    # Clear existing database if present
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print("\n🗑️  Cleared existing vector store.")

    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # --- PDF Knowledge Collection ---
    print(f"\n📚 Embedding {len(pdf_chunks)} PDF chunks...")
    pdf_collection = client.create_collection(
        name=PDF_COLLECTION,
        metadata={"description": "MHT-CET process knowledge from CET Cell PDFs"},
    )

    pdf_texts = [chunk["text"] for chunk in pdf_chunks]
    pdf_embeddings = embedder.encode(pdf_texts, show_progress_bar=True).tolist()
    pdf_metadatas = [chunk["metadata"] for chunk in pdf_chunks]
    pdf_ids = [f"pdf_{i}" for i in range(len(pdf_chunks))]

    # ChromaDB has a batch limit, so we add in batches of 500
    batch_size = 500
    for i in range(0, len(pdf_chunks), batch_size):
        end = min(i + batch_size, len(pdf_chunks))
        pdf_collection.add(
            ids=pdf_ids[i:end],
            embeddings=pdf_embeddings[i:end],
            documents=pdf_texts[i:end],
            metadatas=pdf_metadatas[i:end],
        )

    print(f"   ✅ Stored {len(pdf_chunks)} chunks in '{PDF_COLLECTION}' collection.")

    # --- College Cutoffs Collection ---
    print(f"\n🎓 Embedding {len(college_chunks)} college-branch chunks...")
    college_collection = client.create_collection(
        name=COLLEGE_COLLECTION,
        metadata={"description": "College cutoff data from colleges.json"},
    )

    college_texts = [chunk["text"] for chunk in college_chunks]
    college_embeddings = embedder.encode(college_texts, show_progress_bar=True).tolist()
    college_metadatas = [chunk["metadata"] for chunk in college_chunks]
    college_ids = [f"college_{i}" for i in range(len(college_chunks))]

    for i in range(0, len(college_chunks), batch_size):
        end = min(i + batch_size, len(college_chunks))
        college_collection.add(
            ids=college_ids[i:end],
            embeddings=college_embeddings[i:end],
            documents=college_texts[i:end],
            metadatas=college_metadatas[i:end],
        )

    print(f"   ✅ Stored {len(college_chunks)} chunks in '{COLLEGE_COLLECTION}' collection.")


def main():
    print("=" * 60)
    print("🚀 MHT-CET RAG Chatbot — Knowledge Base Builder")
    print("=" * 60)

    # Step 1: Process PDFs
    pdf_chunks = process_pdfs()

    # Step 2: Process college data
    college_chunks = process_colleges()

    # Step 3: Build vector store
    build_vector_store(pdf_chunks, college_chunks)

    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"   PDF chunks:     {len(pdf_chunks)}")
    print(f"   College chunks: {len(college_chunks)}")
    print(f"   Vector store:   {CHROMA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    # Force UTF-8 encoding for standard output on Windows to support emojis
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
