"""
Hybrid Retrieval Engine — Combines vector search with exact JSON lookup.

Intent classification is keyword-based (zero cost, zero latency).
Cutoff numbers are always fetched from the raw JSON for exactness.
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import json
import difflib

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH,
    COLLEGE_COLLECTION,
    COLLEGES_JSON,
    EMBEDDING_MODEL,
    PDF_COLLECTION,
    SOURCES_JSON,
    TOP_K_CUTOFF,
    TOP_K_MIXED,
    TOP_K_RESULTS,
    get_category_name,
)

# ---- Lazy-loaded globals (initialized on first call) ----
_embedder = None
_chroma_client = None
_pdf_collection = None
_college_collection = None
_colleges_data = None
_college_names = None
_sources_data = None


def _init():
    """Lazy-initialize the embedding model, ChromaDB, and college data."""
    global _embedder, _chroma_client, _pdf_collection, _college_collection
    global _colleges_data, _college_names, _sources_data

    if _embedder is not None:
        return

    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _pdf_collection = _chroma_client.get_collection(PDF_COLLECTION)
    _college_collection = _chroma_client.get_collection(COLLEGE_COLLECTION)

    with open(COLLEGES_JSON, "r", encoding="utf-8") as f:
        _colleges_data = json.load(f)

    _college_names = list(_colleges_data.keys())

    try:
        with open(SOURCES_JSON, "r", encoding="utf-8") as f:
            _sources_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _sources_data = {}


# ============================================================
# Intent Classification (keyword-based, no LLM cost)
# ============================================================

CUTOFF_KEYWORDS = {
    "cutoff", "cut off", "cut-off", "percentile", "rank", "seat",
    "opening", "closing", "merit", "score", "marks", "gopens",
    "gscs", "gsts", "gobcs", "tfws", "ews", "lopens", "cap round",
    "round 1", "round 2", "round 3", "round 4", "r1", "r2", "r3", "r4",
    "category", "reservation", "quota",
}

PROCESS_KEYWORDS = {
    "process", "admission", "counselling", "counseling", "document",
    "verification", "scrutiny", "e-scrutiny", "escrutiny", "brochure",
    "eligibility", "registration", "fee", "schedule", "timeline",
    "rule", "regulation", "procedure", "domicile", "certificate",
    "normalization", "normalisation", "methodology", "syllabus",
    "exam", "pattern", "attempt", "objection", "press note",
    "centre", "center", "option form", "freeze", "float", "slide",
}

COLLEGE_KEYWORDS = {
    "college", "university", "institute", "engineering",
    "pune", "mumbai", "nagpur", "nashik", "amravati", "aurangabad",
    "coep", "vjti", "pict", "cummins", "walchand", "mit",
    "region", "district", "branch", "department", "compare",
    "best", "top",
}


def classify_intent(query: str) -> str:
    """
    Classify user intent based on keyword matching.
    Returns: 'cutoff', 'process', 'college_info', or 'mixed'
    """
    query_lower = query.lower()
    words = set(query_lower.split())

    cutoff_score = sum(1 for kw in CUTOFF_KEYWORDS if kw in query_lower)
    process_score = sum(1 for kw in PROCESS_KEYWORDS if kw in query_lower)
    college_score = sum(1 for kw in COLLEGE_KEYWORDS if kw in query_lower)

    # Cutoff queries often mention college + cutoff keywords
    if cutoff_score >= 1 and (college_score >= 1 or cutoff_score >= 2):
        return "cutoff"
    if cutoff_score >= 2:
        return "cutoff"
    if process_score >= 1 and cutoff_score == 0:
        return "process"
    if college_score >= 1 and cutoff_score == 0 and process_score == 0:
        return "college_info"

    return "mixed"


# ============================================================
# Fuzzy College Name Matching
# ============================================================

# Common abbreviations → full names
COLLEGE_ALIASES = {
    "coep": "College of Engineering, Pune",
    "vjti": "Veermata Jijabai Technological Institute",
    "pict": "Pune Institute of Computer Technology",
    "cummins": "Cummins College of Engineering",
    "walchand": "Walchand College of Engineering",
    "mit": "Maharashtra Institute of Technology",
    "spit": "Sardar Patel Institute of Technology",
    "djsce": "Dwarkadas J. Sanghvi College of Engineering",
    "kjsce": "K. J. Somaiya College of Engineering",
    "tsec": "Thadomal Shahani Engineering College",
}


def find_matching_colleges(query: str, top_n: int = 3) -> list[str]:
    """
    Find college names matching the query using fuzzy matching.
    Handles abbreviations and partial names.
    """
    _init()
    query_lower = query.lower()

    # Check abbreviation aliases first
    for alias, full_name in COLLEGE_ALIASES.items():
        if alias in query_lower:
            # Find the actual key in colleges_data that contains this name
            matches = [
                name for name in _college_names
                if full_name.lower() in name.lower()
            ]
            if matches:
                return matches[:top_n]

    # Extract potential college name parts from the query
    # Try fuzzy matching against all college names
    matches = difflib.get_close_matches(
        query_lower, [n.lower() for n in _college_names], n=top_n, cutoff=0.3
    )

    # Map back to original case names
    result = []
    for match in matches:
        for name in _college_names:
            if name.lower() == match and name not in result:
                result.append(name)
                break

    # Also try substring matching if fuzzy match didn't work well
    if len(result) < top_n:
        query_words = [w for w in query_lower.split() if len(w) > 3]
        for name in _college_names:
            if name in result:
                continue
            name_lower = name.lower()
            if any(word in name_lower for word in query_words):
                result.append(name)
                if len(result) >= top_n:
                    break

    return result[:top_n]


def get_exact_cutoffs(college_name: str, branch_name: str = None) -> dict | None:
    """
    Get exact cutoff data from the JSON for a specific college (and optionally branch).
    Returns the raw data structure from colleges.json.
    """
    _init()

    if college_name not in _colleges_data:
        return None

    college = _colleges_data[college_name]

    if branch_name is None:
        return college

    # Fuzzy match the branch name
    branch_names = list(college.get("offerings", {}).keys())
    branch_matches = difflib.get_close_matches(
        branch_name.lower(),
        [b.lower() for b in branch_names],
        n=1,
        cutoff=0.4,
    )

    if branch_matches:
        for bname in branch_names:
            if bname.lower() == branch_matches[0]:
                return {
                    "college_name": college_name,
                    "branch_name": bname,
                    "data": college["offerings"][bname],
                    "region": college.get("region", ""),
                    "district": college.get("district", ""),
                }
    return None


def format_cutoff_context(college_name: str, cutoff_info: dict) -> str:
    """Format exact cutoff data into a readable text block for the LLM context."""
    lines = [
        f"College: {college_name}",
        f"Branch: {cutoff_info['branch_name']}",
        f"Region: {cutoff_info['region']}",
        f"District: {cutoff_info['district']}",
    ]

    data = cutoff_info["data"]
    for round_key, round_label in [
        ("cutoffsR1", "CAP Round 1"),
        ("cutoffsR2", "CAP Round 2"),
        ("cutoffsR3", "CAP Round 3"),
        ("cutoffsR4", "CAP Round 4"),
    ]:
        if round_key in data:
            cutoff_lines = []
            for code, percentile in data[round_key].items():
                full_name = get_category_name(code)
                cutoff_lines.append(f"    {code} ({full_name}): {percentile}")
            lines.append(f"\n  {round_label} Cutoffs:")
            lines.extend(cutoff_lines)

    return "\n".join(lines)


# ============================================================
# Source Link Resolver
# ============================================================

def get_source_info(source_file: str) -> dict:
    """Look up the source URL and title for a given filename."""
    _init()

    if source_file in _sources_data:
        info = _sources_data[source_file]
        return {
            "title": info.get("title", source_file),
            "url": info.get("url", ""),
        }

    return {"title": source_file, "url": ""}


# ============================================================
# Main Retrieval Function
# ============================================================

def retrieve(query: str) -> dict:
    """
    Main retrieval function. Classifies intent, searches the appropriate
    vector store collection(s), and optionally does exact JSON lookup.

    Returns:
        {
            "context": str,          # Combined text context for the LLM
            "sources": list[dict],   # [{title, url}] for citation
            "intent": str,           # Classified intent
        }
    """
    _init()

    intent = classify_intent(query)
    context_parts = []
    sources = []
    seen_sources = set()

    # Embed the query
    query_embedding = _embedder.encode(query).tolist()

    if intent == "cutoff" or intent == "college_info":
        # --- Search college_cutoffs collection ---
        results = _college_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_CUTOFF,
        )

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                context_parts.append(doc)

                # Get source info
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                src_file = meta.get("source_file", "")
                if src_file and src_file not in seen_sources:
                    sources.append(get_source_info(src_file))
                    seen_sources.add(src_file)

        # --- Also do exact JSON lookup for cutoff queries ---
        if intent == "cutoff":
            matching_colleges = find_matching_colleges(query)
            for college_name in matching_colleges:
                # Try to extract branch name from the query
                branch_query = query.lower()
                college_obj = _colleges_data.get(college_name, {})
                offerings = college_obj.get("offerings", {})

                # Try to match a branch
                branch_found = False
                for bname in offerings:
                    if any(word in branch_query for word in bname.lower().split() if len(word) > 3):
                        cutoff_info = get_exact_cutoffs(college_name, bname)
                        if cutoff_info:
                            exact_context = format_cutoff_context(college_name, cutoff_info)
                            context_parts.insert(0, f"[EXACT DATA]\n{exact_context}")
                            branch_found = True
                            break

                if not branch_found and offerings:
                    # If no specific branch matched, show all branches available
                    branch_list = ", ".join(offerings.keys())
                    context_parts.append(
                        f"College: {college_name} offers the following branches: {branch_list}. "
                        f"Please specify which branch's cutoff you want."
                    )

    elif intent == "process":
        # --- Search pdf_knowledge collection ---
        results = _pdf_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RESULTS,
        )

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                context_parts.append(doc)

                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                src_file = meta.get("source_file", "")
                if src_file and src_file not in seen_sources:
                    sources.append(get_source_info(src_file))
                    seen_sources.add(src_file)

    else:
        # --- Mixed: search both collections ---
        pdf_results = _pdf_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_MIXED,
        )
        college_results = _college_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_MIXED,
        )

        for results in [pdf_results, college_results]:
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    context_parts.append(doc)

                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    src_file = meta.get("source_file", "")
                    if src_file and src_file not in seen_sources:
                        sources.append(get_source_info(src_file))
                        seen_sources.add(src_file)

    # Combine all context
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found in the database."

    return {
        "context": context,
        "sources": sources,
        "intent": intent,
    }
