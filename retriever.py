"""
Lightweight BM25 + Exact JSON Retrieval Engine.

Replaces heavy PyTorch and SentenceTransformers with fast Okapi BM25 and exact JSON lookup.
Memory usage: ~40 MB (fits easily on free tier with 470MB headroom).
"""

import os
import json
import math
import re
import difflib
from collections import Counter

from config import (
    COLLEGES_JSON,
    COLLEGE_NAMES_JSON,
    PDF_CHUNKS_JSON,
    COLLEGE_CHUNKS_JSON,
    SOURCES_JSON,
    TOP_K_CUTOFF,
    TOP_K_MIXED,
    TOP_K_RESULTS,
    get_category_name,
)

# ---- BM25 Search Engine ----
class SimpleBM25:
    """Fast, pure-Python Okapi BM25 search engine with zero external dependencies."""

    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(re.findall(r"\w+", d["text"].lower())) for d in corpus]
        self.avg_len = sum(self.doc_len) / max(len(corpus), 1)
        self.df = Counter()
        for d in corpus:
            words = set(re.findall(r"\w+", d["text"].lower()))
            for w in words:
                self.df[w] += 1
        self.idf = {
            w: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
            for w, count in self.df.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q_tokens = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        if not q_tokens:
            return []
        scores = []
        for i, d in enumerate(self.corpus):
            words = Counter(re.findall(r"\w+", d["text"].lower()))
            score = 0.0
            for q in q_tokens:
                if q in words:
                    tf = words[q]
                    idf = self.idf.get(q, 0.0)
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avg_len))
                    )
            scores.append((score, d))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scores[:top_k] if s > 0]


# ---- Lazy-loaded globals ----
_pdf_bm25 = None
_college_bm25 = None
_colleges_data = None
_college_names = None
_sources_data = None


def _get_colleges_data() -> dict:
    """Lazy-load the 4.5MB colleges.json only when exact cutoff data is requested."""
    global _colleges_data
    if _colleges_data is None:
        with open(COLLEGES_JSON, "r", encoding="utf-8") as f:
            _colleges_data = json.load(f)
    return _colleges_data


def _init():
    """Lazy-initialize BM25 indices and metadata."""
    global _pdf_bm25, _college_bm25, _college_names, _sources_data

    if _pdf_bm25 is not None:
        return

    with open(PDF_CHUNKS_JSON, "r", encoding="utf-8") as f:
        pdf_chunks = json.load(f)
    _pdf_bm25 = SimpleBM25(pdf_chunks)

    with open(COLLEGE_CHUNKS_JSON, "r", encoding="utf-8") as f:
        college_chunks = json.load(f)
    _college_bm25 = SimpleBM25(college_chunks)

    if COLLEGE_NAMES_JSON.exists():
        with open(COLLEGE_NAMES_JSON, "r", encoding="utf-8") as f:
            _college_names = json.load(f)
    else:
        _college_names = list(_get_colleges_data().keys())

    try:
        with open(SOURCES_JSON, "r", encoding="utf-8") as f:
            _sources_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _sources_data = {}


# ============================================================
# Intent Classification (keyword-based, zero cost)
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
    """Classify user intent based on keyword matching."""
    query_lower = query.lower()

    cutoff_score = sum(1 for kw in CUTOFF_KEYWORDS if kw in query_lower)
    process_score = sum(1 for kw in PROCESS_KEYWORDS if kw in query_lower)
    college_score = sum(1 for kw in COLLEGE_KEYWORDS if kw in query_lower)

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
    """Find college names matching the query using fuzzy matching."""
    _init()
    query_lower = query.lower()

    for alias, full_name in COLLEGE_ALIASES.items():
        if alias in query_lower:
            matches = [
                name for name in _college_names
                if full_name.lower() in name.lower()
            ]
            if matches:
                return matches[:top_n]

    matches = difflib.get_close_matches(
        query_lower, [n.lower() for n in _college_names], n=top_n, cutoff=0.3
    )

    result = []
    for match in matches:
        for name in _college_names:
            if name.lower() == match and name not in result:
                result.append(name)
                break

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
    """Get exact cutoff data from colleges.json for a specific college and branch."""
    _init()
    colleges_data = _get_colleges_data()

    if college_name not in colleges_data:
        return None

    college = colleges_data[college_name]

    if branch_name is None:
        return college

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
    Main retrieval function.
    Combines BM25 text search with exact JSON lookup.
    """
    _init()

    intent = classify_intent(query)
    context_parts = []
    sources = []
    seen_sources = set()

    if intent == "cutoff" or intent == "college_info":
        docs = _college_bm25.search(query, top_k=TOP_K_CUTOFF)
        for d in docs:
            context_parts.append(d["text"])
            src_file = d.get("metadata", {}).get("source_file", "")
            if src_file and src_file not in seen_sources:
                sources.append(get_source_info(src_file))
                seen_sources.add(src_file)

        if intent == "cutoff":
            matching_colleges = find_matching_colleges(query)
            for college_name in matching_colleges:
                branch_query = query.lower()
                college_obj = _get_colleges_data().get(college_name, {})
                offerings = college_obj.get("offerings", {})

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
                    branch_list = ", ".join(offerings.keys())
                    context_parts.append(
                        f"College: {college_name} offers the following branches: {branch_list}. "
                        f"Please specify which branch's cutoff you want."
                    )

    elif intent == "process":
        docs = _pdf_bm25.search(query, top_k=TOP_K_RESULTS)
        for d in docs:
            context_parts.append(d["text"])
            src_file = d.get("metadata", {}).get("source_file", "")
            if src_file and src_file not in seen_sources:
                sources.append(get_source_info(src_file))
                seen_sources.add(src_file)

    else:
        pdf_docs = _pdf_bm25.search(query, top_k=TOP_K_MIXED)
        col_docs = _college_bm25.search(query, top_k=TOP_K_MIXED)
        for d in pdf_docs + col_docs:
            context_parts.append(d["text"])
            src_file = d.get("metadata", {}).get("source_file", "")
            if src_file and src_file not in seen_sources:
                sources.append(get_source_info(src_file))
                seen_sources.add(src_file)

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found in the database."

    return {
        "context": context,
        "sources": sources,
        "intent": intent,
    }
