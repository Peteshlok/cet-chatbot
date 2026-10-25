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

# ============================================================
# Indian Engineering & MHT-CET Domain Acronyms & Stopwords
# ============================================================

# Acronyms frequently used by Indian engineering students, mapped to their official brochure / database phrasing
DOMAIN_ACRONYMS_MAP = {
    # Admissions, Centres & Authorities
    "fc": "facilitation centre fc",
    "sc": "scrutiny centre sc",
    "arc": "admission reporting centre arc",
    "cap": "centralized admission process cap",
    "ara": "admissions regulating authority ara",
    "dte": "directorate of technical education dte",
    "cet": "common entrance test mht cet",
    "mht": "mht cet maharashtra",
    "pcm": "physics chemistry mathematics pcm",
    "pcb": "physics chemistry biology pcb",
    "oms": "outside maharashtra state oms",
    "ms": "maharashtra state ms",
    "hu": "home university hu",
    "ohu": "other than home university ohu",
    "sl": "state level sl",
    "ai": "all india artificial intelligence ai",
    "tfws": "tuition fee waiver scheme tfws",
    "ews": "economically weaker section ews",
    "ebc": "economically backward class ebc",
    "ncl": "non creamy layer ncl",
    "cvc": "caste validity certificate cvc",
    "tvic": "tribe validity certificate tvic",
    "tc": "transfer certificate tc",
    "lc": "leaving certificate lc",
    "ssc": "secondary school certificate 10th ssc",
    "hsc": "higher secondary certificate 12th hsc",
    "ils": "institute level seats institutional quota ils",
    "acap": "against cap vacancies acap",
    "without cap": "institutional quota vacant seats after cap non cap",
    "non cap": "institutional quota vacant seats after cap",
    "management quota": "institutional quota seats unaided",
    "institutional quota": "institutional level seats quota",
    "nri quota": "nri supernumerary quota seats non resident indian",
    "nri": "non resident indian nri",
    "oci": "overseas citizen of india oci",
    "pio": "person of indian origin pio",
    "ciwg": "children of indian workers in gulf ciwg",
    "km": "kashmiri migrant km",
    "pwd": "persons with disability pwd",
    "def": "defence category def",

    # CAP Rounds
    "r1": "cap round 1 r1",
    "r2": "cap round 2 r2",
    "r3": "cap round 3 r3",
    "r4": "cap round 4 r4",

    # Engineering Branches / Disciplines
    "cs": "computer science engineering cse cs",
    "cse": "computer science engineering cse",
    "it": "information technology it",
    "ds": "data science ds",
    "ml": "machine learning ml",
    "aiml": "artificial intelligence machine learning aiml",
    "aids": "artificial intelligence data science aids",
    "ce": "civil engineering civil ce",
    "me": "mechanical engineering mechanical me",
    "ee": "electrical engineering electrical ee",
    "ece": "electronics communication engineering ece",
    "entc": "electronics telecommunication engineering entc",
    "extc": "electronics telecommunication engineering extc",
    "etc": "electronics telecommunication engineering etc",
    "che": "chemical engineering chemical che",
    "chem": "chemical engineering chemical",
    "bt": "biotechnology bt",
    "biotech": "biotechnology biotech",
    "aero": "aerospace aeronautical aero",
    "auto": "automobile automotive auto",
    "prod": "production engineering prod",
    "meta": "metallurgy materials engineering meta",
    "inst": "instrumentation control engineering inst",
    "ice": "instrumentation control engineering ice",
    "iot": "internet of things iot",
    "csbs": "computer science business systems csbs",
    "cyber": "cyber security cyber",
    "ro": "robotics automation robotics",

    # Entrance Exams
    "jee": "joint entrance examination jee main",
    "neet": "national eligibility entrance neet",
    "gate": "graduate aptitude test gate",
    "nata": "national aptitude test architecture nata",
    "cuet": "common university entrance test cuet",
    "bitsat": "bitsat",
    "viteee": "viteee",
}

# 2-letter words that are legitimate domain acronyms and must NEVER be dropped
VALID_SHORT_TERMS = {
    # 2-letter Engineering Branches
    "cs", "it", "ai", "ds", "ml", "ce", "me", "ee", "bt", "ro",
    # 2-letter Admissions / Centres / Quotas / Certificates / Categories
    "fc", "sc", "ms", "hu", "sl", "tc", "lc", "jk", "km", "st", "vj", "dt", "nt", "mu", "su",
    # 2-letter Rounds & Standard School Classes
    "r1", "r2", "r3", "r4", "10", "12",
}

# Conversational filler words that skew BM25 relevance if left in
CONVERSATIONAL_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "an", "a", "the", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "by", "over", "there", "here", "that", "this",
    "these", "those", "from", "as", "into", "about", "tell", "give",
    "please", "can", "could", "would", "should", "does", "do", "did",
    "me", "my", "i", "you", "your", "we", "our", "them", "their",
    "any", "some", "need", "needed", "want", "know", "much", "many",
}

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
        # 1. Expand domain acronyms using word boundaries (longest first to match multi-word phrases)
        query_lower = query.lower()
        for abbr in sorted(DOMAIN_ACRONYMS_MAP.keys(), key=len, reverse=True):
            query_lower = re.sub(rf"\b{re.escape(abbr)}\b", DOMAIN_ACRONYMS_MAP[abbr], query_lower)

        # 2. Extract tokens: keep if >= 3 chars or in valid short terms, and exclude conversational stopwords
        raw_words = re.findall(r"\w+", query_lower)
        q_tokens = [
            w for w in raw_words
            if (len(w) >= 3 or w in VALID_SHORT_TERMS) and w not in CONVERSATIONAL_STOPWORDS
        ]

        # Fallback if all tokens filtered
        if not q_tokens:
            q_tokens = [w for w in raw_words if len(w) >= 2 and w not in CONVERSATIONAL_STOPWORDS]
            if not q_tokens:
                q_tokens = [w for w in raw_words if len(w) >= 2]

        scores = []
        for i, d in enumerate(self.corpus):
            words = Counter(re.findall(r"\w+", d["text"].lower()))
            score = 0.0
            # Use set(q_tokens) so expanded synonym terms don't artificially duplicate tf score
            for q in set(q_tokens):
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
    "cutoff", "cut off", "cut-off", "percentile", "closing rank",
    "opening rank", "min rank", "what rank", "what score", "what marks",
    "marks required", "percentile required", "score needed", "rank needed",
    "cutoffs", "last year cutoff", "previous year cutoff",
    "gopens", "gscs", "gsts", "gobcs", "lopens", "lscs", "lsts", "lobcs",
    "gsebcs", "round 1 cutoff", "round 2 cutoff", "round 3 cutoff",
    "r1 cutoff", "r2 cutoff", "r3 cutoff", "cap round cutoff",
}

PROCESS_KEYWORDS = {
    "process", "admission", "counselling", "counseling", "document",
    "verification", "scrutiny", "e-scrutiny", "escrutiny", "brochure",
    "eligibility", "registration", "fee", "schedule", "timeline",
    "rule", "rules", "regulation", "procedure", "domicile", "certificate",
    "normalization", "normalisation", "methodology", "syllabus",
    "exam", "pattern", "attempt", "objection", "press note",
    "centre", "center", "option form", "freeze", "float", "slide",
    "how to", "how can", "can i get", "steps to", "without cap",
    "non cap", "management quota", "institutional", "institute level",
    "ils", "against cap", "acap", "vacancy round", "spot round",
    "nri", "nri quota", "supernumerary", "quota", "reservation",
    "quota rules", "seat matrix", "seats available",
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

    # Check for procedural or "how-to" phrasing
    is_how_to = any(p in query_lower for p in [
        "how to", "how can", "can i get", "procedure", "process", "steps",
        "without cap", "rules", "rule", "eligibility", "documents required"
    ])

    # Check for explicit cutoff queries
    has_cutoff = any(kw in query_lower for kw in CUTOFF_KEYWORDS) or (
        ("cutoff" in query_lower or "cut off" in query_lower or "percentile" in query_lower) and
        not is_how_to
    )
    has_process = any(kw in query_lower for kw in PROCESS_KEYWORDS)
    has_college = any(kw in query_lower for kw in COLLEGE_KEYWORDS) or any(alias in query_lower for alias in COLLEGE_ALIASES)

    # 1. Procedural questions take precedence over pure cutoff lookups
    if is_how_to and not has_cutoff:
        return "process"

    # 2. Strict cutoff questions
    if has_cutoff:
        return "cutoff"

    # 3. General process & admission rule questions
    if has_process:
        return "process"

    # 4. College info / branch offerings
    if has_college:
        return "college_info"

    return "mixed"


# ============================================================
# Fuzzy College Name Matching
# ============================================================

COLLEGE_ALIASES = {
    "coep": "coep technological university",
    "vjti": "veermata jijabai",
    "pict": "pune institute of computer technology",
    "cummins": "cummins",
    "walchand": "walchand college of engineering",
    "wce": "walchand college of engineering",
    "mit": "maharashtra institute of technology",
    "spit": "sardar patel institute",
    "spce": "sardar patel college of engineering",
    "djsce": "dwarkadas j. sanghvi",
    "sanghvi": "dwarkadas j. sanghvi",
    "kjsce": "somaiya",
    "tsec": "thadomal shahani",
    "vit": "vishwakarma institute of technology",
    "viit": "vishwakarma institute of information technology",
    "pccoe": "pimpri chinchwad college of engineering",
    "pccoep": "pimpri chinchwad college of engineering",
    "vesit": "vivekanand education society",
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

        # Quota & policy safety net: if user asked about quotas or rules, always include brochure chunks
        QUOTA_POLICY_KEYWORDS = [
            "quota", "nri", "ils", "institutional", "management",
            "supernumerary", "without cap", "against cap", "reservation",
            "tfws", "minority", "defence", "pwd", "rules", "rule", "procedure"
        ]
        if any(k in query.lower() for k in QUOTA_POLICY_KEYWORDS):
            pdf_docs = _pdf_bm25.search(query, top_k=TOP_K_RESULTS)
            for d in pdf_docs:
                context_parts.append(d["text"])
                src_file = d.get("metadata", {}).get("source_file", "")
                if src_file and src_file not in seen_sources:
                    sources.append(get_source_info(src_file))
                    seen_sources.add(src_file)

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
