"""
Configuration constants for the MHT-CET RAG Chatbot.
Edit these values to tune retrieval quality, chunk sizes, and model selection.
"""

import os
from pathlib import Path

# ----- Paths -----
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLEGES_JSON = DATA_DIR / "colleges.json"
SOURCES_JSON = DATA_DIR / "sources.json"
PROMPT_FILE = BASE_DIR / "prompt.txt"

# ----- Embedding Model (local, free) -----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ----- ChromaDB Collection Names -----
PDF_COLLECTION = "pdf_knowledge"
COLLEGE_COLLECTION = "college_cutoffs"

# ----- Chunking Parameters -----
CHUNK_SIZE = 400          # Target tokens per chunk (~300 words)
CHUNK_OVERLAP = 50        # Overlap between chunks for context continuity
CHUNK_SIZE_CHARS = CHUNK_SIZE * 4    # Approximate character count (1 token ≈ 4 chars)
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * 4

# ----- Retrieval Parameters -----
TOP_K_RESULTS = 5         # Number of chunks to retrieve per query
TOP_K_CUTOFF = 3          # Number of chunks for cutoff-specific queries
TOP_K_MIXED = 3           # Number of chunks per collection for mixed queries

# ----- Conversation -----
MAX_HISTORY_TURNS = 4     # Number of past message pairs to keep in context

# ----- Category Code Mapping -----
# Maps short category codes to human-readable names.
# Structure: PREFIX (G=General, L=Ladies, PWD=Person with Disability, DEF=Defence)
#            + CASTE (OPEN, SC, ST, VJ, NT1, NT2, NT3, OBC, SEBC)
#            + LEVEL (S=State, H=Home University, O=Other than Home University)
CATEGORY_CODES = {
    # --- General (G) - State Level (S) ---
    "GOPENS": "General - Open - State Level",
    "GSCS": "General - Scheduled Caste - State Level",
    "GSTS": "General - Scheduled Tribe - State Level",
    "GVJS": "General - Vimukta Jati (Denotified Tribe) - State Level",
    "GNT1S": "General - Nomadic Tribe 1 - State Level",
    "GNT2S": "General - Nomadic Tribe 2 (Dhangar) - State Level",
    "GNT3S": "General - Nomadic Tribe 3 - State Level",
    "GOBCS": "General - Other Backward Class - State Level",
    "GSEBCS": "General - Socially & Educationally Backward Class - State Level",

    # --- General (G) - Home University (H) ---
    "GOPENH": "General - Open - Home University",
    "GSCH": "General - Scheduled Caste - Home University",
    "GSTH": "General - Scheduled Tribe - Home University",
    "GVJH": "General - Vimukta Jati - Home University",
    "GNT1H": "General - Nomadic Tribe 1 - Home University",
    "GNT2H": "General - Nomadic Tribe 2 - Home University",
    "GNT3H": "General - Nomadic Tribe 3 - Home University",
    "GOBCH": "General - Other Backward Class - Home University",
    "GSEBCH": "General - Socially & Educationally Backward Class - Home University",

    # --- General (G) - Other than Home University (O) ---
    "GOPENO": "General - Open - Other than Home University",
    "GSCO": "General - Scheduled Caste - Other than Home University",
    "GSTO": "General - Scheduled Tribe - Other than Home University",
    "GVJO": "General - Vimukta Jati - Other than Home University",
    "GNT1O": "General - Nomadic Tribe 1 - Other than Home University",
    "GNT2O": "General - Nomadic Tribe 2 - Other than Home University",
    "GNT3O": "General - Nomadic Tribe 3 - Other than Home University",
    "GOBCO": "General - Other Backward Class - Other than Home University",
    "GSEBCO": "General - Socially & Educationally Backward Class - Other than Home University",

    # --- Ladies (L) - State Level (S) ---
    "LOPENS": "Ladies - Open - State Level",
    "LSCS": "Ladies - Scheduled Caste - State Level",
    "LSTS": "Ladies - Scheduled Tribe - State Level",
    "LVJS": "Ladies - Vimukta Jati - State Level",
    "LNT1S": "Ladies - Nomadic Tribe 1 - State Level",
    "LNT2S": "Ladies - Nomadic Tribe 2 - State Level",
    "LNT3S": "Ladies - Nomadic Tribe 3 - State Level",
    "LOBCS": "Ladies - Other Backward Class - State Level",
    "LSEBCS": "Ladies - Socially & Educationally Backward Class - State Level",

    # --- Ladies (L) - Home University (H) ---
    "LOPENH": "Ladies - Open - Home University",
    "LSCH": "Ladies - Scheduled Caste - Home University",
    "LSTH": "Ladies - Scheduled Tribe - Home University",
    "LVJH": "Ladies - Vimukta Jati - Home University",
    "LNT1H": "Ladies - Nomadic Tribe 1 - Home University",
    "LNT2H": "Ladies - Nomadic Tribe 2 - Home University",
    "LNT3H": "Ladies - Nomadic Tribe 3 - Home University",
    "LOBCH": "Ladies - Other Backward Class - Home University",
    "LSEBCH": "Ladies - Socially & Educationally Backward Class - Home University",

    # --- Ladies (L) - Other than Home University (O) ---
    "LOPENO": "Ladies - Open - Other than Home University",
    "LSCO": "Ladies - Scheduled Caste - Other than Home University",
    "LSTO": "Ladies - Scheduled Tribe - Other than Home University",
    "LVJO": "Ladies - Vimukta Jati - Other than Home University",
    "LNT1O": "Ladies - Nomadic Tribe 1 - Other than Home University",
    "LNT2O": "Ladies - Nomadic Tribe 2 - Other than Home University",
    "LNT3O": "Ladies - Nomadic Tribe 3 - Other than Home University",
    "LOBCO": "Ladies - Other Backward Class - Other than Home University",
    "LSEBCO": "Ladies - Socially & Educationally Backward Class - Other than Home University",

    # --- Persons with Disability (PWD) ---
    "PWDOPENS": "PWD - Open - State Level",
    "PWDOBCS": "PWD - OBC - State Level",
    "PWDSCS": "PWD - Scheduled Caste - State Level",
    "PWDSTS": "PWD - Scheduled Tribe - State Level",
    "PWDSEBCS": "PWD - SEBC - State Level",
    "PWDOPENH": "PWD - Open - Home University",
    "PWDOBCH": "PWD - OBC - Home University",
    "PWDSCH": "PWD - Scheduled Caste - Home University",
    "PWDSEBCH": "PWD - SEBC - Home University",
    "PWD": "Persons with Disability (General)",

    # --- PWD Reserved (PWDR) ---
    "PWDRSCS": "PWD Reserved - Scheduled Caste - State Level",
    "PWDRSCH": "PWD Reserved - Scheduled Caste - Home University",
    "PWDROBC": "PWD Reserved - OBC",
    "PWDRNT1S": "PWD Reserved - Nomadic Tribe 1 - State Level",
    "PWDRNT2S": "PWD Reserved - Nomadic Tribe 2 - State Level",
    "PWDRNT3S": "PWD Reserved - Nomadic Tribe 3 - State Level",
    "PWDRSEBC": "PWD Reserved - SEBC",
    "PWDRSTS": "PWD Reserved - Scheduled Tribe - State Level",
    "PWDRVJS": "PWD Reserved - Vimukta Jati - State Level",

    # --- Defence (DEF) ---
    "DEFOPENS": "Defence - Open - State Level",
    "DEFOBCS": "Defence - OBC - State Level",
    "DEFSCS": "Defence - Scheduled Caste - State Level",
    "DEFSTS": "Defence - Scheduled Tribe - State Level",
    "DEFSEBCS": "Defence - SEBC - State Level",

    # --- Defence Reserved (DEFR) ---
    "DEFROBCS": "Defence Reserved - OBC - State Level",
    "DEFRSCS": "Defence Reserved - Scheduled Caste - State Level",
    "DEFRNT1S": "Defence Reserved - Nomadic Tribe 1 - State Level",
    "DEFRNT2S": "Defence Reserved - Nomadic Tribe 2 - State Level",
    "DEFRNT3S": "Defence Reserved - Nomadic Tribe 3 - State Level",
    "DEFRSEBC": "Defence Reserved - SEBC",
    "DEFRSTS": "Defence Reserved - Scheduled Tribe - State Level",
    "DEFRVJS": "Defence Reserved - Vimukta Jati - State Level",

    # --- Special Categories ---
    "TFWS": "Tuition Fee Waiver Scheme",
    "EWS": "Economically Weaker Section",
    "ORPHAN": "Orphan Category",
    "VII": "VII (Visually Impaired / Institution Level)",
}


def get_category_name(code: str) -> str:
    """Return the human-readable name for a category code, or the code itself if unknown."""
    return CATEGORY_CODES.get(code, code)
