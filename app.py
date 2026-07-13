"""
Flask API Server — Entry point for the MHT-CET RAG Chatbot.

Endpoints:
    GET  /              → Serves the chat frontend
    POST /api/chat      → Main chat endpoint (query → RAG → response)
    GET  /api/colleges  → List all college names (for autocomplete)
    GET  /api/suggest   → Get starter question suggestions
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import json

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import COLLEGES_JSON, STATIC_DIR
from generator import generate_response

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# Pre-load college names for the autocomplete endpoint
_college_names = None


def _get_college_names() -> list[str]:
    global _college_names
    if _college_names is None:
        with open(COLLEGES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        _college_names = sorted(data.keys())
    return _college_names


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Serve the chat frontend."""
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static files (CSS, JS, etc.)."""
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request JSON:
        {
            "message": "What is the cutoff for COEP CS?",
            "history": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    Response JSON:
        {
            "response": "The GOPENS cutoff for...",
            "sources": [{"title": "...", "url": "..."}],
            "intent": "cutoff",
            "suggestions": ["Compare with...", "..."]
        }
    """
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body."}), 400

    message = data["message"].strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Generate response through the full RAG pipeline
    result = generate_response(query=message, history=history)

    return jsonify(result)


@app.route("/api/colleges", methods=["GET"])
def colleges():
    """Return all college names for frontend autocomplete."""
    return jsonify({"colleges": _get_college_names()})


@app.route("/api/suggest", methods=["GET"])
def suggest():
    """Return starter question suggestions for new users."""
    return jsonify({
        "suggestions": [
            "What is the MHT-CET admission process?",
            "How does normalisation work in MHT-CET?",
            "What documents are needed for document verification?",
            "What are the cutoffs for COEP Computer Science?",
            "Explain the CAP round counselling process",
            "What is TFWS and how do I apply for it?",
        ]
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys
    # Force UTF-8 encoding for standard output on Windows to support emojis
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("=" * 50)
    print("🤖 MHT-CET RAG Chatbot Server")
    print("   Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
