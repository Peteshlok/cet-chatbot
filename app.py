"""
Flask API Server — Entry point for the MHT-CET RAG Chatbot.

Endpoints:
    GET  /              → Serves the chat frontend
    POST /api/chat      → Main chat endpoint (query → RAG → response)
    GET  /api/colleges  → List all college names (for autocomplete)
    GET  /api/suggest   → Get starter question suggestions
"""

import os
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import json

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import (
    DATA_DIR,
    COLLEGES_JSON,
    COLLEGE_NAMES_JSON,
    STATIC_DIR,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    PDF_COLLECTION,
    COLLEGE_COLLECTION,
)
from generator import generate_response

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# Pre-load college names for the autocomplete endpoint
_college_names = None


def _get_college_names() -> list[str]:
    global _college_names
    if _college_names is None:
        if COLLEGE_NAMES_JSON.exists():
            with open(COLLEGE_NAMES_JSON, "r", encoding="utf-8") as f:
                _college_names = json.load(f)
        else:
            with open(COLLEGES_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            _college_names = sorted(data.keys())
    return _college_names


# ============================================================
# Routes
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Lightweight health check endpoint for uptime pingers & hosting platforms."""
    return jsonify({"status": "healthy", "service": "cet-chatbot"}), 200


@app.route("/api/debug", methods=["GET"])
def debug():
    """Diagnostic endpoint to inspect memory and subsystem loading."""
    steps = {}
    try:
        import psutil
        p = psutil.Process(os.getpid())
        steps["mem_mb"] = round(p.memory_info().rss / 1e6, 1)
    except Exception as e:
        steps["mem_err"] = str(e)

    try:
        from retriever import retrieve
        res = retrieve("What documents are needed for verification?")
        steps["bm25_test"] = "ok"
        steps["context_chars"] = len(res.get("context", ""))
        steps["sources_count"] = len(res.get("sources", []))
    except Exception as e:
        steps["bm25_err"] = str(e)

    return jsonify(steps), 200


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
    user_context = data.get("context")

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # If context (percentile, category, branch) is provided by the frontend, augment the query
    context_prefix = ""
    if user_context and isinstance(user_context, dict):
        ctx_parts = []
        if user_context.get("score"):
            ctx_parts.append(f"Student's MHT-CET Percentile/Score: {user_context['score']}")
        if user_context.get("category"):
            ctx_parts.append(f"Category: {user_context['category']}")
        if user_context.get("branch"):
            ctx_parts.append(f"Preferred Branch: {user_context['branch']}")
        if ctx_parts:
            context_prefix = f"[STUDENT CONTEXT: {', '.join(ctx_parts)}]\n"

    final_query = f"{context_prefix}{message}" if context_prefix else message

    try:
        # Generate response through the full RAG pipeline
        result = generate_response(query=final_query, history=history)
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
        log_query(message, result.get("intent", "unknown"), client_ip)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "response": f"⚠️ An error occurred while processing your request: {str(e)}",
            "sources": [],
            "intent": "error",
            "suggestions": []
        }), 200


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
# Analytics & Usage Tracking
# ============================================================

ANALYTICS_DB = DATA_DIR / "analytics.db"
STATS_SECRET_KEY = os.getenv("STATS_SECRET_KEY", "rm_secret_2026")


def _init_analytics_db():
    import sqlite3
    with sqlite3.connect(ANALYTICS_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                intent TEXT,
                ip TEXT
            )
        """)
        conn.commit()


def log_query(query: str, intent: str, ip: str = ""):
    """Log student questions anonymously to SQLite for usage tracking."""
    try:
        import sqlite3
        _init_analytics_db()
        with sqlite3.connect(ANALYTICS_DB) as conn:
            conn.execute(
                "INSERT INTO queries (query, intent, ip) VALUES (?, ?, ?)",
                (query, intent, ip)
            )
            conn.commit()
    except Exception as e:
        print(f"[Analytics] Error logging query: {e}")


@app.route("/api/stats", methods=["GET"])
def stats():
    """Dashboard showing total queries asked and list of recent questions (password protected)."""
    provided_key = request.args.get("key") or request.headers.get("X-Stats-Key")
    if provided_key != STATS_SECRET_KEY:
        if request.args.get("format") == "json":
            return jsonify({"error": "Unauthorized: Invalid or missing access key."}), 401

        error_banner = "<p style='color: #ef4444; font-size: 13px; margin-bottom: 12px;'>⚠️ Incorrect password. Please try again.</p>" if provided_key else ""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>401 Unauthorized</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b1120; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
        .card {{ background: #131d33; padding: 36px 32px; border-radius: 16px; border: 1px solid #1e293b; max-width: 380px; width: 90%; text-align: center; }}
        h1 {{ font-size: 38px; margin: 0 0 10px; }}
        h2 {{ font-size: 20px; margin: 0 0 8px; color: #38bdf8; }}
        p {{ color: #94a3b8; font-size: 13px; margin: 0 0 20px; }}
        input {{ width: 100%; box-sizing: border-box; padding: 12px 14px; border-radius: 8px; border: 1px solid #334155; background: #0b1120; color: #fff; margin-bottom: 12px; font-size: 14px; outline: none; }}
        input:focus {{ border-color: #38bdf8; }}
        button {{ width: 100%; padding: 12px; border-radius: 8px; border: none; background: #0284c7; color: #fff; font-weight: 600; cursor: pointer; font-size: 14px; }}
        button:hover {{ background: #0369a1; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🔒</h1>
        <h2>Admin Access Required</h2>
        <p>Please enter your access key to view chatbot analytics.</p>
        {error_banner}
        <form method="GET" action="/api/stats">
            <input type="password" name="key" placeholder="Enter admin password..." required autofocus />
            <button type="submit">Unlock Dashboard</button>
        </form>
    </div>
</body>
</html>""", 401

    import sqlite3
    _init_analytics_db()

    with sqlite3.connect(ANALYTICS_DB) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM queries")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as today FROM queries WHERE date(timestamp) = date('now')")
        today = cur.fetchone()["today"]

        cur.execute("SELECT query, intent, timestamp, ip FROM queries ORDER BY id DESC LIMIT 100")
        recent = [dict(r) for r in cur.fetchall()]

    if request.args.get("format") == "json":
        return jsonify({"total_queries": total, "today_queries": today, "recent_queries": recent})

    import html
    rows_html = "".join(
        f"<tr><td>{r['timestamp']}</td><td><strong>{html.escape(r['query'])}</strong></td><td><span class='badge'>{html.escape(r['intent'])}</span></td></tr>"
        for r in recent
    )

    dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RankMatch Chatbot Analytics</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b1120; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }}
        h1 {{ margin: 0; font-size: 22px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
        .refresh-btn {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 6px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; text-decoration: none; }}
        .refresh-btn:hover {{ background: #334155; color: #fff; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #131d33; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; }}
        .card-num {{ font-size: 34px; font-weight: 700; color: #38bdf8; }}
        .card-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
        .table-wrap {{ background: #131d33; border-radius: 12px; border: 1px solid #1e293b; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px 16px; text-align: left; font-size: 13px; border-bottom: 1px solid #1e293b; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: #1a2744; }}
        .badge {{ background: #0284c7; color: #f0f9ff; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RankMatch Chatbot Queries</h1>
            <a href="/api/stats?key={html.escape(provided_key)}" class="refresh-btn">🔄 Refresh</a>
        </div>
        <div class="cards">
            <div class="card">
                <div class="card-num">{total}</div>
                <div class="card-label">Total Questions Asked</div>
            </div>
            <div class="card">
                <div class="card-num">{today}</div>
                <div class="card-label">Questions Today</div>
            </div>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr><th style="width: 170px;">Date & Time (UTC)</th><th>Student Question</th><th style="width: 120px;">Category</th></tr>
                </thead>
                <tbody>
                    {rows_html if rows_html else '<tr><td colspan="3" style="text-align: center; color: #94a3b8; padding: 32px;">No questions asked yet.</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""
    return dashboard


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
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("🤖 MHT-CET RAG Chatbot Server")
    print(f"   Open http://localhost:{port} in your browser")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=port)
