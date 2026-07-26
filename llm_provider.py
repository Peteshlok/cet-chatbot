"""
LLM Provider — The ONLY file you need to edit to switch LLM providers.

Currently configured for: Google Gemini (gemini-2.5-flash-lite)
To switch providers, change LLM_PROVIDER and LLM_MODEL below,
then add the corresponding API key to your .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ⭐ CHANGE THESE TWO LINES TO SWITCH LLM PROVIDERS
# ============================================================
LLM_PROVIDER = "gemini"                # "gemini" | "openai" | "groq" | "ollama"
LLM_MODEL = "gemini-3.5-flash"    # Model name for the chosen provider
# ============================================================


def generate(system_prompt: str, context: str, query: str, history: list[dict] = None) -> str:
    """
    Send a query to the configured LLM with system prompt, retrieved context,
    and optional conversation history. Returns the generated text response.

    Args:
        system_prompt: The system-level instructions for the LLM.
        context: The retrieved RAG context (PDF chunks + cutoff data).
        query: The student's current question.
        history: List of past messages, each as {"role": "user"|"assistant", "content": "..."}.

    Returns:
        The LLM's text response.
    """
    if history is None:
        history = []

    if LLM_PROVIDER == "gemini":
        return _generate_gemini(system_prompt, context, query, history)
    elif LLM_PROVIDER == "openai":
        return _generate_openai(system_prompt, context, query, history)
    elif LLM_PROVIDER == "groq":
        return _generate_groq(system_prompt, context, query, history)
    elif LLM_PROVIDER == "ollama":
        return _generate_ollama(system_prompt, context, query, history)
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Use 'gemini', 'openai', 'groq', or 'ollama'.")


# ============================================================
# Provider Implementations
# ============================================================

def _generate_gemini(system_prompt: str, context: str, query: str, history: list[dict]) -> str:
    """Google Gemini API implementation (using google-genai SDK)."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")

    client = genai.Client(api_key=api_key)

    # Build the conversation contents
    contents = []

    # Add conversation history
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Add the current query with the retrieved context
    user_message = f"""RETRIEVED CONTEXT:
---
{context}
---

STUDENT'S QUESTION:
{query}"""

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,      # Low temperature for factual accuracy
            max_output_tokens=1500,
        ),
    )

    return response.text


def _generate_openai(system_prompt: str, context: str, query: str, history: list[dict]) -> str:
    """OpenAI API implementation. Install: pip install openai"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")

    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current query with context
    user_message = f"""RETRIEVED CONTEXT:
---
{context}
---

STUDENT'S QUESTION:
{query}"""

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def _generate_groq(system_prompt: str, context: str, query: str, history: list[dict]) -> str:
    """Groq API implementation. Install: pip install groq"""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

    client = Groq(api_key=api_key)

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"""RETRIEVED CONTEXT:
---
{context}
---

STUDENT'S QUESTION:
{query}"""

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def _generate_ollama(system_prompt: str, context: str, query: str, history: list[dict]) -> str:
    """Ollama (local) implementation. Install and run Ollama server first."""
    import requests

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"""RETRIEVED CONTEXT:
---
{context}
---

STUDENT'S QUESTION:
{query}"""

    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3},
        },
    )
    response.raise_for_status()

    return response.json()["message"]["content"]
