"""
Response Generator — Takes retrieved context + query → LLM → formatted response.

This module is provider-agnostic. It calls llm_provider.generate() which handles
the actual LLM communication.
"""

from config import MAX_HISTORY_TURNS, PROMPT_FILE
from llm_provider import generate
from retriever import retrieve


# Load system prompt once
_system_prompt = None


def _get_system_prompt() -> str:
    """Load and cache the system prompt from prompt.txt."""
    global _system_prompt
    if _system_prompt is None:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            _system_prompt = f.read().strip()
    return _system_prompt


def generate_response(query: str, history: list[dict] = None) -> dict:
    """
    Full RAG pipeline: retrieve context → generate response → attach sources.

    Args:
        query: The student's current question.
        history: List of past messages [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        {
            "response": str,          # The LLM's response text
            "sources": list[dict],    # [{title, url}] for citations
            "intent": str,            # Detected intent type
            "suggestions": list[str], # Follow-up question suggestions
        }
    """
    if history is None:
        history = []

    # Trim history to configured maximum
    trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]

    # Step 1: Retrieve relevant context
    retrieval_result = retrieve(query)
    context = retrieval_result["context"]
    sources = retrieval_result["sources"]
    intent = retrieval_result["intent"]

    # Step 2: Append source info to context so the LLM can cite them
    source_instruction = ""
    if sources:
        source_lines = []
        for src in sources:
            if src["url"]:
                source_lines.append(f"- {src['title']}: {src['url']}")
            else:
                source_lines.append(f"- {src['title']} (URL not available)")
        source_instruction = (
            "\n\nAVAILABLE SOURCES (use these for citations):\n"
            + "\n".join(source_lines)
        )

    full_context = context + source_instruction

    # Step 3: Generate response via the configured LLM provider
    system_prompt = _get_system_prompt()

    try:
        response_text = generate(
            system_prompt=system_prompt,
            context=full_context,
            query=query,
            history=trimmed_history,
        )
    except Exception as e:
        response_text = (
            f"I'm sorry, I encountered an error while generating a response. "
            f"Please try again in a moment.\n\nError details: {str(e)}"
        )

    # Step 4: Generate follow-up suggestions based on intent
    suggestions = _get_suggestions(intent, query)

    return {
        "response": response_text,
        "sources": sources,
        "intent": intent,
        "suggestions": suggestions,
    }


def _get_suggestions(intent: str, query: str) -> list[str]:
    """Generate contextual follow-up question suggestions."""
    if intent == "cutoff":
        return [
            "Compare this with another college?",
            "What about other branches in this college?",
            "What is the TFWS cutoff here?",
            "Which colleges in this region have lower cutoffs?",
        ]
    elif intent == "process":
        return [
            "What documents do I need for verification?",
            "How does the counselling round work?",
            "What are the important dates?",
            "Explain option form filling",
        ]
    elif intent == "college_info":
        return [
            "What are the cutoffs for this college?",
            "What branches are available here?",
            "Compare with similar colleges",
            "Top colleges in this region?",
        ]
    else:
        return [
            "What is the MHT-CET admission process?",
            "Show me cutoffs for a specific college",
            "How does normalisation work?",
            "What documents are needed for verification?",
        ]
