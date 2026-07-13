/**
 * RankMatch MHT-CET Chatbot — Frontend Logic
 *
 * Handles:
 * - Chat message send/receive
 * - Conversation history management
 * - Markdown → HTML rendering
 * - Source citation rendering
 * - College name autocomplete
 * - Suggested questions
 */

// ============================================================
// State
// ============================================================
const state = {
    history: [],        // [{role: "user"|"assistant", content: "..."}]
    collegeNames: [],   // For autocomplete
    isLoading: false,
};

// ============================================================
// DOM Elements
// ============================================================
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const welcomeContainer = document.getElementById("welcome-container");
const suggestionsGrid = document.getElementById("suggestions-grid");
const autocompleteContainer = document.getElementById("autocomplete-container");
const autocompleteList = document.getElementById("autocomplete-list");

// ============================================================
// Initialization
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    loadSuggestions();
    loadCollegeNames();
    setupInputHandlers();
});

async function loadSuggestions() {
    try {
        const res = await fetch("/api/suggest");
        const data = await res.json();
        renderSuggestions(data.suggestions);
    } catch {
        // Fallback suggestions
        renderSuggestions([
            "What is the MHT-CET admission process?",
            "How does normalisation work?",
            "What documents are needed for verification?",
            "What are the cutoffs for COEP Computer Science?",
            "Explain the CAP round counselling process",
            "What is TFWS and how to apply?",
        ]);
    }
}

async function loadCollegeNames() {
    try {
        const res = await fetch("/api/colleges");
        const data = await res.json();
        state.collegeNames = data.colleges || [];
    } catch {
        state.collegeNames = [];
    }
}

function renderSuggestions(suggestions) {
    suggestionsGrid.innerHTML = suggestions
        .map(
            (s) =>
                `<button class="suggestion-card" onclick="sendSuggestion(this)" data-message="${escapeAttr(s)}">${escapeHtml(s)}</button>`
        )
        .join("");
}

// ============================================================
// Input Handlers
// ============================================================
function setupInputHandlers() {
    // Auto-resize textarea
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";

        // Enable/disable send button
        sendBtn.disabled = chatInput.value.trim() === "";

        // Autocomplete
        handleAutocomplete(chatInput.value);
    });

    // Enter to send, Shift+Enter for newline
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && !state.isLoading) {
                sendMessage();
            }
        }

        // Navigate autocomplete with arrow keys
        if (autocompleteContainer.classList.contains("active")) {
            const items = autocompleteList.querySelectorAll("li");
            const activeItem = autocompleteList.querySelector("li.active");
            let activeIndex = Array.from(items).indexOf(activeItem);

            if (e.key === "ArrowDown") {
                e.preventDefault();
                if (activeIndex < items.length - 1) {
                    items[activeIndex]?.classList.remove("active");
                    items[activeIndex + 1].classList.add("active");
                }
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                if (activeIndex > 0) {
                    items[activeIndex]?.classList.remove("active");
                    items[activeIndex - 1].classList.add("active");
                }
            } else if (e.key === "Tab" || e.key === "Escape") {
                e.preventDefault();
                hideAutocomplete();
            }
        }
    });

    sendBtn.addEventListener("click", () => {
        if (!state.isLoading) sendMessage();
    });

    // Hide autocomplete on click outside
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".input-wrapper")) {
            hideAutocomplete();
        }
    });
}

// ============================================================
// Autocomplete
// ============================================================
function handleAutocomplete(value) {
    const words = value.split(/\s+/);
    const lastWord = words[words.length - 1]?.toLowerCase() || "";

    if (lastWord.length < 2) {
        hideAutocomplete();
        return;
    }

    const matches = state.collegeNames.filter((name) =>
        name.toLowerCase().includes(lastWord)
    ).slice(0, 6);

    if (matches.length === 0) {
        hideAutocomplete();
        return;
    }

    autocompleteList.innerHTML = matches
        .map((name) => {
            const highlighted = name.replace(
                new RegExp(`(${escapeRegex(lastWord)})`, "gi"),
                "<mark>$1</mark>"
            );
            return `<li onclick="selectAutocomplete('${escapeAttr(name)}')">${highlighted}</li>`;
        })
        .join("");

    autocompleteContainer.classList.add("active");
}

function selectAutocomplete(name) {
    const words = chatInput.value.split(/\s+/);
    words[words.length - 1] = name;
    chatInput.value = words.join(" ") + " ";
    chatInput.focus();
    hideAutocomplete();
    sendBtn.disabled = false;
}

function hideAutocomplete() {
    autocompleteContainer.classList.remove("active");
}

// ============================================================
// Send & Receive Messages
// ============================================================
function sendSuggestion(btn) {
    const message = btn.dataset.message;
    chatInput.value = message;
    sendMessage();
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || state.isLoading) return;

    // Hide welcome screen
    if (welcomeContainer) {
        welcomeContainer.style.display = "none";
    }

    // Add user message to UI
    appendMessage("user", message);

    // Add to history
    state.history.push({ role: "user", content: message });

    // Clear input
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;
    hideAutocomplete();

    // Show thinking indicator
    const thinkingEl = showThinking();

    // Send to API
    state.isLoading = true;

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                history: state.history.slice(-8), // Last 4 exchanges (8 messages)
            }),
        });

        const data = await res.json();

        // Remove thinking indicator
        removeThinking(thinkingEl);

        if (data.error) {
            appendMessage("bot", `⚠️ ${data.error}`, [], []);
        } else {
            // Add bot response to UI with sources and suggestions
            appendMessage("bot", data.response, data.sources || [], data.suggestions || []);

            // Add to history
            state.history.push({ role: "assistant", content: data.response });
        }
    } catch (err) {
        removeThinking(thinkingEl);
        appendMessage(
            "bot",
            "I'm sorry, I couldn't connect to the server. Please make sure the backend is running and try again.",
            [],
            []
        );
    }

    state.isLoading = false;
    chatInput.focus();
}

// ============================================================
// UI Rendering
// ============================================================
function appendMessage(role, content, sources = [], suggestions = []) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    if (role === "bot") {
        row.innerHTML = `
            <div class="message-avatar">RM</div>
            <div>
                <div class="message-bubble">${renderMarkdown(content)}</div>
                ${renderSources(sources)}
                ${renderFollowUps(suggestions)}
            </div>
        `;
    } else {
        row.innerHTML = `
            <div class="message-bubble">${escapeHtml(content)}</div>
        `;
    }

    chatMessages.appendChild(row);
    scrollToBottom();
}

function showThinking() {
    const row = document.createElement("div");
    row.className = "thinking-row";
    row.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, hsl(245, 80%, 62%), hsl(170, 75%, 50%)); color: white; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700;">RM</div>
        <div class="thinking-bubble">
            <span>Searching CET Cell documents</span>
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom();
    return row;
}

function removeThinking(el) {
    if (el && el.parentNode) {
        el.parentNode.removeChild(el);
    }
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return "";

    const chips = sources
        .map((src) => {
            const linkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>`;
            if (src.url) {
                return `<a class="source-chip" href="${escapeAttr(src.url)}" target="_blank" rel="noopener">${linkIcon} ${escapeHtml(src.title)}</a>`;
            }
            return `<span class="source-chip">${linkIcon} ${escapeHtml(src.title)}</span>`;
        })
        .join("");

    return `
        <div class="sources-container">
            <div class="sources-label">Sources</div>
            <div class="source-chips">${chips}</div>
        </div>
    `;
}

function renderFollowUps(suggestions) {
    if (!suggestions || suggestions.length === 0) return "";

    const chips = suggestions
        .map(
            (s) =>
                `<button class="follow-up-chip" onclick="sendFollowUp(this)" data-message="${escapeAttr(s)}">${escapeHtml(s)}</button>`
        )
        .join("");

    return `<div class="follow-up-suggestions">${chips}</div>`;
}

function sendFollowUp(btn) {
    chatInput.value = btn.dataset.message;
    sendMessage();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// ============================================================
// Markdown Rendering (lightweight, no external dependencies)
// ============================================================
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Headings (### → h3, ## → h2, # → h1)
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Links: [text](url)
    html = html.replace(
        /\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );

    // Tables
    html = renderTables(html);

    // Unordered lists
    html = html.replace(/^[*\-] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

    // Paragraphs (double newline)
    html = html.replace(/\n\n/g, "</p><p>");
    // Single newlines within paragraphs
    html = html.replace(/\n/g, "<br>");

    // Wrap in paragraph
    html = `<p>${html}</p>`;

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, "");
    // Clean up paragraphs wrapping block elements
    html = html.replace(/<p>(<h[1-3]>)/g, "$1");
    html = html.replace(/(<\/h[1-3]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul>)/g, "$1");
    html = html.replace(/(<\/ul>)<\/p>/g, "$1");
    html = html.replace(/<p>(<table>)/g, "$1");
    html = html.replace(/(<\/table>)<\/p>/g, "$1");

    // Source emoji formatting
    html = html.replace(/📄/g, "📄");

    return html;
}

function renderTables(html) {
    // Match markdown tables: header | row, separator |---|, data rows
    const tableRegex = /(\|.+\|)\n(\|[\s\-:|]+\|)\n((?:\|.+\|\n?)+)/g;

    return html.replace(tableRegex, (match, headerRow, separator, bodyRows) => {
        const headers = headerRow
            .split("|")
            .filter((c) => c.trim())
            .map((c) => `<th>${c.trim()}</th>`)
            .join("");

        const rows = bodyRows
            .trim()
            .split("\n")
            .map((row) => {
                const cells = row
                    .split("|")
                    .filter((c) => c.trim())
                    .map((c) => `<td>${c.trim()}</td>`)
                    .join("");
                return `<tr>${cells}</tr>`;
            })
            .join("");

        return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    });
}

// ============================================================
// Utility Functions
// ============================================================
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    return text.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
