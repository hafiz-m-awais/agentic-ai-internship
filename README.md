# 🤖 Agentic AI: Task-Oriented LLM Agent with Tools & Memory

This project implements a **Task-Oriented Agentic AI System** as part of the KDD LAB Agentic AI Internship Task. Built using LangChain and a local LLM (via Ollama), the agent can reason through tasks using integrated tools and short/long-term memory. A Streamlit-based interface is also included for interactive chat.

---

## Features

- **Conversational ReAct Agent** using LangChain
- **Tool Integration**:
  - DuckDuckGo Search (real-time info)
  - Python REPL Tool (code execution)
  - Conversational Memory (short-term)
- **Local LLM Integration** via Ollama (`mistral`, `llama3`, `gemma`)
- **Streamlit Web Interface**
- **Interaction Logging** for analysis

---

## Setup Instructions

1. **Install Python (>=3.9)**

2. **Clone Repository & Install Requirements**
   ```bash
   git clone https://github.com/hafiz-m-awais/agentic-ai.git
   cd agentic-ai
   pip install -r requirements.txt

