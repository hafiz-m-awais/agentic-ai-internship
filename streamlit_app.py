import streamlit as st
import datetime
import os
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory

# === Setup ===
st.set_page_config(page_title="Agentic Chat with Memory", layout="wide")
st.title("Agentic Chat with LLaMA 3")

# === Sidebar: Model Choice ===
model_choice = st.sidebar.selectbox("Choose LLM Model", ["mistral", "llama3", "gemma"])
embedding_model = "llama3"  # You may switch this too if needed

# === Embedding + Vector Store (Memory) ===
embedding = OllamaEmbeddings(model=embedding_model)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Tools ===
search_tool = DuckDuckGoSearchRun()
calc_tool = PythonREPLTool()

tools = [
    Tool(name="Search", func=search_tool.run, description="Use for real-time information."),
    Tool(name="Calculator", func=calc_tool.run, description="Use for Python-based calculations.")
]

# === LLM Agent ===
llm = Ollama(model=model_choice)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False
)

# === Chat History State ===
if "chat" not in st.session_state:
    st.session_state.chat = []

# === Log to File Function ===
def log_to_file(user_input, reply):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}]\nUser: {user_input}\nBot: {reply}\n\n")

# === Display Chat Messages ===
user_input = st.chat_input("Ask anything...")
for role, message in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(message)

# === Handle Input ===
if user_input:
    st.session_state.chat.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"input": user_input})
                reply = result["output"]
            except Exception as e:
                reply = f"Error: {str(e)}"
            st.markdown(reply)
            st.session_state.chat.append(("assistant", reply))
            log_to_file(user_input, reply)
