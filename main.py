import os
import datetime
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# Initialize the LLM
llm = OllamaLLM(model="mistral")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tools
search_tool = DuckDuckGoSearchRun()
calc_tool = PythonREPLTool()

tools = [
    Tool(name="Search", func=search_tool.run, description="Use for real-time search queries."),
    Tool(name="Calculator", func=calc_tool.run, description="Useful for math and Python calculations.")
]

# Agent initialization
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

# Logger function
def log_interaction(query, result):
    with open("agent_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n---\n[{datetime.datetime.now()}]\nUSER: {query}\nAGENT: {result['output']}\n")

# Main loop
if __name__ == "__main__":
    print("Mistral Agent with Memory is Ready!")
    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            result = agent.invoke({"input": query})
            print(f"\nResponse:\n{result['output']}\n")
            log_interaction(query, result)
        except Exception as e:
            print(f"Error: {e}")
