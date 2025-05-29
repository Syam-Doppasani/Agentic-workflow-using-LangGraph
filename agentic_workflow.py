!pip install -q langgraph
!pip install -q langsmith langchain langgraph transformers accelerate duckduckgo-search
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Agent-FreeModel"
os.environ["LANGCHAIN_API_KEY"] = "# 🔑 Replace with your LangSmith API key"  
from langchain_core.runnables import RunnableLambda
from transformers import pipeline
import re
from typing import TypedDict, List, Dict, Any

# Free Hugging Face model
generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b", max_new_tokens=150)

# Plan Agent
def plan_agent(state):
    user_query = state["query"]
    prompt = f"""You are a planner agent. Break down the user query into clear subtasks.

User Query: "{user_query}"

Subtasks:
1."""
    generated = generator(prompt, do_sample=True, truncation=True)[0]["generated_text"]
    subtasks = re.findall(r"\d+\.\s+(.*)", generated)
    return {
        "query": user_query,
        "subtasks": [{"task": t.strip()} for t in subtasks],
        "results": []
    }

# Tool Agent (Search or simulate result)
from duckduckgo_search import DDGS
def tool_agent(state):
    results = state["results"]
    subtasks = state["subtasks"]

    if len(results) < len(subtasks):
        task = subtasks[len(results)]["task"]

        if "search" in task.lower() or "what is" in task.lower():
            with DDGS() as ddgs:
                result = next(ddgs.text(task), {}).get("body", "No result found.")
        else:
            result = f"Simulated result for: {task}"

        results.append({"task": task, "result": result})

    return {
        "query": state["query"],
        "subtasks": subtasks,
        "results": results
    }
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    query: str
    subtasks: List[Dict[str, str]]
    results: List[Dict[str, str]]

plan_node = RunnableLambda(plan_agent)
tool_node = RunnableLambda(tool_agent)

builder = StateGraph(GraphState)
builder.add_node("PlanAgent", plan_node)
builder.add_node("ToolAgent", tool_node)

builder.set_entry_point("PlanAgent")
builder.add_edge("PlanAgent", "ToolAgent")
builder.add_conditional_edges(
    "ToolAgent",
    lambda state: END if len(state["results"]) == len(state["subtasks"]) else "ToolAgent"
)

builder.set_finish_point("ToolAgent")
graph = builder.compile()
user_input = input("🔹 Enter your query: ")

initial_state = {
    "query": user_input,
    "subtasks": [],
    "results": []
}

output = graph.invoke(initial_state)

print("\n📌 Subtasks:")
for task in output["subtasks"]:
    print(f"🔹 {task['task']}")

print("\n✅ Results:")
for r in output["results"]:
    print(f"🔹 {r['task']} → {r['result']}")
