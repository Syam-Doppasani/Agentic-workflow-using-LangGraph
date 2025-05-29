!pip install -q langgraph
!pip install -q langsmith langchain openai --upgrade

import os
from typing import TypedDict, List, Dict, Any
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

# Optional LangSmith integration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Agentic-Workflow"

# Define Graph State
class GraphState(TypedDict):
    query: str
    subtasks: List[Dict[str, str]]
    results: List[Dict[str, str]]
    current_task: Dict[str, Any] | None

# Step 1: Plan Agent — splits user query into general subtasks
def plan_agent(state: GraphState) -> GraphState:
    query = state["query"]
    # You can replace this with a real LLM plan generation
    subtasks = [
        {"task": f"Understand and analyze: {query}"},
        {"task": f"Break down requirements of: {query}"},
        {"task": f"Suggest implementation plan for: {query}"}
    ]
    return {
        "query": query,
        "subtasks": subtasks,
        "results": [],
        "current_task": None
    }

plan_node = RunnableLambda(plan_agent)

# Step 2: Tool Agent — simulates or solves each subtask
def tool_agent(state: GraphState) -> GraphState:
    subtasks = state["subtasks"]
    results = state.get("results", [])

    if len(results) < len(subtasks):
        task = subtasks[len(results)]["task"]
        result = f"Simulated result for: {task}"
        results.append({"task": task, "result": result})

    return {
        "query": state["query"],
        "subtasks": subtasks,
        "results": results,
        "current_task": state.get("current_task")
    }

tool_node = RunnableLambda(tool_agent)

# Optional: Reflection node (currently stubbed)
def reflect(input: dict) -> dict:
    return {"feedback": "Looks good", "new_task": {"task": "Verify correctness"}}

reflect_node = RunnableLambda(reflect)

# Optional: Refine plan with new task
def refine_plan(input: dict) -> dict:
    subtasks = input["subtasks"]
    if input.get("new_task"):
        subtasks.append(input["new_task"])
    return {"subtasks": subtasks}

refine_node = RunnableLambda(refine_plan)

# Conditional logic
def should_continue(state: GraphState) -> str:
    return "tool" if len(state["results"]) < len(state["subtasks"]) else END

# Build LangGraph
builder = StateGraph(GraphState)

builder.add_node("plan", plan_node)
builder.add_node("tool", tool_node)

builder.set_entry_point("plan")
builder.add_edge("plan", "tool")
builder.add_conditional_edges("tool", should_continue)

builder.set_finish_point("tool")

graph = builder.compile()

# 🔹 Prompt user for input
user_query = input("📝 Enter your query: ")

initial_state = {
    "query": user_query,
    "subtasks": [],
    "results": [],
    "current_task": None
}

output = graph.invoke(initial_state)

# ✅ Final Output Display
print("\n📌 Subtasks:")
for task in output["subtasks"]:
    print(f"- {task['task']}")

print("\n✅ Final Results by Subtask:")
for r in output["results"]:
    print(f"🔹 {r['task']} → {r['result']}")
