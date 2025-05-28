!pip install -q langsmith langchain openai --upgrade
import os

os.environ["langsmith_01"] = "Your_API_KEY"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Agentic-Workflow"
from langchain_core.runnables import RunnableLambda

def plan_agent(state: GraphState) -> GraphState:
    query = state["query"]
    subtasks = [
        {"task": "Search IPO UI ideas"},
        {"task": "Choose color and layout"},
        {"task": "Recommend features"}
    ]
    return {
        "query": query,
        "subtasks": subtasks,
        "results": []
    }

plan_node = RunnableLambda(plan_agent)


def tool_agent(state: GraphState) -> GraphState:
    subtasks = state["subtasks"]
    results = state.get("results", [])

    if len(results) < len(subtasks):
        task = subtasks[len(results)]["task"]

        # Simulate or generate a response
        if "search" in task.lower():
            result = "Use a clean layout with tables for listing IPOs."
        elif "layout" in task.lower():
            result = "Choose a modern color scheme like blue/white with sidebar navigation."
        elif "feature" in task.lower():
            result = "Include filters by industry/date and allow bookmarking IPOs."
        else:
            result = f"Simulated result for: {task}"

        results.append({"task": task, "result": result})

    return {
        "query": state["query"],
        "subtasks": subtasks,
        "results": results
    }

tool_node = RunnableLambda(tool_agent)


def reflect(input: dict) -> dict:
    return {"feedback": "Looks good", "new_task": {"task": "Verify correctness"}}

reflect_node = RunnableLambda(reflect)

def refine_plan(input: dict) -> dict:
    subtasks = input["subtasks"]
    if input.get("new_task"):
        subtasks.append(input["new_task"])
    return {"subtasks": subtasks}

refine = RunnableLambda(refine_plan)
plan_node = RunnableLambda(plan_agent)
tool_node = RunnableLambda(tool_agent)
reflect_node = RunnableLambda(reflect)
refine_node = RunnableLambda(refine_plan)
# ipython-input-12-bf804c3a31d2
from langgraph.graph import END, StateGraph
from typing import TypedDict, List, Dict

class GraphState(TypedDict):
    query: str
    subtasks: List[Dict[str, str]]
    results: List[Dict[str, str]]  # changed from List[str]

 # Assuming results are strings
    # Add a field to hold the current task being processed
    current_task: Dict[str, Any] | None


# Define new functions to manage the task flow
def get_next_task(state: GraphState) -> Dict[str, Any]:
    """Selects the next task from the subtasks list."""
    subtasks = state.get("subtasks", [])
    if subtasks:
        # Take the first task and update the state to reflect that it's being processed
        next_task = subtasks.pop(0) # Modify subtasks in state (important for graph state updates)
        return {"current_task": next_task, "subtasks": subtasks}
    else:
        # No more tasks
        return {"current_task": None}

def process_task(state: GraphState) -> Dict[str, Any]:
    """Processes the current task using the tool_agent."""
    current_task = state.get("current_task")
    if current_task:
        # The tool_agent expects {"task": "task description"}
        processed_result = tool_agent(current_task)
        # Return the result to be added to the state
        return {"results": state.get("results", []) + [processed_result["result"]]}
    else:
        # Should not happen if routed correctly, but handle defensively
        return {"results": state.get("results", [])}

def should_continue(state: GraphState) -> str:
    """Determines if there are more subtasks or if reflection added a new one."""
    # Continue if there are still subtasks to process or if a new task was added
    # The get_next_task node already modified the subtasks list in the state when it took a task.
    # We need to check if the list is empty *after* a task was potentially taken.
    # Also check if reflect added a new task via refine_plan which would be in subtasks.
    return "GetNextTask" if state.get("subtasks") or state.get("new_task") else END


# Initialize StateGraph with the defined schema class
from langgraph.graph import StateGraph

builder = StateGraph(GraphState)

builder.add_node("plan", plan_node)
builder.add_node("execute", tool_node)

# Edges
builder.set_entry_point("plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges(
    "execute",
    lambda state: "end" if len(state["results"]) == len(state["subtasks"]) else "execute"
)

builder.set_finish_point("execute")

graph = builder.compile()
user_query = input("📝 Enter your query: ")

initial_state = {
    "query": user_query,
    "subtasks": [],
    "results": []
}

output = graph.invoke(initial_state)

print("\n✅ Final Output:")
for r in output["results"]:
    print(r)

print("\n📌 Subtasks:")
for task in output["subtasks"]:
    print(f"- {task['task']}")
print("\n✅ Final Results by Subtask:")
for r in output["results"]:
    print(f"🔹 {r['task']} → {r['result']}")
