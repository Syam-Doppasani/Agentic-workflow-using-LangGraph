# Agentic workflow using LangGraph
This repository contains a Python example demonstrating a simple agentic workflow using LangGraph. The workflow orchestrates multiple steps to process a user query by breaking it down into subtasks, executing each subtask, and accumulating results.

# Features
Task Planning: Divides an initial query into a list of distinct subtasks.

Sequential Execution: Processes subtasks one by one.

Simulated Tool Use: Includes a placeholder tool_agent function to simulate external tool interactions and generate results for each subtask.

State Management: Leverages LangGraph's StateGraph to manage the workflow's state, including the original query, subtasks, and accumulated results.
# How it Works
The core of the workflow is built using langgraph.graph.StateGraph. The graph defines a sequence of nodes and edges:

plan node: Takes the initial query and generates a list of subtasks.

execute node: Processes the current task using a simulated tool and adds the result to the state. The graph then conditionally transitions back to the execute node if there are more tasks to process, or to the END state when all tasks are completed.
