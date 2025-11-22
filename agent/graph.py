import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from prompts import planner_prompt, architect_prompt, coder_system_prompt
from states import Plan, TaskPlan, CoderState
from tools import write_file, read_file, list_files, get_current_directory


# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------
#  LLM
# ---------------------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b")


# ---------------------------------------------------------------------
#  PLANNER → returns Plan schema
# ---------------------------------------------------------------------
def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]

    # Structured output: Plan
    response = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )
    if response is None:
        raise ValueError("Planner agent returned no structured output.")

    return {"plan": response}


# ---------------------------------------------------------------------
#  ARCHITECT → returns TaskPlan schema
# ---------------------------------------------------------------------
def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]

    # Pass the JSON of Plan
    response = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan.model_dump_json())
    )
    if response is None:
        raise ValueError("Architect agent failed to produce TaskPlan.")

    # Preserve original plan
    response.plan = plan

    return {"task_plan": response}


# ---------------------------------------------------------------------
#  CODER → uses ReAct tool agent to write files
# ---------------------------------------------------------------------
def coder_agent(state: dict) -> dict:
    # Get task_plan from state
    task_plan: TaskPlan = state.get("task_plan")
    
    if task_plan is None:
        raise ValueError("task_plan not found in state. Architect must run first.")
    
    coder_state: CoderState = state.get("coder_state")

    # Initialize coder_state if first run
    if coder_state is None:
        coder_state = CoderState(
            task_plan=task_plan,
            current_step_idx=0
        )

    steps = coder_state.task_plan.implementation_steps
    
    # Check if all steps are complete
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    
    # Try to read existing file content
    try:
        existing_content = read_file.invoke({"filepath": current_task.filepath})
    except Exception as e:
        existing_content = "File does not exist yet."

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"{system_prompt}\n\n"
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n\n"
        "Use the write_file tool to save your changes. Provide the complete file content."
    )

    tools = [read_file, write_file, list_files, get_current_directory]

    # Create ReAct agent
    agent = create_react_agent(llm, tools)

    # Invoke agent with the task
    try:
        agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    except Exception as e:
        print(f"Error in coder agent: {e}")

    # Move to next step
    coder_state.current_step_idx += 1
    
    # Return updated state - preserve task_plan
    return {
        "coder_state": coder_state,
        "task_plan": task_plan
    }


# ---------------------------------------------------------------------
#  BUILD GRAPH
# ---------------------------------------------------------------------
graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

# Conditional edge: loop coder until all steps done
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {
        "END": END,
        "coder": "coder"
    }
)

graph.set_entry_point("planner")
agent = graph.compile()


# ---------------------------------------------------------------------
#  RUN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting AI Agent Coder...")
    prompt = input('enter your prompt')
    result = agent.invoke(
        {"user_prompt": prompt},
        {"recursion_limit": 10}
    )
    print("\n" + "="*50)
    print("FINAL STATE:")
    print("="*50)
    print(f"Status: {result.get('status', 'IN PROGRESS')}")
    if result.get('coder_state'):
        print(f"Steps completed: {result['coder_state'].current_step_idx}")
    print("="*50)