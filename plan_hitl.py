from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from typing import Literal
from langgraph.types import interrupt, Command
from langgraph.graph import MessagesState, StateGraph

_ = load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)


class PlannerState(MessagesState):
    plan: str = ""
    human_action: Literal["approve", "reject", "modify", ""] = ""
    plan_approved: bool = False
    task: str


def create_plan_node(state: PlannerState):
    task = state.get("task", "")

    if not task or task.strip() == "":
        return Command(
            update={
                "plan": "ERROR: No task provided. Please provide a task in the input.",
                "messages": [AIMessage(content="ERROR: No task provided.")],
            },
            goto="human_review_node",
        )

    if state.get("human_action"):
        prompt = f"""TASK TO ACCOMPLISH: {task}

Previous plan:
{state.get('plan', 'No prior plan.')}

User feedback:
{state.get('human_action', 'No feedback provided.')}

Instructions:
1. Revise the plan based on user feedback.
2. Provide 3–7 numbered steps (each step must be ONE imperative sentence, ≤120 chars).
3. After the plan, list any "Changes" made (added/removed/reordered steps).
4. End with a one-sentence rationale.

Output format (exact):
<NUMBERED PLAN>
Changes: <summary of edits>
Rationale: <one sentence>
"""
    else:
        prompt = f"""TASK TO ACCOMPLISH: {task}

Create a specific plan for the task above. Do NOT create a generic project plan.

Requirements:
- Provide 4–8 numbered steps
- Each step must be ONE imperative sentence starting with a verb
- Include realistic time estimates (e.g., 1w, 3d, 2h)
- Steps must directly address the specific task
- End with one-line success criterion

Output format:
1. <step> (<duration>)
...
Success criterion: <outcome>"""

    messages = [
        SystemMessage(
            content=(
                "You are a professional planner assistant. "
                "Create plans that directly address the specific task provided. "
                "Do NOT create generic project management plans."
            )
        ),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)

    return Command(
        update={
            "plan": response.content,
            "messages": [AIMessage(content=f"Generated Plan:\n{response.content}")],
            "human_action": "",
            "plan_approved": False,
        },
        goto="human_review_node",
    )


def human_review_node(state: PlannerState):
    plan = state.get("plan", "No plan available.")
    feedback_prompt = f"""
Current Plan
--------------------------
{plan}

Actions:
[a] Approve
[r] Reject (generate new plan)
[m] Modify (provide feedback)
"""
    response = interrupt(feedback_prompt).strip().lower()

    if response in ["yes", "y", "approve", "a"]:
        return Command(update={"plan_approved": True}, goto="end_node")
    elif response in ["no", "r", "n", "reject"]:
        return Command(update={"human_action": "reject"}, goto="create_plan_node")
    elif response in ["modify", "m", "edit"]:
        feedback = interrupt("Enter your feedback for plan revision: ").strip()
        return Command(
            update={"human_action": "modify", "task": feedback}, goto="create_plan_node"
        )
    else:
        return Command(update={"human_action": "reject"}, goto="create_plan_node")


graph = StateGraph(PlannerState)
graph.add_node("create_plan_node", create_plan_node)
graph.add_node("human_review_node", human_review_node)
graph.add_edge("create_plan_node", "human_review_node")
graph.set_entry_point("create_plan_node")

agent = graph.compile()
