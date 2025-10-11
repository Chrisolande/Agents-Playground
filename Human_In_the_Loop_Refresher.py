from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

_ = load_dotenv()


@lru_cache(maxsize=None)
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Cannot calculate the factorial of a negative number")
    return n * factorial(n - 1) if n else 1

@tool
def calculate_factorial(n: int) -> int:
    """Calculate the factorial of n

    Args:
        n: Value to calculate the factorial of

    Returns:
        int: Result of the factorial calculation
    """
    return factorial(n)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply the input a b y b

    Args:
        a: First number to multiply
        b: Second number to multiply

    Returns:
        int: Result of the multiplication
    """
    return a * b

tools = [calculate_factorial, multiply]

model = ChatDeepSeek(model="deepseek-reasoner", temperature=1.7)
model_with_tools = model.bind_tools(tools)

def math_llm(state):
    system_message = SystemMessage(
            content="You are a helpful math assistant that explains what it does."
    )

    full_messages = [system_message] + state["messages"]

    response = model_with_tools.invoke(full_messages)

    return {"messages": state["messages"] + [response]}


def human_approval(state):
    tool_call = state["messages"][-1].tool_calls[0] if state["messages"][-1].tool_calls else None

    if tool_call:
        approval_message = f"Tool: {tool_call['name']}\nArgs: {tool_call['args']}\n\nApprove? (yes/no)"
    else:
        approval_message = "No tool call to approve"

    response = interrupt(approval_message)

    if response.lower() in ["yes", "y", "approve"]:
        return Command(goto="tools")
    else:
        return Command(goto=END)


graph = StateGraph(MessagesState)
graph.add_node("math_llm", math_llm)
graph.add_node("human_approval", human_approval)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("math_llm")
graph.add_edge("tools", "math_llm")
graph.add_edge("math_llm", "human_approval")
graph.add_conditional_edges("human_approval", lambda x: x)

agent = graph.compile()