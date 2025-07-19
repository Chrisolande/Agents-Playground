import json
import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize session state
if "app" not in st.session_state:
    st.session_state.app = None
if "thread" not in st.session_state:
    st.session_state.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
if "waiting_for_approval" not in st.session_state:
    st.session_state.waiting_for_approval = False
if "tool_call_message" not in st.session_state:
    st.session_state.tool_call_message = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Define tools


@tool
def add_numbers(x: float, y: float) -> float:
    """Adds two floating-point numbers and returns their sum.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The sum of x and y.
    """
    return x + y


@tool
def multiply_numbers(x: float, y: float) -> float:
    """Multiplies two floating-point numbers and returns the product.

    Args:
        x (float): The first number.
        y (float): The second number.

    Returns:
        float: The product of x and y.
    """
    return x * y


def initialize_app():
    try:
        tools = [TavilySearch(max_results=1), add_numbers, multiply_numbers]
        model = ChatOpenAI(model="mistralai/mistral-small-3.2-24b-instruct")
        # model = model.bind_tools(tools)

        workflow = StateGraph(MessagesState)
        model_with_tools = model.bind_tools(
            tools
        )  # Tell the model which tools are available at its disposal

        workflow.add_node(
            "agent",
            lambda state: {"messages": [model_with_tools.invoke(state["messages"])]},
        )
        workflow.add_node("tools", ToolNode(tools))

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])
        return app

    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        return None


def generate_verification_message(message: AIMessage):  # type: ignore
    serialized_tool_calls = json.dumps(message.tool_calls, indent=2)
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n\n"
            "Type 'yes' if you do, anything else to stop.\n\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )


def stream_app_catch_tool_calls(inputs, thread, app):
    pass


def main():
    st.title("Hooman in the loop agent")
    st.markdown("This agent will ask for your approval before executing tools.")
    if st.session_state.app is None:
        with st.spinner("Initializing the agent ..."):
            st.session_state.app = initialize_app()

    if st.session_state.app is None:
        st.error("Failed to initialize the agent. Please check your API keys.")
        return

    if st.session_state.conversation_history is None:
        st.subheader("Conversation history")
        for msg in st.session_state.conversation_history:
            st.write(msg)
        st.divider()


def render_api_key_input():
    st.markdown("### :key: API Configuration")
    api_key_exists = bool(os.getenv("OPENAI_API_KEY")) and bool(
        os.getenv("TAVILY_API_KEY")
    )
    if api_key_exists:
        st.success(":white_check_mark: API key found in environment")
    else:
        st.warning(":warning: API key not found in environment")

        api_key = st.text_input(
            "Openrouter API Key",
            type="password",
            help="Enter your openrouter API Key",
        )

        tavily_key = st.text_input(
            "Tavily API Key", type="password", help="Enter your Tavily API Key"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"] = api_key
            os.environ["TAVILY_API_KEY"] = tavily_key
            st.session_state.app = (
                initialize_app()
            )  # Reinitialize the app with the new API key
            st.success(":white_check_mark: API key set successfully")


with st.sidebar:
    render_api_key_input()

if __name__ == "__main__":
    main()
