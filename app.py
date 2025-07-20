import json
import os
import uuid

import streamlit as st

# LangChain and LangGraph imports
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
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

        workflow = StateGraph(MessagesState)
        model_with_tools = model.bind_tools(tools)

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


def generate_verification_message(message: AIMessage):
    serialized_tool_calls = json.dumps(message.tool_calls, indent=2)
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n\n"
            "Type 'yes' if you do, anything else to stop.\n\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )


def render_api_key_input():
    st.markdown("### :key: API Configuration")
    api_key_exists = bool(os.getenv("OPENAI_API_KEY")) and bool(
        os.getenv("TAVILY_API_KEY")
    )
    if api_key_exists:
        st.success(":white_check_mark: API keys found in environment")
    else:
        st.warning(":warning: API keys not found in environment")

        api_key = st.text_input(
            "Openrouter API Key",
            type="password",
            help="Enter your openrouter API Key",
        )

        tavily_key = st.text_input(
            "Tavily API Key", type="password", help="Enter your Tavily API Key"
        )

        if api_key and tavily_key:
            os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"] = api_key
            os.environ["TAVILY_API_KEY"] = tavily_key
            st.session_state.app = initialize_app()
            st.success(":white_check_mark: API keys set successfully")


def stream_app_catch_tool_calls(inputs, thread, app):
    """Stream app, catching tool calls."""
    tool_call_message = None
    messages_to_display = []

    try:
        for event in app.stream(inputs, thread, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_call_message = message
            messages_to_display.append(message)

        # Display the messages
        for msg in messages_to_display:
            if isinstance(msg, AIMessage) and msg.content:
                st.session_state.conversation_history.append(
                    f"Assistant: {msg.content}"
                )
            elif isinstance(msg, ToolMessage) and msg.content:
                st.session_state.conversation_history.append(
                    f"Tool Result: {msg.content}"
                )

    except Exception as e:
        st.error(f"Error in stream: {str(e)}")

    return tool_call_message


def handle_approval(approval):
    """Handle user approval or rejection of tool execution."""
    try:
        verification_message = generate_verification_message(
            st.session_state.tool_call_message
        )
        input_message = HumanMessage(approval)

        # Get current state
        snapshot = st.session_state.app.get_state(st.session_state.thread)
        snapshot.values["messages"] += [verification_message, input_message]

        if approval.lower() == "yes":
            # Generate new ID for tool call message
            st.session_state.tool_call_message.id = str(uuid.uuid4())
            snapshot.values["messages"] += [st.session_state.tool_call_message]
            st.session_state.app.update_state(
                st.session_state.thread, snapshot.values, as_node="agent"
            )

            with st.spinner("Executing tool..."):
                tool_call_message = stream_app_catch_tool_calls(
                    None, st.session_state.thread, st.session_state.app
                )

            if tool_call_message:
                st.session_state.tool_call_message = tool_call_message
                st.rerun()
            else:
                st.success("Tool executed successfully!")
                reset_state()
        else:
            st.session_state.app.update_state(
                st.session_state.thread, snapshot.values, as_node="__start__"
            )
            st.warning("Tool execution rejected. You can ask a new question.")
            reset_state()

    except Exception as e:
        st.error(f"Error handling approval: {str(e)}")
        st.warning("Please try asking a new question.")
        reset_state()


def reset_state():
    """Reset the approval state."""
    st.session_state.waiting_for_approval = False
    st.session_state.tool_call_message = None
    try:
        st.rerun()
    except AttributeError:
        st.rerun()


def main():
    st.title(":robot: Human-in-the-Loop Agent")
    st.markdown("This agent will ask for your approval before executing tools.")

    # Initialize app if not already done
    if st.session_state.app is None:
        with st.spinner("Initializing the agent..."):
            st.session_state.app = initialize_app()

    if st.session_state.app is None:
        st.error("Failed to initialize the agent. Please check your API keys.")
        return

    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for msg in st.session_state.conversation_history:
            st.write(msg)
        st.divider()

    # Handle different states
    if not st.session_state.waiting_for_approval:
        st.subheader("Ask a Question")
        user_input = st.text_input("Enter your question:", key="user_input")

        if st.button("Submit Question", type="primary"):
            if user_input:
                st.session_state.conversation_history.append(f"You: {user_input}")

                inputs = [HumanMessage(content=user_input)]

                with st.spinner("Agent is thinking..."):
                    tool_call_message = stream_app_catch_tool_calls(
                        {"messages": inputs},
                        st.session_state.thread,
                        st.session_state.app,
                    )

                if tool_call_message:
                    st.session_state.tool_call_message = tool_call_message
                    st.session_state.waiting_for_approval = True
                    try:
                        st.rerun()
                    except AttributeError:
                        st.rerun()
                else:
                    st.success("Response completed without tool calls!")

    else:
        # Handle waiting for approval state
        st.subheader("🛠️ Tool Approval Required")
        if st.session_state.tool_call_message:
            tool_calls = st.session_state.tool_call_message.tool_calls

            for i, tool_call in enumerate(tool_calls):
                st.info(f"**Tool {i+1}:** {tool_call['name']}")
                st.code(json.dumps(tool_call["args"], indent=2), language="json")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Approve", type="primary"):
                    handle_approval("yes")

            with col2:
                if st.button("❌ Reject", type="secondary"):
                    handle_approval("no")


# Sidebar for API key configuration
with st.sidebar:
    render_api_key_input()
    st.subheader("Controls")
    if st.button("🔄 Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
        reset_state()

    st.subheader("Setup")
    st.markdown(
        """
    **Required Environment Variables:**
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `TAVILY_API_KEY`: Your Tavily API key (for web search)

    **Available Tools:**
    - Web search (Tavily)
    - Add numbers
    - Multiply numbers
    """
    )

if __name__ == "__main__":
    main()
