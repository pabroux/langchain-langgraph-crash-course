import os
from typing import Annotated

from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

llm = init_chat_model("openai:gpt-4o-mini")


# Memory
memory = MemorySaver() # Not to use in production. Use real DB instead
config = {"configurable": {"thread_id": "1"}} # Use a fixed memory


# Tools
@tool
# ↳ Note that because we are generating a ToolMessage for a state update, we
#   generally require the ID of the corresponding tool call. We can use
#   LangChain's InjectedToolCallId to signal that this argument should not
#   be revealed to the model in the tool's schema
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state
    return Command(update=state_update)

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


# Nodes
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# Graph
graph_builder = StateGraph(State)

# ↳ The first argument is the unique node name
#   The second argument is the function or object that will be called whenever
#   the node is used
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
# ↳ Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
# ↳ The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
#   it is fine directly responding. This conditional routing defines the main agent loop
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph = graph_builder.compile(checkpointer=memory)


# Stream with human interaction required
user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# Display the next node
snapshot = graph.get_state(config)
print("Snapshot next node →", snapshot.next)


# Stream (resume) by passing a human interaction
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# Manually update the state (can be done at any stage)
# ↳ Can also be used to control human-in-the-loop workflows.
#   Use of the `interrupt` function is generally recommended
#   instead, as it allows data to be transmitted in a
#   human-in-the-loop interaction independently of state updates
graph.update_state(config, {"name": "LangGraph (library)"})
