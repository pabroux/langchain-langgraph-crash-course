import os
from typing import Annotated

from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")
os.environ["TAVILY_API_KEY"] = input("Tavily API key: ")

llm = init_chat_model("openai:gpt-4o-mini")


# Memory
memory = MemorySaver() # Not to use in production. Use real DB instead
config = {"configurable": {"thread_id": "1"}} # Use a fixed memory


# Tools
tool = TavilySearch(max_results=2)
tools = [tool]
# ↳ Tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


# State
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# Nodes
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


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


# Stream first pass
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# Display history
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state
        to_replay = state


# Display the chosen past state
print(to_replay.next)
print(to_replay.config)


# Stream by resuming from the chosen past state
# ↳ The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
