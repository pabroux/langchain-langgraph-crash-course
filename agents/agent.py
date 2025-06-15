import os
from typing import Annotated

from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

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


# Stream
def stream_graph_updates(user_input: str):
    for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
