import os
from typing import Annotated, List, Literal

import bs4  # BeauifulSoup to parse HTML
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict

# LangChain Hub is a centralized platform for uploading,
# browsing, pulling, and managing prompts to help developers
# discover and share polished prompt templates for various large
# language models (LLMs)
from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Vector store
vector_store = InMemoryVectorStore(embeddings)


# Load and chunk contents of the blog
loader = WebBaseLoader(  # A document loader for web pages
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )  # Customize HTML parsing
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


# Add documents to the vector store
_ = vector_store.add_documents(documents=all_splits)


# Define prompt for question-answering
# ↳ You can pull prompts from LangChain Hub with `hub.pull`
prompt = hub.pull("rlm/rag-prompt")


# Define the state for the application
# ↳ The state is typically a Python `TypedDict` but can also
#   be a `pydantic` model
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


# Define application steps
# ↳ The retrieve step retrieves documents from the vector store
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# ↳ The generate step generates an answer
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # The two following instructions could be rewritten using
    # the LCEL (LangChain Expression Language)
    # ```python
    # chain = prompt | llm
    # response = chain.invoke({"question": state["question"],
    #                          "context": docs_content})
    # ```
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application
graph_builder = StateGraph(State)

# ↳ `add_sequence` adds a list of nodes to the graph that
#   will be executed sequentially in the order provided
graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Test it out
# ↳ LangGraph supports multiple invocation modes, including
#   sync, async and streaming. Here is with the stream tokens:
for message, metadata in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="messages"
):
    print(message.content, end="|")


# Query analysis (self-querying)
# ↳ Query analysis employs models to transform or construct
#   optimized search queries from raw user input. For illustrative
#   purposes, let's add some metadata to the documents in your
#   vector store. You will add some (contrived) sections to the
#   document which you can filter on later
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# ↳ Update the vector store
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)


# ↳ Define a schema for your search query
#   ↳ You will use structured output for this purpose (i.e. using
#     function call mode or JSON mode underneath). Here you
#     define a query as containing a string query and a document
#     section (either "beginning", "middle", or "end")
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]


# ↳ Update the state
class State(TypedDict):
    question: str
    query: Search
    context: list[Document]
    answer: str


# ↳ Add a new application step
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


# ↳ Update the retrieve step
def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


# ↳ Recompile the application
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# ↳ Test it out
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")
