import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

# Documents
# ↳ LangChain implements a `Document` abstraction, which
#   is intended to represent a unit of text and associated
#   metadata. 3 attributes:
#    - `page_content`: a string representing the content;
#    - `metadata`: a dict containing arbitrary metadata;
#    - `id` (opt): a string identifier for the document
# ↳ Note that an individual `Document` object often
#   represents a chunk of a larger document
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


# Document loaders
# ↳ LangChain supports many different `Document` loaders that
#   you can use to read and parse documents from different
#   sources (e.g. `AsyncHtmlLoader` for HTML files, `PyPDFLoader`
#   for PDF files, etc.)
loader = PyPDFLoader(
    str(Path(__file__).resolve().parent.parent) + "/data/nke-10k-2023.pdf"
)
docs = loader.load()  # Returns a list of `Document`, one per page


# Chunking documents
# ↳ For both information retrieval and downstream question-answering
#   purposes, a page may be too coarse as a representation. Chunking
#   helps ensure that the meanings of relevant portions of the
#   document are not "washed out" by surrounding text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,  # Add a `start_index` metadata to each chunk that indicates the start index of the chunk in the original document
)
all_splits = text_splitter.split_documents(
    docs
)  # Returns a list of `Document`, one per chunk


# Embeddings
# ↳ LangChain supports many different embedding models that
#   you can use to generate embeddings for your documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

print(f"Generated vectors of length {len(vector_1)}\n")


# Vector stores
# ↳ LangChain `VectorStore` objects contain methods for adding
#   text and `Document` objects to the store, and querying them
#   using various similarity metrics. They are often initialized
#   with embedding models, which determine how text data is translated
#   to numeric vectors
# ↳ LangChain supports many different vector stores that
#   you can use to store and retrieve embeddings
vector_store = InMemoryVectorStore(embeddings)

# ↳ Add documents to the vector store
ids = vector_store.add_documents(documents=all_splits)

# ↳ Query the vector store
#   ↳ `VectorStore` includes methods for querying:
#       - Synchronously and asynchronously;
#       - By string query and by vector;
#       - With and without returning similarity scores;
#       - By similarity and maximum marginal relevance (to balance
#         similarity with query to diversity in retrieved results).
#   ↳ Note that providers implement different scores. Below the
#     score is a distance metric that varies inversely with
#     similarity
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]

print(f"Score: {score}\n")
print(doc)

# ↳ You can also query by vector
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])


# Retrievers
# ↳ LangChain `VectorStore` objects do not subclass `Runnable`. LangChain
#   `Retrievers` are `Runnable`, so they implement a standard set of
#   methods (e.g. synchronous and asynchronous `invoke` and `batch`
#   operations). Although we can construct retrievers from vector stores,
#   retrievers can interface with non-vector store sources of data, as
#   well (such as external APIs)
# ↳ You can create a simple version of this yourselves, without subclassing
#   `Retriever`. If you choose what method you wish to use to retrieve
#   documents, you can create a runnable easily. Here is an example:
@chain
def retriever(query: str) -> list[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)


# ↳ `VectorStore` implement an `as_retriever` method that returns a
#   `Retriever` object, specifically a `VectorStoreRetriever`. These
#   include specific `search_type` and `search_kwargs` attributes that
#   identify what methods of the underlying vector store to call. The
#   following is a replicate of the above
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
