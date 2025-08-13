# 🦜🔗 LangChain LangGraph Crash Course

<p align="left">
  <a href="https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/LICENSE">
    <picture>
      <img src="https://img.shields.io/badge/License-MIT-green" alt="License Badge">
    </picture>
  </a>
  <a href="https://pixi.sh">
    <picture>
      <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json" alt="Pixi Badge">
    </picture>
  </a>
  <a href="https://github.com/pabroux/langchain-langgraph-crash-course/actions/workflows/ci.yml">
    <picture>
      <img src="https://github.com/pabroux/langchain-langgraph-crash-course/actions/workflows/ci.yml/badge.svg" alt="CI Badge">
    </picture>
  </a>
</p>

LangChain LangGraph Crash Course (LLCC) is a hands-on course designed to teach you [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) from the ground up. In this course, you will learn how to build Retrieval-Augmented Generation (RAG) workflows, intelligent agents doing web searches, text classifiers, and much more.

> [!NOTE] 
> Unlike official tutorials, LLCC provides pure Python scripts instead of Jupyter or Colab notebooks, delivering clean, ready-to-run code. Each script is extensively commented for complete clarity and ease of understanding. Additionally, LLCC sometimes includes extra steps not found in official tutorials, such as LCEL.

## Table of contents

- [Requirements](#requirements)
- [Install](#install)
- [Cost](#cost)
- [Course](#course)
- [Resources](#resources)

## Requirements

If you want to run the examples, you will need to install the following:

- [Pixi](https://pixi.sh)

You'll also need to have an [OpenAI key](https://platform.openai.com/settings/organization/api-keys).

## Install 

Inside the repository, install the dependencies as follows:
```shell
pixi install -a
```

## Cost

All examples utilize the OpenAI API, specifically employing the `gpt-4o-mini` model as the language model and/or the `text-embedding-3-large` model for embeddings. You can access these services under the free tier, which typically incurs no cost.

## Course

> [!TIP]
> It's recommended to follow the course in the given order.

### 1. LangChain

Familiarize yourself with LangChain components by building simple applications.

| Task | Description | File |
|------|-------------|------|
| Chat models & prompts | Simple LLM application with prompt templates and chat models | [simple_llm_application.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langchain/simple_llm_application.py) |
| Semantic search | Search over a PDF with document loaders, embedding models and vector stores | [semantic_search_engine.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langchain/semantic_search_engine.py) |
| Classification | Classify text into tags using chat models with structured outputs | [text_classifier.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langchain/text_classifier.py) |
| Extraction | Extract structured data from text using chat models and few-shot examples | [data_extractor.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langchain/data_extractor.py) |

### 2. LangGraph

Use LangGraph to assemble and orchestrate LangChain components into full-featured applications.

> [!NOTE] 
> LangGraph is not required to build an agent or a RAG application. You can implement each through invocations of only LangChain components.

#### 2.1 Agents

| Task | Description | File |
|------|-------------|------|
| Agent | Simple agent with a memory and able to do web searches | [agent.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langgraph/agents/agent.py) |
| Agent & human in the loop | Agent empowered to request a human intervention | [agent_human_assitance.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langgraph/agents/agent_human_assitance.py) | 
| Agent & time travel | Altering an agent output by changing its memory | [agent_time_travel.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langgraph/agents/agent_time_travel.py) |

#### 2.2 Retrieval augmented generation (RAG)

| Task | Description | File |
|------|-------------|------|
| RAG | Simple RAG with an introduction to self-query (an advanced RAG technique) | [rag.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langgraph/rag/rag.py) | 
| RAG & conversations | Delegating (multi-step) RAG calls to a LLM | [rag_delegation.py](https://github.com/pabroux/langchain-langgraph-crash-course/blob/master/langgraph/rag/rag_delegation.py) |

## Resources

Here are the resources I highly recommend.

### LLM

- [LLM course](https://github.com/mlabonne/llm-course?tab=readme-ov-file#-the-llm-scientist)
- [LLM's Engineer Handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/?_encoding=UTF8&pd_rd_w=TdT64&content-id=amzn1.sym.46807d81-91bd-498b-9732-d523d8e7a752%3Aamzn1.symc.fc11ad14-99c1-406b-aa77-051d0ba1aade&pf_rd_p=46807d81-91bd-498b-9732-d523d8e7a752&pf_rd_r=ZRWE6KNJ1MWQT6JCGNQQ&pd_rd_wg=F82Rn&pd_rd_r=d1fd6111-7922-469e-8bb0-e0a31dd91141&ref_=pd_hp_d_atf_ci_mcx_mr_ca_hp_atf_d)

### Agent
- [Agent course](https://huggingface.co/learn/agents-course)

### LangChain

- [LangChain tutorials](https://python.langchain.com/docs/tutorials/)
- [LangChain expression language (LCEL)](https://python.langchain.com/docs/versions/migrating_chains/llm_chain/#legacy)

### LangGraph

- [LangGraph tutorials](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
