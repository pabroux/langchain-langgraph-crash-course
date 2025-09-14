import os
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import tool_example_to_messages
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# Extract data from text
# ↳ In LangChain, extracting relies on function call mode (also called
#   tool calling) of supported LLM
#   ↳ JSON mode is also supported for LLMs that support it and is used
#     as a fallback if the LLM doesn't support function call mode. You
#     can even explicitly use JSON mode if you want to
# ↳ Create a schema defining the structure and fields you want to extract
#   ↳ Best practices when defining schema:
#      1. Document the attributes and the schema itself: This
#         information is sent to the LLM and is used to improve the
#         quality of information extraction;
#      2. Do not force the LLM to make up information! Below, you use
#         `Optional` for the attributes allowing the LLM to output
#         `None` if it doesn't know the answer
class Person(BaseModel):
    """Information about a person."""

    # The doc-string above is sent to the LLM as the description of
    # the schema Person, and it can help to improve extraction results

    # Note that:
    #  1. Each field is an `optional` -- this allows the model to
    #     decline to extract it!
    #  2. Each field has a `description` -- this description is used
    #     by the LLM
    # Having a good description can help improve extraction results
    name: str | None = Field(default=None, description="The name of the person")
    hair_color: str | None = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: str | None = Field(
        default=None, description="Height measured in meters"
    )


# ↳ Create a prompt to provide instructions and any additional context
#   ↳ Note that you can:
#      1. Add examples into the prompt template to improve extraction
#         quality (see the few-shot section below);
#      2. Introduce additional parameters to take context into account
#         (e.g. include metadata about the document from which the text
#         was extracted)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

# ↳ Prompt the model to return the structured data as a function call
structured_llm = llm.with_structured_output(schema=Person)

text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
result = structured_llm.invoke(prompt)

print(result)

# ↳ The previous instructions could be rewritten using the LCEL
#   (LangChain Expression Language)
chain = prompt_template | structured_llm
result = chain.invoke({"text": text})

print(result)


# Multiple entities
# ↳ In most cases, you should be extracting a list of entities
#   rather than a single entity. This can be easily achieved using
#   Pydantic by nesting models inside one another
#   ↳ Note that when the schema accommodates the extraction of multiple
#     entities, it also allows the model to extract no entities if
#     no relevant information is in the text by providing an empty
#     list
class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that you can extract multiple entities
    people: list[Person]


# ↳ Prompt the model to return the structured data as a function call
structured_llm = llm.with_structured_output(schema=Data)

text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
structured_llm.invoke(prompt)


# Reference examples (few-shot)
# ↳ LLM can be steered using few-shot prompting. For chat models, this
#   can take the form of a sequence of pairs of input and response
#   messages demonstrating desired behaviors. For example, we can convey
#   the meaning of a symbol with alternating `user` and `assistant`
#   messages:
messages = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 🦜 4"},
]

response = llm.invoke(messages)
print(response.content)

# ↳ Since different chat model providers impose different requirements
#   for valid message sequences, LangChain includes a utility function
#   `tool_example_to_messages` that will generate a valid sequence for
#   most model providers. It simplifies the generation of structured
#   few-shot examples by just requiring Pydantic representations of the
#   corresponding tool calls. You can convert pairs of input strings
#   and desired Pydantic objects to a sequence of messages that can be
#   provided to a chat model. Under the hood, LangChain will format the
#   tool calls to each provider's required format
examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

messages = []

for txt, tool_call in examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

for message in messages:
    message.pretty_print()  # inspects messages

# ↳ Test it out
message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}

structured_llm = llm.with_structured_output(schema=Data)

#   ↳ Without the few-shot examples, the model will fail
result = structured_llm.invoke([message_no_extraction])

print(result)

#   ↳ With the few-shot examples, the model will succeed
result = structured_llm.invoke(messages + [message_no_extraction])

print(result)
