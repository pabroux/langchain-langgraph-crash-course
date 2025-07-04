import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Tagging
# ↳ Tagging has a few components:
#    - `function`: Tagging relies on function call mode of
#      supported LLMs to specify how to tag a document. The
#      function signature (based on the shema) is sent to the LLM
#      which then returns structured output accordingly;
#    - `schema`: defines the structure and fields you want to extract
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


# ↳ Let's specify a Pydantic model with a few properties and
#   their expected type in your schema
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# ↳ Prompt the model to return the structured data as a function call
structured_llm = llm.with_structured_output(Classification)

ipt = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": ipt})
response = structured_llm.invoke(prompt)

print(response)

# ↳ If you want dictionary output, use the `model_dump` method
print(response.model_dump())


# Finer control
# ↳ Careful schema definition gives us more control over the model's
#   output. Specifically, you can define:
#    - Possible values for each property;
#    - Description to make sure that the model understands the
#      property;
#    - Required properties to be returned.
#   Without it the results vary so that you may get, for example,
#   sentiments in different languages ('positive', 'enojado' etc.)
# ↳ Let's redeclare your schema with more fine-grained control
class Classification(BaseModel):
    sentiment: str = Field(
        description="The sentiment of the text", enum=["happy", "neutral", "sad"]
    )
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        description="The language the text is written in",
        enum=["spanish", "english", "french", "german", "italian"],
    )


# ↳ Now, let's prompt the model again
structured_llm = llm.with_structured_output(Classification)
prompt = tagging_prompt.invoke({"input": ipt})
response = structured_llm.invoke(prompt)

print(response)
