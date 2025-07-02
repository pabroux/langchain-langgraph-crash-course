import os 

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")


# Using LLM
# ↳ LangChain supports many different language models that you 
#   can use interchangeably. ChatModels are instances of 
#   LangChain Runnables, which means they expose a standard
#   interface for interacting with them
model = init_chat_model("gpt-4o-mini", model_provider="openai")


# ↳ To simply call the model, you can pass in a list of messages
#   to the `.invoke` method
#   ↳ In addition to text content, message objects convey
#     conversational roles and hold important data, such as tool
#     calls and token usage counts
#   ↳ LangChain supports chat model inputs via strings or 
#     provider (e.g. OpenAi) format. The following are equivalent:
#     ```python
#     model.invoke("Hello")
#     model.invoke([{"role": "user", "content": "Hello"}])
#     model.invoke([HumanMessage("Hello")])
#     ```
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

#print(model.invoke(messages))

# ↳ Because chat models are Runnables, they expose a standard
#   interface that includes async and streaming modes of
#   invocation. This allows us to stream individual tokens from
#   a chat model
for token in model.stream(messages):
   print(token.content, end="|")


# Prompt template
# ↳ Allow us to create a prompt template for the chat model instead
#   of passing in a list of messages directly
system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# ↳ The input of the prompt template is dict
prompt = prompt_template.invoke({"language":"Italian", "text":"hi!"})

# ↳ As you can see, we get again a list of messages (i.e.
#   SystemMessage and HumanMessage)
print(prompt)

# ↳ If you want to access the messages directly, use the 
#   `to_messages` method
print(prompt.to_messages())

# ↳ Now invoke the chat model on the formatted prompt.
output = model.invoke(prompt)

# ↳ You can use the StrOutputParser to parse the output
parser = StrOutputParser()
print(parser.parse(output))

# ↳ All the previous steps can be chained together in a short way
chain = prompt_template | model | parser
print(chain.invoke({"language": "Italian", "text": "hi!"}))
