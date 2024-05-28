import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage, ToolMessage




from langchain.globals import set_verbose, set_debug

# set_verbose()
# set_debug(True)

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]



# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [Add, Multiply]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

llm_with_tools = llm.bind_tools(tools)


query = "What is 3 * 12? Also, what is 11 + 49?"

llm_with_tools.invoke(query)
# content='' additional_kwargs={'tool_calls': [{'id': 'call_PGvECQm0Vt3JnSHZ7gbE3xhD', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'Multiply'}, 'type': 'function'}, {'id': 'call_eQZBqmm15IrHYPgf4VhDFXDi', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'Add'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 105, 'total_tokens': 155}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None} id='run-12d0f8da-125b-4106-9d2b-b7e0e674a9fb-0' tool_calls=[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_PGvECQm0Vt3JnSHZ7gbE3xhD'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_eQZBqmm15IrHYPgf4VhDFXDi'}]

# llm_with_tools.invoke(query).tool_calls
# [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_HU30ioaV4keYcg6b0xs8Aytf'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_4VrCiI7hVrG6bucEye0GdlcI'}]


## Messages example

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
messages
# [HumanMessage(content='What is 3 * 12? Also, what is 11 + 49?'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Zq24VOwEMNV7KgwSVFe9ZrSo', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'Multiply'}, 'type': 'function'}, {'id': 'call_jR3Ht6Yv8CIKfxGKfQJI9pCV', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'Add'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 105, 'total_tokens': 155}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-ea379595-99b4-4215-b4e2-345867614916-0', tool_calls=[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_Zq24VOwEMNV7KgwSVFe9ZrSo'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_jR3Ht6Yv8CIKfxGKfQJI9pCV'}]), ToolMessage(content='36', tool_call_id='call_Zq24VOwEMNV7KgwSVFe9ZrSo'), ToolMessage(content='60', tool_call_id='call_jR3Ht6Yv8CIKfxGKfQJI9pCV')]

llm_with_tools.invoke(messages)
# content='3 * 12 = 36\n\n11 + 49 = 60' response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 171, 'total_tokens': 187}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-8be4571b-1cd9-4bdd-8941-59eb8111b5ab-0'


## Few-shot learning
llm_with_tools.invoke(
    "Whats 119 times 8 minus 20. Don't do any math yourself, only use tools for math. Respect order of operations"
).tool_calls
# [{'name': 'Multiply', 'args': {'a': 119, 'b': 8}, 'id': 'call_9bZGA5Dojs8ovKtoQxZmzvLP'}, {'name': 'Add', 'args': {'a': 952, 'b': -20}, 'id': 'call_FTojj3FtRK7rYtatIYx80bkS'}]




examples = [
    HumanMessage(
        "What's the product of 317253 and 128472 plus four", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "Multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "Add", "args": {"x": 16505054784, "y": 4}, "id": "2"}],
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant",
    ),
]

system = """You are bad at math but are an expert at using a calculator. 

Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
print(chain.invoke("Whats 119 times 8 minus 20").tool_calls)

