from langchain_core.prompts import (ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain_core.tools import tool
from models.llm_model import gemini_model


@tool
def add(a: float, b: float) -> float:
    """Returns the sum of two numbers"""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Returns the difference of two numbers"""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Returns the product of two numbers"""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Returns the division of two numbers"""
    if b == 0:
        return "Error: Division by zero"
    return a / b


# --- Collect all tools ---
tools = [add, subtract, multiply, divide]

llm_with_tools = gemini_model.bind_tools(tools=tools)

while True:
    query = input(">> ").replace("\n", "").strip()

    if not query:
        continue

    result = llm_with_tools.invoke(query)

    print("RESULT:::\n", result)

