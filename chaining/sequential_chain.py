from models.llm_model import gemini_model

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1 = PromptTemplate.from_template(
    'Generate a detailed report on {topic}'
)

prompt2 = PromptTemplate.from_template(
    'Generate a 5 pointer summary from the following text\n {text}'
)

parser = StrOutputParser()

chain = prompt1 | gemini_model | parser | prompt2 | gemini_model | parser

while True:
    topic = input('>> ')
    result = chain.invoke({'topic': topic})
    print("Answer: ", result)