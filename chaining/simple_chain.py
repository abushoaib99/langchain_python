from models.llm_model import gemini_model

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    'Generate 5 interseting facts about {topic}'
)

parser = StrOutputParser()

chain = prompt | gemini_model | parser

while True:
    topic = input('>> ')
    result = chain.invoke({'topic': topic})
    print("Answer: ", result)