import re
from langchain_core.prompts import (ChatPromptTemplate, 
                                    SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from models.embedding_model import embedding_model

from langchain_community.vectorstores import FAISS


from models.llm_model import gemini_model

SYSTEM_MESSAGE = """You are a helpful assistant.
Answer the following question based only on the providied context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
"""

# Collapse multiple whitespaces into one and strip leading/trailing spaces
SYSTEM_MESSAGE = re.sub(r"\s+", " ", SYSTEM_MESSAGE).strip()

system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE)
human_message_prompt = HumanMessagePromptTemplate.from_template("{asked}")

prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
])

parser = StrOutputParser()

db = FAISS.load_local(
    folder_path="faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)


# REPL loop
while True:
    try:
        query = input(">> ")

        print("\nWait for answer...\n")

        retriever = db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 1}
        )

        db_response = retriever.invoke(query)

        if not db_response:
            print("No context found in DB. Try another query.\n")
            continue

        content1 = db_response[0].page_content
        # content2 = db_response[1].page_content
        # content3 = db_response[2].page_content
        # content4 = db_response[3].page_content
        # context = f"{content1}{content2}{content3}{content4}"
        context = f"{content1}"
        
        print("db_response->>:\n", context)
        
        # Chaining
        chain = prompt | gemini_model | parser

        response = chain.invoke({
            "context": context,   # retrieved docs
            "asked": query        # for human_message_prompt
        })

        print("\n\nAnswer::\n", response, "\n")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
