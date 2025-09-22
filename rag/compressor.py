import re
from langchain_core.prompts import (ChatPromptTemplate, 
                                    SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA

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

# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# REPL loop
while True:
    try:
        query = input(">> ")
        query = query.replace("\n", '').strip()

        if not query:
            continue

        retriever = db.as_retriever()

        _extractor = LLMChainExtractor.from_llm(gemini_model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_extractor, base_retriever=retriever
        )

        print("\nWait for answer...\n")

        qa_chain = RetrievalQA.from_chain_type(
            llm=gemini_model,
            retriever=compression_retriever,
            chain_type="stuff"
        )

        answer = qa_chain.invoke({"query": query})
        print(answer['result'])

    except KeyboardInterrupt:
        print("\nExiting...")
        break
