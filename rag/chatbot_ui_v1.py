import sys
import os

# Add the parent directory (langchain_python/) to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from langchain_core.prompts import (ChatPromptTemplate, 
                                    SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate)

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from models.llm_model import gemini_model
from models.embedding_model import embedding_model


db = FAISS.load_local(
    folder_path="faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

def get_conversational_chain():

    SYSTEM_MESSAGE = """You are a helpful assistant.
    Answer the following question based only on the providied context.
    Think step by step before providing a detailed answer. Make sure to 
    provide all the details, if the answer is not in provided context just say, 
    "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE)
    human_message_prompt = HumanMessagePromptTemplate.from_template("Question: {question}")

    prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt,
    ])

    chain = load_qa_chain(gemini_model, chain_type="stuff", prompt=prompt)

    return chain


chain = get_conversational_chain()


def answer(question):
    docs = db.similarity_search(query=question)
    output = "No context found in DB. Try another query."
    
    if docs:
        # Chaining
        response = chain.invoke(
            {
                "input_documents": docs,
                "question": question
            },
            return_only_outputs=True
        )
        print(response)
        output = response["output_text"]

    return output

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        output = answer(user_question)
        st.write("Reply: ", output)


if __name__ == "__main__":
    main()