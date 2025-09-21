import sys
import os

# Add the parent directory (langchain_python/) to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS

from models.llm_model import gemini_model
from models.embedding_model import embedding_model


db = FAISS.load_local(
    folder_path="faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

def get_qa_chain():

    prompt_message = """
    You are a helpful customer support assistant.
    Answer the user's question based only on the provided context.
    If the answer is not in the context, say: 
    "Sorry, I don't know the answer to that. Please contact support."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_messages([prompt_message])

    qa_chain = RetrievalQA.from_chain_type(
        llm=gemini_model, 
        retriever=db.as_retriever(), 
        chain_type="stuff", 
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


qa_chain = get_qa_chain()

def answer(query):    
    # Chaining
    answer = qa_chain.run(query)
    print('Answer->>\n', answer)
    return answer

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        output = answer(user_question)
        st.write("Reply: ", output)


if __name__ == "__main__":
    main()