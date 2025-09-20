from models.embedding_model import embedding_model

from langchain_community.vectorstores import FAISS

db = FAISS.load_local(
    folder_path="faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

while True:
    query = input('>> ')
    if not query.strip():
        continue

    retrieve_result = db.similarity_search(query=query)
    print(retrieve_result[0].page_content)
