from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from models.embedding_model import embedding_model

pdf_path = 'documents/attention.pdf'

# Load PDF file
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Splite the documents into chunck
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk_documents = text_splitter.split_documents(documents=docs)

# Vector Store
db = FAISS.from_documents(
    documents=chunk_documents,
    embedding=embedding_model,
)

# Save to local folder
db.save_local(folder_path="faiss_db")