from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

from models.embedding_model import embedding_model

pdf_paths = [
    'documents/workflow_admin_manual.pdf',
    'documents/dms_admin_manual.pdf'
]

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(full_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    return chunks


def store_vector_db(text_chunks):
    vector_store = FAISS.from_texts(
        texts=text_chunks, 
        embedding=embedding_model
    )
    vector_store.save_local("faiss_db")


def main():
    print('Processing...\n')
    raw_text = get_pdf_text(pdf_docs=pdf_paths)
    text_chunks = get_text_chunks(full_text=raw_text)
    store_vector_db(text_chunks=text_chunks)
    print('Done...')



if __name__ == "__main__":
    main()