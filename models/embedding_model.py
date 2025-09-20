# from decouple import config

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# GEMINI_API_KEY =config("GEMINI_API_KEY")

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001",
#     google_api_key=GEMINI_API_KEY
# )

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")