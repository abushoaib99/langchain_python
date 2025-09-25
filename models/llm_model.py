from decouple import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

GEMINI_API_KEY =config("GEMINI_API_KEY")
GROQ_API_KEY = config("GROQ_API_KEY")

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

groq_model = ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=GROQ_API_KEY,
    temperature=0
)