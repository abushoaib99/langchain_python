from decouple import config

from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY =config("GEMINI_API_KEY")

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)