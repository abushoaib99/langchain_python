from models.llm_model import gemini_model

while True:
    query = input('>> ')
    response = gemini_model.invoke(query)
    print("Response: ", response.content)
