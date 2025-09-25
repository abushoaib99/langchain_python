from models.llm_model import groq_model

resp = groq_model.invoke("Who invented AC?")
print(resp.content)