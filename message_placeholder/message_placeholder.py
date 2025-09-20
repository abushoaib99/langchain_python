from langchain_core.prompts import (ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)


SYSTEM_MESSAGE = 'You are a helpful customer support agent'

# chat template
system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE)
message_placeholder = MessagesPlaceholder(variable_name="chat_history")
human_message_prompt = HumanMessagePromptTemplate.from_template("{query}")

prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    message_placeholder,
    human_message_prompt
])

# load chat history

chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


# create prompt
prompt = prompt.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

print(prompt)