from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple words, what is {topic}?')
])

prompt = chat_template.invoke({
    'domain': 'cricket','topic': 'Duckworth-Lewis method'})

print(prompt)