from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("Enter your prompt:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: " + response.content)

print(chat_history)

# types of messages in langchain : system, human, ai, message