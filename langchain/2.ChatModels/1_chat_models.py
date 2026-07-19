from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


# Initialize the Gemini chat model
# It automatically picks up your GOOGLE_API_KEY environment variable
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Invoke the model with a prompt
response = llm.invoke("What are three core benefits of using LangChain?")
print(response.content)