from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)
prompt = "Write a haiku about a server running out of memory."
response = llm.invoke(prompt)

print(response)