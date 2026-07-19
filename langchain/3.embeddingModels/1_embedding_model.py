import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Grab your token from the .env file
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 2. Use HuggingFaceEndpointEmbeddings and pass the token directly
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token  # <-- Pass your token variable here
)

text = "delhi is capital of india"

# 3. Request the vector from the cloud API
vector = embedding.embed_query(text)

print(str(vector))