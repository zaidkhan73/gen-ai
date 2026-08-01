from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    PydanticOutputParser
)
from pydantic import BaseModel, Field

load_dotenv()

print("HuggingFace API Key:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", # this model cannot give structured output
    task="text-generation",
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(
    llm=llm
)

class Person(BaseModel):
    name: str = Field( description="The name of the person")
    age: int = Field( gt=18,description="The age of the person")
    city: str = Field( description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me the name, age and city of a fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"place": "Indian"})

model_output = model.invoke(prompt)

result = parser.parse(model_output.content)

print("Final Result:\n", result)