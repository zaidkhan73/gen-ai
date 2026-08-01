from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)

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

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about the following topic \n {format_instructions}',
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"topic": "The impact of climate change on agriculture."})

result = model.invoke(prompt)

print(result.content)      # optional, dekhne ke liye model ne kya return kiya

final_result = parser.parse(result.content)

print(final_result)

# with structured output parser, you can enforce a schema for the output of the model. The model will return the output in the form of a dictionary with keys as defined in the schema.
# but you can not perform validation of the output of the model with structured output parser. If you want to perform validation of the output of the model, you can use pydantic models instead of response schemas.